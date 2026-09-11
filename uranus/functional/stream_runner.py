from dataclasses import dataclass
from typing import Any

import torch
from uranus.utils.video_tensor_utils import (
    preprocess_video,
    preprocess_videos,
)
from uranus.functional.text_encoder import encode_prompt
from uranus.functional.lightweight_skeleton import (
    build_lightweight_skeleton_reference_context,
    build_lightweight_skeleton_step_context,
)
from uranus.functional.stream_state import UranusStreamState
from uranus.functional.plucker import (
    build_reference_plucker_embeddings,
    build_step_plucker_embeddings,
)
from uranus.functional.dit_inference import (
    dit_prefill,
    dit_decode_step,
    dit_kv_cache_fill,
)
from uranus.functional.ropes import precompute_freqs_cis_3d
from uranus.functional.vae_decoder import decode_latents_chunk
from einops import rearrange, repeat


@dataclass
class UranusStreamConfig:
    num_cameras: int = 3
    height: int = 224
    width: int = 224
    num_inference_steps: int = 25
    sigma_shift: float = 5.0
    cfg_scale: float = 0.0
    uranus_scale: float = 1.0
    teacher_forcing_window_size: int = 8
    skeleton_mode: str = "lightweight"

def get_prompt_context(prompt, models, state, device, dtype):
    # Idempotent: skip re-encoding when this state already holds the same prompt
    # (step calls always pass the session's original prompt).
    if state.context is not None and state.prompt == prompt:
        return state
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    prompt_emb = encode_prompt(tokenizer, text_encoder, prompt, device=device)
    state.context = prompt_emb
    state.prompt = prompt
    return state


def get_ref_latents(reference_images, models, state, device, dtype):
    num_cameras = state.num_cameras
    vae = models["vae"]
    if not isinstance(reference_images, list):
        reference_images = [reference_images]
    n_ref = len(reference_images)
    reference_images = preprocess_video(reference_images).to(device=device, dtype=dtype)
    reference_images = rearrange(reference_images, "B C N_REF H W -> (B N_REF) C 1 H W")
    reference_latents = vae.encode(reference_images, device=device, tiled=True)
    reference_latents = rearrange(
        reference_latents, "(B N_REF) C 1 H W -> B C N_REF H W", N_REF=n_ref
    )
    reference_latents = repeat(
        reference_latents, "1 C N_REF H W -> 1 N_CAM C N_REF H W", N_CAM=num_cameras
    )
    return n_ref, reference_latents.to(device=device, dtype=dtype)


def get_ref_skeleton_latents(reference_skeleton_images, models, state, device, dtype):
    use_lightweight_skeleton_controller = state.skeleton_mode == "lightweight"
    num_cameras = state.num_cameras
    if use_lightweight_skeleton_controller:
        ref_imgs = (
            reference_skeleton_images
            if isinstance(reference_skeleton_images, list)
            else [reference_skeleton_images]
        )
        ref_imgs_t = preprocess_video(ref_imgs).to(device=device, dtype=dtype)
        reference_skeleton = rearrange(ref_imgs_t, "B C N_REF H W -> B 1 C N_REF H W")
        reference_skeleton = repeat(
            reference_skeleton,
            "B 1 C N_REF H W -> B N_CAM C N_REF H W",
            N_CAM=num_cameras,
        )
        reference_skeleton_latents = build_lightweight_skeleton_reference_context(
            reference_skeleton
        )
    else:
        _, reference_skeleton_latents = get_ref_latents(
            reference_skeleton_images, models, state, device, dtype
        )
    return reference_skeleton_latents


def get_ref_plucker_embeddings(
    reference_camera_extrinsics, reference_camera_intrinsics, state, device, dtype
):
    height = state.height
    width = state.width
    reference_plucker_embedding = build_reference_plucker_embeddings(
        reference_camera_extrinsics=reference_camera_extrinsics,
        reference_camera_intrinsics=reference_camera_intrinsics,
        num_cameras=state.num_cameras,
        height=height,
        width=width,
        device=device,
    ).to(device=device, dtype=dtype)
    return reference_plucker_embedding


def get_plucker_embeddings(camera_extrinsics, camera_intrinsics, state, device, dtype):
    height = state.height
    width = state.width
    plucker_embedding = build_step_plucker_embeddings(
        camera_extrinsics,
        camera_intrinsics,
        height,
        width,
        device,
    ).to(device=device, dtype=dtype)
    return plucker_embedding


def get_skeleton_latents(skeleton_images, device, dtype):
    skeleton_video_t = preprocess_videos(skeleton_images).to(device=device, dtype=dtype)
    skeleton_latents = build_lightweight_skeleton_step_context(skeleton_video_t)
    return skeleton_latents


def get_fused_context(
    plucker_embedding,
    skeleton_latents,
    models,
):
    plucker_adapter = models["plucker_adapter"]
    skeleton_embedding = models["skeleton_embedding"]
    y_plucker = [plucker_adapter(u) for u in plucker_embedding]
    c = [skeleton_embedding(u) for u in skeleton_latents]
    c = [u + v for u, v in zip(c, y_plucker)]
    c = torch.stack(c, dim=0)
    return c


def build_step_fused_context(
    skeleton_images,
    camera_extrinsics,
    camera_intrinsics,
    models,
    state,
    device,
    dtype,
):
    plucker_embedding = get_plucker_embeddings(
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        state=state,
        device=device,
        dtype=dtype,
    )
    skeleton_latents = get_skeleton_latents(
        skeleton_images=skeleton_images,
        device=device,
        dtype=dtype,
    )
    return get_fused_context(
        plucker_embedding=plucker_embedding,
        skeleton_latents=skeleton_latents,
        models=models,
    )


def prefill(
    reference_images,
    reference_skeleton_images,
    reference_camera_extrinsics,
    reference_camera_intrinsics,
    models,
    state,
    device,
    dtype,
):
    assert (
        len(reference_images)
        == len(reference_skeleton_images)
        == len(reference_camera_extrinsics)
        == len(reference_camera_intrinsics)
    ), (
        "Number of reference images, reference skeletons, camera extrinsics, and camera intrinsics must be the same."
    )

    # NOTE: reference_latents / fused_context are returned to the caller and
    # never stored on the state: create re-uses them as the first solved frame
    # (skipping the generate step) and keeping them on the state would bloat
    # every offload/onload round-trip.
    n_ref, reference_latents = get_ref_latents(
        reference_images, models, state, device, dtype
    )
    state.num_reference = n_ref

    if state.reference_skeleton_latents is None:
        reference_skeleton_latents = get_ref_skeleton_latents(
            reference_skeleton_images, models, state, device, dtype
        )

    if state.reference_plucker_embedding is None:
        reference_plucker_embedding = get_ref_plucker_embeddings(
            reference_camera_extrinsics,
            reference_camera_intrinsics,
            state,
            device,
            dtype,
        )

    fused_context = get_fused_context(
        plucker_embedding=reference_plucker_embedding,
        skeleton_latents=reference_skeleton_latents,
        models=models,
    )

    dit = models["dit"]
    kv_cache = dit_prefill(
        dit=dit,
        ref_latents=reference_latents,
        fused_context=fused_context,
        rope_freqs=state.rope_freqs,
        context=state.context,
        current_frame_start_index=0,
        clean_latents=None,
    )
    state.kv_cache = kv_cache
    state.current_frame_start_index += state.num_reference
    state.clear_prefill()
    # Return the reference latents + fused context to the caller: create skips
    # the generate step (the first generated frame is ~identical to the
    # reference latents) and writes them into the KV-cache slot directly.
    return state, reference_latents, fused_context


@torch.no_grad()
def generate_frame(
    skeleton_images,
    camera_extrinsics,
    camera_intrinsics,
    models,
    state,
    device,
    dtype,
    config: UranusStreamConfig | None = None,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise one new latent frame. Read-only on ``state`` (no KV, no index bump).

    Returns ``(generated_latents, fused_context)``. ``fused_context`` must be
    handed to :func:`fill_frame_kv_cache` — it is the same tensor the DiT
    consumed, and recomputing it would double the per-step conditioning cost.
    """
    if config is None:
        config = UranusStreamConfig()

    fused_context = build_step_fused_context(
        skeleton_images=skeleton_images,
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        models=models,
        state=state,
        device=device,
        dtype=dtype,
    )

    shape = (
        1,
        state.num_cameras,
        models["vae"].model.z_dim,
        1,
        state.height // state.spatial_interval,
        state.width // state.spatial_interval,
    )
    if noise is None:
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    else:
        if tuple(noise.shape) != shape:
            raise ValueError(
                f"noise must have shape {shape}, got {tuple(noise.shape)}"
            )
        noise = noise.to(device=device, dtype=dtype)

    generated_latents = dit_decode_step(
        dit=models["dit"],
        scheduler=models["scheduler"],
        noisy_latents=noise,
        fused_context=fused_context,
        rope_freqs=state.rope_freqs,
        num_inference_steps=state.num_inference_steps,
        context=state.context,
        current_frame_start_index=state.current_frame_start_index,
        kv_cache=state.kv_cache,
        sigma_shift=config.sigma_shift,
        cfg_scale=config.cfg_scale,
        uranus_scale=config.uranus_scale,
    )
    return generated_latents, fused_context


@torch.no_grad()
def fill_frame_kv_cache(
    generated_latents,
    fused_context,
    models,
    state,
    config: UranusStreamConfig | None = None,
) -> UranusStreamState:
    """Write the solved frame's KV, trim the sliding window, advance frame index.

    Order matters: the fill reads ``current_frame_start_index`` to slice the rope
    freqs, so the index bump must come last. Window parameters come from
    ``state`` (the serialized source of truth), never from ``config``.
    """
    if config is None:
        config = UranusStreamConfig()

    kv_cache = dit_kv_cache_fill(
        dit=models["dit"],
        clean_latents=generated_latents,
        fused_context=fused_context,
        context=state.context,
        rope_freqs=state.rope_freqs,
        current_frame_start_index=state.current_frame_start_index,
        kv_cache=state.kv_cache,
        uranus_scale=config.uranus_scale,
    )
    kv_cache = state.kv_cache if kv_cache is None else kv_cache

    kv_cache.sliding_window_drop(
        sliding_window_size=state.teacher_forcing_window_size,
        num_ref=state.num_reference,
        tokens_per_frame=state.tokens_per_frame,
    )
    state.kv_cache = kv_cache
    state.current_frame_start_index += 1
    return state


@torch.no_grad()
def decode_frame(
    latents,
    models,
    state,
    device,
    config: UranusStreamConfig | None = None,
) -> tuple[torch.Tensor, UranusStreamState]:
    """Decode one latent frame via the incremental VAE decoder; returns CPU video."""
    if config is None:
        config = UranusStreamConfig()

    (
        video,
        state.dec_feat_map,
        state.dec_feat_idx,
        state.decoded_latent_frames,
    ) = decode_latents_chunk(
        vae=models["vae"],
        latents=latents,
        device=device,
        dec_feat_map=state.dec_feat_map,
        dec_feat_idx=state.dec_feat_idx,
        decoded_latent_frames=state.decoded_latent_frames,
        tiled=False,
    )
    video = video.detach().to("cpu").contiguous()
    return video, state


def initialize_stream_state(models, config: UranusStreamConfig, device):
    dit = models["dit"]
    state = UranusStreamState()
    spatial_interval = state.spatial_interval
    state.num_cameras = config.num_cameras
    state.skeleton_mode = config.skeleton_mode
    state.height = config.height
    state.width = config.width
    state.rope_freqs = precompute_freqs_cis_3d(
        dit.dim // dit.blocks[0].num_heads,
        100000,
        device=device,
    )
    state.num_inference_steps = config.num_inference_steps
    state.tokens_per_frame = (
        config.height * config.width // (spatial_interval * spatial_interval * 4)
    )
    state.current_frame_start_index = 0
    state.teacher_forcing_window_size = config.teacher_forcing_window_size
    return state


def check_data_validity(skeleton_images, camera_extrinsics, camera_intrinsics, state):
    if (
        skeleton_images is None
        or camera_extrinsics is None
        or camera_intrinsics is None
    ):
        raise ValueError(
            "skeleton_images, camera_extrinsics, and camera_intrinsics must be provided"
        )
    num_cameras = state.num_cameras
    temporal_interval = state.temporal_interval
    if (
        len(skeleton_images) != num_cameras
        or len(camera_extrinsics) != num_cameras
        or len(camera_intrinsics) != num_cameras
    ):
        raise ValueError(
            f"skeleton_images, camera_extrinsics, and camera_intrinsics all must have length {num_cameras}, but got len {len(skeleton_images)}, {len(camera_extrinsics)}, {len(camera_intrinsics)}"
        )
    num_condition_frames = len(skeleton_images[0])
    assert num_condition_frames in [temporal_interval, 1], (
        f"skeleton_images must have shape (temporal_interval, ...), but got shape (1, ...), or got shape {skeleton_images[0].shape}"
    )
    for i in range(num_cameras):
        assert (
            len(skeleton_images[i])
            == len(camera_extrinsics[i])
            == len(camera_intrinsics[i])
        ), (
            f"skeleton_images, camera_extrinsics, and camera_intrinsics all must have shape (temporal_interval, ...), but got shape (1, ...), or got shape {skeleton_images[i].shape}"
        )
