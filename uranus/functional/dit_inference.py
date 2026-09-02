import torch
from einops import rearrange

from uranus.modules.kv_cache import KVCache
from uranus.modules.dit import (
    SpatialTemporalWanModel,
    sinusoidal_embedding_1d,
)
from uranus.modules.scheduler import FlowMatchScheduler

# Internal helpers


def _unpatchify(
    x: torch.Tensor, n_cam: int, f: int, h: int, w: int, patch_size: tuple
) -> torch.Tensor:
    """Unpatchify multi-camera token sequence back to spatio-temporal tensor."""
    x = rearrange(
        x,
        "B (N_CAM F H W) (X Y Z C) -> (B N_CAM) C (F X) (H Y) (W Z)",
        N_CAM=n_cam,
        F=f,
        H=h,
        W=w,
        X=patch_size[0],
        Y=patch_size[1],
        Z=patch_size[2],
    )
    return x    


def _run_dit_blocks(
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    dit: SpatialTemporalWanModel,
    kv_cache: KVCache | None,
    kv_cache_filling: bool,
    n_cam: int,
    f: int,
    h: int,
    w: int,
):
    for block_id, block in enumerate(dit.blocks):
        x = block(
            x,
            context,
            t_mod,
            freqs,
            kv_cache=kv_cache,
            kv_cache_filling=kv_cache_filling,
            layer_idx=block_id,
            N_CAM=n_cam,
            F=f,
            H=h,
            W=w,
        )
    return x


# Prefill: build KV cache from reference + first clean frames


def dit_prefill(
    dit: SpatialTemporalWanModel,
    ref_latents: torch.Tensor,  # [1, N_CAM, C, N_REF, H, W]
    clean_latents: torch.Tensor,  # [1, N_CAM, C, F_clean, H, W]
    context: torch.Tensor,  # [1, L, dim]  — already text_embedded
    rope_freqs: tuple[torch.Tensor, ...],
    fused_context: torch.Tensor | None,  # [1, N_CAM, dim, F, H', W']
    current_frame_start_index: int = 0,
    uranus_scale: float = 1.0,
) -> KVCache:
    """Run DiT with timestep=0 and kv_cache_filling=True for reference + clean frames."""
    kv_cache = KVCache()
    n_cam = ref_latents.shape[1]
    num_ref = ref_latents.shape[3]
    num_clean = clean_latents.shape[3] if clean_latents is not None else 0

    all_x = [ref_latents]
    if clean_latents is not None and num_clean > 0:
        all_x.append(clean_latents)
    x = torch.concat(all_x, dim=3)

    tokens_per_frame = x.shape[4] * x.shape[5] // 4
    t_tokens = torch.zeros(
        (num_ref + num_clean, tokens_per_frame), dtype=x.dtype, device=x.device
    )
    t_tokens = t_tokens.unsqueeze(0).expand(n_cam, -1, -1).flatten()
    t_emb = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, t_tokens).unsqueeze(0)
    )
    t_mod = dit.time_projection(t_emb).unflatten(2, (6, dit.dim))

    context_emb = dit.text_embedding(context)

    x = rearrange(x, "B N_CAM C F H W -> (B N_CAM) C F H W")
    x, (f, h, w) = dit.patchify(x)
    x = rearrange(x, "(B N_CAM) T C -> B (N_CAM T) C", B=1, N_CAM=n_cam)

    if fused_context is not None:
        fc = rearrange(fused_context, "B N_CAM C F H W -> B (N_CAM F H W) C")
        x = x + fc * uranus_scale

    rope_f_parts = [rope_freqs[0][:num_ref]]
    if num_clean > 0:
        rope_f_parts.append(
            rope_freqs[0][
                current_frame_start_index + num_ref : current_frame_start_index
                + num_ref
                + num_clean
            ]
        )
    rope_f_cat = torch.cat(rope_f_parts)
    freqs = (
        torch.cat(
            [
                rope_f_cat.view(1, f, 1, 1, -1).expand(n_cam, f, h, w, -1),
                rope_freqs[1][:h].view(1, 1, h, 1, -1).expand(n_cam, f, h, w, -1),
                rope_freqs[2][:w].view(1, 1, 1, w, -1).expand(n_cam, f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(n_cam * f * h * w, 1, -1)
        .to(x.device)
        .view(torch.float64)
        .to(torch.float32)
    )

    _run_dit_blocks(
        x,
        context_emb,
        t_mod,
        freqs,
        dit,
        kv_cache,
        True,
        n_cam,
        f,
        h,
        w,
    )

    return kv_cache


# Decode: full denoising loop for one latent frame


def dit_decode_step(
    dit: SpatialTemporalWanModel,
    scheduler: FlowMatchScheduler,
    noisy_latents: torch.Tensor,  # [1, N_CAM, C, 1, H, W]
    context: torch.Tensor,  # [1, L, dim]  — already text_embedded
    rope_freqs: tuple[torch.Tensor, ...],
    kv_cache: KVCache | None,
    fused_context: torch.Tensor | None,  # [1, N_CAM, dim, 1, H', W']
    negative_context: torch.Tensor | None = None,
    num_inference_steps: int = 50,
    sigma_shift: float = 5.0,
    cfg_scale: float = 0.0,
    uranus_scale: float = 1.0,
    current_frame_start_index: int = 0,
) -> torch.Tensor:
    """Run full denoising loop for a single latent frame.

    kv_cache is READ (past K,V concatenated via past_kv) but NOT written.
    """
    scheduler.set_timesteps(
        num_inference_steps, shift=sigma_shift, device=noisy_latents.device
    )
    latents = noisy_latents
    n_cam = latents.shape[1]
    context_emb_pos = dit.text_embedding(context)
    context_emb_neg = (
        dit.text_embedding(negative_context)
        if isinstance(negative_context, torch.Tensor)
        else None
    )

    for t_idx, timestep_val in enumerate(scheduler.timesteps):
        t = timestep_val.unsqueeze(0).to(dtype=latents.dtype, device=latents.device)
        is_final = t_idx == len(scheduler.timesteps) - 1

        x = latents
        tpv = x.shape[4] * x.shape[5] // 4
        t_tokens = torch.ones((1, tpv), dtype=x.dtype, device=x.device) * t
        t_tokens = t_tokens.unsqueeze(0).expand(n_cam, -1, -1).flatten()
        t_emb = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, t_tokens).unsqueeze(0)
        )
        t_mod = dit.time_projection(t_emb).unflatten(2, (6, dit.dim))

        x_in = rearrange(x, "B N_CAM C F H W -> (B N_CAM) C F H W")
        x_flat, (f, h, w) = dit.patchify(x_in)
        x_flat = rearrange(x_flat, "(B N_CAM) T C -> B (N_CAM T) C", B=1, N_CAM=n_cam)

        if fused_context is not None:
            fc = rearrange(fused_context, "B N_CAM C F H W -> B (N_CAM F H W) C")
            x_flat = x_flat + fc * uranus_scale

        freqs_flat = (
            torch.cat(
                [
                    rope_freqs[0][
                        current_frame_start_index : current_frame_start_index + f
                    ]
                    .view(1, f, 1, 1, -1)
                    .expand(n_cam, f, h, w, -1),
                    rope_freqs[1][:h].view(1, 1, h, 1, -1).expand(n_cam, f, h, w, -1),
                    rope_freqs[2][:w].view(1, 1, 1, w, -1).expand(n_cam, f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(n_cam * f * h * w, 1, -1)
            .to(x_flat.device)
            .view(torch.float64)
            .to(torch.float32)
        )

        x_in_tokens = x_flat
        x_out_pos = _run_dit_blocks(
            x_in_tokens,
            context_emb_pos,
            t_mod,
            freqs_flat,
            dit,
            kv_cache,
            False,
            n_cam,
            f,
            h,
            w,
        )
        x_out_pos = dit.head(x_out_pos, t_emb)
        noise_pred_pos = _unpatchify(x_out_pos, n_cam, f, h, w, dit.patch_size)

        if cfg_scale > 0 and isinstance(context_emb_neg, torch.Tensor):
            x_out_neg = _run_dit_blocks(
                x_in_tokens.clone(),
                context_emb_neg,
                t_mod,
                freqs_flat,
                dit,
                kv_cache,
                False,
                n_cam,
                f,
                h,
                w,
            )
            x_out_neg = dit.head(x_out_neg, t_emb)
            noise_pred_neg = _unpatchify(x_out_neg, n_cam, f, h, w, dit.patch_size)
            noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)
        else:
            noise_pred = noise_pred_pos

        latents = scheduler.step(noise_pred, t, latents, to_final=is_final)

    return latents


# KV Cache fill: add newly solved frame to cache


def dit_kv_cache_fill(
    dit: SpatialTemporalWanModel,
    clean_latents: torch.Tensor,  # [1, N_CAM, C, 1, H, W]
    context: torch.Tensor,  # [1, L, dim]  — already text_embedded
    rope_freqs: tuple[torch.Tensor, ...],
    fused_context: torch.Tensor | None,  # [1, N_CAM, dim, 1, H', W']
    kv_cache: KVCache,
    current_frame_start_index: int,
    uranus_scale: float = 1.0,
) -> None:
    """Run DiT with timestep=0 and kv_cache_filling=True to store the new frame's KV."""
    n_cam = clean_latents.shape[1]

    x = clean_latents
    tpv = x.shape[4] * x.shape[5] // 4
    t_tokens = torch.zeros((1, tpv), dtype=x.dtype, device=x.device)
    t_tokens = t_tokens.unsqueeze(0).expand(n_cam, -1, -1).flatten()
    t_emb = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, t_tokens).unsqueeze(0)
    )
    t_mod = dit.time_projection(t_emb).unflatten(2, (6, dit.dim))
    context_emb = dit.text_embedding(context)

    x_in = rearrange(x, "B N_CAM C F H W -> (B N_CAM) C F H W")
    x_flat, (f, h, w) = dit.patchify(x_in)
    x_flat = rearrange(x_flat, "(B N_CAM) T C -> B (N_CAM T) C", B=1, N_CAM=n_cam)

    if fused_context is not None:
        fc = rearrange(fused_context, "B N_CAM C F H W -> B (N_CAM F H W) C")
        x_flat = x_flat + fc * uranus_scale

    freqs_flat = (
        torch.cat(
            [
                rope_freqs[0][current_frame_start_index : current_frame_start_index + f]
                .view(1, f, 1, 1, -1)
                .expand(n_cam, f, h, w, -1),
                rope_freqs[1][:h].view(1, 1, h, 1, -1).expand(n_cam, f, h, w, -1),
                rope_freqs[2][:w].view(1, 1, 1, w, -1).expand(n_cam, f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(n_cam * f * h * w, 1, -1)
        .to(x_flat.device)
        .view(torch.float64)
        .to(torch.float32)
    )

    _run_dit_blocks(
        x_flat,
        context_emb,
        t_mod,
        freqs_flat,
        dit,
        kv_cache,
        True,
        n_cam,
        f,
        h,
        w,
    )

    return kv_cache
