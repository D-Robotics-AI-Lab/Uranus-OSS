"""
Functional API for building skeleton context with streaming VAE encode.

The VAE's CausalConv3d layers maintain feat_cache across chunks for temporal
continuity.  Streaming splits must preserve this: a single cache is advanced
exactly once per chunk for all cameras and inactive/reactive channels combined.
"""

import torch
from einops import rearrange
from uranus.modules.vae import WanVideoVAE


def init_skeleton_cache(vae: WanVideoVAE) -> tuple[list, list]:
    """Initialise VAE encoder cache for streaming skeleton encoding.

    Returns (feat_map, feat_idx).  Pass both to prefill / step and persist
    the returned values across calls.
    """
    return vae.init_encoder_cache()


# helpers

def _encode_one_chunk(
    vae: WanVideoVAE,
    skeleton_video: torch.Tensor,       # [N_CAM, 3, T_chunk, H, W] in [-1, 1]
    skeleton_mask: torch.Tensor | None,  # [N_CAM, 1, T_chunk, H, W]
    num_cameras: int,
    enc_cache: tuple[list, list],
    device: torch.device,
    dtype: torch.dtype,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, tuple[list, list]]:
    """Encode one temporal chunk, inactive + reactive together in one batch.

    Stacking [inactive_0, ..., inactive_N, reactive_0, ..., reactive_N]
    into a single batch call ensures the cache advances exactly once.
    """
    if skeleton_mask is None:
        skeleton_mask = torch.ones_like(skeleton_video)

    inactive = skeleton_video * (1 - skeleton_mask)   # [N_CAM, 3, T, H, W]
    reactive = skeleton_video * skeleton_mask

    # Batch: [2 * N_CAM, 3, T, H, W]
    combined = torch.cat([inactive, reactive], dim=0)

    feat_map, feat_idx = enc_cache
    latents, feat_map, feat_idx = vae.encode_chunk(
        combined, feat_map, feat_idx, device=device,
        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
    )
    # latents: [2 * N_CAM, z_dim, 1, H_lat, W_lat]
    latents = latents.to(dtype=dtype, device=device)
    inactive_lat = latents[:num_cameras]
    reactive_lat = latents[num_cameras:]

    return inactive_lat, reactive_lat, (feat_map, feat_idx)


def _build_mask_latent(
    skeleton_mask: torch.Tensor,    # [N_CAM, 1, T, H, W]  (cam 0 slice used)
    latent_frames: int,
) -> torch.Tensor:
    """Downsample mask to VAE latent space: 8x8 spatial patch -> 64 channels."""
    m = skeleton_mask[0:1, :1]              # take cam 0 only  [1, 1, T, H, W]
    mask_lat = rearrange(m, "B C T (H P) (W Q) -> B (C P Q) T H W", P=8, Q=8)
    mask_lat = torch.nn.functional.interpolate(
        mask_lat,
        size=(latent_frames, mask_lat.shape[3], mask_lat.shape[4]),
        mode="nearest-exact",
    )
    return mask_lat  # [1, 64, F, H, W]


# public API

def build_skeleton_context_prefill(
    vae: WanVideoVAE,
    skeleton_video: torch.Tensor,        # [N_CAM, 3, T_prefill, H, W]  in [-1, 1]
    skeleton_mask: torch.Tensor | None,   # [N_CAM, 1, T_prefill, H, W]
    num_cameras: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tiled: bool = True,
    tile_size: tuple[int, int] = (30, 52),
    tile_stride: tuple[int, int] = (15, 26),
) -> tuple[torch.Tensor, tuple[list, list]]:
    """Build skeleton context for prefilling frames.

    All T_prefill frames are consumed in the VAE's native chunking pattern
    (first frame alone, then groups of 4) while preserving feat_cache continuity.

    Returns:
        uranus_context  [1, N_CAM, 2*z_dim + 64, F_lat, H_lat, W_lat]
        enc_cache       (feat_map, feat_idx)  for subsequent decode steps
    """
    if skeleton_mask is None:
        skeleton_mask = torch.ones_like(skeleton_video)

    enc_cache = init_skeleton_cache(vae)
    T = skeleton_video.shape[2]
    num_chunks = 1 + (T - 1) // 4

    inactive_parts = []
    reactive_parts = []
    mask_parts = []

    for i in range(num_chunks):
        if i == 0:
            chunk = slice(0, 1)
            latent_frames = 1
        else:
            start = 1 + 4 * (i - 1)
            end = min(1 + 4 * i, T)
            chunk = slice(start, end)
            latent_frames = 1

        skel = skeleton_video[:, :, chunk]     # [N_CAM, 3, 1|4, H, W]
        msk = skeleton_mask[:, :, chunk]        # [N_CAM, 1, 1|4, H, W]

        inactive_lat, reactive_lat, enc_cache = _encode_one_chunk(
            vae, skel, msk, num_cameras, enc_cache, device, dtype, tiled, tile_size, tile_stride,
        )
        # Each: [N_CAM, z_dim, 1, H_lat, W_lat]
        inactive_parts.append(inactive_lat.unsqueeze(0))  # add batch dim: [1, N_CAM, z_dim, 1, H_lat, W_lat]
        reactive_parts.append(reactive_lat.unsqueeze(0))

        mask_lat = _build_mask_latent(msk, latent_frames)  # [1, 64, 1, H_lat, W_lat]
        mask_parts.append(mask_lat)

    # Concat along F dimension
    inactive_all = torch.cat(inactive_parts, dim=3)   # [1, N_CAM, z_dim, F_lat, H_lat, W_lat]
    reactive_all = torch.cat(reactive_parts, dim=3)
    mask_all = torch.cat(mask_parts, dim=2)            # [1, 64, F_lat, H_lat, W_lat]
    mask_all = mask_all.unsqueeze(1).expand(-1, num_cameras, -1, -1, -1, -1)

    uranus_video = torch.cat([inactive_all, reactive_all], dim=2)  # [1, N_CAM, 2*z_dim, F_lat, H_lat, W_lat]
    uranus_context = torch.cat([uranus_video, mask_all], dim=2)

    return uranus_context, enc_cache


def build_skeleton_context_step(
    vae: WanVideoVAE,
    skeleton_video: torch.Tensor,        # [N_CAM, 3, T_step, H, W]  (1 or 4 frames)
    skeleton_mask: torch.Tensor | None,   # [N_CAM, 1, T_step, H, W]
    num_cameras: int,
    enc_cache: tuple[list, list],         # from prefill or previous step
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tiled: bool = True,
    tile_size: tuple[int, int] = (30, 52),
    tile_stride: tuple[int, int] = (15, 26),
) -> tuple[torch.Tensor, tuple[list, list]]:
    """Build skeleton context for one decoding step, reusing temporal cache.

    Returns:
        uranus_context_step  [1, N_CAM, 2*z_dim + 64, 1, H_lat, W_lat]
        enc_cache            updated cache for the next step
    """
    if skeleton_mask is None:
        skeleton_mask = torch.ones_like(skeleton_video)

    inactive_lat, reactive_lat, enc_cache = _encode_one_chunk(
        vae, skeleton_video, skeleton_mask, num_cameras, enc_cache,
        device, dtype, tiled, tile_size, tile_stride,
    )
    # Add batch dims
    inactive_lat = inactive_lat.unsqueeze(0)  # [1, N_CAM, z_dim, 1, H_lat, W_lat]
    reactive_lat = reactive_lat.unsqueeze(0)

    mask_lat = _build_mask_latent(skeleton_mask, latent_frames=1)
    mask_lat = mask_lat.unsqueeze(1).expand(-1, num_cameras, -1, -1, -1, -1)

    uranus_video = torch.cat([inactive_lat, reactive_lat], dim=2)
    uranus_context = torch.cat([uranus_video, mask_lat], dim=2)

    return uranus_context, enc_cache
