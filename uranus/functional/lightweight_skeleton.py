"""
Functional API for lightweight skeleton conditioning (pixel-space patchify).
"""

import torch
from einops import rearrange


def _validate_lightweight_skeleton_video(x: torch.Tensor, spatial_interval: int) -> tuple[int, int]:
    if x.ndim != 6:
        raise ValueError(f"expected [B,N_CAM,3,T,H,W], got {tuple(x.shape)}")
    if x.shape[2] != 3:
        raise ValueError(f"expected C=3, got C={x.shape[2]}")
    h = int(x.shape[4])
    w = int(x.shape[5])
    if h % spatial_interval != 0 or w % spatial_interval != 0:
        raise ValueError(f"Expected H/W divisible by {spatial_interval}, got H={h}, W={w}")
    return h, w


def _pack_lightweight_skeleton(
    x: torch.Tensor,
    *,
    temporal_interval: int,
    spatial_interval: int,
) -> torch.Tensor:
    h, w = _validate_lightweight_skeleton_video(x, spatial_interval)
    if x.shape[3] % temporal_interval != 0:
        raise ValueError(f"Expected T % {temporal_interval} == 0, got T={x.shape[3]}")
    return rearrange(
        x,
        "B N_CAM C (F I) (H PH) (W PW) -> B N_CAM (C PH PW I) F H W",
        I=temporal_interval,
        PH=spatial_interval,
        PW=spatial_interval,
        F=x.shape[3] // temporal_interval,
        H=h // spatial_interval,
        W=w // spatial_interval,
    )


def build_lightweight_skeleton_reference_context(
    reference_skeleton: torch.Tensor,  # [B, N_CAM, 3, N_REF, H, W]
    temporal_interval: int = 4,
    spatial_interval: int = 8,
) -> torch.Tensor:
    ref = reference_skeleton.repeat_interleave(temporal_interval, dim=3)
    return _pack_lightweight_skeleton(
        ref,
        temporal_interval=temporal_interval,
        spatial_interval=spatial_interval,
    )


def build_lightweight_skeleton_step_context(
    skeleton_chunk: torch.Tensor,  # [B, N_CAM, 3, T, H, W] where T is 1 or 4
    temporal_interval: int = 4,
    spatial_interval: int = 8,
) -> torch.Tensor:
    t = skeleton_chunk.shape[3]
    if t == 1:
        x = skeleton_chunk.repeat_interleave(temporal_interval, dim=3)
    elif t == temporal_interval:
        x = skeleton_chunk
    else:
        raise ValueError(f"skeleton_chunk T must be 1 or {temporal_interval}, got T={t}")
    return _pack_lightweight_skeleton(
        x,
        temporal_interval=temporal_interval,
        spatial_interval=spatial_interval,
    )
