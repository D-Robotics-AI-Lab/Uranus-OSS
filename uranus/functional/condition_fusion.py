"""
Functional API for fusing skeleton context and Plucker embeddings
into a unified condition token for DiT injection.

Ported from the uranus_adapter_with_vace=False branch of UranusAdapter.forward.
"""

import torch
from torch import nn
from uranus.modules.plucker_adapter import PluckerAdapter


def fuse_conditions(
    skeleton_embedding: nn.Module,
    plucker_adapter: PluckerAdapter,
    uranus_context: torch.Tensor,
    plucker_embeddings: torch.Tensor,   # [1, N_CAM, 24, F, H_pix, W_pix]
) -> torch.Tensor:
    """
    Fuse skeleton context and Plucker embeddings per camera.

    Each camera's skeleton latents and Plucker embeddings are independently
    projected into DiT token space and summed.

    Args:
        skeleton_embedding: VacePatchEmbedding (Conv3d) for skeleton context
        plucker_adapter: PluckerAdapter for Plucker embedding
        uranus_context: Skeleton context [1, N_CAM, C_ctx, F, H, W]
        plucker_embeddings: Plucker embeddings [1, N_CAM, 24, F, H_pix, W_pix]

    Returns:
        Fused context tokens suitable for injection into DiT block input
    """
    n_cam = uranus_context.shape[1]
    fused_parts = []
    for cam_idx in range(n_cam):
        ctx = uranus_context[:, cam_idx:cam_idx+1]       # [1, 1, C_ctx, F, H, W]
        plk = plucker_embeddings[:, cam_idx:cam_idx+1]    # [1, 1, 24, F, H_pix, W_pix]

        # Squeeze camera dim temporarily
        ctx_sq = ctx.squeeze(1)   # [1, C_ctx, F, H, W]
        plk_sq = plk.squeeze(1)   # [1, 24, F, H_pix, W_pix]

        c = skeleton_embedding(ctx_sq)           # [1, dim, F, H', W']
        y = plucker_adapter(plk_sq)               # [1, dim, F, H', W']

        fused = c + y                              # element-wise sum
        fused_parts.append(fused)

    # Stack per-camera results back
    fused_all = torch.stack(fused_parts, dim=1)   # [1, N_CAM, dim, F, H', W']
    return fused_all
