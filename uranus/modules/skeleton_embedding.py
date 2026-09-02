"""
Skeleton context patch embedding for projecting VAE-encoded skeleton latents
into DiT token space. Inherits from nn.Conv3d to preserve weight key compatibility
with the original UranusAdapter.vace_patch_embedding checkpoint.

Migrated from uranus.models.controller.UranusAdapter.vace_patch_embedding.
"""

from torch import nn


class VacePatchEmbedding(nn.Conv3d):
    """
    Conv3d that maps skeleton context into DiT token space.

    Input:  [N_CAM, in_dim, F, H, W]  — skeleton latents (2*z_dim + 64 channels)
    Output: [N_CAM, dim, F, H/2, W/2] — tokens matching DiT patch_size spatial stride

    Inherits from nn.Conv3d so state_dict key ``vace_patch_embedding.weight``
    matches the original checkpoint without nesting.
    """

    def __init__(self, in_dim=96, dim=1536):
        super().__init__(in_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))


class LightweightSkeletonPatchEmbedding(nn.Conv3d):
    def __init__(self, in_dim=768, dim=1536):
        super().__init__(in_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
