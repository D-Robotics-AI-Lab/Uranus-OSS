"""
Plucker embedding adapter for spatial downsampling into DiT token space.

Migrated from uranus.models.controller.PluckerAdapter.
Processes 6-channel Plucker ray embeddings through:
  PixelUnshuffle(8) -> Conv2d -> ResidualBlocks
"""

from torch import nn


class ResidualBlock2d(nn.Module):
    """2D residual block used inside PluckerAdapter (distinct from VAE's 3D ResidualBlock)."""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class PluckerAdapter(nn.Module):
    """
    Adapts Plucker ray embeddings [N_CAM, 24, F, H, W] into DiT token space
    [N_CAM, out_dim, F, H/patch, W/patch].

    PixelUnshuffle(8) collapses 8x spatial patches into channels (24→1536),
    then Conv2d+ResBlocks project to out_dim with stride-2 spatial reduction
    to match DiT patch_size=(1,2,2).
    """

    def __init__(self, in_dim=24, out_dim=1536, kernel_size=(2, 2), stride=(2, 2), num_residual_blocks=1):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        self.conv = nn.Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock2d(out_dim) for _ in range(num_residual_blocks)]
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        for block in self.residual_blocks:
            nn.init.zeros_(block.conv2.weight)
            if block.conv2.bias is not None:
                nn.init.zeros_(block.conv2.bias)

    def forward(self, x):
        n_cam, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(n_cam * f, c, h, w)
        x = self.pixel_unshuffle(x)
        x = self.conv(x)
        x = self.residual_blocks(x)
        x = x.view(n_cam, f, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4)
        return x
