"""
Functional API for VAE encoding of reference images.
"""

import torch
from einops import rearrange, repeat
from uranus.modules.vae import WanVideoVAE


def encode_reference_images(
    vae: WanVideoVAE,
    ref_images: list[torch.Tensor],  # N_REF tensors of shape [3, H, W] in [-1, 1]
    device: torch.device,
    tiled: bool = True,
    tile_size: tuple[int, int] = (30, 52),
    tile_stride: tuple[int, int] = (15, 26),
) -> torch.Tensor:
    """
    VAE-encode reference images into latent space.

    Args:
        vae: WanVideoVAE instance
        ref_images: List of reference images, each [3, H, W] in [-1, 1]
        device: Target device
        tiled: Whether to use tiled VAE encoding
        tile_size: Tile size in latent space
        tile_stride: Tile stride in latent space

    Returns:
        Reference latents of shape [1, N_CAM, z_dim, N_REF, H_lat, W_lat]
        where N_CAM is 1 (replicated per camera by the caller if needed).
    """
    if not isinstance(ref_images, list):
        ref_images = [ref_images]
    n_ref = len(ref_images)
    # Stack: [N_REF, 3, H, W] -> [B=1, 3, N_REF, H, W]
    ref_tensor = torch.stack(ref_images, dim=0).unsqueeze(0)
    ref_tensor = rearrange(ref_tensor, "B N_REF C H W -> B C N_REF H W")
    # VAE encode expects [B, C, T, H, W] with T=1 per image
    ref_tensor = rearrange(ref_tensor, "B C N_REF H W -> (B N_REF) C 1 H W")
    ref_latents = vae.encode(ref_tensor, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    # Reshape: [B*N_REF, z_dim, 1, H_lat, W_lat] -> [1, 1, z_dim, N_REF, H_lat, W_lat]
    ref_latents = rearrange(ref_latents, "(B N_REF) C 1 H W -> B N_REF C H W", N_REF=n_ref)
    ref_latents = rearrange(ref_latents, "B N_REF C H W -> B C N_REF H W").unsqueeze(1)
    return ref_latents
