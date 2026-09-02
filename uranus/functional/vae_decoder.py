"""
Functional API for VAE decoding of video latents to pixel frames.
"""

import inspect

import torch
from einops import rearrange

from uranus.modules.vae import VideoVAE38_, WanVideoVAE, count_conv3d, unpatchify


def decode_latents(
    vae: WanVideoVAE,
    latents: torch.Tensor,  # [B, N_CAM, z_dim, F_lat, H_lat, W_lat]
    device: torch.device,
    tiled: bool = True,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
) -> torch.Tensor:
    """
    VAE-decode latent frames to video frames.

    VAE temporal expansion:
      - 1st latent frame → 1 video frame
      - subsequent latent frames → 4 video frames each

    Args:
        vae: WanVideoVAE instance
        latents: Video latents [B, N_CAM, z_dim, F_lat, H_lat, W_lat]
        device: Target device
        tiled: Whether to use tiled VAE decoding
        tile_size: Tile size in latent space
        tile_stride: Tile stride in latent space

    Returns:
        Video frames [B, N_CAM, 3, F_video, H_pix, W_pix] in [-1, 1]
    """
    batch, n_cam, _, _, _, _ = latents.shape

    latents_flat = rearrange(latents, "B N_CAM C F H W -> (B N_CAM) C F H W")

    videos = vae.decode(
        latents_flat,
        device=device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )

    videos = rearrange(
        videos, "(B N_CAM) C F H W -> B N_CAM C F H W", B=batch, N_CAM=n_cam
    )
    return videos.clamp_(-1, 1)


def _apply_vae_scale(z: torch.Tensor, vae: WanVideoVAE) -> torch.Tensor:
    scale = vae.scale
    if isinstance(scale[0], torch.Tensor):
        scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
        return z / scale[1].view(1, vae.z_dim, 1, 1, 1) + scale[0].view(
            1, vae.z_dim, 1, 1, 1
        )
    scale = scale.to(dtype=z.dtype, device=z.device)
    return z / scale[1] + scale[0]


def _decode_single_chunk(
    vae: WanVideoVAE,
    latents: torch.Tensor,
    dec_feat_map,
    decoded_latent_frames: int,
) -> tuple[torch.Tensor, list, list]:
    model = vae.model
    z = _apply_vae_scale(latents, vae)
    x = model.conv2(z)

    # Flash-VAED decoder exposes its own cache-slot count (its causal convs
    # are depthwise-separable); fall back to the legacy count otherwise.
    count_fn = getattr(model.decoder, "count_conv3d", None) or count_conv3d
    if dec_feat_map is None:
        dec_feat_map = [None] * count_fn(model.decoder)

    decoder_signature = inspect.signature(model.decoder.forward)
    supports_first_chunk = "first_chunk" in decoder_signature.parameters

    outputs = []
    for frame_idx in range(x.shape[2]):
        feat_idx = [0]
        decoder_kwargs = {
            "feat_cache": dec_feat_map,
            "feat_idx": feat_idx,
        }
        if supports_first_chunk:
            decoder_kwargs["first_chunk"] = (
                decoded_latent_frames == 0 and frame_idx == 0
            )
        out, dec_feat_map, feat_idx = model.decoder(
            x[:, :, frame_idx : frame_idx + 1], **decoder_kwargs
        )
        outputs.append(out)

    video = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
    if isinstance(model, VideoVAE38_):
        video = unpatchify(video, patch_size=2)
    return video, dec_feat_map, feat_idx


def decode_latents_chunk(
    vae: WanVideoVAE,
    latents: torch.Tensor,
    device: torch.device,
    dec_feat_map=None,
    dec_feat_idx: list | None = None,
    decoded_latent_frames: int = 0,
    tiled: bool = False,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
) -> tuple[torch.Tensor, list, list, int]:
    """
    Incrementally decode a latent chunk while reusing decoder feature cache.

    This mirrors `VideoVAE_.decode()` / `VideoVAE38_.decode()` but externalizes
    decoder cache so streaming callers can carry it in their own state.
    """
    del tiled, tile_size, tile_stride

    batch, n_cam, _, f_lat, _, _ = latents.shape
    latents_flat = rearrange(latents, "B N_CAM C F H W -> (B N_CAM) C F H W").to(
        device=device
    )
    video, dec_feat_map, dec_feat_idx = _decode_single_chunk(
        vae=vae,
        latents=latents_flat,
        dec_feat_map=dec_feat_map,
        decoded_latent_frames=decoded_latent_frames,
    )
    videos = rearrange(
        video, "(B N_CAM) C F H W -> B N_CAM C F H W", B=batch, N_CAM=n_cam
    )
    return (
        videos.clamp_(-1, 1),
        dec_feat_map,
        dec_feat_idx,
        decoded_latent_frames + f_lat,
    )
