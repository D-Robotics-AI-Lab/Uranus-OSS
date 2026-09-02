"""
Functional API for Plucker ray encoding from camera parameters.

Ported from uranus.utils.plucker_encoder. Includes both the raw per-camera
ray computation (``compute_plucker_embeddings``) and the helpers that build
embeddings aligned to the WanVideo latent time axis (multi-camera packing,
reference/step builders).
"""
import numpy as np
import torch
from einops import rearrange, repeat


def compute_plucker_embeddings(
    extrinsics: list[np.ndarray],   # list of [4, 4] world-to-camera matrices
    intrinsics: list[np.ndarray],    # list of [3, 3] intrinsic matrices
    height: int,
    width: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Compute Plucker ray embeddings from camera parameters.

    Plucker coordinates represent each pixel's ray as a 6D vector:
    [moment_x, moment_y, moment_z, direction_x, direction_y, direction_z]
    where moment = origin × direction.

    Args:
        extrinsics: List of F world-to-camera 4x4 matrices
        intrinsics: List of F camera intrinsic 3x3 matrices
        height: Image height in pixels
        width: Image width in pixels
        device: Target device

    Returns:
        Plucker embeddings of shape [1, F, H, W, 6]
    """
    F = len(extrinsics)

    # w2c -> c2w and extract K params
    c2w_list = []
    k_params_list = []
    for ext, K in zip(extrinsics, intrinsics):
        c2w = np.linalg.inv(ext)
        c2w_list.append(c2w)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        k_params_list.append([fx, fy, cx, cy])

    # Build the small camera tensors on CPU (cast + pin_memory are pure-CPU,
    # no GPU sync), then H2D asynchronously from pinned memory so the caller
    # does not block on the copy; the GPU work in ``_ray_condition`` is
    # stream-ordered after it automatically.
    if torch.device(device).type == "cuda":
        c2w = (
            torch.from_numpy(np.stack(c2w_list))
            .to(dtype=torch.float32)
            .pin_memory()
            .to(device=device, non_blocking=True)
        )  # [F, 4, 4]
        K_tensor = (
            torch.from_numpy(np.array(k_params_list))
            .to(dtype=torch.float32)
            .pin_memory()
            .to(device=device, non_blocking=True)
        )  # [F, 4]
    else:
        c2w = torch.from_numpy(np.stack(c2w_list)).to(dtype=torch.float32, device=device)
        K_tensor = torch.from_numpy(np.array(k_params_list)).to(dtype=torch.float32, device=device)
    c2w = c2w.unsqueeze(0)  # [1, F, 4, 4]
    K_tensor = K_tensor.unsqueeze(0)  # [1, F, 4]

    return _ray_condition(K_tensor, c2w, height, width)


def _ray_condition(
    K: torch.Tensor,       # [B, F, 4]   fx, fy, cx, cy
    c2w: torch.Tensor,     # [B, F, 4, 4]  camera-to-world
    H: int,
    W: int,
) -> torch.Tensor:
    """Compute Plucker coordinates via ray conditioning."""
    B, F = K.shape[:2]
    device = K.device
    dtype = c2w.dtype

    # Pixel grid
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
        torch.linspace(0, H - 1, H, device=device, dtype=dtype),
        indexing="xy",
    )
    i = i.reshape(1, 1, H * W).expand(B, F, H * W) + 0.5
    j = j.reshape(1, 1, H * W).expand(B, F, H * W) + 0.5

    fx = K[..., 0:1]
    fy = K[..., 1:2]
    cx = K[..., 2:3]
    cy = K[..., 3:4]

    # Camera-space ray directions
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack([xs, ys, zs], dim=-1)  # [B, F, H*W, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform to world space
    R = c2w[..., :3, :3]  # [B, F, 3, 3]
    rays_d = directions @ R.transpose(-1, -2)  # [B, F, H*W, 3]
    rays_d = rays_d.to(torch.float32)

    # Ray origins (camera centers in world space)
    rays_o = c2w[..., :3, 3].unsqueeze(-2).expand_as(rays_d)  # [B, F, H*W, 3]

    # Plucker: moment = origin x direction
    rays_dxo = torch.linalg.cross(rays_o, rays_d)

    plucker = torch.cat([rays_dxo, rays_d], dim=-1)  # [B, F, H*W, 6]
    plucker = plucker.reshape(B, F, H, W, 6)

    return plucker


def compute_multi_camera_plucker_embeddings(
    camera_extrinsics: list[list[np.ndarray]],
    camera_intrinsics: list[list[np.ndarray]],
    height: int,
    width: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if len(camera_extrinsics) != len(camera_intrinsics):
        raise ValueError("camera_extrinsics and camera_intrinsics must have same length (N_CAM)")
    pluckers = []
    for extrinsics, intrinsics in zip(camera_extrinsics, camera_intrinsics):
        plk = compute_plucker_embeddings(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            height=height,
            width=width,
            device=device,
        )
        pluckers.append(plk)
    return torch.stack(pluckers, dim=1)


def pack_plucker_frames_to_latent_embeddings(
    plucker_frames: torch.Tensor,  # [B, N_CAM, T, H, W, 6]
    temporal_interval: int = 4,
) -> torch.Tensor:
    if plucker_frames.ndim != 6:
        raise ValueError(f"plucker_frames must be [B,N_CAM,T,H,W,6], got {tuple(plucker_frames.shape)}")
    if plucker_frames.shape[-1] != 6:
        raise ValueError(f"plucker_frames last dim must be 6, got {plucker_frames.shape[-1]}")

    t = plucker_frames.shape[2]
    if t == 1:
        plucker_frames = plucker_frames.repeat_interleave(temporal_interval, dim=2)
        t = temporal_interval
    if t % temporal_interval != 0:
        raise ValueError(f"Expected T % {temporal_interval} == 0, got T={t}")

    x = rearrange(plucker_frames, "B N_CAM (F I) H W C -> B N_CAM (C I) F H W", I=temporal_interval)
    return x.contiguous()


def build_reference_plucker_embeddings(
    reference_camera_extrinsics: list[np.ndarray],
    reference_camera_intrinsics: list[np.ndarray],
    num_cameras: int,
    height: int,
    width: int,
    device: str | torch.device = "cpu",
    temporal_interval: int = 4,
) -> torch.Tensor:
    reference_plk = compute_plucker_embeddings(
        extrinsics=reference_camera_extrinsics,
        intrinsics=reference_camera_intrinsics,
        height=height,
        width=width,
        device=device,
    )
    reference_plk = repeat(reference_plk, "B F H W C -> B N_CAM F H W C", N_CAM=num_cameras)
    reference_plk = reference_plk.repeat_interleave(repeats=temporal_interval, dim=2)
    return pack_plucker_frames_to_latent_embeddings(plucker_frames=reference_plk, temporal_interval=temporal_interval)


def build_step_plucker_embeddings(
    camera_extrinsics_chunk: list[list[np.ndarray]],
    camera_intrinsics_chunk: list[list[np.ndarray]],
    height: int,
    width: int,
    device: str | torch.device = "cpu",
    temporal_interval: int = 4,
) -> torch.Tensor:
    plk = compute_multi_camera_plucker_embeddings(
        camera_extrinsics=camera_extrinsics_chunk,
        camera_intrinsics=camera_intrinsics_chunk,
        height=height,
        width=width,
        device=device,
    )
    return pack_plucker_frames_to_latent_embeddings(plk, temporal_interval=temporal_interval)