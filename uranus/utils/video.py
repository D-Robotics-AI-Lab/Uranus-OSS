"""Convert the model's raw decoded video tensor into per-camera uint8 frames.

Clamps to [-1, 1], maps to [0, 255], and returns one ``HxWx3`` RGB
``np.ndarray`` per (camera, time). PNG encoding / base64 are a transport
concern and deliberately not handled here.
"""


import numpy as np
import torch


def video_to_frames(video: torch.Tensor, camera_names) -> dict[str, list[np.ndarray]]:
    """Convert a 6-D video tensor to ``{camera: [H, W, 3] uint8}``.

    ``video`` layout is ``[batch=1, camera, channel, time, H, W]`` (dim 3 is
    time). Output frames are RGB uint8, one ``np.ndarray`` per (camera, time).
    """
    if video.ndim != 6:
        raise ValueError(f"Expected generated video with 6 dims, got {tuple(video.shape)}")
    frames = video[0].detach().to("cpu", dtype=torch.float32).clamp(-1, 1)
    frames = ((frames + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    result: dict[str, list[np.ndarray]] = {}
    for camera_index, camera_name in enumerate(camera_names):
        result[camera_name] = [
            frames[camera_index, :, t].permute(1, 2, 0).contiguous().numpy()
            for t in range(frames.shape[2])
        ]
    return result
