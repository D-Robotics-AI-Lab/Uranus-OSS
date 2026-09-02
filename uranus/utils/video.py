"""Convert and encode Uranus video frames.

Clamps to [-1, 1], maps to [0, 255], and returns one ``HxWx3`` RGB
``np.ndarray`` per (camera, time). H.264 encoding uses the system FFmpeg so the
codec is explicit rather than depending on OpenCV's build-time codec support.
"""


from pathlib import Path
import shutil
import subprocess

import cv2
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


def plucker_to_rgb_frames(plucker: torch.Tensor) -> list[np.ndarray]:
    """Visualize 6-D Plücker rays as moment-RGB | direction-RGB frames."""
    if plucker.ndim != 5 or plucker.shape[0] != 1 or plucker.shape[-1] != 6:
        raise ValueError(
            f"Expected Plücker tensor [1,F,H,W,6], got {tuple(plucker.shape)}"
        )
    values = plucker[0].detach().to("cpu", dtype=torch.float32).numpy()
    result = []
    for value in values:
        moment = value[..., :3]
        direction = value[..., 3:]
        scale = max(float(np.quantile(np.abs(moment), 0.99)), 1e-6)
        moment_rgb = (
            (np.clip(moment / scale, -1.0, 1.0) + 1.0) * 127.5
        ).astype(np.uint8)
        direction_rgb = (
            (np.clip(direction, -1.0, 1.0) + 1.0) * 127.5
        ).astype(np.uint8)
        height, width = moment_rgb.shape[:2]
        left_width = max(1, width // 2)
        right_width = max(1, width - left_width)
        moment_rgb = cv2.resize(
            moment_rgb,
            (left_width, height),
            interpolation=cv2.INTER_AREA,
        )
        direction_rgb = cv2.resize(
            direction_rgb,
            (right_width, height),
            interpolation=cv2.INTER_AREA,
        )
        result.append(np.concatenate((moment_rgb, direction_rgb), axis=1))
    return result


def write_h264_video(
    frames: list[np.ndarray],
    path: str | Path,
    *,
    fps: float,
) -> None:
    """Write RGB uint8 frames as an H.264/yuv420p MP4."""
    if not frames:
        return
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    height, width = frames[0].shape[:2]
    for index, frame in enumerate(frames):
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"frame {index} must be RGB uint8 with shape "
                f"{(height, width, 3)}, got {frame.shape} {frame.dtype}"
            )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to write H.264 videos")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed to write {path}: {message}")
