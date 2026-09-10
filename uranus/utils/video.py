"""Convert the model's raw decoded video tensor into per-camera uint8 frames.

Clamps to [-1, 1], maps to [0, 255], and returns one ``HxWx3`` RGB
``np.ndarray`` per (camera, time). PNG encoding / base64 are a transport
concern and deliberately not handled here.
"""


import numpy as np
import torch
import subprocess
from pathlib import Path


def encode_h264(
    frames: list[np.ndarray],
    destination: str | Path,
    *,
    fps: int | float,
    crf: int = 23,
    preset: str = "medium",
) -> None:
    """Stream RGB frames directly to a browser-friendly H.264 MP4."""
    if not frames:
        return
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(
            f"Expected RGB frames with shape (H, W, 3), got {first.shape}"
        )
    height, width = int(first.shape[0]), int(first.shape[1])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame in frames:
            array = np.asarray(frame)
            if array.shape != first.shape:
                raise ValueError(
                    f"Frame shape changed while encoding {destination}: "
                    f"expected {first.shape}, got {array.shape}"
                )
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            process.stdin.write(np.ascontiguousarray(array).tobytes())
        process.stdin.close()
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(
                stderr.strip() or f"ffmpeg exited with status {returncode}"
            )
        temporary.replace(destination)
    except BaseException as error:
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(
            f"Failed to encode H.264 video {destination}: {error}"
        ) from error


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
