"""Image decode and camera-calibration helpers (v2 rig pipeline).

CPU-side preprocessing used by the runner's create/step flow: raw image bytes
are decoded with cv2 (only the raw size matters for the rig — reference
pixels feed the VAE, not the skeleton). Camera intrinsics live in the rig and
are scaled once per session from the raw sensor size to the model's target
size; extrinsics are *composed per frame* by ``SkeletonEngine`` and arrive
here already in world frame. All helpers run on CPU; only their outputs move
to the GPU later.

Validation errors raise ``ValueError`` (no HTTP error taxonomy here).
"""


from typing import Any, NamedTuple

import cv2
import numpy as np
import torch


def _require(condition: bool, *, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_hw(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair of [height, width]")
    return int(value[0]), int(value[1])


def scale_intrinsic(intrinsic: np.ndarray, *, raw_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    raw_height, raw_width = raw_size
    target_height, target_width = target_size
    if raw_height <= 0 or raw_width <= 0:
        raise ValueError(f"Invalid raw image size for intrinsic scaling: {(raw_height, raw_width)}")
    scaled = intrinsic.astype(np.float64, copy=True)
    scaled[0, :] *= target_width / raw_width
    scaled[1, :] *= target_height / raw_height
    return scaled


def pad_list(values: list[Any], target_len: int, *, name: str) -> list[Any]:
    if not values:
        raise ValueError(f"Sequence must not be empty: {name}")
    if len(values) >= target_len:
        return list(values[:target_len])
    return list(values) + [values[-1]] * (target_len - len(values))


def read_image_bytes(data: bytes, *, target_size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    if not data:
        raise ValueError("Empty image upload")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image upload")
    raw_height, raw_width = bgr.shape[0], bgr.shape[1]
    if (raw_height, raw_width) != (target_size[0], target_size[1]):
        bgr = cv2.resize(bgr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous(), (raw_height, raw_width)


class DecodedReference(NamedTuple):
    reference_images: list[torch.Tensor]
    raw_sizes: dict[str, tuple[int, int]]


def decode_reference_images(
    *,
    ref_images: tuple[bytes, ...],
    camera_names: tuple[str, ...],
    target_size: tuple[int, int],
) -> DecodedReference:
    """Decode the per-camera reference images; return tensors + raw sizes.

    Reference pixels feed the VAE (identity/appearance conditioning); the raw
    sizes drive the per-camera intrinsics scaling. Camera calibration itself
    lives in the rig (``uranus.skeleton.specs``), not here.
    """
    reference_images: list[torch.Tensor] = []
    raw_sizes: dict[str, tuple[int, int]] = {}
    for index, camera_name in enumerate(camera_names):
        image_tensor, raw_size = read_image_bytes(ref_images[index], target_size=target_size)
        raw_sizes[camera_name] = raw_size
        reference_images.append(image_tensor)
    return DecodedReference(
        reference_images=reference_images,
        raw_sizes=raw_sizes,
    )


def resolve_step_raw_sizes(
    camera_names: tuple[str, ...],
    height: int,
    width: int,
    raw_sizes: dict[str, tuple[int, int]] | None,
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for camera_name in camera_names:
        if raw_sizes is None or camera_name not in raw_sizes:
            result[camera_name] = (height, width)
        else:
            result[camera_name] = normalize_hw(f"raw_sizes[{camera_name}]", raw_sizes[camera_name])
    return result


def pad_qpos_sequence(
    qpos_sequence: list[np.ndarray],
    T_sequence: list[np.ndarray | None] | None,
    requested_steps: int,
    actual_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray | None] | None]:
    """Repeat-last pad the step inputs to the requested then chunk-aligned length.

    qpos and the optional robot-to-world transforms are padded in lockstep so
    frame i of one always pairs with frame i of the other.
    """
    qpos_sequence = pad_list(list(qpos_sequence), requested_steps, name="qpos")
    qpos_sequence = pad_list(qpos_sequence, actual_steps, name="qpos.chunk_align")
    if T_sequence is not None:
        T_sequence = pad_list(list(T_sequence), requested_steps, name="robot2world")
        T_sequence = pad_list(T_sequence, actual_steps, name="robot2world.chunk_align")
    return qpos_sequence, T_sequence
