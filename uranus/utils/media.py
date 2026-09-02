"""Image decode, skeleton rendering, and camera-calibration helpers.

CPU-side preprocessing used by the runner's create/step flow: raw image bytes
are decoded with cv2, reference/step skeletons are rendered on top of the
robot models, and camera intrinsics are scaled between raw sensor size and
the model's target size. All helpers run on CPU; only their outputs move to
the GPU later.

WARNING: ``ChunkCalibration.per_camera_intrinsics`` holds the *raw* intrinsics
chunk while ``ChunkCalibration.intrinsics_list`` holds the *scaled* intrinsics
chunk. The names give no hint; the raw set feeds skeleton rendering, the
scaled set feeds the model — do not "fix" without re-validating end-to-end.

Validation errors raise ``ValueError`` (no HTTP error taxonomy here).
"""


from dataclasses import dataclass
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
from uranus.robot import make_robot_for_type, render_skeleton_frames
from uranus.robot.unifiedrobot import precompute_fk


def _require(condition: bool, *, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ensure_square_matrix(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


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


def to_numpy_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, tuple):
        return np.asarray(list(value), dtype=np.float64)
    if isinstance(value, dict):
        return {key: to_numpy_tree(item) for key, item in value.items()}
    return value


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


def require_robot_obs(robot_type: str, robot_obs: Any | None, *, field_name: str) -> None:
    """Raise if the robot is movable but no robot observation was provided."""
    robot = make_robot_for_type(robot_type)
    if robot.is_movable() and robot_obs is None:
        raise ValueError(
            f"robot_type={robot_type!r} requires {field_name} for skeleton rendering"
        )


class DecodedReference(NamedTuple):
    reference_images: list[torch.Tensor]
    reference_camera_extrinsics: list[np.ndarray]
    reference_camera_intrinsics: list[np.ndarray]  # scaled to target
    raw_sizes: dict[str, tuple[int, int]]
    camera_extrinsics_dict: dict[str, list[np.ndarray]]  # raw
    camera_intrinsics_dict: dict[str, list[np.ndarray]]  # raw


@dataclass(slots=True)
class ChunkCalibration:
    # per_camera_* are RAW (skeleton render); *_list are SCALED (model) — see module docstring.
    per_camera_extrinsics: dict[str, list[np.ndarray]]
    per_camera_intrinsics: dict[str, list[np.ndarray]]
    extrinsics_list: list[list[np.ndarray]]
    intrinsics_list: list[list[np.ndarray]]


def decode_reference_images(
    *,
    ref_images: tuple[bytes, ...],
    calib_ext: dict[str, np.ndarray],
    calib_int: dict[str, np.ndarray],
    camera_names: tuple[str, ...],
    target_size: tuple[int, int],
) -> DecodedReference:
    reference_images: list[torch.Tensor] = []
    reference_camera_extrinsics: list[np.ndarray] = []
    reference_camera_intrinsics: list[np.ndarray] = []
    raw_sizes: dict[str, tuple[int, int]] = {}
    camera_extrinsics_dict: dict[str, list[np.ndarray]] = {}
    camera_intrinsics_dict: dict[str, list[np.ndarray]] = {}
    for index, camera_name in enumerate(camera_names):
        image_tensor, raw_size = read_image_bytes(ref_images[index], target_size=target_size)
        raw_sizes[camera_name] = raw_size
        reference_images.append(image_tensor)
        extrinsic = _ensure_square_matrix(f"ref_cam_extrinsics[{camera_name}]", calib_ext.get(camera_name), (4, 4))
        intrinsic = _ensure_square_matrix(f"ref_cam_intrinsics[{camera_name}]", calib_int.get(camera_name), (3, 3))
        reference_camera_extrinsics.append(extrinsic)
        reference_camera_intrinsics.append(scale_intrinsic(intrinsic, raw_size=raw_size, target_size=target_size))
        camera_extrinsics_dict[camera_name] = [extrinsic]
        camera_intrinsics_dict[camera_name] = [intrinsic]
    return DecodedReference(
        reference_images=reference_images,
        reference_camera_extrinsics=reference_camera_extrinsics,
        reference_camera_intrinsics=reference_camera_intrinsics,
        raw_sizes=raw_sizes,
        camera_extrinsics_dict=camera_extrinsics_dict,
        camera_intrinsics_dict=camera_intrinsics_dict,
    )


def build_render_frames(
    *,
    qpos_sequence: list[Any],
    robot_obs_sequence: list[Any] | None,
    camera_extrinsics: dict[str, list[np.ndarray]],
    camera_intrinsics: dict[str, list[np.ndarray]],
    camera_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for frame_index, qpos in enumerate(qpos_sequence):
        frame: dict[str, Any] = {"observation.state": to_numpy_tree(qpos)}
        if robot_obs_sequence is not None:
            frame["observation.robot"] = to_numpy_tree(robot_obs_sequence[frame_index])
        for camera_name in camera_names:
            frame[f"camera_extrinsics.{camera_name}"] = np.asarray(
                camera_extrinsics[camera_name][frame_index], dtype=np.float64
            )
            frame[f"camera_intrinsics.{camera_name}"] = np.asarray(
                camera_intrinsics[camera_name][frame_index], dtype=np.float64
            )
        frames.append(frame)
    return frames


def render_skeleton_images(
    *,
    robot_type: str,
    frames: list[dict[str, Any]],
    camera_names: tuple[str, ...],
    raw_sizes: dict[str, tuple[int, int]],
    target_size: tuple[int, int],
) -> list[list[torch.Tensor]]:
    robot = make_robot_for_type(robot_type)
    fk_cache = precompute_fk(robot, frames)
    rendered: list[list[torch.Tensor]] = []
    for camera_name in camera_names:
        raw_height, raw_width = raw_sizes[camera_name]
        camera_video = render_skeleton_frames(
            robot, frames, camera_name, raw_height, raw_width, fk_cache=fk_cache, target_size=target_size
        )
        rendered.append([frame.contiguous() for frame in camera_video])
    return rendered


def render_reference_skeleton(
    *,
    robot_type: str,
    ref_qpos: np.ndarray,
    ref_robot_obs: np.ndarray | None,
    decoded: DecodedReference,
    camera_names: tuple[str, ...],
    target_size: tuple[int, int],
) -> list[torch.Tensor]:
    """Render the single reference skeleton frame per camera (the [0] frame)."""
    reference_frames = build_render_frames(
        qpos_sequence=[ref_qpos],
        robot_obs_sequence=[ref_robot_obs] if ref_robot_obs is not None else None,
        camera_extrinsics=decoded.camera_extrinsics_dict,
        camera_intrinsics=decoded.camera_intrinsics_dict,
        camera_names=camera_names,
    )
    rendered_reference = render_skeleton_images(
        robot_type=robot_type,
        frames=reference_frames,
        camera_names=camera_names,
        raw_sizes=decoded.raw_sizes,
        target_size=target_size,
    )
    return [camera_frames[0] for camera_frames in rendered_reference]


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


def scale_camera_intrinsics(
    *,
    camera_names: tuple[str, ...],
    camera_intrinsics: dict[str, list[np.ndarray]],
    raw_sizes: dict[str, tuple[int, int]],
    target_size: tuple[int, int],
) -> dict[str, list[np.ndarray]]:
    return {
        camera_name: [
            scale_intrinsic(intrinsic, raw_size=raw_sizes[camera_name], target_size=target_size)
            for intrinsic in camera_intrinsics[camera_name]
        ]
        for camera_name in camera_names
    }


def pad_step_inputs(
    *,
    camera_names: tuple[str, ...],
    qpos_sequence: list[Any],
    robot_obs_sequence: list[Any] | None,
    camera_extrinsics: dict[str, list[np.ndarray]],
    camera_intrinsics: dict[str, list[np.ndarray]],
    scaled_camera_intrinsics: dict[str, list[np.ndarray]],
    requested_steps: int,
    actual_steps: int,
) -> tuple[list[Any], list[Any] | None]:
    qpos_sequence = pad_list(qpos_sequence, requested_steps, name="qpos")
    if robot_obs_sequence is not None:
        robot_obs_sequence = pad_list(robot_obs_sequence, requested_steps, name="robot_obs")
    for camera_name in camera_names:
        camera_extrinsics[camera_name] = pad_list(
            camera_extrinsics[camera_name], requested_steps, name=f"cam_calib_ext[{camera_name}]"
        )
        camera_intrinsics[camera_name] = pad_list(
            camera_intrinsics[camera_name], requested_steps, name=f"cam_calib_int_raw[{camera_name}]"
        )
        scaled_camera_intrinsics[camera_name] = pad_list(
            scaled_camera_intrinsics[camera_name], requested_steps, name=f"cam_calib_int_scaled[{camera_name}]"
        )

    qpos_sequence = pad_list(qpos_sequence, actual_steps, name="qpos.chunk_align")
    if robot_obs_sequence is not None:
        robot_obs_sequence = pad_list(robot_obs_sequence, actual_steps, name="robot_obs.chunk_align")
    for camera_name in camera_names:
        camera_extrinsics[camera_name] = pad_list(
            camera_extrinsics[camera_name], actual_steps, name=f"cam_calib_ext[{camera_name}].chunk_align"
        )
        camera_intrinsics[camera_name] = pad_list(
            camera_intrinsics[camera_name], actual_steps, name=f"cam_calib_int_raw[{camera_name}].chunk_align"
        )
        scaled_camera_intrinsics[camera_name] = pad_list(
            scaled_camera_intrinsics[camera_name], actual_steps, name=f"cam_calib_int_scaled[{camera_name}].chunk_align"
        )
    return qpos_sequence, robot_obs_sequence


def build_chunk_calibration(
    *,
    camera_names: tuple[str, ...],
    camera_extrinsics: dict[str, list[np.ndarray]],
    camera_intrinsics: dict[str, list[np.ndarray]],
    scaled_camera_intrinsics: dict[str, list[np.ndarray]],
    start: int,
    chunk_size: int,
) -> ChunkCalibration:
    camera_extrinsics_chunk_dict: dict[str, list[np.ndarray]] = {}
    camera_intrinsics_chunk_dict: dict[str, list[np.ndarray]] = {}
    camera_extrinsics_chunk: list[list[np.ndarray]] = []
    camera_intrinsics_chunk: list[list[np.ndarray]] = []

    for camera_name in camera_names:
        extrinsics_chunk = [
            _ensure_square_matrix(
                f"cam_calib_ext[{camera_name}][{start + frame_offset}]",
                camera_extrinsics[camera_name][start + frame_offset],
                (4, 4),
            )
            for frame_offset in range(chunk_size)
        ]
        intrinsics_chunk_raw = [
            _ensure_square_matrix(
                f"cam_calib_int_raw[{camera_name}][{start + frame_offset}]",
                camera_intrinsics[camera_name][start + frame_offset],
                (3, 3),
            )
            for frame_offset in range(chunk_size)
        ]
        intrinsics_chunk_scaled = [
            _ensure_square_matrix(
                f"cam_calib_int_scaled[{camera_name}][{start + frame_offset}]",
                scaled_camera_intrinsics[camera_name][start + frame_offset],
                (3, 3),
            )
            for frame_offset in range(chunk_size)
        ]
        camera_extrinsics_chunk_dict[camera_name] = extrinsics_chunk
        camera_intrinsics_chunk_dict[camera_name] = intrinsics_chunk_raw
        camera_extrinsics_chunk.append(extrinsics_chunk)
        camera_intrinsics_chunk.append(intrinsics_chunk_scaled)

    return ChunkCalibration(
        per_camera_extrinsics=camera_extrinsics_chunk_dict,
        per_camera_intrinsics=camera_intrinsics_chunk_dict,
        extrinsics_list=camera_extrinsics_chunk,
        intrinsics_list=camera_intrinsics_chunk,
    )


def render_step_skeleton_chunk(
    *,
    qpos_chunk: list[np.ndarray],
    robot_obs_chunk: list[np.ndarray | None] | None,
    camera_names: tuple[str, ...],
    camera_extrinsics: dict[str, list[np.ndarray]],
    camera_intrinsics: dict[str, list[np.ndarray]],
    scaled_camera_intrinsics: dict[str, list[np.ndarray]],
    raw_sizes: dict[str, tuple[int, int]],
    target_size: tuple[int, int],
    robot_type: str,
    start: int,
    chunk_size: int,
) -> tuple[list[list[torch.Tensor]], ChunkCalibration]:
    """Build chunk calibration + render the per-camera skeleton frames for one chunk."""
    calib = build_chunk_calibration(
        camera_names=camera_names,
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        scaled_camera_intrinsics=scaled_camera_intrinsics,
        start=start,
        chunk_size=chunk_size,
    )
    frames = build_render_frames(
        qpos_sequence=qpos_chunk,
        robot_obs_sequence=robot_obs_chunk,
        camera_extrinsics=calib.per_camera_extrinsics,  # raw → skeleton render
        camera_intrinsics=calib.per_camera_intrinsics,  # raw → skeleton render
        camera_names=camera_names,
    )
    skeleton_images = render_skeleton_images(
        robot_type=robot_type,
        frames=frames,
        camera_names=camera_names,
        raw_sizes=raw_sizes,
        target_size=target_size,
    )
    return skeleton_images, calib
