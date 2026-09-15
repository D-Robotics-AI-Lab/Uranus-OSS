"""Export Lance episodes as simple XML-environment Uranus samples.

Each exported sample contains:

* one self-contained MJCF with camera calibration and mounts;
* ``meta.json`` with prompt, paths, and rendering configuration;
* ``temporal.json`` whose ``step_qpos`` frames contain complete MuJoCo qpos
  vectors and per-frame robot-to-world transforms.

The exporter reads both local and TOS Lance datasets through ``uranus-data``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mujoco
import numpy as np
import torch

from uranus_dataset import UranusMultiDataset
from uranus_dataset.robot import ROBOT_REGISTRY

from uranus.utils.video import write_h264_video


DATASET_ROOT = "tos://uranus-data/las/lance-dataset/robotics/"
DATASET_CONFIGS = {
    "agibot": {"camera_names": ["head", "hand_left", "hand_right"]},
    "agibot-challenge-2026": {
        "camera_names": ["head", "hand_left", "hand_right"],
        "source_repo_id": "agibot-world-2026",
    },
    "droid": {"camera_names": ["ext1", "ext2", "wrist"]},
    "rc-table30-v1/rc-aloha": {
        "camera_names": ["global", "left_wrist", "right_wrist"]
    },
    "rc-table30-v1/rc-arx5": {"camera_names": ["global", "side", "wrist"]},
    "rc-table30-v1/rc-franka": {"camera_names": ["main", "side", "wrist"]},
    "rc-table30-v1/rc-ur5": {"camera_names": ["global", "wrist"]},
    "rc-table30-v2/rc-aloha": {
        "camera_names": ["global", "left_wrist", "right_wrist"]
    },
    "rc-table30-v2/rc-arx5": {"camera_names": ["global", "side", "wrist"]},
    "rc-table30-v2/rc-dos-w1": {
        "camera_names": ["global", "left_wrist", "right_wrist"]
    },
    "rc-table30-v2/rc-ur5": {"camera_names": ["global", "wrist", "pad"]},
}

# Wrist/hand cameras are attached to these flange bodies. External cameras are
# attached to the robot base; a robot-mounted head camera keeps its head mount.
ROBOT_FLANGES = {
    "Panda Franka": ("link7",),
    "Agibot G1": ("Link7_l", "Link7_r"),
    "Agibot G2": ("arm_l_link7", "arm_r_link7"),
    "ARX5": ("link6",),
    "UR5": ("wrist_3_link",),
    "ALOHA": ("fl_link6", "fr_link6"),
    "DOS-W1": ("left_flange", "right_flange"),
    "RoboTwin2-ALOHA": ("fl_link6", "fr_link6"),
}

ROBOT_BASES = {
    "Panda Franka": "link0",
    "Agibot G1": None,
    "Agibot G2": "base_link",
    "ARX5": "base_link",
    "UR5": "base",
    "ALOHA": "footprint",
    "DOS-W1": None,
    "RoboTwin2-ALOHA": "footprint",
}

ROBOT_HEADS = {"Agibot G2": "head_link3"}

MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
ROBOT_BASE_BODY = "uranus_robot_base"


def _np(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_image(value) -> np.ndarray:
    image = _np(value)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {image.shape}")
    if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"expected 1, 3, or 4 image channels, got {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def _mat4(pos, rot) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return matrix


def _pose_to_attrs(matrix: np.ndarray) -> tuple[str, str]:
    """Return MuJoCo ``pos`` and ``quat`` (wxyz) attributes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(
                max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-15)
            ) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                max(1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2], 1e-15)
            ) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(
                max(1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2], 1e-15)
            ) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return (
        " ".join(f"{value:.12g}" for value in matrix[:3, 3]),
        " ".join(f"{value:.12g}" for value in quaternion),
    )


def _serialize_robot_transform(matrix: np.ndarray) -> list[list[float]]:
    """Serialize the robot-to-world transform as a 4x4 matrix."""
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()


def _asset_for_robot(robot_name: str) -> Path:
    try:
        robot_class, relative = ROBOT_REGISTRY[robot_name]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset robot {robot_name!r}") from exc
    del robot_class
    # Use the asset shipped with uranus-data first: its robot adapter and MJCF
    # are versioned together (for example, its ARX5 adds gripper joints that an
    # older copy of the asset does not contain).
    import uranus_dataset

    package_asset = Path(uranus_dataset.__file__).resolve().parents[1] / relative
    if package_asset.is_file():
        return package_asset
    project_asset = PROJECT_ROOT / "assets" / Path(relative).name
    if project_asset.is_file():
        return project_asset
    raise FileNotFoundError(f"robot asset not found: {package_asset} or {project_asset}")


def _make_robot(robot_name: str, asset: Path):
    robot_class, _ = ROBOT_REGISTRY[robot_name]
    return robot_class(mjcf_path=str(asset))


def _mobile_frame(item: dict, child, frame_index: int, episode_index: int = 0):
    reference = item.get("reference.observation.mobile_base")
    if frame_index == 0 and isinstance(reference, dict):
        return {key: _np(value).copy() for key, value in reference.items()}
    sequence = item.get("observation.mobile_base")
    if frame_index > 0 and isinstance(sequence, dict):
        return {key: _np(value)[frame_index - 1].copy() for key, value in sequence.items()}
    if frame_index == 0:
        episode_position = child._episode_position(episode_index)
        start = int(child._episode_from[episode_position])
        row = child._take_rows([start])[start]
        value = row["observation"].get("mobile_base")
        if value is not None:
            return {key: np.asarray(item).copy() for key, item in value.items()}
    return None


def _episode_frames(item: dict, child, episode_index: int = 0) -> list[dict]:
    states = [
        _np(item["reference.observation.state"]).copy(),
        *[_np(value).copy() for value in item["observation.state"]],
    ]
    frames = []
    for index, state in enumerate(states):
        frame = {"observation.state": state}
        mobile = _mobile_frame(item, child, index, episode_index)
        if mobile is not None:
            frame["observation.mobile_base"] = mobile
        frames.append(frame)
    return frames


def _resample_episode(item: dict, child, target_fps: float | None) -> tuple[dict, float]:
    source_fps = float(child.fps)
    if target_fps is None or target_fps <= 0 or target_fps >= source_fps:
        return item, source_fps
    timestamps = [float(_np(item["reference.timestamp"]).reshape(-1)[0])]
    timestamps.extend(float(value) for value in _np(item["timestamp"]).reshape(-1))
    duration = timestamps[-1] - timestamps[0]
    targets = np.arange(0.0, duration + 1e-9, 1.0 / target_fps)
    indices = sorted(
        {
            0,
            *(
                int(
                    np.argmin(
                        np.abs(np.asarray(timestamps) - (timestamps[0] + target))
                    )
                )
                for target in targets
            ),
        }
    )
    temporal_indices = [index - 1 for index in indices if index > 0] or [0]
    sampled = dict(item)
    sampled["observation.state"] = item["observation.state"][temporal_indices]
    sampled["timestamp"] = item["timestamp"][temporal_indices]
    for camera in child.camera_names:
        for prefix in ("camera_intrinsics", "camera_extrinsics", "observation.images"):
            key = f"{prefix}.{camera}"
            if key in item:
                sampled[key] = item[key][temporal_indices]
    mobile = item.get("observation.mobile_base")
    if isinstance(mobile, dict):
        sampled["observation.mobile_base"] = {
            key: value[temporal_indices] for key, value in mobile.items()
        }
    return sampled, float(target_fps)


def _slice_episode(item: dict, child, start_frame: int) -> dict:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    temporal_length = int(_np(item["observation.state"]).shape[0])
    if start_frame >= temporal_length + 1:
        raise ValueError(
            f"start_frame {start_frame} is outside episode of {temporal_length + 1} frames"
        )
    if start_frame == 0:
        return item
    reference_index = start_frame - 1
    sliced = dict(item)
    sliced["reference.observation.state"] = item["observation.state"][reference_index]
    sliced["observation.state"] = item["observation.state"][reference_index + 1 :]
    sliced["reference.timestamp"] = item["timestamp"][reference_index]
    sliced["timestamp"] = item["timestamp"][reference_index + 1 :]
    mobile = item.get("observation.mobile_base")
    if isinstance(mobile, dict):
        sliced["reference.observation.mobile_base"] = {
            key: value[reference_index] for key, value in mobile.items()
        }
        sliced["observation.mobile_base"] = {
            key: value[reference_index + 1 :] for key, value in mobile.items()
        }
    for camera in child.camera_names:
        for prefix in ("camera_intrinsics", "camera_extrinsics", "observation.images"):
            key = f"{prefix}.{camera}"
            if key in item:
                sliced[f"reference.{prefix}.{camera}"] = item[key][reference_index]
                sliced[key] = item[key][reference_index + 1 :]
    return sliced


def _camera_calibration(item: dict, camera: str) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = _np(item[f"reference.camera_intrinsics.{camera}"]).astype(np.float64)
    extrinsic = _np(item[f"reference.camera_extrinsics.{camera}"]).astype(np.float64)
    return intrinsic, extrinsic


def _build_robot_frames(robot, frames: list[dict]):
    model_to_robot = np.asarray(
        getattr(robot, "_model_to_robot", np.eye(4)), dtype=np.float64
    )
    qpos, poses, robot_to_world = [], [], []
    for frame in frames:
        robot.build_qpos(frame)
        qpos.append(np.asarray(robot.data.qpos, dtype=np.float64).copy())
        pose_map = {}
        for body_id in range(1, robot.model.nbody):
            name = mujoco.mj_id2name(
                robot.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            pose_map[name] = model_to_robot @ _mat4(
                robot.data.xpos[body_id], robot.data.xmat[body_id]
            )
        poses.append(pose_map)
        robot_to_world.append(
            np.asarray(robot.get_robot_to_world_transform(), dtype=np.float64).reshape(4, 4)
        )
    return qpos, poses, robot_to_world, model_to_robot


def _camera_mount(robot_name: str, camera: str) -> str:
    lowered = camera.lower()
    if lowered == "head" and robot_name in ROBOT_HEADS:
        return ROBOT_HEADS[robot_name]
    wrist_camera = "wrist" in lowered or "hand" in lowered
    if not wrist_camera:
        return ROBOT_BASES[robot_name] or ROBOT_BASE_BODY
    flanges = ROBOT_FLANGES[robot_name]
    if len(flanges) == 1:
        return flanges[0]
    if "left" in lowered:
        return flanges[0]
    if "right" in lowered:
        return flanges[1]
    raise ValueError(
        f"camera {camera!r} is wrist-mounted but does not identify left/right for {robot_name}"
    )


def _bake_model_frame(root: ET.Element, asset: Path, model_to_robot: np.ndarray) -> None:
    if np.allclose(model_to_robot, np.eye(4), atol=1e-10):
        return
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{asset} has no worldbody")
    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for element in worldbody.findall("body"):
        name = element.get("name")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id <= 0:
            continue
        pose = model_to_robot @ _mat4(data.xpos[body_id], data.xmat[body_id])
        pos, quat = _pose_to_attrs(pose)
        element.set("pos", pos)
        element.set("quat", quat)
        element.attrib.pop("euler", None)


def _ensure_robot_base(root: ET.Element, robot_name: str) -> ET.Element:
    configured_name = ROBOT_BASES[robot_name]
    if configured_name is not None:
        configured = root.find(f".//body[@name='{configured_name}']")
        if configured is None:
            raise ValueError(
                f"robot base body {configured_name!r} not found for {robot_name}"
            )
        return configured
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    existing = root.find(f".//body[@name='{ROBOT_BASE_BODY}']")
    if existing is not None:
        return existing
    base = ET.Element("body", {"name": ROBOT_BASE_BODY})
    bodies = list(worldbody.findall("body"))
    for body in bodies:
        worldbody.remove(body)
        base.append(body)
    worldbody.append(base)
    return base


def _inject_camera(
    root: ET.Element,
    *,
    name: str,
    mount: str,
    camera_from_mount: np.ndarray,
    intrinsics: np.ndarray,
    raw_size: tuple[int, int],
) -> None:
    parent = root.find(f".//body[@name='{mount}']")
    if parent is None:
        raise ValueError(f"camera mount body {mount!r} not found")
    height, width = raw_size
    camera_pose_mj = np.linalg.inv(camera_from_mount) @ MUJOCO_TO_CV
    pos, quat = _pose_to_attrs(camera_pose_mj)
    ET.SubElement(
        parent,
        "camera",
        {
            "name": name,
            "resolution": f"{width} {height}",
            "sensorsize": f"{width} {height}",
            "focal": f"{float(intrinsics[0, 0]):.12g} {float(intrinsics[1, 1]):.12g}",
            "principal": (
                f"{float(intrinsics[0, 2]):.12g} {float(intrinsics[1, 2]):.12g}"
            ),
            "pos": pos,
            "quat": quat,
        },
    )


def _ee_and_skeleton(robot_name: str, robot) -> tuple[list[dict], dict]:
    corrections = robot.get_ee_sh_corrections()
    if hasattr(robot, "ee_names"):
        names = tuple(robot.ee_names)
        are_sites = bool(getattr(robot, "ee_are_sites", False))
        pairs = tuple(robot.gripper_body_pairs)
        end_effectors = [
            {
                "object_type": "site" if are_sites else "body",
                "object_name": name,
                "radius_mode": "pad_pair",
                "pad_bodies": list(pair),
                "sh_correction": np.asarray(correction).tolist(),
            }
            for name, pair, correction in zip(names, pairs, corrections, strict=True)
        ]
        skeleton = {
            "mode": "chains",
            "chains": [list(chain) for chain in robot.skeleton_body_names],
            "skip_bodies": [],
        }
        if getattr(robot, "render_gripper_keypoints_from_width", False):
            skeleton["gripper_keypoint_overrides"] = [
                {
                    "ee_object_type": "site" if are_sites else "body",
                    "ee_object_name": name,
                    "finger_bodies": list(pair),
                    "closing_axis": 1,
                }
                for name, pair in zip(names, pairs, strict=True)
            ]
        return end_effectors, skeleton
    if robot_name == "Panda Franka":
        return [
            {
                "object_type": "site",
                "object_name": "pinch",
                "radius_mode": "pad_pair",
                "pad_bodies": ["right_silicone_pad", "left_silicone_pad"],
                "sh_correction": np.asarray(corrections[0]).tolist(),
            }
        ], {"mode": "full_tree", "skip_bodies": ["gripper_center"]}
    if robot_name in {"Agibot G1", "Agibot G2"}:
        if robot_name == "Agibot G1":
            from uranus_dataset.robot.robot_g1_120s import (
                LEFT_ARM_BODIES,
                LEFT_GRIP_BODIES,
                RIGHT_ARM_BODIES,
                RIGHT_GRIP_BODIES,
            )
        else:
            from uranus_dataset.robot.robot_g2_120s import (
                LEFT_ARM_BODIES,
                LEFT_GRIP_BODIES,
                RIGHT_ARM_BODIES,
                RIGHT_GRIP_BODIES,
            )
        end_effectors = [
            {
                "object_type": "site",
                "object_name": name,
                "radius_mode": "pad_pair",
                "pad_bodies": [f"{side}_Left_Pad_Link", f"{side}_Right_Pad_Link"],
                "sh_correction": np.asarray(correction).tolist(),
            }
            for name, side, correction in zip(
                ("gripper_center", "right_gripper_center"),
                ("left", "right"),
                corrections,
                strict=True,
            )
        ]
        return end_effectors, {
            "mode": "chains",
            "chains": [
                [*LEFT_ARM_BODIES, *LEFT_GRIP_BODIES],
                [*RIGHT_ARM_BODIES, *RIGHT_GRIP_BODIES],
            ],
            "skip_bodies": [],
        }
    raise ValueError(f"no end-effector mapping for robot {robot_name!r}")


def _write_video(frames: list[np.ndarray], path: Path, *, fps: float) -> None:
    write_h264_video(frames, path, fps=fps)


def export_repo(
    repo_id: str,
    config: dict,
    output_root: Path,
    data_root: str,
    target_fps: float | None = None,
    episode_index: int = 0,
    start_frame: int = 0,
) -> Path:
    """Export one Lance episode and return its new sample directory."""
    cameras = list(config["camera_names"])
    source_repo_id = str(config.get("source_repo_id", repo_id))
    dataset = UranusMultiDataset(
        [source_repo_id],
        root=data_root,
        camera_names={source_repo_id: cameras},
        shared_features_only=False,
        return_uint8=True,
    )
    item = dataset.get_episode(episode_index)
    child = dataset._datasets[0]
    item, export_fps = _resample_episode(item, child, target_fps)
    item = _slice_episode(item, child, start_frame)

    robot_name = child.meta.get_episode_robot(episode_index) or child.meta.robot
    asset = _asset_for_robot(robot_name)
    robot = _make_robot(robot_name, asset)
    frames = _episode_frames(item, child, episode_index)
    qpos, body_poses, robot_to_world, model_to_robot = _build_robot_frames(robot, frames)

    step_qpos = [
        {
            "state": values.tolist(),
            "robot2world_transform": _serialize_robot_transform(transform),
        }
        for values, transform in zip(qpos, robot_to_world, strict=True)
    ]

    first_images = {
        camera: _rgb_image(item[f"reference.observation.images.{camera}"])
        for camera in cameras
    }
    sample_id = len(list(output_root.glob("*/meta.json")))
    sample_dir = output_root / f"{sample_id:06d}"
    ref_dir = sample_dir / "ref_images"
    xml_path = sample_dir / "mjcf" / asset.name

    ref_images = {}
    gt_videos = {}
    for camera in cameras:
        ref_path = ref_dir / f"{camera}.png"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(ref_path), cv2.cvtColor(first_images[camera], cv2.COLOR_RGB2BGR)
        ):
            raise OSError(f"failed to write {ref_path}")
        ref_images[camera] = str(ref_path.relative_to(sample_dir))
        target_frames = [
            _rgb_image(frame) for frame in item[f"observation.images.{camera}"]
        ]
        video_path = sample_dir / "gt" / f"{camera}.mp4"
        _write_video(target_frames, video_path, fps=export_fps)
        gt_videos[camera] = str(video_path.relative_to(sample_dir))

    root = ET.parse(asset).getroot()
    _bake_model_frame(root, asset, model_to_robot)
    _ensure_robot_base(root, robot_name)
    for camera in cameras:
        intrinsic, world_to_camera = _camera_calibration(item, camera)
        mount = _camera_mount(robot_name, camera)
        if mount == (ROBOT_BASES[robot_name] or ROBOT_BASE_BODY):
            camera_from_mount = world_to_camera @ robot_to_world[0]
        else:
            camera_from_mount = (
                world_to_camera @ robot_to_world[0] @ body_poses[0][mount]
            )
        raw_size = first_images[camera].shape[:2]
        _inject_camera(
            root,
            name=camera,
            mount=mount,
            camera_from_mount=camera_from_mount,
            intrinsics=intrinsic,
            raw_size=(int(raw_size[0]), int(raw_size[1])),
        )
    ET.indent(root, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    # Fail at export time if the generated environment or cameras are invalid.
    mujoco.MjModel.from_xml_path(str(xml_path))

    end_effectors, skeleton = _ee_and_skeleton(robot_name, robot)
    meta = {
        "format_version": 8,
        "prompt": str(item.get("task", "")),
        "mjcf_path": str(xml_path.relative_to(sample_dir)),
        "cameras": cameras,
        "end_effectors": end_effectors,
        "skeleton": skeleton,
        "height": int(next(iter(first_images.values())).shape[0]),
        "width": int(next(iter(first_images.values())).shape[1]),
        "fps": export_fps,
        "ref_cam_images": ref_images,
        "gt_videos": gt_videos,
        "migrated_from": {
            "repo_id": repo_id,
            "source_repo_id": source_repo_id,
            "episode_index": episode_index,
            "start_frame": start_frame,
            "source_fps": float(child.fps),
            "robot_type": robot_name,
        },
    }
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (sample_dir / "temporal.json").write_text(
        json.dumps({"step_qpos": step_qpos}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sample_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Lance episodes to XML-environment Uranus samples"
    )
    parser.add_argument("--output-root", type=Path, default=Path("examples/data"))
    parser.add_argument("--data-root", default=DATASET_ROOT)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--repo", action="append", choices=sorted(DATASET_CONFIGS))
    args = parser.parse_args()
    for repo_id in args.repo or list(DATASET_CONFIGS):
        path = export_repo(
            repo_id,
            DATASET_CONFIGS[repo_id],
            args.output_root,
            args.data_root,
            args.fps,
            args.episode_index,
            args.start_frame,
        )
        print(f"exported {repo_id} -> {path}", flush=True)


if __name__ == "__main__":
    main()
