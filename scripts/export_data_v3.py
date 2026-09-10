"""Export one complete episode per Uranus dataset in the XML-camera format.

The exporter deliberately uses ``get_episode`` rather than ``__getitem__``.
Robot FK turns the dataset's native observation state into complete MJCF qpos.
The XML retains a portable camera rig fallback, while exact per-frame camera
extrinsics and measured gripper geometry are kept in temporal.json so Uranus
inference receives the same conditions as the dataset episode API.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


# ``python scripts/export_data_v3.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add the root before importing local packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mujoco
import numpy as np
import torch

from uranus_dataset import UranusMultiDataset
from uranus_dataset.robot import ROBOT_REGISTRY
from uranus.utils.video import encode_h264


DATASET_ROOT = "tos://uranus-data/las/lance-dataset/robotics/"
DATASET_CONFIGS = {
    "agibot": {"camera_names": ["head", "hand_left", "hand_right"]},
    # TOS retains the historical storage path; v3 exposes the challenge name.
    "agibot-challenge-2026": {
        "camera_names": ["head", "hand_left", "hand_right"],
        "source_repo_id": "agibot-world-2026",
    },
    "droid": {"camera_names": ["ext1", "ext2", "wrist"]},
    "rc-table30-v1/rc-aloha": {"camera_names": ["global", "left_wrist", "right_wrist"]},
    "rc-table30-v1/rc-arx5": {"camera_names": ["global", "side", "wrist"]},
    "rc-table30-v1/rc-franka": {"camera_names": ["main", "side", "wrist"]},
    "rc-table30-v1/rc-ur5": {"camera_names": ["global", "wrist"]},
    "rc-table30-v2/rc-aloha": {"camera_names": ["global", "left_wrist", "right_wrist"]},
    "rc-table30-v2/rc-arx5": {"camera_names": ["global", "side", "wrist"]},
    "rc-table30-v2/rc-dos-w1": {"camera_names": ["global", "left_wrist", "right_wrist"]},
    "rc-table30-v2/rc-ur5": {"camera_names": ["global", "wrist", "pad"]},
}

PROJECT_ASSETS = PROJECT_ROOT / "assets"
MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
SH_CORRECTIONS = {
    "Agibot G1": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64),
    "Agibot G2": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64),
    "Panda Franka": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64),
    "UR5": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64),
    "ALOHA": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64),
    "RoboTwin2-ALOHA": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64),
    "ARX5": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64),
    "DOS-W1": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64),
}


def _np(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_image(value) -> np.ndarray:
    """Normalize one dataset image to contiguous HWC uint8 RGB."""
    image = _np(value)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {image.shape}")
    if image.shape[-1] in (1, 3, 4):
        pass
    elif image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"expected 1, 3, or 4 image channels, got shape {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def _mat4(pos, rot) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return out


def _pose_to_attrs(matrix: np.ndarray) -> tuple[str, str]:
    """Return MuJoCo ``pos`` and ``quat`` (wxyz) attributes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    R = matrix[:3, :3]
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        diagonal = np.diag(R)
        i = int(np.argmax(diagonal))
        if i == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-15)) * 2.0
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(max(1.0 - R[0, 0] + R[1, 1] - R[2, 2], 1e-15)) * 2.0
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 - R[0, 0] - R[1, 1] + R[2, 2], 1e-15)) * 2.0
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    pos = matrix[:3, 3]
    return " ".join(f"{v:.12g}" for v in pos), " ".join(f"{v:.12g}" for v in quat)


def _asset_for_robot(robot_name: str) -> Path:
    try:
        _, relative = ROBOT_REGISTRY[robot_name]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset robot {robot_name!r}") from exc
    path = PROJECT_ASSETS / Path(relative).name
    if not path.is_file():
        raise FileNotFoundError(f"project asset is missing: {path}")
    return path


def _make_robot(robot_name: str, asset: Path):
    cls, _ = ROBOT_REGISTRY[robot_name]
    return cls(mjcf_path=str(asset))


def _mobile_frame(item: dict, child, frame_index: int, total: int, episode_index: int = 0):
    reference = item.get("reference.observation.mobile_base")
    if frame_index == 0 and isinstance(reference, dict):
        return {key: _np(value).copy() for key, value in reference.items()}
    sequence = item.get("observation.mobile_base")
    if isinstance(sequence, dict):
        if frame_index > 0:
            return {
                key: _np(value)[frame_index - 1].copy()
                for key, value in sequence.items()
            }
    # Current uranus-data releases omit reference.mobile_base from get_episode.
    # Recover that one row without invoking __getitem__.
    if frame_index == 0:
        episode_pos = child._episode_position(episode_index)
        start = int(child._episode_from[episode_pos])
        rows = child._take_rows([start])
        value = rows[start]["observation"].get("mobile_base")
        if value is not None:
            return {key: np.asarray(val).copy() for key, val in value.items()}
    return None


def _episode_frames(item: dict, child, episode_index: int = 0) -> list[dict]:
    reference_state = _np(item["reference.observation.state"]).copy()
    states = [_np(value).copy() for value in item["observation.state"]]
    all_states = [reference_state, *states]
    frames = []
    for index, state in enumerate(all_states):
        frame = {"observation.state": state}
        mobile = _mobile_frame(item, child, index, len(all_states), episode_index)
        if mobile is not None:
            frame["observation.mobile_base"] = mobile
        frames.append(frame)
    return frames


def _resample_episode(item: dict, child, target_fps: float | None) -> tuple[dict, float]:
    """Synchronously subsample an episode to ``target_fps``.

    The reference frame is index 0; all temporal state, camera calibration,
    images, and mobile-base fields are selected by the same source indices.
    """
    source_fps = float(child.fps)
    if target_fps is None or target_fps <= 0 or target_fps >= source_fps:
        return item, source_fps
    timestamps = [float(_np(item["reference.timestamp"]).reshape(-1)[0])]
    timestamps.extend(float(v) for v in _np(item["timestamp"]).reshape(-1))
    duration = timestamps[-1] - timestamps[0]
    targets = np.arange(0.0, duration + 1e-9, 1.0 / target_fps)
    indices = []
    for target in targets:
        indices.append(int(np.argmin(np.abs(np.asarray(timestamps) - (timestamps[0] + target)))))
    indices = sorted(set([0, *indices]))
    temporal_indices = [index - 1 for index in indices if index > 0]
    if not temporal_indices:
        temporal_indices = [0]
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
    """Rebase an already-sampled episode so ``start_frame`` becomes frame 0."""
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    temporal_len = int(_np(item["observation.state"]).shape[0])
    if start_frame >= temporal_len + 1:
        raise ValueError(
            f"start_frame {start_frame} is outside episode of {temporal_len + 1} frames"
        )
    if start_frame == 0:
        return item
    # The complete sequence is [reference, temporal[0], temporal[1], ...].
    ref_temporal_index = start_frame - 1
    sliced = dict(item)
    sliced["reference.observation.state"] = item["observation.state"][ref_temporal_index]
    sliced["observation.state"] = item["observation.state"][ref_temporal_index + 1 :]
    sliced["reference.timestamp"] = item["timestamp"][ref_temporal_index]
    sliced["timestamp"] = item["timestamp"][ref_temporal_index + 1 :]
    mobile = item.get("observation.mobile_base")
    if isinstance(mobile, dict):
        sliced["reference.observation.mobile_base"] = {
            key: value[ref_temporal_index] for key, value in mobile.items()
        }
        sliced["observation.mobile_base"] = {
            key: value[ref_temporal_index + 1 :] for key, value in mobile.items()
        }
    for camera in child.camera_names:
        for prefix in ("camera_intrinsics", "camera_extrinsics", "observation.images"):
            key = f"{prefix}.{camera}"
            if key in item:
                sliced[f"reference.{prefix}.{camera}"] = item[key][ref_temporal_index]
                sliced[key] = item[key][ref_temporal_index + 1 :]
    return sliced


def _camera_sequence(item: dict, camera: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    K = _np(item[f"reference.camera_intrinsics.{camera}"]).astype(np.float64)
    E_ref = _np(item[f"reference.camera_extrinsics.{camera}"]).astype(np.float64)
    Ks = _np(item[f"camera_intrinsics.{camera}"]).astype(np.float64)
    Es = _np(item[f"camera_extrinsics.{camera}"]).astype(np.float64)
    if Ks.ndim == 2:
        Ks = Ks[None]
    if Es.ndim == 2:
        Es = Es[None]
    all_K = np.concatenate([K[None], Ks], axis=0)
    all_E = np.concatenate([E_ref[None], Es], axis=0)
    if not np.allclose(all_K, all_K[0], atol=1e-4, rtol=1e-6):
        raise ValueError(f"camera {camera!r} has time-varying intrinsics")
    return all_K[0], all_E[0], [np.asarray(value) for value in all_E]


def _body_poses(robot, frames: list[dict]):
    C = np.asarray(getattr(robot, "_model_to_robot", np.eye(4)), dtype=np.float64)
    Ts, poses = [], []
    qpos, ee_radii, gripper_widths = [], [], []
    for frame in frames:
        robot.build_qpos(frame)
        qpos.append(np.asarray(robot.data.qpos, dtype=np.float64).copy())
        ee_radii.append(
            [float(state[2]) for state in robot.get_ee_states()]
        )
        widths = getattr(robot, "_gripper_widths", None)
        gripper_widths.append(
            np.asarray(widths, dtype=np.float64).reshape(-1).copy()
            if widths is not None
            else None
        )
        T = np.asarray(robot.get_robot_to_world_transform(), dtype=np.float64).reshape(4, 4)
        Ts.append(T)
        body_map = {}
        for body_id in range(1, robot.model.nbody):
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            body_map[name] = C @ _mat4(robot.data.xpos[body_id], robot.data.xmat[body_id])
        poses.append(body_map)
    return qpos, poses, Ts, ee_radii, gripper_widths


def _camera_mount(robot_name: str, camera: str, E_all: list[np.ndarray], poses, Ts) -> tuple[str | None, np.ndarray]:
    """Fit external/mounted topology and return (body name, camera-from-body)."""
    external_error = max(float(np.linalg.norm(E - E_all[0])) for E in E_all)
    lowered = camera.lower()
    movable_all = robot_name in {"Agibot G1", "Agibot G2"}
    hints = []
    if "head" in lowered:
        hints = ["head_link3", "head_link2", "head_link1"]
    elif "left" in lowered:
        hints = ["arm_l_link7", "Link7_l", "fl_link6", "left_link6", "left_end_link"]
    elif "right" in lowered:
        hints = ["arm_r_link7", "Link7_r", "fr_link6", "right_link6", "right_end_link"]
    elif "wrist" in lowered or "hand" in lowered or "pad" in lowered:
        hints = ["wrist_3_link", "link6", "fl_link6", "fr_link6", "left_link6", "right_link6", "gripper_center"]

    names = list(poses[0])
    scores = []
    for name in names:
        rel = [E @ (T @ pose[name]) for E, T, pose in zip(E_all, Ts, poses)]
        error = max(float(np.linalg.norm(value - rel[0])) for value in rel)
        hint_rank = hints.index(name) if name in hints else len(hints)
        scores.append((error, hint_rank, name, rel[0]))
    best = None
    if scores:
        minimum_error = min(value[0] for value in scores)
        # Treat names as a tie-breaker among transforms that fit equally well;
        # never let a semantic hint override a substantially better FK fit.
        candidates = [value for value in scores if value[0] <= minimum_error + 1e-3]
        candidates.sort(
            key=lambda value: (
                value[1] if value[1] < len(hints) else 99,
                value[0],
            )
        )
        best = candidates[0]
    if movable_all:
        if best is None:
            raise ValueError(f"cannot find a mounted body for {robot_name}/{camera}")
        return best[2], best[3]
    # Wrist/hand names are an explicit semantic hint.  If the arm did not
    # move enough to disambiguate a mounted camera numerically, retain that
    # known topology; ``pad`` is deliberately excluded because UR5 records
    # contain both external and wrist-mounted pad cameras.
    wrist_hint = "wrist" in lowered or "hand" in lowered
    if external_error <= 1e-4:
        if wrist_hint and best is not None and best[0] <= 1e-3:
            return best[2], best[3]
        return None, E_all[0]
    if best is None or best[0] > max(external_error * 2.0, 1e-3):
        return None, E_all[0]
    return best[2], best[3]


def _bake_model_to_robot(root: ET.Element, asset: Path, C: np.ndarray) -> None:
    if np.allclose(C, np.eye(4), atol=1e-10):
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
        pose = C @ _mat4(data.xpos[body_id], data.xmat[body_id])
        pos, quat = _pose_to_attrs(pose)
        element.set("pos", pos)
        element.set("quat", quat)
        element.attrib.pop("euler", None)


def _inject_sh_corrections(xml_path: Path, robot_name: str, ees: list[dict]) -> None:
    """Canonicalize each EE frame in-place; SH then needs no meta matrix."""
    correction = SH_CORRECTIONS.get(robot_name)
    if correction is None:
        return
    tree = ET.parse(xml_path)
    root = tree.getroot()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for ee in ees:
        name = ee["object_name"]
        if ee["object_type"] == "site":
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"EE site {name!r} not found in {xml_path}")
            parent = int(model.site_bodyid[sid])
            parent_pose = _mat4(data.xpos[parent], data.xmat[parent])
            site_pose = _mat4(data.site_xpos[sid], data.site_xmat[sid])
            local = np.linalg.inv(parent_pose) @ site_pose
            local[:3, :3] = local[:3, :3] @ correction
            element = root.find(f".//site[@name='{name}']")
        else:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid <= 0:
                raise ValueError(f"EE body {name!r} not found in {xml_path}")
            parent = int(model.body_parentid[bid])
            parent_pose = _mat4(data.xpos[parent], data.xmat[parent])
            body_pose = _mat4(data.xpos[bid], data.xmat[bid])
            local = np.linalg.inv(parent_pose) @ body_pose
            local[:3, :3] = local[:3, :3] @ correction
            element = root.find(f".//body[@name='{name}']")
        if element is None:
            raise ValueError(f"EE XML element {name!r} not found in {xml_path}")
        _, quat = _pose_to_attrs(local)
        element.set("quat", quat)
        element.attrib.pop("euler", None)
    ET.indent(root, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _inject_cameras(asset: Path, output: Path, cameras: list[str], Ks, mounts, rels, raw_sizes) -> None:
    root = ET.parse(asset).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{asset} has no worldbody")
    # Inject the camera pose in MuJoCo's native camera frame.  F converts it
    # back to the OpenCV frame used by the dataset and skeleton renderer.
    for camera, K, (mount, E_or_rel), (height, width) in zip(cameras, Ks, zip(mounts, rels), raw_sizes, strict=True):
        E_or_rel = np.asarray(E_or_rel, dtype=np.float64)
        M_mj = np.linalg.inv(E_or_rel) @ MUJOCO_TO_CV
        element = ET.Element("camera")
        element.set("name", camera)
        element.set("resolution", f"{width} {height}")
        element.set("sensorsize", f"{width} {height}")
        element.set("focal", f"{float(K[0, 0]):.12g} {float(K[1, 1]):.12g}")
        element.set("principal", f"{float(K[0, 2]):.12g} {float(K[1, 2]):.12g}")
        pos, quat = _pose_to_attrs(M_mj)
        element.set("pos", pos)
        element.set("quat", quat)
        if mount is None:
            worldbody.append(element)
        else:
            body = root.find(f".//body[@name='{mount}']")
            if body is None:
                raise ValueError(f"camera mount body {mount!r} not found in {asset}")
            body.append(element)
    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def _explicit_mobile_skeleton(robot_name: str, robot) -> dict:
    if robot_name == "Agibot G1":
        from uranus_dataset.robot.robot_g1_120s import (
            LEFT_ARM_BODIES,
            LEFT_GRIP_BODIES,
            RIGHT_ARM_BODIES,
            RIGHT_GRIP_BODIES,
        )

        color_offset = 0
        sides = (
            (LEFT_ARM_BODIES, LEFT_GRIP_BODIES, "Link7_l"),
            (RIGHT_ARM_BODIES, RIGHT_GRIP_BODIES, "Link7_r"),
        )
    else:
        from uranus_dataset.robot.robot_g2_120s import (
            BODY_BODIES,
            LEFT_ARM_BODIES,
            LEFT_GRIP_BODIES,
            RIGHT_ARM_BODIES,
            RIGHT_GRIP_BODIES,
        )

        color_offset = len(BODY_BODIES)
        sides = (
            (LEFT_ARM_BODIES, LEFT_GRIP_BODIES, "arm_l_link7"),
            (RIGHT_ARM_BODIES, RIGHT_GRIP_BODIES, "arm_r_link7"),
        )

    keypoints = []
    for arm_bodies, gripper_bodies, wrist_name in sides:
        arm_indices = {}
        for offset, name in enumerate(arm_bodies):
            parent = None if offset == 0 else arm_indices[arm_bodies[offset - 1]]
            arm_indices[name] = len(keypoints)
            keypoints.append(
                {"body_name": name, "color": color_offset + offset, "parent": parent}
            )
        wrist_id = mujoco.mj_name2id(
            robot.model, mujoco.mjtObj.mjOBJ_BODY, wrist_name
        )
        body_to_keypoint = {wrist_id: arm_indices[wrist_name]}
        for offset, name in enumerate(gripper_bodies):
            body_id = mujoco.mj_name2id(
                robot.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id < 0:
                continue
            parent = body_to_keypoint.get(
                int(robot.model.body_parentid[body_id]), arm_indices[wrist_name]
            )
            body_to_keypoint[body_id] = len(keypoints)
            keypoints.append(
                {
                    "body_name": name,
                    "color": color_offset + len(arm_bodies) + offset,
                    "parent": parent,
                }
            )
    return {"mode": "explicit", "keypoints": keypoints}


def _ee_and_skeleton(robot_name: str, robot) -> tuple[list[dict], dict]:
    corrections = robot.get_ee_sh_corrections()
    if hasattr(robot, "ee_names"):
        names = tuple(robot.ee_names)
        are_sites = bool(getattr(robot, "ee_are_sites", False))
        pairs = tuple(robot.gripper_body_pairs)
        ees = [
            {
                "object_type": "site" if are_sites else "body",
                "object_name": name,
                "radius_mode": "frame",
                "pad_bodies": list(pair),
                "sh_correction": np.asarray(correction).tolist(),
            }
            for name, pair, correction in zip(
                names, pairs, corrections, strict=True
            )
        ]
        chains = [list(chain) for chain in getattr(robot, "skeleton_body_names", ())]
        skeleton = {"mode": "chains", "chains": chains, "skip_bodies": []}
        if getattr(robot, "render_gripper_keypoints_from_width", False):
            skeleton["gripper_keypoint_overrides"] = [
                {
                    "ee_object_type": "site" if are_sites else "body",
                    "ee_object_name": name,
                    "finger_bodies": list(pair),
                    "closing_axis": 1,
                    "width_index": index,
                }
                for index, (name, pair) in enumerate(
                    zip(names, pairs, strict=True)
                )
            ]
        return ees, skeleton
    if robot_name == "Panda Franka":
        # The adapter's physical EE body is a parent of the gripper.  Use the
        # co-located pinch site as the marker so canonicalizing its orientation
        # cannot rotate the downstream finger geometry.
        return [{"object_type": "site", "object_name": "pinch", "radius_mode": "pad_pair", "pad_bodies": ["right_silicone_pad", "left_silicone_pad"], "sh_correction": np.asarray(corrections[0]).tolist()}], {"mode": "full_tree", "skip_bodies": ["gripper_center"]}
    if robot_name in {"Agibot G1", "Agibot G2"}:
        names = ("gripper_center", "right_gripper_center")
        return [
            {"object_type": "site", "object_name": name, "radius_mode": "pad_pair", "pad_bodies": [f"{side}_Left_Pad_Link", f"{side}_Right_Pad_Link"], "sh_correction": np.asarray(correction).tolist()}
            for name, side, correction in zip(
                names, ("left", "right"), corrections, strict=True
            )
        ], _explicit_mobile_skeleton(robot_name, robot)
    raise ValueError(f"no EE mapping for robot {robot_name!r}")


def export_repo(
    repo_id: str,
    config: dict,
    output_root: Path,
    data_root: str,
    target_fps: float | None = None,
    episode_index: int = 0,
    start_frame: int = 0,
) -> Path:
    cameras = list(config["camera_names"])
    source_repo_id = str(config.get("source_repo_id", repo_id))
    dataset = UranusMultiDataset(
        [source_repo_id], root=data_root, camera_names={source_repo_id: cameras},
        shared_features_only=False, return_uint8=True,
    )
    item = dataset.get_episode(episode_index)
    child = dataset._datasets[0]
    item, export_fps = _resample_episode(item, child, target_fps)
    item = _slice_episode(item, child, start_frame)
    robot_name = child.meta.get_episode_robot(episode_index) or child.meta.robot
    asset = _asset_for_robot(robot_name)
    robot = _make_robot(robot_name, asset)
    frames = _episode_frames(item, child, episode_index)
    qpos, body_poses, Ts, ee_radii, gripper_widths = _body_poses(robot, frames)
    K_all, mounts, rels = [], [], []
    camera_extrinsics = {}
    for camera in cameras:
        K, _, E_all = _camera_sequence(item, camera)
        camera_extrinsics[camera] = E_all
        mount, E_or_rel = _camera_mount(robot_name, camera, E_all, body_poses, Ts)
        if mount is not None:
            # E_or_rel is camera-from-body in OpenCV coordinates.
            rels.append(E_or_rel)
        else:
            rels.append(E_or_rel)
        K_all.append(K)
        mounts.append(mount)

    first_images = {}
    gt_frames = {}
    raw_sizes = []
    for camera in cameras:
        image = _rgb_image(item[f"reference.observation.images.{camera}"])
        first_images[camera] = image
        gt_frames[camera] = [
            _rgb_image(frame) for frame in item[f"observation.images.{camera}"]
        ]
        if not gt_frames[camera]:
            raise ValueError(
                f"episode has no target frames for GT video after start_frame={start_frame}"
            )
        raw_sizes.append((int(image.shape[0]), int(image.shape[1])))

    sample_id = len(list(output_root.glob("*/meta.json")))
    sample_dir = output_root / f"{sample_id:06d}"
    mjcf_dir = sample_dir / "mjcf"
    ref_dir = sample_dir / "ref_images"
    mjcf_path = mjcf_dir / asset.name
    root = ET.parse(asset).getroot()
    C = np.asarray(getattr(robot, "_model_to_robot", np.eye(4)), dtype=np.float64)
    _bake_model_to_robot(root, asset, C)
    temp_asset = sample_dir / ".asset_base.xml"
    ET.indent(root, space="  ")
    temp_asset.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(temp_asset, encoding="utf-8", xml_declaration=True)

    ref_images = {}
    for camera, image in first_images.items():
        path = ref_dir / f"{camera}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write {path}")
        ref_images[camera] = str(path.relative_to(sample_dir))

    gt_videos = {}
    for camera, camera_frames in gt_frames.items():
        path = sample_dir / "gt" / f"{camera}.mp4"
        encode_h264(camera_frames, path, fps=export_fps)
        gt_videos[camera] = str(path.relative_to(sample_dir))

    ee, skeleton = _ee_and_skeleton(robot_name, robot)
    _inject_cameras(temp_asset, mjcf_path, cameras, K_all, mounts, rels, raw_sizes)
    temp_asset.unlink()
    meta = {
        "format_version": 4,
        "prompt": str(item.get("task", "")),
        "mjcf_path": str(mjcf_path.relative_to(sample_dir)),
        "cameras": [{"name": camera} for camera in cameras],
        "end_effectors": ee,
        "skeleton": skeleton,
        "height": raw_sizes[0][0],
        "width": raw_sizes[0][1],
        "fps": export_fps,
        "ref_cam_images": ref_images,
        # These begin at step_qpos[1]. The reference/conditioning image is
        # intentionally kept only in ref_cam_images, so frame 0 is aligned
        # with the first frame returned by UranusRunner.step().
        "gt_videos": gt_videos,
        "migrated_from": {
            "repo_id": repo_id,
            "source_repo_id": source_repo_id,
            "episode_index": episode_index,
            "start_frame": start_frame,
            "source_fps": float(child.fps),
            "robot": robot_name,
            "robot_type": robot_name,
        },
    }
    temporal_frames = []
    mujoco_qpos = []
    movable = robot_name in {"Agibot G1", "Agibot G2"}
    for index, frame in enumerate(frames):
        # Keep the dataset-native arm+gripper state as the public observation.
        # Full qpos is an implementation cache for the generic MJCF engine;
        # it contains mimic/follower joints that are not observations (notably
        # G1/G2's many gripper joints). Keep it out of each public step_qpos
        # entry so observation.state remains the native compact signal.
        compact_state = np.asarray(frame["observation.state"], dtype=np.float64).reshape(-1).tolist()
        entry = {
            "observation.state": compact_state,
            "camera_extrinsics": {
                camera: camera_extrinsics[camera][index].tolist()
                for camera in cameras
            },
            "end_effector_radii": ee_radii[index],
        }
        mujoco_qpos.append(qpos[index].tolist())
        if movable:
            entry["observation.robot2world_trans"] = Ts[index].tolist()
        if gripper_widths[index] is not None:
            entry["gripper_widths"] = gripper_widths[index].tolist()
        temporal_frames.append(entry)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (sample_dir / "temporal.json").write_text(
        json.dumps({"step_qpos": temporal_frames}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    # Keep the renderer's expanded MJCF state out of temporal.json entirely;
    # the latter mirrors Uranus-data's native state stream.
    (sample_dir / "mujoco_qpos.json").write_text(
        json.dumps(mujoco_qpos, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return sample_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("examples/data_v3"))
    parser.add_argument("--data-root", default=DATASET_ROOT)
    parser.add_argument("--fps", type=float, default=None, help="output sampling FPS (default: source FPS)")
    parser.add_argument("--episode-index", type=int, default=0, help="episode index to export (default: 0)")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame after FPS sampling (default: 0)")
    parser.add_argument("--repo", action="append", choices=sorted(DATASET_CONFIGS))
    args = parser.parse_args()
    repos = args.repo or list(DATASET_CONFIGS)
    for repo_id in repos:
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
