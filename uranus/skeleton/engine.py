"""Generic MuJoCo FK engine: qpos in, world-frame geometry out.

Replaces the per-robot ``UnifiedRobot`` subclasses.  One engine is built per
session (``create``) and persists for its lifetime; every ``step`` frame goes
through ``set_state`` once and all cameras share the resulting FK.

The MJCF is the environment source of truth: robot geometry, default state,
state-to-joint mapping, camera mounts, camera poses, and camera intrinsics all
live in XML. ``temporal.json`` carries joint positions, compact gripper state,
and a base-to-world robot transform.

Validation errors raise ``ValueError``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .specs import (
    CameraSpec,
    EESpec,
    RigSpec,
    SkeletonSpec,
)


# MuJoCo cameras look along local -Z with +Y up.  Dataset calibration and the
# renderer use the OpenCV convention (+Z forward, +Y down).
_MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
_STATE_FIELDS = ("joint_positions", "gripper")
_FRAME_FIELDS = (*_STATE_FIELDS, "robot_transform")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mat4(xpos, xmat) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    M[:3, 3] = np.asarray(xpos, dtype=np.float64).reshape(3)
    return M


def _robot_transform(value) -> np.ndarray:
    """Build a matrix from ``{xyz, quaternion}`` with an xyzw quaternion."""
    _require(isinstance(value, dict), "robot_transform must be an object")
    _require(
        set(value) == {"xyz", "quaternion"},
        "robot_transform must contain exactly xyz and quaternion",
    )
    xyz = np.asarray(value["xyz"], dtype=np.float64).reshape(-1)
    quaternion = np.asarray(value["quaternion"], dtype=np.float64).reshape(-1)
    _require(xyz.shape == (3,), f"robot_transform.xyz must have shape (3,), got {xyz.shape}")
    _require(
        quaternion.shape == (4,),
        f"robot_transform.quaternion must have shape (4,), got {quaternion.shape}",
    )
    _require(
        bool(np.all(np.isfinite(xyz))) and bool(np.all(np.isfinite(quaternion))),
        "robot_transform must be finite",
    )
    x, y, z, w = quaternion
    norm = float(np.dot(quaternion, quaternion))
    _require(norm > 1e-12, "robot_transform.quaternion must be non-zero")
    scale = 2.0 / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = xyz
    return matrix


class SkeletonEngine:
    """MuJoCo FK engine bound to one MJCF model and one rig configuration."""

    def __init__(self, rig: RigSpec):
        rig.validate()
        self.rig = rig
        self.model = mujoco.MjModel.from_xml_path(rig.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self._has_state = False
        self._T = np.eye(4, dtype=np.float64)
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "uranus_default"
        )
        self._default_qpos = (
            np.asarray(self.model.key_qpos[key_id], dtype=np.float64).copy()
            if key_id >= 0
            else np.asarray(self.model.qpos0, dtype=np.float64).copy()
        )
        self._state_qpos_addresses = self._load_state_qpos_addresses(rig.mjcf_path)
        self._gripper_mapping = self._load_gripper_mapping(rig.mjcf_path)

        # Bind every spec name -> id once, fail fast (mirrors legacy _require_name).
        self._body_ids: dict[str, int] = {}
        self._site_ids: dict[str, int] = {}
        self._mount_ids: list[int | None] = []
        self._camera_ids: dict[str, int] = {}

        for camera in rig.cameras:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name
            )
            if camera_id >= 0:
                self._camera_ids[camera.name] = camera_id
            elif camera.extrinsic_rel is None:
                raise ValueError(f"camera {camera.name!r} not found in {rig.mjcf_path}")
            if camera.mount_body is not None:
                self._mount_ids.append(self._require_body(camera.mount_body))
            else:
                self._mount_ids.append(None)
        for ee in rig.end_effectors:
            if ee.object_type == "site":
                self._require_site(ee.object_name)
                if ee.pad_bodies is not None:
                    self._require_body(ee.pad_bodies[0])
                    self._require_body(ee.pad_bodies[1])
            else:
                self._require_body(ee.object_name)
                if ee.pad_bodies is not None:
                    self._require_body(ee.pad_bodies[0])
                    self._require_body(ee.pad_bodies[1])
        if rig.skeleton.mode == "chains":
            for chain in rig.skeleton.chains:
                for name in chain:
                    self._require_body(name)
        for name in rig.skeleton.skip_bodies:
            self._require_body(name)
        for ov in rig.skeleton.gripper_keypoint_overrides:
            if ov.ee_object_type == "site":
                self._require_site(ov.ee_object_name)
            else:
                self._require_body(ov.ee_object_name)
            self._require_body(ov.finger_bodies[0])
            self._require_body(ov.finger_bodies[1])

        mujoco.mj_forward(self.model, self.data)

    def _load_state_qpos_addresses(self, mjcf_path: str) -> dict[str, tuple[int, ...]]:
        """Read the exported joint/gripper order from MJCF ``<custom>``."""
        root = ET.parse(mjcf_path).getroot()
        joint_names: dict[str, list[str]] = {field: [] for field in _STATE_FIELDS}
        for field in _STATE_FIELDS:
            xml_name = (
                "uranus_joint_names"
                if field == "joint_positions"
                else "uranus_gripper_joints"
            )
            element = root.find(f"./custom/text[@name='{xml_name}']")
            if field == "joint_positions" and element is None:
                # Compatibility with format-version 7 development exports.
                element = root.find("./custom/text[@name='uranus_arm_joints']")
            if element is not None:
                joint_names[field] = element.get("data", "").split()

        addresses: dict[str, tuple[int, ...]] = {}
        seen: set[int] = set()
        for field, names in joint_names.items():
            values = []
            for name in names:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                _require(joint_id >= 0, f"state joint {name!r} not found in {mjcf_path}")
                joint_type = int(self.model.jnt_type[joint_id])
                _require(
                    joint_type
                    in (
                        int(mujoco.mjtJoint.mjJNT_HINGE),
                        int(mujoco.mjtJoint.mjJNT_SLIDE),
                    ),
                    f"state joint {name!r} must have one qpos value",
                )
                address = int(self.model.jnt_qposadr[joint_id])
                _require(address not in seen, f"state joint {name!r} is listed more than once")
                seen.add(address)
                values.append(address)
            addresses[field] = tuple(values)
        return addresses

    def _load_gripper_mapping(self, mjcf_path: str) -> dict | None:
        """Read optional compact-gripper expansion parameters from MJCF."""
        root = ET.parse(mjcf_path).getroot()
        inputs = root.find("./custom/text[@name='uranus_gripper_inputs']")
        if inputs is None:
            return None
        input_names = inputs.get("data", "").split()
        _require(input_names, "uranus_gripper_inputs must not be empty")
        modes_element = root.find(
            "./custom/text[@name='uranus_gripper_input_modes']"
        )
        modes = modes_element.get("data", "").split() if modes_element is not None else []
        _require(
            len(modes) == len(input_names),
            "uranus_gripper_input_modes must match uranus_gripper_inputs",
        )

        def numbers(name: str) -> np.ndarray:
            element = root.find(f"./custom/numeric[@name='{name}']")
            _require(element is not None, f"missing XML custom numeric {name}")
            return np.asarray(
                [float(value) for value in element.get("data", "").split()],
                dtype=np.float64,
            )

        ranges = numbers("uranus_gripper_input_ranges")
        target_indices_raw = numbers("uranus_gripper_target_indices")
        target_scales = numbers("uranus_gripper_target_scales")
        target_offsets = numbers("uranus_gripper_target_offsets")
        target_count = len(self._state_qpos_addresses["gripper"])
        _require(
            ranges.shape == (2 * len(input_names),),
            "uranus_gripper_input_ranges must contain low/high per input",
        )
        _require(
            target_indices_raw.shape == target_scales.shape == target_offsets.shape == (target_count,),
            "gripper target mapping must match uranus_gripper_joints",
        )
        target_indices = target_indices_raw.astype(np.int64)
        _require(
            bool(np.all(target_indices_raw == target_indices))
            and bool(np.all((0 <= target_indices) & (target_indices < len(input_names)))),
            "uranus_gripper_target_indices contains an invalid input index",
        )
        _require(
            all(mode in {"clip", "g1_closure"} for mode in modes),
            f"unsupported gripper input mode in {modes}",
        )
        return {
            "input_names": tuple(input_names),
            "modes": tuple(modes),
            "ranges": ranges.reshape(-1, 2),
            "target_indices": target_indices,
            "target_scales": target_scales,
            "target_offsets": target_offsets,
        }

    # ---------------------------------------------------------------- naming

    def _require_body(self, name: str) -> int:
        bid = self._body_ids.get(name)
        if bid is not None:
            return bid
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        _require(bid >= 0, f"body {name!r} not found in {self.rig.mjcf_path}")
        self._body_ids[name] = bid
        return bid

    def _require_site(self, name: str) -> int:
        sid = self._site_ids.get(name)
        if sid is not None:
            return sid
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        _require(sid >= 0, f"site {name!r} not found in {self.rig.mjcf_path}")
        self._site_ids[name] = sid
        return sid

    # ---------------------------------------------------------------- state

    def set_state(self, state) -> None:
        """Set one frame and run FK.

        New samples pass ``{"joint_positions": [...], "gripper": [...]}``;
        each vector is mapped through joint names declared in the XML. A
        complete qpos vector and the old ``arm`` key are accepted for
        compatibility with existing callers.
        """
        structured = isinstance(state, dict) and any(
            field in state for field in (*_FRAME_FIELDS, "arm")
        )
        if structured:
            state = dict(state)
            if "arm" in state and "joint_positions" not in state:
                state["joint_positions"] = state.pop("arm")
            missing = [field for field in _FRAME_FIELDS if field not in state]
            _require(not missing, f"state is missing fields: {missing}")
            extra = sorted(set(state) - set(_FRAME_FIELDS))
            _require(not extra, f"state has unsupported fields: {extra}")
            _require(
                any(self._state_qpos_addresses.values()),
                "structured state requires uranus_joint_names and "
                "uranus_gripper_joints in the MJCF <custom> section",
            )
            vector = self._default_qpos.copy()
            joint_positions = np.asarray(
                state["joint_positions"], dtype=np.float64
            ).reshape(-1)
            joint_addresses = self._state_qpos_addresses["joint_positions"]
            _require(
                len(joint_positions) == len(joint_addresses),
                "state.joint_positions must have length "
                f"{len(joint_addresses)}, got {len(joint_positions)}",
            )
            vector[list(joint_addresses)] = joint_positions

            gripper = np.asarray(state["gripper"], dtype=np.float64).reshape(-1)
            gripper_addresses = self._state_qpos_addresses["gripper"]
            if self._gripper_mapping is None:
                _require(
                    len(gripper) == len(gripper_addresses),
                    f"state.gripper must have length {len(gripper_addresses)}, got {len(gripper)}",
                )
                vector[list(gripper_addresses)] = gripper
            else:
                mapping = self._gripper_mapping
                expected = len(mapping["input_names"])
                _require(
                    len(gripper) == expected,
                    f"state.gripper must have length {expected}, got {len(gripper)}",
                )
                _require(
                    bool(np.all(np.isfinite(gripper))),
                    "state.gripper must be finite",
                )
                inputs = gripper.copy()
                for index, mode in enumerate(mapping["modes"]):
                    if mode == "g1_closure":
                        value = inputs[index]
                        inputs[index] = (
                            np.clip(value, 0.0, 1.0)
                            if abs(value) <= 1.0
                            else np.clip((value - 35.0) / 85.0, 0.0, 1.0)
                        )
                    else:
                        low, high = mapping["ranges"][index]
                        inputs[index] = np.clip(inputs[index], low, high)
                expanded = (
                    mapping["target_scales"]
                    * inputs[mapping["target_indices"]]
                    + mapping["target_offsets"]
                )
                vector[list(gripper_addresses)] = expanded
            self._T = _robot_transform(state["robot_transform"])
        else:
            vector = np.asarray(state, dtype=np.float64).reshape(-1)
            self._T = np.eye(4, dtype=np.float64)
        _require(
            vector.shape[0] == self.model.nq,
            f"qpos must have length {self.model.nq}, got {vector.shape[0]}",
        )
        _require(bool(np.all(np.isfinite(vector))), "qpos must be finite")

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = vector
        # Clip every limited joint to its range (idempotent on already-clipped
        # migration output; protects against out-of-range telemetry).
        for jid in range(self.model.njnt):
            if self.model.jnt_limited[jid]:
                adr = self.model.jnt_qposadr[jid]
                lo, hi = self.model.jnt_range[jid]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr], lo, hi)

        mujoco.mj_forward(self.model, self.data)
        self._has_state = True

    def _require_state(self) -> None:
        _require(self._has_state, "set_state() must be called before reading FK results")

    # ---------------------------------------------------------------- frames

    def body_pose_model(self, body_name: str) -> np.ndarray:
        """4x4 world-from-body in the *model* frame (no C, no T)."""
        self._require_state()
        bid = self._require_body(body_name)
        return _mat4(self.data.xpos[bid], self.data.xmat[bid])

    def body_pose_world(self, body_name: str) -> np.ndarray:
        """4x4 environment-from-body pose."""
        return self._T @ self.body_pose_model(body_name)

    def compose_camera_extrinsics(self) -> list[np.ndarray]:
        """World-to-camera 4x4 for every rig camera, in rig order.

        XML cameras use their MuJoCo-composed world pose.  Deprecated v2
        ``CameraSpec`` transforms remain readable for old samples.
        """
        self._require_state()
        result: list[np.ndarray] = []
        for camera, mount_id in zip(self.rig.cameras, self._mount_ids):
            if camera.extrinsic_rel is not None:
                if mount_id is None:
                    result.append(np.asarray(camera.extrinsic_rel, dtype=np.float64).copy())
                else:
                    M = _mat4(self.data.xpos[mount_id], self.data.xmat[mount_id])
                    result.append(
                        np.asarray(camera.extrinsic_rel, dtype=np.float64) @ np.linalg.inv(M)
                    )
                continue
            camera_id = self._camera_ids[camera.name]
            M_cam = _mat4(self.data.cam_xpos[camera_id], self.data.cam_xmat[camera_id])
            M_cam = self._T @ M_cam
            result.append(_MUJOCO_TO_CV @ np.linalg.inv(M_cam))
        return result

    # ---------------------------------------------------------------- ees

    def get_ee_states(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """[(pos, rot, radius)] per EE spec, in the XML environment frame."""
        self._require_state()
        rotation = self._T[:3, :3]
        translation = self._T[:3, 3]
        result: list[tuple[np.ndarray, np.ndarray, float]] = []
        for ee in self.rig.end_effectors:
            if ee.object_type == "site":
                sid = self._require_site(ee.object_name)
                pos_m = self.data.site_xpos[sid]
                rot_m = self.data.site_xmat[sid]
            else:
                bid = self._require_body(ee.object_name)
                pos_m = self.data.xpos[bid]
                rot_m = self.data.xmat[bid]
            pos = (rotation @ np.asarray(pos_m) + translation).astype(np.float32)
            rot = (rotation @ np.asarray(rot_m).reshape(3, 3)).astype(np.float32)

            if ee.radius_mode == "pad_pair":
                a = self._require_body(ee.pad_bodies[0])
                b = self._require_body(ee.pad_bodies[1])
                # legacy semantics: radius = max(pad_distance / 2, 1e-3) — the
                # 1mm floor keeps a visible dot for closed grippers
                radius = max(
                    float(np.linalg.norm(self.data.xpos[a] - self.data.xpos[b])) * 0.5,
                    1e-3,
                )
            else:
                radius = max(float(ee.radius), 1e-3)
            result.append((pos, rot, radius))
        return result

    def get_ee_sh_corrections(self) -> list[np.ndarray]:
        """Per-EE SH correction matrices (identity where unset)."""
        return [
            (
                np.asarray(ee.sh_correction, dtype=np.float32)
                if ee.sh_correction is not None
                else np.eye(3, dtype=np.float32)
            )
            for ee in self.rig.end_effectors
        ]

    # ---------------------------------------------------------------- keypoints

    def get_keypoints(self) -> list[dict]:
        """Skeleton keypoints in world frame; format matches the legacy renderer.

        Each keypoint: {"pos": (3,) float32 world, "color": int, "parent": int | None}.
        """
        self._require_state()
        spec = self.rig.skeleton
        rotation = self._T[:3, :3]
        translation = self._T[:3, 3]

        def to_world(pos_m) -> np.ndarray:
            return (rotation @ np.asarray(pos_m) + translation).astype(np.float32)

        keypoints: list[dict] = []
        if spec.mode == "full_tree":
            skip_ids = {self._require_body(name) for name in spec.skip_bodies}
            body_to_kp: dict[int, int] = {}
            for bid in range(self.model.nbody):
                if bid in skip_ids:
                    continue
                parent_bid = self.model.body_parentid[bid]
                keypoints.append(
                    {
                        "pos": to_world(self.data.xpos[bid]),
                        "color": bid,
                        "parent": body_to_kp.get(parent_bid),
                    }
                )
                body_to_kp[bid] = len(keypoints) - 1
            return keypoints

        # chains mode: color = position within the chain (the legacy variable
        # is named chain_index but enumerates the *inner* loop); parent =
        # nearest keypoint ancestor walking up the body tree (Table30 rule).
        body_to_kp: dict[int, int] = {}
        for chain in spec.chains:
            for chain_index, name in enumerate(chain):
                bid = self._require_body(name)
                parent_id = self.model.body_parentid[bid]
                while parent_id > 0 and parent_id not in body_to_kp:
                    parent_id = self.model.body_parentid[parent_id]
                keypoints.append(
                    {
                        "pos": to_world(self.data.xpos[bid]),
                        "color": chain_index,
                        "parent": body_to_kp.get(parent_id),
                    }
                )
                body_to_kp[bid] = len(keypoints) - 1

        # legacy ``render_gripper_keypoints_from_width``: reposition the two
        # finger-body keypoints to EE_pos +/- EE_rot[:, closing_axis] * w/2,
        # where w is recovered as the pad-pair distance (exact under the
        # adjusted slider geometry).
        for ov in spec.gripper_keypoint_overrides:
            first_kp = body_to_kp[self._require_body(ov.finger_bodies[0])]
            second_kp = body_to_kp[self._require_body(ov.finger_bodies[1])]
            if ov.ee_object_type == "site":
                sid = self._require_site(ov.ee_object_name)
                center_m = self.data.site_xpos[sid]
                rot_m = np.asarray(self.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
            else:
                bid = self._require_body(ov.ee_object_name)
                center_m = self.data.xpos[bid]
                rot_m = np.asarray(self.data.xmat[bid], dtype=np.float64).reshape(3, 3)
            center = rotation @ np.asarray(center_m) + translation
            closing_axis = (rotation @ rot_m)[:, ov.closing_axis]
            a = self._require_body(ov.finger_bodies[0])
            b = self._require_body(ov.finger_bodies[1])
            width = float(np.linalg.norm(self.data.xpos[a] - self.data.xpos[b]))
            half_width = width * 0.5
            keypoints[first_kp]["pos"] = (center - closing_axis * half_width).astype(np.float32)
            keypoints[second_kp]["pos"] = (center + closing_axis * half_width).astype(np.float32)
        return keypoints

    # ---------------------------------------------------------------- helpers

    def scaled_intrinsics(self, camera: CameraSpec, raw_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
        """Scale a camera's raw intrinsics to the target resolution."""
        raw_h, raw_w = raw_size
        target_h, target_w = target_size
        K = self.camera_intrinsics(camera)
        K[0, :] *= target_w / raw_w
        K[1, :] *= target_h / raw_h
        return K

    def camera_intrinsics(self, camera: CameraSpec | str) -> np.ndarray:
        """Return the 3x3 OpenCV intrinsic matrix declared in the XML."""
        name = camera if isinstance(camera, str) else camera.name
        camera_spec = next((item for item in self.rig.cameras if item.name == name), None)
        if camera_spec is None:
            raise KeyError(f"unknown camera {name!r}")
        camera_id = self._camera_ids.get(name)
        if camera_id is None:
            if camera_spec.intrinsics is None:
                raise ValueError(f"camera {name!r} has no intrinsic calibration")
            return np.asarray(camera_spec.intrinsics, dtype=np.float64).copy()
        fx, fy, cx, cy = np.asarray(self.model.cam_intrinsic[camera_id], dtype=np.float64)
        if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
            raise ValueError(f"invalid intrinsic calibration for camera {name!r}")
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def camera_resolution(self, camera: CameraSpec | str) -> tuple[int, int]:
        """Return XML camera resolution as ``(height, width)``."""
        name = camera if isinstance(camera, str) else camera.name
        camera_id = self._camera_ids[name]
        width, height = (int(value) for value in self.model.cam_resolution[camera_id])
        if width <= 0 or height <= 0:
            raise ValueError(f"camera {name!r} has invalid XML resolution {width}x{height}")
        return height, width

    @property
    def nq(self) -> int:
        return int(self.model.nq)
