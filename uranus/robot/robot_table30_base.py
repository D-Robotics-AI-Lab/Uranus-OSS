"""Shared FK and skeleton helpers for RoboChallenge Table30v2 robots."""

from __future__ import annotations

from abc import abstractmethod

import mujoco
import numpy as np

from .unifiedrobot import UnifiedRobot


class Table30FixedRobot(UnifiedRobot):
    """Base class for fixed-base Table30v2 embodiments.

    Table30v2 stores each arm as ``[joint1..joint6, gripper, gripper]``.
    The duplicate gripper value is retained by the data converter for the
    Uranus state convention; only the first copy is consumed here.
    """

    arm_joint_names: tuple[tuple[str, ...], ...] = ()
    skeleton_body_names: tuple[tuple[str, ...], ...] = ()
    ee_names: tuple[str, ...] = ()
    ee_are_sites: bool = False
    gripper_body_pairs: tuple[tuple[str, str] | None, ...] = ()
    arm_base_body_names: tuple[str, ...] = ()
    render_gripper_keypoints_from_width: bool = False
    gripper_closing_axis: int = 1
    default_gripper_radius: float = 0.02
    reference_body_name: str | None = None

    def __init__(self, mjcf_path: str, cameras: list[str], raw_size: tuple[int, int]):
        super().__init__(mjcf_path, cameras=cameras, raw_size=raw_size)
        self._gripper_widths: np.ndarray | None = None
        self._validate_model_names()
        mujoco.mj_forward(self.model, self.data)
        self._model_to_robot = np.eye(4, dtype=np.float64)
        if self.reference_body_name is not None:
            body_id = self._require_name(mujoco.mjtObj.mjOBJ_BODY, self.reference_body_name)
            robot_to_model = np.eye(4, dtype=np.float64)
            robot_to_model[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
            robot_to_model[:3, 3] = self.data.xpos[body_id]
            self._model_to_robot = np.linalg.inv(robot_to_model)

    def has_gripper(self) -> bool:
        return True

    def has_arm(self) -> bool:
        return True

    def num_arms(self) -> int:
        return len(self.arm_joint_names)

    def is_movable(self) -> bool:
        return False

    def get_robot_to_world_transform(self) -> np.ndarray:
        return np.eye(4, dtype=np.float64)

    def get_robot_to_arm_base_transforms(self) -> tuple[np.ndarray, ...]:
        """Return canonical robot/root -> per-arm base transforms.

        MuJoCo body poses are arm-base -> model transforms. Compose them with
        ``_model_to_robot`` and invert so the result can be appended to a
        world/root -> camera projection chain.
        """
        if not self.arm_base_body_names:
            return tuple(np.eye(4, dtype=np.float64) for _ in range(self.num_arms()))

        transforms = []
        for name in self.arm_base_body_names:
            body_id = self._require_name(mujoco.mjtObj.mjOBJ_BODY, name)
            model_from_arm = np.eye(4, dtype=np.float64)
            model_from_arm[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
            model_from_arm[:3, 3] = self.data.xpos[body_id]
            robot_from_arm = self._model_to_robot @ model_from_arm
            transforms.append(np.linalg.inv(robot_from_arm))
        return tuple(transforms)

    def _validate_model_names(self) -> None:
        for name in (joint for arm in self.arm_joint_names for joint in arm):
            self._require_name(mujoco.mjtObj.mjOBJ_JOINT, name)
        ee_object = mujoco.mjtObj.mjOBJ_SITE if self.ee_are_sites else mujoco.mjtObj.mjOBJ_BODY
        for name in self.ee_names:
            self._require_name(ee_object, name)
        for name in (body for chain in self.skeleton_body_names for body in chain):
            self._require_name(mujoco.mjtObj.mjOBJ_BODY, name)
        if self.arm_base_body_names and len(self.arm_base_body_names) != self.num_arms():
            raise ValueError(
                f"{type(self).__name__} defines {len(self.arm_base_body_names)} arm bases "
                f"for {self.num_arms()} arms"
            )
        for name in self.arm_base_body_names:
            self._require_name(mujoco.mjtObj.mjOBJ_BODY, name)

    def _require_name(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"{name!r} not found in {self.mjcf_path}")
        return object_id

    def _set_joint_clipped(self, name: str, value: float) -> None:
        joint_id = self._require_name(mujoco.mjtObj.mjOBJ_JOINT, name)
        if self.model.jnt_limited[joint_id]:
            low, high = self.model.jnt_range[joint_id]
            value = float(np.clip(value, low, high))
        self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

    def build_qpos(self, frame_data: dict) -> None:
        state = np.asarray(frame_data["observation.state"], dtype=np.float64)
        expected_shape = (8,) if self.num_arms() == 1 else (self.num_arms(), 8)
        if state.shape != expected_shape:
            raise ValueError(
                f"{type(self).__name__} expects observation.state shape "
                f"{expected_shape}, got {state.shape}"
            )
        if self.num_arms() == 1:
            state = state[None, :]

        mujoco.mj_resetData(self.model, self.data)
        for arm_index, joint_names in enumerate(self.arm_joint_names):
            for joint_index, joint_name in enumerate(joint_names):
                self._set_joint_clipped(joint_name, float(state[arm_index, joint_index]))
        # Table30v2 records the total clear opening between the two fingers.
        # Keep this measured signal for visualization: body origins are not
        # necessarily located on the finger contact surfaces.
        self._gripper_widths = np.maximum(state[:, 6], 0.0)
        self._set_grippers(self._gripper_widths)
        mujoco.mj_forward(self.model, self.data)

    @abstractmethod
    def _set_grippers(self, widths: np.ndarray) -> None:
        """Map measured total gripper widths in meters to MJCF joints."""

    def _object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if self.ee_are_sites:
            object_id = self._require_name(mujoco.mjtObj.mjOBJ_SITE, name)
            position = self.data.site_xpos[object_id]
            rotation = self.data.site_xmat[object_id]
        else:
            object_id = self._require_name(mujoco.mjtObj.mjOBJ_BODY, name)
            position = self.data.xpos[object_id]
            rotation = self.data.xmat[object_id]
        position = self._model_to_robot[:3, :3] @ position + self._model_to_robot[:3, 3]
        rotation = self._model_to_robot[:3, :3] @ rotation.reshape(3, 3)
        return position.astype(np.float32), rotation.astype(np.float32)

    def get_ee_states(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        result = []
        for arm_index, ee_name in enumerate(self.ee_names):
            position, rotation = self._object_pose(ee_name)
            if self._gripper_widths is not None:
                # The renderer represents the total opening as a sphere
                # centered at the end effector, hence radius = width / 2.
                radius = max(float(self._gripper_widths[arm_index]) * 0.5, 1e-3)
            else:
                pair = self.gripper_body_pairs[arm_index] if self.gripper_body_pairs else None
                if pair is None:
                    radius = self.default_gripper_radius
                else:
                    first = self._require_name(mujoco.mjtObj.mjOBJ_BODY, pair[0])
                    second = self._require_name(mujoco.mjtObj.mjOBJ_BODY, pair[1])
                    radius = max(
                        float(np.linalg.norm(self.data.xpos[first] - self.data.xpos[second])) * 0.5,
                        1e-3,
                    )
            result.append((position, rotation, radius))
        return result

    def build_keypoints(self) -> list[dict]:
        keypoints: list[dict] = []
        body_to_keypoint: dict[int, int] = {}
        for chain in self.skeleton_body_names:
            for chain_index, name in enumerate(chain):
                body_id = self._require_name(mujoco.mjtObj.mjOBJ_BODY, name)
                parent_id = self.model.body_parentid[body_id]
                while parent_id > 0 and parent_id not in body_to_keypoint:
                    parent_id = self.model.body_parentid[parent_id]
                keypoints.append(
                    {
                        "pos": (
                            self._model_to_robot[:3, :3] @ self.data.xpos[body_id]
                            + self._model_to_robot[:3, 3]
                        ).astype(np.float32),
                        "color": chain_index,
                        "parent": body_to_keypoint.get(parent_id),
                    }
                )
                body_to_keypoint[body_id] = len(keypoints) - 1

        if self.render_gripper_keypoints_from_width and self._gripper_widths is not None:
            for arm_index, pair in enumerate(self.gripper_body_pairs):
                if pair is None:
                    continue
                first_id = self._require_name(mujoco.mjtObj.mjOBJ_BODY, pair[0])
                second_id = self._require_name(mujoco.mjtObj.mjOBJ_BODY, pair[1])
                first_kp = body_to_keypoint[first_id]
                second_kp = body_to_keypoint[second_id]
                center, rotation = self._object_pose(self.ee_names[arm_index])
                closing_axis = rotation[:, self.gripper_closing_axis]
                half_width = float(self._gripper_widths[arm_index]) * 0.5
                keypoints[first_kp]["pos"] = (center - closing_axis * half_width).astype(
                    np.float32
                )
                keypoints[second_kp]["pos"] = (center + closing_axis * half_width).astype(
                    np.float32
                )
        return keypoints

    def _set_symmetric_slider(self, negative_joint: str, positive_joint: str, width: float) -> None:
        half_width = max(float(width), 0.0) * 0.5
        self._set_joint_clipped(negative_joint, -half_width)
        self._set_joint_clipped(positive_joint, half_width)
