"""RoboTwin2 ALOHA/AgileX dual-arm embodiment."""

from __future__ import annotations

import mujoco
import numpy as np

from .robot_table30_base import Table30FixedRobot
from .unified_robot import _SH_CORR_RC_ALOHA


class RobotRoboTwin2ALOHA(Table30FixedRobot):
    arm_joint_names = (
        tuple(f"fl_joint{i}" for i in range(1, 7)),
        tuple(f"fr_joint{i}" for i in range(1, 7)),
    )
    arm_base_body_names = ("fl_base_link", "fr_base_link")
    skeleton_body_names = (
        ("fl_base_link", *(f"fl_link{i}" for i in range(1, 9))),
        ("fr_base_link", *(f"fr_link{i}" for i in range(1, 9))),
    )
    ee_names = ("fl_gripper_center", "fr_gripper_center")
    ee_are_sites = True
    gripper_body_pairs = (("fl_link7", "fl_link8"), ("fr_link7", "fr_link8"))
    render_gripper_keypoints_from_width = True

    # RoboTwin2 stores poses in the simulator world frame. The Isaac URDF/MJCF
    # model frame differs by an approximately +90 degree yaw and translation.
    _MODEL_TO_ROBOT = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -0.652],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _GRIPPER_JOINT_MAX = 0.04765

    def __init__(self, mjcf_path: str = "assets/arx5_description_isaac_no_mesh.xml"):
        super().__init__(
            mjcf_path,
            cameras=["front", "head", "left_wrist", "right_wrist"],
            raw_size=(240, 320),
        )
        self._model_to_robot = self._MODEL_TO_ROBOT.copy()

    def get_ee_sh_corrections(self):
        return [_SH_CORR_RC_ALOHA] * len(self.ee_names)

    def build_qpos(self, frame_data: dict) -> None:
        state = np.asarray(frame_data["observation.state"], dtype=np.float64)
        if state.shape == (14,):
            state = np.stack([state[:7], state[7:]], axis=0)
        if state.shape != (2, 7):
            raise ValueError(
                f"{type(self).__name__} expects observation.state shape (2, 7) "
                f"or flat (14,), got {state.shape}"
            )

        mujoco.mj_resetData(self.model, self.data)
        for arm_index, joint_names in enumerate(self.arm_joint_names):
            for joint_index, joint_name in enumerate(joint_names):
                self._set_joint_clipped(joint_name, float(state[arm_index, joint_index]))

        self._gripper_widths = np.clip(state[:, 6], 0.0, 1.0) * self._GRIPPER_JOINT_MAX
        self._set_grippers(self._gripper_widths)
        mujoco.mj_forward(self.model, self.data)

    def _set_grippers(self, widths: np.ndarray) -> None:
        self._set_normalized_gripper("fl_joint7", "fl_joint8", widths[0])
        self._set_normalized_gripper("fr_joint7", "fr_joint8", widths[1])

    def _set_normalized_gripper(self, first_joint: str, second_joint: str, width: float) -> None:
        value = float(np.clip(width, 0.0, self._GRIPPER_JOINT_MAX))
        self._set_joint_clipped(first_joint, value)
        self._set_joint_clipped(second_joint, value)
