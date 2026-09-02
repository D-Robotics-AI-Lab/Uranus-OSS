"""RoboChallenge Table30v2 DOS-W1 dual-arm embodiment."""

import numpy as np

from .robot_table30_base import Table30FixedRobot
from .unifiedrobot import _SH_CORR_RC_DOSW1


class RobotTable30DOSW1(Table30FixedRobot):
    arm_joint_names = (
        tuple(f"left_joint{i}" for i in range(1, 7)),
        tuple(f"right_joint{i}" for i in range(1, 7)),
    )
    arm_base_body_names = ("left_base_link", "right_base_link")
    skeleton_body_names = (
        (
            "left_base_link",
            *(f"left_link{i}" for i in range(1, 7)),
            "left_eef_base_link",
            "left_eef_link1",
            "left_eef_link2",
        ),
        (
            "right_base_link",
            *(f"right_link{i}" for i in range(1, 7)),
            "right_eef_base_link",
            "right_eef_link1",
            "right_eef_link2",
        ),
    )
    ee_names = ("left_end_link", "right_end_link")
    gripper_body_pairs = (
        ("left_eef_link1", "left_eef_link2"),
        ("right_eef_link1", "right_eef_link2"),
    )
    render_gripper_keypoints_from_width = True
    reference_body_name = "left_base_link"

    def get_ee_sh_corrections(self):
        return [_SH_CORR_RC_DOSW1] * len(self.ee_names)

    def __init__(self, mjcf_path: str = "assets/dos-w1_no_mesh.xml"):
        super().__init__(
            mjcf_path,
            cameras=["global", "left_wrist", "right_wrist"],
            raw_size=(480, 640),
        )

    def _set_grippers(self, widths: np.ndarray) -> None:
        self._set_symmetric_slider("left_eef_joint1", "left_eef_joint2", widths[0])
        self._set_symmetric_slider("right_eef_joint1", "right_eef_joint2", widths[1])
