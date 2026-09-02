"""RoboChallenge Table30v2 ALOHA dual-arm embodiment."""

import numpy as np

from .robot_table30_base import Table30FixedRobot
from .unified_robot import _SH_CORR_RC_ALOHA


class RobotTable30ALOHA(Table30FixedRobot):
    # Table30v2 uses the front pair; these are the two arms carrying wrist cameras.
    arm_joint_names = (
        tuple(f"fl_joint{i}" for i in range(1, 7)),
        tuple(f"fr_joint{i}" for i in range(1, 7)),
    )
    arm_base_body_names = ("fl_base_link", "fr_base_link")
    skeleton_body_names = (
        ("fl_base_link", *(f"fl_link{i}" for i in range(1, 9))),
        ("fr_base_link", *(f"fr_link{i}" for i in range(1, 9))),
    )
    ee_names = ("gripper_center", "fr_gripper_center")
    ee_are_sites = True
    gripper_body_pairs = (("fl_link7", "fl_link8"), ("fr_link7", "fr_link8"))
    render_gripper_keypoints_from_width = True
    reference_body_name = "fl_base_link"

    def get_ee_sh_corrections(self):
        return [_SH_CORR_RC_ALOHA] * len(self.ee_names)

    def __init__(self, mjcf_path: str = "assets/aloha_tracer2_dabai_dark_no_mesh.xml"):
        super().__init__(
            mjcf_path,
            cameras=["global", "left_wrist", "right_wrist"],
            raw_size=(480, 640),
        )

    def _set_grippers(self, widths: np.ndarray) -> None:
        self._set_symmetric_slider("fl_joint8", "fl_joint7", widths[0])
        self._set_symmetric_slider("fr_joint8", "fr_joint7", widths[1])
