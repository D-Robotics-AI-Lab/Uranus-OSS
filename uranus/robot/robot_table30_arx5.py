"""RoboChallenge Table30v2 ARX5 embodiment."""

import numpy as np

from .robot_table30_base import Table30FixedRobot
from .unified_robot import _SH_CORR_RC_ARX5


class RobotTable30ARX5(Table30FixedRobot):
    arm_joint_names = (tuple(f"joint{i}" for i in range(1, 7)),)
    skeleton_body_names = (("base_link", *(f"link{i}" for i in range(1, 7)), "eef_link"),)
    ee_names = ("eef_link",)

    def get_ee_sh_corrections(self):
        return [_SH_CORR_RC_ARX5]

    def __init__(self, mjcf_path: str = "assets/X5_no_mesh.xml"):
        super().__init__(mjcf_path, cameras=["global", "side", "wrist"], raw_size=(720, 1280))

    def _set_grippers(self, widths: np.ndarray) -> None:
        # X5_no_mesh.xml ends at eef_link and contains no articulated finger joints.
        pass
