"""RoboChallenge Table30v2 UR5 + Robotiq 2F-85 embodiment."""

import numpy as np

from .robot_table30_base import Table30FixedRobot
from .unified_robot import _SH_CORR_RC_UR5


class RobotTable30UR5(Table30FixedRobot):
    arm_joint_names = (
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
    )
    # Table30v2 uses the UR5 base frame as its root. This MuJoCo model's `base`
    # body is at the correct origin but carries an Rz(pi) rotation. Canonicalize
    # through that body to fix the mirrored links without changing their height.
    arm_base_body_names = ("base",)
    reference_body_name = "base"
    skeleton_body_names = (
        (
            "base",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "base_mount",
            "gripper_base",
            "right_driver",
            "right_coupler",
            "right_spring_link",
            "right_follower",
            "right_pad",
            "left_driver",
            "left_coupler",
            "left_spring_link",
            "left_follower",
            "left_pad",
        ),
    )
    ee_names = ("gripper_center",)
    ee_are_sites = True
    gripper_body_pairs = (("right_pad", "left_pad"),)

    def get_ee_sh_corrections(self):
        return [_SH_CORR_RC_UR5]

    def __init__(self, mjcf_path: str = "assets/ur5_2f85_no_mesh.xml"):
        super().__init__(mjcf_path, cameras=["global", "wrist"], raw_size=(480, 640))

    def _set_grippers(self, widths: np.ndarray) -> None:
        width = float(np.clip(widths[0], 0.0, 0.085))
        driver_angle = 0.8 * (1.0 - width / 0.085)
        follower_angle = -0.964 * driver_angle
        self._set_joint_clipped("right_driver_joint",      driver_angle)
        self._set_joint_clipped("left_driver_joint",       driver_angle)
        self._set_joint_clipped("right_coupler_joint",     0.0)
        self._set_joint_clipped("left_coupler_joint",      0.0)
        self._set_joint_clipped("right_spring_link_joint", driver_angle)
        self._set_joint_clipped("left_spring_link_joint",  driver_angle)
        self._set_joint_clipped("right_follower_joint",    follower_angle)
        self._set_joint_clipped("left_follower_joint",     follower_angle)
