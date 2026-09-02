import numpy as np
import mujoco

from .unified_robot import UnifiedRobot, _SH_CORR_AGIBOT_G1


LEFT_ARM_BODIES  = [f"Link{i}_l" for i in range(1, 8)]
RIGHT_ARM_BODIES = [f"Link{i}_r" for i in range(1, 8)]

LEFT_GRIP_BODIES = [
    "left_Left_01_Link", "left_Left_00_Link", "left_Left_Support_Link", "left_Left_Pad_Link", "left_Left_2_Link",
    "left_Right_2_Link", "left_Right_01_Link", "left_Right_00_Link", "left_Right_Support_Link", "left_Right_Pad_Link",
]
RIGHT_GRIP_BODIES = [
    "right_Left_01_Link", "right_Left_00_Link", "right_Left_Support_Link", "right_Left_Pad_Link", "right_Left_2_Link",
    "right_Right_2_Link", "right_Right_01_Link", "right_Right_00_Link", "right_Right_Support_Link", "right_Right_Pad_Link",
]

GRIPPER_OPEN_ANGLE,GRIPPER_OPEN_MM, GRIPPER_CLOSED_MM = 0.6, 35.0, 120.0

RIGHT_GRIPPER_JOINT_COEFS = {
    "right_Left_1_Joint":        1.0,
    "right_Left_0_Joint":        1.0,
    "right_Left_Support_Joint":  1.0,
    "right_Left_2_Joint":        1.0,
    "right_Right_2_Joint":      -1.0,
    "right_Right_1_Joint":      -1.0,
    "right_Right_0_Joint":      -1.0,
    "right_Right_Support_Joint": 1.0,
}
LEFT_GRIPPER_JOINT_COEFS = {
    "left_Left_1_Joint":        1.0,
    "left_Left_0_Joint":       -1.0,
    "left_Left_Support_Joint":  1.0,
    "left_Left_2_Joint":        1.0,
    "left_Right_2_Joint":      -1.0,
    "left_Right_1_Joint":      -1.0,
    "left_Right_0_Joint":       1.0,
    "left_Right_Support_Joint": 1.0,
}


def quat_xyzw_to_rot(q) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y + z*z), s*(x*y - z*w),     s*(x*z + y*w)],
        [s*(x*y + z*w),     1 - s*(x*x + z*z), s*(y*z - x*w)],
        [s*(x*z - y*w),     s*(y*z + x*w),     1 - s*(x*x + y*y)],
    ], dtype=np.float64)


def get_base_transform(base_pos, base_quat) -> np.ndarray:
    base_pos  = np.asarray(base_pos,  dtype=np.float64).reshape(3)
    base_quat = np.asarray(base_quat, dtype=np.float64).reshape(4)
    if not np.any(base_pos) and not np.any(base_quat):
        return np.eye(4, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(base_quat)
    T[:3, 3]  = base_pos
    return T


def get_normalize_gripper_mm(val: float) -> float:
    """Map gripper value to [0, 1]: 0 = open, 1 = closed."""
    val = float(val)
    if not np.isfinite(val):
        return 0.0
    if abs(val) <= 1.0:
        return float(np.clip(val, 0.0, 1.0))
    return float(np.clip((val - GRIPPER_OPEN_MM) / (GRIPPER_CLOSED_MM - GRIPPER_OPEN_MM), 0.0, 1.0))


class RobotG1120s(UnifiedRobot):
    """AgiBot G1 torso + 120s dual-arm + grippers.

    observation.state:  ndarray(2, 9) — [left(7 joints + gripper_mm + ?), right(...)]
    observation.robot:  dict with waist(2,), robot_base_position(3,),
                        robot_base_orientation(4, xyzw)
    """

    def __init__(
        self,
        mjcf_path: str,
        cameras: list[str] | None = None,
        raw_size: tuple[int, int] | None = None,
    ):
        super().__init__(
            mjcf_path,
            cameras=cameras or ["head", "hand_left", "hand_right"],
            raw_size=raw_size or (480, 640),
        )
        self._T: np.ndarray = np.eye(4, dtype=np.float64)  # base-to-world transform


    def has_gripper(self) -> bool: return True
    def has_arm(self)    -> bool: return True
    def num_arms(self)   -> int:  return 2
    def is_movable(self) -> bool: return True

    def get_ee_sh_corrections(self) -> list[np.ndarray]:
        return [_SH_CORR_AGIBOT_G1, _SH_CORR_AGIBOT_G1]


    def build_qpos(self, frame_data: dict) -> None:
        obs_robot = frame_data["observation.robot"]
        state = np.asarray(frame_data["observation.state"], dtype=np.float64).reshape(2, 9)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        waist = np.asarray(obs_robot["waist"], dtype=np.float64).reshape(2)
        self._set_joint("joint_lift_body",  waist[1])
        self._set_joint("joint_body_pitch", waist[0])
        for i in range(7):
            self._set_joint(f"Joint{i+1}_l", state[0, i])
            self._set_joint(f"Joint{i+1}_r", state[1, i])

        angle_l = GRIPPER_OPEN_ANGLE * (1.0 - get_normalize_gripper_mm(state[0, 7]))
        angle_r = GRIPPER_OPEN_ANGLE * (1.0 - get_normalize_gripper_mm(state[1, 7]))
        self.set_gripper_side(LEFT_GRIPPER_JOINT_COEFS,  angle_l)
        self.set_gripper_side(RIGHT_GRIPPER_JOINT_COEFS, angle_r)

        mujoco.mj_forward(self.model, self.data)
        self._T = get_base_transform(obs_robot["robot_base_position"],
                                  obs_robot["robot_base_orientation"])

    def get_robot_to_world_transform(self) -> np.ndarray:
        return self._T.copy()


    def get_ee_states(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        left_pos,  left_rot  = self.get_ee_pose_from_site("gripper_center")
        right_pos, right_rot = self.get_ee_pose_from_site("right_gripper_center")
        return [
            (left_pos,  left_rot,  self.get_pad_radius("left_Left_Pad_Link",  "left_Right_Pad_Link")),
            (right_pos, right_rot, self.get_pad_radius("right_Left_Pad_Link", "right_Right_Pad_Link")),
        ]


    def build_keypoints(self) -> list[dict]:
        keypoints = []

        left_kps, left_name_to_kp = self.build_arm_keypoints(
            LEFT_ARM_BODIES, color_offset=0, root_parent_kp=None, index_offset=0,
        )
        keypoints.extend(left_kps)
        keypoints.extend(self.build_gripper_keypoints(
            LEFT_GRIP_BODIES, "Link7_l",
            color_offset=len(LEFT_ARM_BODIES), wrist_kp=left_name_to_kp["Link7_l"],
            index_offset=len(keypoints),
        ))

        right_kps, right_name_to_kp = self.build_arm_keypoints(
            RIGHT_ARM_BODIES, color_offset=0, root_parent_kp=None, index_offset=len(keypoints),
        )
        keypoints.extend(right_kps)
        keypoints.extend(self.build_gripper_keypoints(
            RIGHT_GRIP_BODIES, "Link7_r",
            color_offset=len(RIGHT_ARM_BODIES), wrist_kp=right_name_to_kp["Link7_r"],
            index_offset=len(keypoints),
        ))

        return keypoints


    def set_gripper_side(self, coef_dict: dict, angle: float) -> None:
        for name, coef in coef_dict.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            val = coef * angle
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                val = float(np.clip(val, lo, hi))
            self.data.qpos[self.model.jnt_qposadr[jid]] = val

    def get_ee_pose_from_site(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        R, t = self._T[:3, :3], self._T[:3, 3]
        pos = (R @ self.data.site_xpos[sid] + t).astype(np.float32)
        rot = (R @ self.data.site_xmat[sid].reshape(3, 3)).astype(np.float32)
        return pos, rot

    def get_pad_radius(self, pad_a: str, pad_b: str) -> float:
        bid_a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pad_a)
        bid_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pad_b)
        return float(np.linalg.norm(self.data.xpos[bid_a] - self.data.xpos[bid_b])) * 0.5

    def build_arm_keypoints(self, body_names: list[str], color_offset: int,
                              root_parent_kp, index_offset: int):
        keypoints, name_to_kp = [], {}
        R, t = self._T[:3, :3], self._T[:3, 3]
        for j, name in enumerate(body_names):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            parent = root_parent_kp if j == 0 else name_to_kp[body_names[j - 1]]
            keypoints.append({"pos": R @ self.data.xpos[bid] + t,
                               "color": color_offset + j, "parent": parent})
            name_to_kp[name] = index_offset + len(keypoints) - 1
        return keypoints, name_to_kp

    def build_gripper_keypoints(self, grip_body_names: list[str], wrist_body_name: str,
                                  color_offset: int, wrist_kp: int, index_offset: int):
        keypoints = []
        R, t = self._T[:3, :3], self._T[:3, 3]
        wrist_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, wrist_body_name)
        body_to_kp = {wrist_bid: wrist_kp}
        for j, name in enumerate(grip_body_names):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            parent_kp = body_to_kp.get(self.model.body_parentid[bid], wrist_kp)
            body_to_kp[bid] = index_offset + len(keypoints)
            keypoints.append({"pos": R @ self.data.xpos[bid] + t,
                               "color": color_offset + j, "parent": parent_kp})
        return keypoints
