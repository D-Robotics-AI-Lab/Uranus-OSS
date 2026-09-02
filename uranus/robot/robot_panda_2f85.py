import numpy as np
import mujoco

from .unified_robot import UnifiedRobot, _SH_CORR_DROID_PANDA


class RobotPanda2F85(UnifiedRobot):
    """Franka Panda + Robotiq 2F-85 single-arm robot.

    observation.state: ndarray(9,) — 7 arm joints + 2 finger values
    """

    def __init__(
        self,
        mjcf_path: str,
        cameras: list[str] | None = None,
        raw_size: tuple[int, int] | None = None,
    ):
        super().__init__(
            mjcf_path,
            cameras=cameras or ["ext1", "ext2", "wrist"],
            raw_size=raw_size or (720, 1280),
        )

    @property
    def skip_body_names(self) -> list[str]:
        return ["gripper_center"]

    def has_gripper(self) -> bool:
        return True

    def has_arm(self) -> bool:
        return True

    def num_arms(self) -> int:
        return 1

    def is_movable(self) -> bool:
        return False

    def get_ee_sh_corrections(self) -> list[np.ndarray]:
        return [_SH_CORR_DROID_PANDA]

    def build_qpos(self, frame_data: dict) -> None:
        qpos = np.asarray(frame_data["observation.state"], dtype=np.float64).copy()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.qpos[:7] = qpos[:7]

        panda_finger = float(qpos[7])
        panda_finger = float(np.clip(panda_finger, 0.0, 0.04))

        # panda finger: 0=closed 0.04=open  →  2F-85 ctrl: 255=closed 0=open
        driver_angle = 0.8 * (1.0 - panda_finger / 0.04)

        follower_angle = -0.964 * driver_angle
        self._set_joint("right_driver_joint",      driver_angle)
        self._set_joint("left_driver_joint",       driver_angle)
        self._set_joint("right_coupler_joint",     0.0)
        self._set_joint("left_coupler_joint",      0.0)
        self._set_joint("right_spring_link_joint", driver_angle)
        self._set_joint("left_spring_link_joint",  driver_angle)
        self._set_joint("right_follower_joint",    follower_angle)
        self._set_joint("left_follower_joint",     follower_angle)

        mujoco.mj_forward(self.model, self.data)

    def get_robot_to_world_transform(self) -> np.ndarray:
        return np.eye(4, dtype=np.float64)

    def get_ee_states(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_center")
        if body_id < 0:
            raise ValueError("'gripper_center' body not found in MJCF")
        pos = self.data.xpos[body_id].copy().astype(np.float32)
        rot = self.data.xmat[body_id].reshape(3, 3).copy().astype(np.float32)

        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_silicone_pad")
        left_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_silicone_pad")
        if right_id < 0 or left_id < 0:
            raise ValueError("2F85 silicone pad bodies not found in MJCF")
        radius = float(np.linalg.norm(self.data.xpos[right_id] - self.data.xpos[left_id])) * 0.5

        return [(pos, rot, radius)]

    def build_keypoints(self) -> list[dict]:
        skip_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.skip_body_names
        }
        keypoints = []
        body_to_kp = {}
        for bid in range(self.model.nbody):
            if bid in skip_ids:
                continue
            parent_bid = self.model.body_parentid[bid]
            pos = self.data.xpos[bid].copy().astype(np.float32)
            keypoints.append({"pos": pos, "color": bid, "parent": body_to_kp.get(parent_bid)})
            body_to_kp[bid] = len(keypoints) - 1
        return keypoints
