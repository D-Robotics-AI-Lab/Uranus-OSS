from abc import ABC, abstractmethod
import mujoco
import numpy as np

_SH_CORR_DROID_PANDA = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32
)
_SH_CORR_DROID_UR5 = _SH_CORR_DROID_PANDA
_SH_CORR_AGIBOT_G1 = _SH_CORR_DROID_PANDA
_SH_CORR_RC_ALOHA = _SH_CORR_DROID_PANDA
_SH_CORR_RC_UR5 = _SH_CORR_DROID_PANDA
_SH_CORR_RC_ARX5 = np.array(
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32
)
_SH_CORR_RC_DOSW1 = _SH_CORR_RC_ARX5


class UnifiedRobot(ABC):
    """Abstract base for all robots.

    Usage pattern per frame:
        robot.build_qpos(frame_data)  # FK + cache base transform
        poses  = robot.get_ee_poses()
        radii  = robot.get_gripper_radii()
        kpts   = robot.build_keypoints()  # None → use body-tree renderer
    """

    def __init__(
        self,
        mjcf_path: str,
        cameras: list[str],
        raw_size: tuple[int, int],
    ):
        self.mjcf_path = mjcf_path
        self.cameras   = cameras
        self.raw_size  = raw_size
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data  = mujoco.MjData(self.model)


    @property
    def skip_body_names(self) -> list[str]:
        return []

    @abstractmethod
    def has_gripper(self) -> bool: ...

    @abstractmethod
    def has_arm(self) -> bool: ...

    @abstractmethod
    def num_arms(self) -> int: ...

    @abstractmethod
    def is_movable(self) -> bool:
        """Whether the robot base can translate/rotate during a trajectory.

        Movable robots (e.g. G1 with wheel base) carry a per-frame base
        transform in observation.robot; fixed robots (e.g. Panda) always
        have identity base transform.
        """
        ...


    def _set_joint(self, name: str, val: float) -> None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(val)


    @abstractmethod
    def build_qpos(self, frame_data: dict) -> None:
        """Set qpos from frame_data and call mj_forward.

        Implementations should also cache any frame-level state (e.g. base
        transform) needed by get_ee_poses / build_keypoints.
        """
        ...

    @abstractmethod
    def get_robot_to_world_transform(self) -> np.ndarray:
        """After build_qpos: return the robot/base-to-world transform."""
        ...


    @abstractmethod
    def get_ee_states(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """After build_qpos: return [(pos (3,), rot (3,3), radius), ...] per end-effector.

        pos/rot are in world frame (base transform applied).
        """
        ...

    def get_ee_sh_corrections(self) -> list[np.ndarray]:
        """Per-EE rotation correction applied only to SH sphere rendering.

        Returns a list of (3, 3) matrices, one per end-effector.  The SH
        direction is computed as ``dirs @ (ee_rot_world @ correction)`` so
        that grippers with different flange conventions produce the same
        colour when pointing in the same physical direction.

        Override in subclasses where the native gripper_center frame does
        not follow the canonical convention (X=left, Y=forward, Z=up from
        the wrist-camera viewpoint).  The default returns identity for
        every EE.
        """
        return [np.eye(3, dtype=np.float32)] * self.num_arms


    @abstractmethod
    def build_keypoints(self) -> list[dict]:
        """After build_qpos: build skeleton keypoints for rendering.

        Each keypoint is {"pos": ndarray(3,), "color": int, "parent": int | None}.
        """
        ...


def precompute_fk(robot: UnifiedRobot, frames_data: list[dict]):
    ee_states_all, keypoints_all, robot_to_world_all = [], [], []
    for frame_data in frames_data:
        robot.build_qpos(frame_data)
        ee_states_all.append(robot.get_ee_states())
        keypoints_all.append(robot.build_keypoints())
        robot_to_world_all.append(robot.get_robot_to_world_transform())
    return ee_states_all, keypoints_all, robot_to_world_all
