from pathlib import Path
from types import SimpleNamespace

from .unified_robot import UnifiedRobot
from .robot_panda_2f85 import RobotPanda2F85
from .robot_g1_120s import RobotG1120s
from .robot_table30_arx5 import RobotTable30ARX5
from .robot_table30_ur5 import RobotTable30UR5
from .robot_table30_aloha import RobotTable30ALOHA
from .robot_table30_dos_w1 import RobotTable30DOSW1
from .robot_robotwin2_aloha import RobotRoboTwin2ALOHA

from .skeleton_render import render_skeleton_frames

# parents[0]=uranus/robot, parents[1]=uranus, parents[2]=repo root — the assets/
# dir lives next to the checkout (kept out of the package); wheel installs
# would need a different resolution strategy.
_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

ROBOT_CONFIGS = SimpleNamespace(
    droid=str(_ASSETS_DIR / "panda_2f85_no_mesh.xml"),
    agibot=str(_ASSETS_DIR / "g1_120s_no_mesh.xml"),
    arx5=str(_ASSETS_DIR / "X5_no_mesh.xml"),
    ur5=str(_ASSETS_DIR / "ur5_2f85_no_mesh.xml"),
    aloha=str(_ASSETS_DIR / "aloha_tracer2_dabai_dark_no_mesh.xml"),
    dos_w1=str(_ASSETS_DIR / "dos-w1_no_mesh.xml"),
    robotwin2_aloha=str(_ASSETS_DIR / "arx5_description_isaac_no_mesh.xml"),
)

def normalize_robot_type(robot_type: str) -> str:
    aliases = {
        "droid": "droid",
        "agibot": "agibot",
        "arx5": "arx5",
        "ur5": "ur5",
        "aloha": "aloha",
        "robotwin2_aloha": "robotwin2_aloha",
        "dos-w1": "dos_w1",
        "dos_w1": "dos_w1",
    }
    key = str(robot_type).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported robot_type: {robot_type!r}")
    return aliases[key]


def make_robot_for_type(robot_type):
    key = normalize_robot_type(robot_type)
    factories = {
        "droid": RobotPanda2F85,
        "agibot": RobotG1120s,
        "arx5": RobotTable30ARX5,
        "ur5": RobotTable30UR5,
        "aloha": RobotTable30ALOHA,
        "robotwin2_aloha": RobotRoboTwin2ALOHA,
        "dos_w1": RobotTable30DOSW1,
    }
    return factories[key](getattr(ROBOT_CONFIGS, key))


def make_robot_for_path(data_path: str, robot_cfg):
    if 'droid' in data_path:
        return RobotPanda2F85(robot_cfg.droid)
    elif 'agibot' in data_path:
        return RobotG1120s(robot_cfg.agibot)
    elif 'arx5' in data_path:
        return RobotTable30ARX5(robot_cfg.arx5)
    elif 'ur5' in data_path:
        return RobotTable30UR5(robot_cfg.ur5)
    elif 'aloha' in data_path:
        return RobotTable30ALOHA(robot_cfg.aloha)
    elif 'dos_w1' in data_path:
        return RobotTable30DOSW1(robot_cfg.dos_w1)
    elif 'robotwin2_aloha' in data_path:
        return RobotRoboTwin2ALOHA(robot_cfg.robotwin2_aloha)
    return None


__all__ = [
    "UnifiedRobot",
    "RobotPanda2F85",
    "RobotG1120s",
    "RobotTable30ARX5",
    "RobotTable30UR5",
    "RobotTable30ALOHA",
    "RobotRoboTwin2ALOHA",
    "RobotTable30DOSW1",
    "normalize_robot_type",
    "make_robot_for_type",
    "make_robot_for_path",
    "render_skeleton_frames",
]
