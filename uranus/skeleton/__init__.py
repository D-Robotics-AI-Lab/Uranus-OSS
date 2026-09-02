"""Skeleton rendering package: generic MuJoCo FK engine + rig specs.

Replaces the per-robot ``uranus.robot`` classes.  See
``skeleton_refactor_plan.md`` for the frame conventions and design rationale.
"""

from .engine import SkeletonEngine
from .render import render_skeleton_frames, batch_render_sh_on_image
from .specs import (
    CameraSpec,
    EESpec,
    GripperKeypointOverride,
    RigSpec,
    SkeletonSpec,
    rigid_transform,
)

__all__ = [
    "SkeletonEngine",
    "CameraSpec",
    "EESpec",
    "GripperKeypointOverride",
    "RigSpec",
    "SkeletonSpec",
    "rigid_transform",
    "render_skeleton_frames",
    "batch_render_sh_on_image",
]
