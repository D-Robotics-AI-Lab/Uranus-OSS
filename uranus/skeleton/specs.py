"""Rig / end-effector / skeleton specifications (create-time configuration).

The specs replace the per-robot ``UnifiedRobot`` subclasses: everything the
runtime needs to render skeletons and compose camera extrinsics is expressed
here as plain data, validated once at construction time against the MJCF
model.  ``from_dict`` is the meta.json (v2) deserialization entry point.

Frame conventions (see skeleton_refactor_plan.md §2):

* ``world_from_model`` (C) maps the MJCF model frame to the annotation/world
  frame; identity by default.
* ``CameraSpec.extrinsic_rel`` is camera-from-mount (world-from-camera for
  external cameras, ``mount_body=None``).
* Per-frame robot-to-world (``observation.robot2world_trans``) is *data*, not
  rig state — it enters at ``SkeletonEngine.set_state`` time.

Validation errors raise ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_matrix(name: str, value, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == shape, f"{name} must have shape {shape}, got {array.shape}")
    _require(bool(np.all(np.isfinite(array))), f"{name} must be finite")
    return array


def _check_rigid(name: str, matrix: np.ndarray) -> np.ndarray:
    """Validate a 4x4 as a rigid transform (orthonormal R, det=+1).

    Tolerance is 1e-5: dataset extrinsics carry ~1e-6 non-orthogonality noise
    (float64 round-trips through JSON); real scaling/shear errors are orders
    of magnitude larger.
    """
    R = matrix[:3, :3]
    _require(
        np.allclose(R @ R.T, np.eye(3), atol=1e-5),
        f"{name} rotation block is not orthonormal",
    )
    _require(abs(np.linalg.det(R) - 1.0) < 1e-5, f"{name} rotation determinant != +1")
    _require(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9),
        f"{name} bottom row must be [0, 0, 0, 1]",
    )
    return matrix


def rigid_transform(name: str, value) -> np.ndarray:
    """Validate and return a rigid 4x4 transform (world_from_model, robot2world_trans)."""
    return _check_rigid(name, _as_matrix(name, value, (4, 4)))


@dataclass(frozen=True)
class CameraSpec:
    """One camera of the rig.

    ``mount_body=None`` means external (rigidly attached to the world):
    ``extrinsic_rel`` is then the world-to-camera transform itself and must be
    the constant extrinsic recorded in the data.
    """

    name: str
    intrinsics: np.ndarray          # 3x3, raw sensor resolution
    mount_body: str | None
    extrinsic_rel: np.ndarray       # 4x4 camera-from-mount

    def validate(self) -> None:
        _require(bool(self.name), "camera name must be non-empty")
        _as_matrix(f"cameras[{self.name!r}].intrinsics", self.intrinsics, (3, 3))
        _check_rigid(f"cameras[{self.name!r}].extrinsic_rel", self.extrinsic_rel)


@dataclass(frozen=True)
class EESpec:
    """One end-effector SH sphere: anchor object + radius source."""

    object_type: str                # "body" | "site"
    object_name: str
    radius_mode: str                # "pad_pair" | "fixed"
    pad_bodies: tuple[str, str] | None = None
    radius: float | None = None     # fixed mode, meters
    sh_correction: np.ndarray | None = None   # 3x3, default identity

    def validate(self) -> None:
        _require(
            self.object_type in ("body", "site"),
            f"end_effectors[{self.object_name!r}].object_type must be 'body' or 'site'",
        )
        _require(bool(self.object_name), "end_effector object_name must be non-empty")
        if self.radius_mode == "pad_pair":
            _require(
                self.pad_bodies is not None and len(self.pad_bodies) == 2,
                f"end_effectors[{self.object_name!r}] pad_pair requires two pad bodies",
            )
        elif self.radius_mode == "fixed":
            _require(
                self.radius is not None and self.radius > 0.0,
                f"end_effectors[{self.object_name!r}] fixed radius must be > 0",
            )
        else:
            raise ValueError(
                f"end_effectors[{self.object_name!r}].radius_mode must be 'pad_pair' or 'fixed'"
            )
        if self.sh_correction is not None:
            corr = _as_matrix(
                f"end_effectors[{self.object_name!r}].sh_correction", self.sh_correction, (3, 3)
            )
            _require(
                abs(np.linalg.det(corr)) > 1e-9,
                f"end_effectors[{self.object_name!r}].sh_correction must be invertible",
            )


@dataclass(frozen=True)
class GripperKeypointOverride:
    """Legacy ``render_gripper_keypoints_from_width`` semantics, expressed via
    the rig: the two finger-body keypoints are repositioned to
    ``EE_pos +/- EE_rot[:, closing_axis] * width/2`` where ``width`` is
    recovered as the pad-pair distance (exact under the adjusted slider
    geometry)."""

    ee_object_type: str            # "body" | "site"
    ee_object_name: str
    finger_bodies: tuple[str, str]
    closing_axis: int = 1          # column index into the EE rotation

    def validate(self) -> None:
        _require(self.ee_object_type in ("body", "site"), "gripper override ee_object_type")
        _require(len(self.finger_bodies) == 2, "gripper override needs two finger bodies")
        _require(0 <= self.closing_axis < 3, "closing_axis must be 0/1/2")


@dataclass(frozen=True)
class SkeletonSpec:
    """Keypoint topology for rendering.

    ``mode="chains"``: render the named body chains (color = chain index);
    a keypoint's parent is the nearest ancestor body that is already a
    keypoint (Table30 rule).

    ``mode="full_tree"``: render every body (color = body id) except
    ``skip_bodies`` (Panda rule).

    ``gripper_keypoint_overrides``: optional legacy-width finger repositioning
    (see ``GripperKeypointOverride``).
    """

    mode: str = "full_tree"
    chains: tuple[tuple[str, ...], ...] = ()
    skip_bodies: tuple[str, ...] = ()
    gripper_keypoint_overrides: tuple[GripperKeypointOverride, ...] = ()

    def validate(self) -> None:
        if self.mode not in ("chains", "full_tree"):
            raise ValueError(f"skeleton.mode must be 'chains' or 'full_tree', got {self.mode!r}")
        if self.mode == "chains":
            _require(len(self.chains) > 0, "skeleton chains must be non-empty in 'chains' mode")
            for i, chain in enumerate(self.chains):
                _require(len(chain) > 0, f"skeleton chains[{i}] must be non-empty")
        for ov in self.gripper_keypoint_overrides:
            ov.validate()


@dataclass(frozen=True)
class RigSpec:
    """Full create-time configuration (meta.json v2 ``rig`` payload)."""

    mjcf_path: str
    cameras: tuple[CameraSpec, ...]                 # ordered — camera order is rig order
    world_from_model: np.ndarray                    # 4x4 (default identity)
    end_effectors: tuple[EESpec, ...] = ()
    skeleton: SkeletonSpec = field(default_factory=SkeletonSpec)

    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)

    def validate(self) -> None:
        _require(bool(self.mjcf_path), "mjcf_path must be non-empty")
        _require(len(self.cameras) > 0, "cameras must be non-empty")
        names = self.camera_names()
        _require(len(names) == len(set(names)), f"camera names must be unique, got {names}")
        for camera in self.cameras:
            camera.validate()
        _check_rigid("world_from_model", self.world_from_model)
        for ee in self.end_effectors:
            ee.validate()
        self.skeleton.validate()

    # ------------------------------------------------------------------ io

    @classmethod
    def from_dict(cls, payload: dict) -> "RigSpec":
        """Build from a meta.json (v2) dict (see skeleton_refactor_plan.md §4)."""
        cameras = tuple(
            CameraSpec(
                name=str(cam["name"]),
                intrinsics=_as_matrix(f"cameras[{cam['name']!r}].intrinsics", cam["intrinsics"], (3, 3)),
                mount_body=cam.get("mount_body"),
                extrinsic_rel=_as_matrix(
                    f"cameras[{cam['name']!r}].extrinsic_rel", cam["extrinsic_rel"], (4, 4)
                ),
            )
            for cam in payload["cameras"]
        )
        world_from_model = (
            rigid_transform("world_from_model", payload["world_from_model"])
            if payload.get("world_from_model") is not None
            else np.eye(4)
        )
        end_effectors = tuple(
            EESpec(
                object_type=str(ee["object_type"]),
                object_name=str(ee["object_name"]),
                radius_mode=str(ee["radius_mode"]),
                pad_bodies=tuple(ee["pad_bodies"]) if ee.get("pad_bodies") is not None else None,
                radius=float(ee["radius"]) if ee.get("radius") is not None else None,
                sh_correction=(
                    _as_matrix(
                        f"end_effectors[{ee['object_name']!r}].sh_correction",
                        ee["sh_correction"], (3, 3),
                    )
                    if ee.get("sh_correction") is not None
                    else None
                ),
            )
            for ee in payload.get("end_effectors", [])
        )
        skel = payload.get("skeleton") or {}
        overrides = tuple(
            GripperKeypointOverride(
                ee_object_type=str(ov["ee_object_type"]),
                ee_object_name=str(ov["ee_object_name"]),
                finger_bodies=tuple(ov["finger_bodies"]),
                closing_axis=int(ov.get("closing_axis", 1)),
            )
            for ov in skel.get("gripper_keypoint_overrides", [])
        )
        skeleton = SkeletonSpec(
            mode=str(skel.get("mode", "full_tree")),
            chains=tuple(tuple(chain) for chain in skel.get("chains", [])),
            skip_bodies=tuple(skel.get("skip_bodies", [])),
            gripper_keypoint_overrides=overrides,
        )
        rig = cls(
            mjcf_path=str(payload["mjcf_path"]),
            cameras=cameras,
            world_from_model=world_from_model,
            end_effectors=end_effectors,
            skeleton=skeleton,
        )
        rig.validate()
        return rig
