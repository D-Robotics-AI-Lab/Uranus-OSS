"""Generic MuJoCo FK engine: qpos in, world-frame geometry out.

Replaces the per-robot ``UnifiedRobot`` subclasses.  One engine is built per
session (``create``) and persists for its lifetime; every ``step`` frame goes
through ``set_state`` once and all cameras share the resulting FK.

Frame conventions (skeleton_refactor_plan.md §2):

    p_w      = T @ C @ p_model                    (points)
    M_B_w(t) = T @ C @ M_B_model(t)               (body poses)
    E_cam_w(t) = E_rel @ inv(M_B_w(t))            (camera extrinsics)

where ``T`` is the per-frame robot-to-world transform (data, default I) and
``C`` is the create-time ``world_from_model`` (default I).  The composed
``E_cam_w`` is the single source of truth fed to *both* the skeleton renderer
and the Plücker conditioning — it must never be recomputed differently in the
two consumers.

Validation errors raise ``ValueError``.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .specs import (
    CameraSpec,
    EESpec,
    RigSpec,
    SkeletonSpec,
    rigid_transform,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mat4(xpos, xmat) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    M[:3, 3] = np.asarray(xpos, dtype=np.float64).reshape(3)
    return M


class SkeletonEngine:
    """MuJoCo FK engine bound to one MJCF model and one rig configuration."""

    def __init__(self, rig: RigSpec):
        rig.validate()
        self.rig = rig
        self.model = mujoco.MjModel.from_xml_path(rig.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self._C = np.asarray(rig.world_from_model, dtype=np.float64)
        self._T = np.eye(4, dtype=np.float64)  # last set robot2world (default I)
        self._has_state = False

        # Bind every spec name -> id once, fail fast (mirrors legacy _require_name).
        self._body_ids: dict[str, int] = {}
        self._site_ids: dict[str, int] = {}
        self._mount_ids: list[int | None] = []

        for camera in rig.cameras:
            if camera.mount_body is not None:
                self._mount_ids.append(self._require_body(camera.mount_body))
            else:
                self._mount_ids.append(None)
        for ee in rig.end_effectors:
            if ee.object_type == "site":
                self._require_site(ee.object_name)
                if ee.pad_bodies is not None:
                    self._require_body(ee.pad_bodies[0])
                    self._require_body(ee.pad_bodies[1])
            else:
                self._require_body(ee.object_name)
                if ee.pad_bodies is not None:
                    self._require_body(ee.pad_bodies[0])
                    self._require_body(ee.pad_bodies[1])
        if rig.skeleton.mode == "chains":
            for chain in rig.skeleton.chains:
                for name in chain:
                    self._require_body(name)
        for name in rig.skeleton.skip_bodies:
            self._require_body(name)
        for ov in rig.skeleton.gripper_keypoint_overrides:
            if ov.ee_object_type == "site":
                self._require_site(ov.ee_object_name)
            else:
                self._require_body(ov.ee_object_name)
            self._require_body(ov.finger_bodies[0])
            self._require_body(ov.finger_bodies[1])

        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- naming

    def _require_body(self, name: str) -> int:
        bid = self._body_ids.get(name)
        if bid is not None:
            return bid
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        _require(bid >= 0, f"body {name!r} not found in {self.rig.mjcf_path}")
        self._body_ids[name] = bid
        return bid

    def _require_site(self, name: str) -> int:
        sid = self._site_ids.get(name)
        if sid is not None:
            return sid
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        _require(sid >= 0, f"site {name!r} not found in {self.rig.mjcf_path}")
        self._site_ids[name] = sid
        return sid

    # ---------------------------------------------------------------- state

    def set_state(self, qpos, robot2world=None) -> None:
        """Set the full qpos vector (with joint-range clipping) and run FK.

        ``qpos`` must be the complete model qpos (length ``model.nq``) — the
        format produced by the migration script.  ``robot2world`` is the
        optional per-frame robot-to-world 4x4 (default identity).
        """
        vector = np.asarray(qpos, dtype=np.float64).reshape(-1)
        _require(
            vector.shape[0] == self.model.nq,
            f"qpos must have length {self.model.nq}, got {vector.shape[0]}",
        )
        _require(bool(np.all(np.isfinite(vector))), "qpos must be finite")

        self.data.qpos[:] = vector
        # Clip every limited joint to its range (idempotent on already-clipped
        # migration output; protects against out-of-range telemetry).
        for jid in range(self.model.njnt):
            if self.model.jnt_limited[jid]:
                adr = self.model.jnt_qposadr[jid]
                lo, hi = self.model.jnt_range[jid]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr], lo, hi)

        if robot2world is None:
            self._T = np.eye(4, dtype=np.float64)
        else:
            self._T = rigid_transform("robot2world", robot2world).copy()

        mujoco.mj_forward(self.model, self.data)
        self._has_state = True

    def _require_state(self) -> None:
        _require(self._has_state, "set_state() must be called before reading FK results")

    # ---------------------------------------------------------------- frames

    def body_pose_model(self, body_name: str) -> np.ndarray:
        """4x4 world-from-body in the *model* frame (no C, no T)."""
        self._require_state()
        bid = self._require_body(body_name)
        return _mat4(self.data.xpos[bid], self.data.xmat[bid])

    def body_pose_world(self, body_name: str) -> np.ndarray:
        """4x4 world-from-body: T @ C @ M_B_model."""
        return self._T @ self._C @ self.body_pose_model(body_name)

    def compose_camera_extrinsics(self) -> list[np.ndarray]:
        """World-to-camera 4x4 for every rig camera, in rig order.

        Mounted: E_cam_w = E_rel @ inv(T @ C @ M_B_model).
        External: mount is the world, E_cam_w = E_rel (constant).
        """
        self._require_state()
        result: list[np.ndarray] = []
        for camera, mount_id in zip(self.rig.cameras, self._mount_ids):
            if mount_id is None:
                result.append(np.asarray(camera.extrinsic_rel, dtype=np.float64).copy())
            else:
                M = _mat4(self.data.xpos[mount_id], self.data.xmat[mount_id])
                M_w = self._T @ self._C @ M
                result.append(
                    np.asarray(camera.extrinsic_rel, dtype=np.float64) @ np.linalg.inv(M_w)
                )
        return result

    # ---------------------------------------------------------------- ees

    def get_ee_states(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """[(pos, rot, radius)] per EE spec, in world frame (T @ C applied)."""
        self._require_state()
        R_wc, t_wc = self._T[:3, :3] @ self._C[:3, :3], self._T[:3, :3] @ self._C[:3, 3]
        result: list[tuple[np.ndarray, np.ndarray, float]] = []
        for ee in self.rig.end_effectors:
            if ee.object_type == "site":
                sid = self._require_site(ee.object_name)
                pos_m = self.data.site_xpos[sid]
                rot_m = self.data.site_xmat[sid]
            else:
                bid = self._require_body(ee.object_name)
                pos_m = self.data.xpos[bid]
                rot_m = self.data.xmat[bid]
            pos = (R_wc @ np.asarray(pos_m) + t_wc).astype(np.float32)
            rot = (R_wc @ np.asarray(rot_m).reshape(3, 3)).astype(np.float32)

            if ee.radius_mode == "pad_pair":
                a = self._require_body(ee.pad_bodies[0])
                b = self._require_body(ee.pad_bodies[1])
                # legacy semantics: radius = max(pad_distance / 2, 1e-3) — the
                # 1mm floor keeps a visible dot for closed grippers
                radius = max(
                    float(np.linalg.norm(self.data.xpos[a] - self.data.xpos[b])) * 0.5,
                    1e-3,
                )
            else:
                radius = max(float(ee.radius), 1e-3)
            result.append((pos, rot, radius))
        return result

    def get_ee_sh_corrections(self) -> list[np.ndarray]:
        """Per-EE SH correction matrices (identity where unset)."""
        return [
            (
                np.asarray(ee.sh_correction, dtype=np.float32)
                if ee.sh_correction is not None
                else np.eye(3, dtype=np.float32)
            )
            for ee in self.rig.end_effectors
        ]

    # ---------------------------------------------------------------- keypoints

    def get_keypoints(self) -> list[dict]:
        """Skeleton keypoints in world frame; format matches the legacy renderer.

        Each keypoint: {"pos": (3,) float32 world, "color": int, "parent": int | None}.
        """
        self._require_state()
        spec = self.rig.skeleton
        R_wc, t_wc = self._T[:3, :3] @ self._C[:3, :3], self._T[:3, :3] @ self._C[:3, 3]

        def to_world(pos_m) -> np.ndarray:
            return (R_wc @ np.asarray(pos_m) + t_wc).astype(np.float32)

        keypoints: list[dict] = []
        if spec.mode == "full_tree":
            skip_ids = {self._require_body(name) for name in spec.skip_bodies}
            body_to_kp: dict[int, int] = {}
            for bid in range(self.model.nbody):
                if bid in skip_ids:
                    continue
                parent_bid = self.model.body_parentid[bid]
                keypoints.append(
                    {
                        "pos": to_world(self.data.xpos[bid]),
                        "color": bid,
                        "parent": body_to_kp.get(parent_bid),
                    }
                )
                body_to_kp[bid] = len(keypoints) - 1
            return keypoints

        # chains mode: color = position within the chain (the legacy variable
        # is named chain_index but enumerates the *inner* loop); parent =
        # nearest keypoint ancestor walking up the body tree (Table30 rule).
        body_to_kp: dict[int, int] = {}
        for chain in spec.chains:
            for chain_index, name in enumerate(chain):
                bid = self._require_body(name)
                parent_id = self.model.body_parentid[bid]
                while parent_id > 0 and parent_id not in body_to_kp:
                    parent_id = self.model.body_parentid[parent_id]
                keypoints.append(
                    {
                        "pos": to_world(self.data.xpos[bid]),
                        "color": chain_index,
                        "parent": body_to_kp.get(parent_id),
                    }
                )
                body_to_kp[bid] = len(keypoints) - 1

        # legacy ``render_gripper_keypoints_from_width``: reposition the two
        # finger-body keypoints to EE_pos +/- EE_rot[:, closing_axis] * w/2,
        # where w is recovered as the pad-pair distance (exact under the
        # adjusted slider geometry).
        for ov in spec.gripper_keypoint_overrides:
            first_kp = body_to_kp[self._require_body(ov.finger_bodies[0])]
            second_kp = body_to_kp[self._require_body(ov.finger_bodies[1])]
            if ov.ee_object_type == "site":
                sid = self._require_site(ov.ee_object_name)
                center_m = self.data.site_xpos[sid]
                rot_m = np.asarray(self.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
            else:
                bid = self._require_body(ov.ee_object_name)
                center_m = self.data.xpos[bid]
                rot_m = np.asarray(self.data.xmat[bid], dtype=np.float64).reshape(3, 3)
            center = R_wc @ np.asarray(center_m) + t_wc
            closing_axis = (R_wc @ rot_m)[:, ov.closing_axis]
            a = self._require_body(ov.finger_bodies[0])
            b = self._require_body(ov.finger_bodies[1])
            width = float(np.linalg.norm(self.data.xpos[a] - self.data.xpos[b]))
            half_width = width * 0.5
            keypoints[first_kp]["pos"] = (center - closing_axis * half_width).astype(np.float32)
            keypoints[second_kp]["pos"] = (center + closing_axis * half_width).astype(np.float32)
        return keypoints

    # ---------------------------------------------------------------- helpers

    def scaled_intrinsics(self, camera: CameraSpec, raw_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
        """Scale a camera's raw intrinsics to the target resolution."""
        raw_h, raw_w = raw_size
        target_h, target_w = target_size
        K = np.asarray(camera.intrinsics, dtype=np.float64).copy()
        K[0, :] *= target_w / raw_w
        K[1, :] *= target_h / raw_h
        return K

    @property
    def nq(self) -> int:
        return int(self.model.nq)
