"""Parity checks between v3 samples and the Uranus-data episode API.

Run against a real sample with:

    URANUS_PARITY_SAMPLE=examples/data_v3_fps3/000004 \
    .venv/bin/python -m unittest tests.test_data_v3_parity -v

The test uses ``get_episode`` only; it never calls ``__getitem__``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

import cv2
import mujoco
import numpy as np

from main import load_sample
from scripts.export_data_v3 import (
    _asset_for_robot,
    _episode_frames,
    _make_robot,
    _np,
    _resample_episode,
    _slice_episode,
)
from uranus.runner import parse_frame_conditions
from uranus_dataset import UranusMultiDataset
from uranus.skeleton import CameraSpec, RigSpec, SkeletonEngine


DATA_ROOT = os.environ.get(
    "URANUS_DATA_ROOT", "tos://uranus-data/las/lance-dataset/robotics/"
)


class DataV3ParityTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("URANUS_PARITY_SAMPLE", "examples/data_v3_fps3/000004")
        cls.sample_dir = Path(configured)
        if not cls.sample_dir.is_dir():
            raise unittest.SkipTest(f"sample directory not found: {cls.sample_dir}")
        cls.meta = json.loads((cls.sample_dir / "meta.json").read_text())
        cls.format_version = int(cls.meta.get("format_version", 3))
        cls.temporal = json.loads((cls.sample_dir / "temporal.json").read_text())
        migrated = cls.meta.get("migrated_from", {})
        cls.repo_id = migrated.get("source_repo_id", migrated.get("repo_id"))
        if not cls.repo_id:
            raise unittest.SkipTest("sample has no migrated_from.repo_id")
        cls.episode_index = int(migrated.get("episode_index", 0))
        cls.target_fps = float(cls.meta.get("fps", 0.0)) or None
        cls.start_frame = int(
            migrated.get("start_frame", os.environ.get("URANUS_PARITY_START_FRAME", 0))
        )
        cls.cameras = [camera["name"] for camera in cls.meta["cameras"]]
        try:
            ds = UranusMultiDataset(
                [cls.repo_id],
                root=DATA_ROOT,
                camera_names={cls.repo_id: cls.cameras},
                shared_features_only=False,
                return_uint8=True,
            )
            raw = ds.get_episode(cls.episode_index)
        except Exception as exc:  # network/auth is an environment prerequisite
            raise unittest.SkipTest(f"unable to read Uranus-data: {exc}") from exc
        child = ds._datasets[0]
        cls.child = child
        sampled, _ = _resample_episode(raw, child, cls.target_fps)
        cls.source = _slice_episode(sampled, child, cls.start_frame)
        cls.robot_name = (
            child.meta.get_episode_robot(cls.episode_index) or child.meta.robot
        )
        cls.source_frames = _episode_frames(
            cls.source, child, cls.episode_index
        )
        if cls.source.get("reference.observation.mobile_base") is not None:
            ref_mobile = cls.source["reference.observation.mobile_base"]
        else:
            episode_start = int(child._episode_from[cls.episode_index])
            ref_mobile = child._take_rows([episode_start])[episode_start]["observation"].get(
                "mobile_base"
            )
        temporal_mobile = cls.source.get("observation.mobile_base")
        cls.source_mobile = None
        if isinstance(ref_mobile, dict) and isinstance(temporal_mobile, dict):
            cls.source_mobile = [
                {key: _np(value).copy() for key, value in ref_mobile.items()}
            ] + [
                {key: _np(value)[i].copy() for key, value in temporal_mobile.items()}
                for i in range(len(_np(next(iter(temporal_mobile.values())))))
            ]

    def test_native_state_stream_matches_get_episode(self):
        expected = np.concatenate(
            [
                _np(self.source["reference.observation.state"])[None].reshape(1, -1),
                _np(self.source["observation.state"]).reshape(
                    len(self.source["observation.state"]), -1
                ),
            ],
            axis=0,
        )
        exported = self.temporal["step_qpos"]
        actual = np.asarray(
            [frame["observation.state"] if isinstance(frame, dict) else frame for frame in exported],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(actual.shape[1], expected.shape[1])

    def test_reference_images_match_get_episode(self):
        for camera in self.cameras:
            source = _np(self.source[f"reference.observation.images.{camera}"])
            source = np.moveaxis(source[:3], 0, -1).astype(np.uint8)
            encoded = cv2.imread(
                str(self.sample_dir / self.meta["ref_cam_images"][camera]),
                cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(encoded, camera)
            actual = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(actual, source, err_msg=camera)

    def test_main_inputs_use_exported_qpos_cache(self):
        loaded = load_sample(self.sample_dir, step_length=1, num_chunks=1)
        cache = json.loads((self.sample_dir / "mujoco_qpos.json").read_text())
        np.testing.assert_allclose(
            np.asarray(loaded["create"]["ref_qpos"]["mujoco_qpos"]), cache[0]
        )
        first_step = loaded["steps"][0]["qpos"][0]
        np.testing.assert_allclose(np.asarray(first_step["mujoco_qpos"]), cache[1])

    def test_reference_camera_extrinsics_match_source(self):
        if self.format_version < 4:
            self.skipTest("exact per-frame camera extrinsics require format_version >= 4")
        first = self.temporal["step_qpos"][0]
        actual, _, _ = parse_frame_conditions(first, tuple(self.cameras))
        self.assertIsNotNone(actual)
        for camera, extrinsic in zip(self.cameras, actual, strict=True):
            expected = _np(self.source[f"reference.camera_extrinsics.{camera}"])
            np.testing.assert_array_equal(extrinsic, expected, err_msg=camera)

    def test_camera_calibration_matches_source_every_frame(self):
        """Check per-frame K/E, including cameras moving with the base."""
        if self.format_version < 4:
            self.skipTest("exact per-frame camera extrinsics require format_version >= 4")
        xml = self.sample_dir / self.meta["mjcf_path"]
        engine = SkeletonEngine(
            RigSpec(str(xml), tuple(CameraSpec(camera) for camera in self.cameras))
        )
        cache = json.loads((self.sample_dir / "mujoco_qpos.json").read_text())
        exported = self.temporal["step_qpos"]
        synthetic = bool(self.meta.get("synthetic_transform"))
        for index, (frame, qpos) in enumerate(zip(exported, cache, strict=True)):
            transform = (
                np.asarray(frame["observation.robot2world_trans"], dtype=np.float64)
                if isinstance(frame, dict) and "observation.robot2world_trans" in frame
                else None
            )
            engine.set_state(qpos, transform)
            actual_extrinsics, _, _ = parse_frame_conditions(
                frame, tuple(self.cameras)
            )
            self.assertIsNotNone(actual_extrinsics)
            for camera, actual_extrinsic in zip(self.cameras, actual_extrinsics, strict=True):
                if index == 0:
                    expected_k = _np(self.source[f"reference.camera_intrinsics.{camera}"])
                    expected_e = _np(self.source[f"reference.camera_extrinsics.{camera}"])
                else:
                    expected_k = _np(self.source[f"camera_intrinsics.{camera}"])[index - 1]
                    expected_e = _np(self.source[f"camera_extrinsics.{camera}"])[index - 1]
                np.testing.assert_allclose(
                    engine.camera_intrinsics(camera), expected_k,
                    atol=2e-5, rtol=2e-5, err_msg=f"{camera} K frame {index}",
                )
                if not synthetic:
                    np.testing.assert_array_equal(
                        actual_extrinsic,
                        expected_e,
                        err_msg=f"{camera} E frame {index}",
                    )

    def test_ee_and_skeleton_geometry_matches_source(self):
        if self.format_version < 4:
            self.skipTest("exact frame geometry requires format_version >= 4")
        loaded = load_sample(self.sample_dir, step_length=1, num_chunks=1)
        create = loaded["create"]
        engine = SkeletonEngine(
            RigSpec(
                mjcf_path=create["mjcf_path"],
                cameras=tuple(create["cameras"]),
                world_from_model=create["world_from_model"],
                end_effectors=tuple(create["end_effectors"]),
                skeleton=create["skeleton"],
            )
        )
        source_robot = _make_robot(
            self.robot_name, _asset_for_robot(self.robot_name)
        )
        cache = json.loads((self.sample_dir / "mujoco_qpos.json").read_text())
        np.testing.assert_allclose(
            engine.get_ee_sh_corrections(),
            source_robot.get_ee_sh_corrections(),
            atol=0.0,
            rtol=0.0,
        )
        for index, (source_frame, exported_frame, qpos) in enumerate(
            zip(self.source_frames, self.temporal["step_qpos"], cache, strict=True)
        ):
            source_robot.build_qpos(source_frame)
            transform = exported_frame.get("observation.robot2world_trans")
            _, radii, widths = parse_frame_conditions(
                exported_frame, tuple(self.cameras)
            )
            engine.set_state(
                qpos,
                transform,
                end_effector_radii=radii,
                gripper_widths=widths,
            )
            expected_ee = source_robot.get_ee_states()
            actual_ee = engine.get_ee_states()
            self.assertEqual(len(actual_ee), len(expected_ee))
            for expected, actual in zip(expected_ee, actual_ee, strict=True):
                np.testing.assert_allclose(
                    actual[0], expected[0], atol=3e-7, rtol=1e-6,
                    err_msg=f"EE position frame {index}",
                )
                np.testing.assert_allclose(
                    actual[1], expected[1], atol=3e-7, rtol=1e-6,
                    err_msg=f"EE rotation frame {index}",
                )
                self.assertAlmostEqual(actual[2], expected[2], places=8)

            expected_keypoints = source_robot.build_keypoints()
            actual_keypoints = engine.get_keypoints()
            self.assertEqual(len(actual_keypoints), len(expected_keypoints))
            for expected, actual in zip(
                expected_keypoints, actual_keypoints, strict=True
            ):
                self.assertEqual(actual["parent"], expected["parent"])
                self.assertEqual(actual["color"], expected["color"])
                np.testing.assert_allclose(
                    actual["pos"], expected["pos"], atol=3e-7, rtol=1e-6,
                    err_msg=f"keypoint frame {index}",
                )

    def test_mobile_base_transform_matches_source(self):
        """Verify the exported robot2world sequence, including translation."""
        first = self.temporal["step_qpos"][0]
        if not isinstance(first, dict) or "observation.robot2world_trans" not in first:
            self.skipTest("sample has no mobile-base transform")
        if self.meta.get("synthetic_transform"):
            self.skipTest("synthetic transform intentionally differs from source")
        if self.source_mobile is None:
            self.skipTest("source episode has no mobile_base fields")

        actual = [
            np.asarray(frame["observation.robot2world_trans"], dtype=np.float64)
            for frame in self.temporal["step_qpos"]
        ]
        robot = _make_robot(self.robot_name, _asset_for_robot(self.robot_name))
        expected = []
        for frame in self.source_frames:
            robot.build_qpos(frame)
            expected.append(robot.get_robot_to_world_transform())
        self.assertEqual(len(actual), len(expected))
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
