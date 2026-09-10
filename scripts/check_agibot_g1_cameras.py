"""Diagnostic: compare Agibot G1 source camera extrinsics to MJCF FK.

Uses UranusMultiDataset.get_episode and samples frames across selected episodes.
No files are exported; prints per-camera best mount and residual statistics.
"""
from __future__ import annotations

import argparse
import numpy as np

from uranus_dataset import UranusMultiDataset
from scripts.export_data_v3 import (
    DATASET_ROOT,
    _asset_for_robot,
    _body_poses,
    _camera_sequence,
    _episode_frames,
    _make_robot,
    _np,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2, 10, 100, 200, 287, 300, 500, 1000])
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--data-root", default=DATASET_ROOT)
    args = ap.parse_args()
    cameras = ["head", "hand_left", "hand_right"]
    ds = UranusMultiDataset(["agibot"], root=args.data_root,
                            camera_names={"agibot": cameras},
                            shared_features_only=False, return_uint8=True)
    child = ds._datasets[0]
    asset = _asset_for_robot("Agibot G1")
    print(f"dataset episodes={child.num_episodes} fps={child.fps} asset={asset}")
    for ep in args.episodes:
        if ep < 0 or ep >= child.num_episodes:
            print(f"episode {ep}: SKIP (out of range)")
            continue
        item = ds.get_episode(ep)
        n = int(_np(item["observation.state"]).shape[0]) + 1
        inds = np.unique(np.linspace(0, n - 1, min(args.samples, n), dtype=int))
        # Keep the complete item contract but use a sampled subset for FK.
        frames_all = _episode_frames(item, child, ep)
        frames = [frames_all[int(i)] for i in inds]
        robot = _make_robot("Agibot G1", asset)
        qpos, poses, Ts = _body_poses(robot, frames)
        print(f"\nepisode={ep} frames={n} sampled={len(inds)}")
        for camera in cameras:
            _, _, Es_all = _camera_sequence(item, camera)
            # Source E sequence is [reference, temporal...].
            Es = [_np(item[f"reference.camera_extrinsics.{camera}"]).astype(float)]
            Es.extend(_np(item[f"camera_extrinsics.{camera}"]).astype(float))
            Es = [Es[int(i)] for i in inds]
            rows = []
            for body in poses[0]:
                rel = [E @ (T @ pose[body]) for E, T, pose in zip(Es, Ts, poses)]
                d = np.asarray([np.linalg.norm(x - rel[0]) for x in rel])
                rows.append((float(np.max(d)), float(np.mean(d)), body, rel[0]))
            rows.sort(key=lambda x: x[0])
            best = rows[:5]
            print(f"  {camera}: best=" + ", ".join(f"{b[2]} max={b[0]:.4g} mean={b[1]:.4g}" for b in best))
            # Explicitly report semantic wrist/head candidates when present.
            names = {"head": ["head_link3", "head_link2", "head_link1"],
                     "hand_left": ["arm_l_link7", "Link7_l"],
                     "hand_right": ["arm_r_link7", "Link7_r"]}[camera]
            for name in names:
                match = [r for r in rows if r[2] == name]
                if match:
                    r = match[0]
                    print(f"    semantic {name}: max={r[0]:.6g} mean={r[1]:.6g}")


if __name__ == "__main__":
    main()
