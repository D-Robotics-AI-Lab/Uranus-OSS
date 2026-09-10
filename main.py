"""CLI driver for the single-GPU Uranus streaming runner (v2 rig format).

Loads one sample directory holding ``meta.json`` (rig: mjcf_path / XML camera
names / end_effectors / skeleton / prompt) and ``temporal.json``
(``step_qpos``: native compact observations, optionally accompanied by a
``mujoco_qpos.json`` renderer cache and per-frame
``observation.robot2world_trans``), plus optional per-camera ``gt_videos``, drives
``uranus.runner.UranusRunner`` through ``create → N x step`` (models are
assembled lazily inside the first ``create``), and writes per-camera mp4
files plus a comparison ``preview.mp4`` with GT, generated, and skeleton rows.

The reference (create) frame is always frame 0 of the time series; chunk ``c``
of ``step`` covers frames ``[c * step_length + 1, (c + 1) * step_length]``.

Usage:
    uv run python main.py --sample-dir <dir> --weights-dir <dir> [--num-chunks 10]
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from uranus.runner import UranusRunner
from uranus.skeleton import (
    CameraSpec,
    EESpec,
    GripperKeypointOverride,
    SkeletonKeypointSpec,
    SkeletonSpec,
)
from uranus.utils.video import encode_h264


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uranus single-GPU streaming runner CLI")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="converted weights dir (dit.pt/vae.pt/text_encoder.pt/plucker_adapter.pt/"
        "vace_patch_embedding.pt/tokenizer/)",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="sample dir with meta.json / temporal.json / ref_images (v2 rig format)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--teacher-forcing-window-size", type=int, default=2)
    parser.add_argument("--step-length", type=int, default=4, help="frames per step call")
    parser.add_argument("--num-chunks", type=int, default=10, help="step calls (= chunks)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 quality setting; lower is higher quality (default: 23)",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        help="libx264 encoding speed/size preset (default: medium)",
    )
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def _resolve_path(path_str: str, sample_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else sample_dir / path


def _as_matrix(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _build_ee_spec(ee: dict) -> EESpec:
    return EESpec(
        object_type=str(ee["object_type"]),
        object_name=str(ee["object_name"]),
        radius_mode=str(ee["radius_mode"]),
        pad_bodies=tuple(ee["pad_bodies"]) if ee.get("pad_bodies") is not None else None,
        radius=float(ee["radius"]) if ee.get("radius") is not None else None,
        sh_correction=(
            _as_matrix(ee["sh_correction"]) if ee.get("sh_correction") is not None else None
        ),
    )


def load_sample(sample_dir: Path, *, step_length: int, num_chunks: int) -> dict:
    """Load a v2 sample dir (meta.json + temporal.json) into runner-ready data."""
    sample_dir = Path(sample_dir)
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (sample_dir / "temporal.json").open("r", encoding="utf-8") as f:
        temporal = json.load(f)

    cameras = []
    for cam in meta["cameras"]:
        if isinstance(cam, str):
            cameras.append(CameraSpec(name=cam))
            continue
        cameras.append(
            CameraSpec(
                name=str(cam["name"]),
                intrinsics=(
                    _as_matrix(cam["intrinsics"])
                    if cam.get("intrinsics") is not None
                    else None
                ),
                mount_body=cam.get("mount_body"),
                extrinsic_rel=(
                    _as_matrix(cam["extrinsic_rel"])
                    if cam.get("extrinsic_rel") is not None
                    else None
                ),
            )
        )
    camera_names = tuple(cam.name for cam in cameras)

    skel = meta.get("skeleton") or {}
    overrides = tuple(
        GripperKeypointOverride(
            ee_object_type=str(override["ee_object_type"]),
            ee_object_name=str(override["ee_object_name"]),
            finger_bodies=tuple(override["finger_bodies"]),
            closing_axis=int(override.get("closing_axis", 1)),
            width_index=(
                int(override["width_index"])
                if override.get("width_index") is not None
                else None
            ),
        )
        for override in skel.get("gripper_keypoint_overrides", [])
    )
    skeleton = SkeletonSpec(
        mode=str(skel.get("mode", "full_tree")),
        chains=tuple(tuple(chain) for chain in skel.get("chains", [])),
        skip_bodies=tuple(skel.get("skip_bodies", [])),
        gripper_keypoint_overrides=overrides,
        keypoints=tuple(
            SkeletonKeypointSpec(
                body_name=str(keypoint["body_name"]),
                color=int(keypoint["color"]),
                parent=(
                    int(keypoint["parent"])
                    if keypoint.get("parent") is not None
                    else None
                ),
            )
            for keypoint in skel.get("keypoints", [])
        ),
    )

    step_qpos = temporal["step_qpos"]
    cache_file = sample_dir / "mujoco_qpos.json"
    if cache_file.is_file():
        with cache_file.open("r", encoding="utf-8") as f:
            cached_qpos = json.load(f)
    else:
        # Transitional support for the first v3 prototype.
        cached_qpos = temporal.get("mujoco_qpos")
    if cached_qpos is not None and len(cached_qpos) != len(step_qpos):
        raise ValueError("temporal.mujoco_qpos must have the same length as step_qpos")
    total_step_frames = num_chunks * step_length
    if len(step_qpos) < total_step_frames + 1:
        raise ValueError(
            f"Not enough qpos frames for num_chunks={num_chunks} * step_length={step_length}: "
            f"need >= {total_step_frames + 1}, got {len(step_qpos)}"
        )

    height = int(meta.get("height", 384))
    width = int(meta.get("width", 640))
    def frame_with_cache(index: int) -> dict:
        frame = step_qpos[index]
        if not isinstance(frame, dict):
            frame = {"observation.state": frame}
        else:
            frame = dict(frame)
        if cached_qpos is not None:
            frame["mujoco_qpos"] = cached_qpos[index]
        return frame

    create = {
        "prompt": str(meta.get("prompt", "")),
        "mjcf_path": str(_resolve_path(meta["mjcf_path"], sample_dir)),
        "ref_cam_images": tuple(
            _resolve_path(meta["ref_cam_images"][name], sample_dir).read_bytes()
            for name in camera_names
        ),
        "ref_qpos": frame_with_cache(0),
        "cameras": cameras,
        "world_from_model": (
            _as_matrix(meta["world_from_model"])
            if meta.get("world_from_model") is not None
            else None
        ),
        "end_effectors": [_build_ee_spec(ee) for ee in meta.get("end_effectors", [])],
        "skeleton": skeleton,
        "camera_names": camera_names,
    }
    steps = []
    for chunk_index in range(num_chunks):
        start = chunk_index * step_length + 1
        steps.append(
            {
                "qpos": tuple(frame_with_cache(index) for index in range(start, start + step_length)),
                "num_step": step_length,
            }
        )
    gt_video_paths = {
        name: _resolve_path(path, sample_dir)
        for name, path in (meta.get("gt_videos") or {}).items()
    }
    if gt_video_paths:
        missing_gt = [name for name in camera_names if name not in gt_video_paths]
        if missing_gt:
            raise ValueError(f"meta.gt_videos is missing cameras: {missing_gt}")
        missing_files = [
            str(path) for path in gt_video_paths.values() if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(f"GT videos not found: {missing_files}")

    return {
        "camera_names": camera_names,
        "robot_type": (meta.get("migrated_from") or {}).get("robot_type"),
        "height": height,
        "width": width,
        "gt_video_paths": gt_video_paths,
        "create": create,
        "steps": steps,
    }


def _encode_h264(
    frames: list[np.ndarray],
    destination: Path,
    *,
    fps: int,
    crf: int,
    preset: str,
) -> None:
    """Compatibility wrapper around the shared H.264 encoder."""
    encode_h264(frames, destination, fps=fps, crf=crf, preset=preset)


def _save_frames(
    frames: dict[str, list[np.ndarray]],
    output_dir: Path,
    *,
    fps: int,
    crf: int,
    preset: str,
    save_images: bool,
    save_preview: bool = True,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    if not frames:
        return saved

    for camera_name, cam_frames in frames.items():
        if not cam_frames:
            continue
        if save_images:
            cam_dir = output_dir / "obs" / camera_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(cam_frames):
                cv2.imwrite(
                    str(cam_dir / f"{index:04d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
        output_path = output_dir / f"{camera_name}.mp4"
        _encode_h264(
            cam_frames,
            output_path,
            fps=fps,
            crf=crf,
            preset=preset,
        )
        saved.append(str(output_path))

    frame_count = min(len(f) for f in frames.values() if f)
    if save_preview and frame_count > 0:
        cameras = list(frames)
        preview_frames = [
            np.concatenate([frames[c][i] for c in cameras], axis=1)
            for i in range(frame_count)
        ]
        output_path = output_dir / "preview.mp4"
        _encode_h264(
            preview_frames,
            output_path,
            fps=fps,
            crf=crf,
            preset=preset,
        )
        saved.append(str(output_path))
    return saved


def _load_video_frames(path: Path, *, frame_count: int) -> list[np.ndarray]:
    """Decode exactly the prefix needed for the generated rollout."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open GT video: {path}")
    frames = []
    try:
        while len(frames) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < frame_count:
        raise ValueError(
            f"GT video {path} has only {len(frames)} frames; need {frame_count}"
        )
    return frames


def _resize_like(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    target_height, target_width = reference.shape[:2]
    if frame.shape[:2] == (target_height, target_width):
        return frame
    return cv2.resize(
        frame, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


def _save_comparison_preview(
    *,
    gt_frames: dict[str, list[np.ndarray]],
    generated_frames: dict[str, list[np.ndarray]],
    skeleton_frames: dict[str, list[np.ndarray]],
    camera_names: tuple[str, ...],
    destination: Path,
    fps: int,
    crf: int,
    preset: str,
) -> None:
    """Write camera columns with GT/generated/skeleton rows, in that order."""
    rows = (gt_frames, generated_frames, skeleton_frames)
    missing = [
        camera
        for camera in camera_names
        if any(camera not in row or not row[camera] for row in rows)
    ]
    if missing:
        raise ValueError(f"Cannot build comparison preview; missing frames for {missing}")
    frame_count = len(generated_frames[camera_names[0]])
    mismatched = {
        f"{row_name}/{camera}": len(row[camera])
        for row_name, row in zip(("gt", "generated", "skeleton"), rows, strict=True)
        for camera in camera_names
        if len(row[camera]) != frame_count
    }
    if mismatched:
        raise ValueError(
            f"Cannot build synchronized comparison preview; expected {frame_count} "
            f"frames per stream, got {mismatched}"
        )
    preview_frames = []
    for index in range(frame_count):
        tiled_rows = []
        for row in rows:
            tiles = [
                _resize_like(row[camera][index], generated_frames[camera][index])
                for camera in camera_names
            ]
            tiled_rows.append(np.concatenate(tiles, axis=1))
        preview_frames.append(np.concatenate(tiled_rows, axis=0))
    _encode_h264(
        preview_frames,
        destination,
        fps=fps,
        crf=crf,
        preset=preset,
    )


def main() -> None:
    args = parse_args()
    weights_dir = args.weights_dir or os.environ.get("URANUS_WEIGHTS_DIR")
    sample_dir = args.sample_dir or os.environ.get("URANUS_SAMPLE_DIR")
    if not weights_dir:
        raise SystemExit("--weights-dir is required (or set URANUS_WEIGHTS_DIR)")
    if not sample_dir:
        raise SystemExit("--sample-dir is required (or set URANUS_SAMPLE_DIR)")
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise SystemExit(f"sample dir not found: {sample_dir}")
    for required in ("meta.json", "temporal.json"):
        if not (sample_dir / required).is_file():
            raise SystemExit(
                f"{required} not found in {sample_dir} — expected the v2 rig format "
                f"(meta.json with cameras/mjcf_path, temporal.json with step_qpos; "
                f"migrate legacy samples with scripts/migrate_samples.py)"
            )
    if not Path(weights_dir).is_dir():
        raise SystemExit(f"weights dir not found: {weights_dir}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for direct H.264 output but was not found")
    if args.crf < 0 or args.crf > 51:
        raise SystemExit("--crf must be between 0 and 51")

    sample = load_sample(sample_dir, step_length=args.step_length, num_chunks=args.num_chunks)
    print(
        f"[uranus-cli] sample={sample_dir} cameras={sample['camera_names']} "
        f"robot={sample['robot_type']} chunks={len(sample['steps'])}",
        flush=True,
    )
    if (sample["height"], sample["width"]) != (args.height, args.width):
        print(
            f"[uranus-cli] WARNING: sample target is {sample['height']}x{sample['width']} "
            f"but generating at {args.height}x{args.width}",
            flush=True,
        )

    runner = UranusRunner(
        weights_dir,
        device=args.device,
        dtype=args.dtype,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        teacher_forcing_window_size=args.teacher_forcing_window_size,
    )

    meta: dict = {"phases": {}}
    try:
        t0 = perf_counter()
        runner.create(seed=args.seed, **sample["create"])
        create_s = perf_counter() - t0
        print(f"[uranus-cli] create done in {create_s:.1f} s (no video; includes first-time model load)", flush=True)
        meta["phases"]["create"] = {"seconds": create_s}

        aggregated: dict[str, list[np.ndarray]] = {}
        aggregated_skeleton: dict[str, list[np.ndarray]] = {}
        total_step_s = 0.0
        for i, step_data in enumerate(sample["steps"]):
            t0 = perf_counter()
            frames = runner.step(seed=args.seed, **step_data)
            step_s = perf_counter() - t0
            total_step_s += step_s
            ssummary = {c: len(f) for c, f in frames.items()}
            print(
                f"[uranus-cli] step_{i:02d} done in {step_s:.1f} s, frames: {ssummary}",
                flush=True,
            )
            meta["phases"][f"step_{i:02d}"] = {"seconds": step_s}
            for camera, cam_frames in frames.items():
                aggregated.setdefault(camera, []).extend(cam_frames)
            for camera, cam_frames in runner.last_skeleton_frames.items():
                aggregated_skeleton.setdefault(camera, []).extend(cam_frames)
    finally:
        runner.close()

    output_dir = Path(args.output_dir)
    saved = _save_frames(
        aggregated,
        output_dir,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        save_images=args.save_images,
        save_preview=not bool(sample["gt_video_paths"]),
    )
    gt_frames: dict[str, list[np.ndarray]] = {}
    gt_video_paths = sample["gt_video_paths"]
    if gt_video_paths:
        for camera in sample["camera_names"]:
            gt_frames[camera] = _load_video_frames(
                gt_video_paths[camera], frame_count=len(aggregated[camera])
            )
    skeleton_dir = output_dir / "skeleton"
    skeleton_saved = _save_frames(
        aggregated_skeleton,
        skeleton_dir,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        save_images=args.save_images,
    )
    comparison_preview = None
    if gt_frames:
        comparison_path = output_dir / "preview.mp4"
        _save_comparison_preview(
            gt_frames=gt_frames,
            generated_frames=aggregated,
            skeleton_frames=aggregated_skeleton,
            camera_names=sample["camera_names"],
            destination=comparison_path,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
        )
        comparison_preview = str(comparison_path)
        saved.append(comparison_preview)
    else:
        print(
            "[uranus-cli] WARNING: sample has no meta.gt_videos; "
            "preview.mp4 contains only the generated camera row",
            flush=True,
        )
    meta["rollout"] = {c: len(f) for c, f in aggregated.items()}
    meta["outputs"] = saved
    meta["skeleton_outputs"] = skeleton_saved
    meta["comparison_preview"] = comparison_preview
    meta["preview_layout"] = (
        {
            "rows": ["gt", "generated", "skeleton"],
            "columns": list(sample["camera_names"]),
        }
        if comparison_preview
        else {"rows": ["generated"], "columns": list(sample["camera_names"])}
    )
    meta["video_encoding"] = {
        "container": "mp4",
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "crf": args.crf,
        "preset": args.preset,
        "faststart": True,
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[uranus-cli] done: create={create_s:.1f} s, steps={total_step_s:.1f} s over "
        f"{len(sample['steps'])} chunks. Artifacts in {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
