"""CLI driver for the single-GPU Uranus streaming runner.

Loads one sample directory holding ``meta.json`` (MJCF path, camera names,
end-effectors, skeleton, prompt) and ``temporal.json`` (a one-time joint-group
description plus per-frame ``joint_positions``, compact ``gripper``, and
``robot_transform`` state). Camera calibration and mounting live in the MJCF.
It drives ``uranus.runner.UranusRunner`` through ``create → N x step`` (models
are assembled lazily inside the first ``create``), then writes aligned H.264
GT, generated, skeleton, and Plücker videos plus one comparison preview.

The reference (create) frame is always frame 0 of the time series; chunk ``c``
of ``step`` covers frames ``[c * step_length + 1, (c + 1) * step_length]``.

Usage:
    uv run python main.py --sample-dir <dir> --weights-dir <dir> [--num-chunks 10]
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from uranus.runner import UranusRunner
from uranus.skeleton import CameraSpec, EESpec, GripperKeypointOverride, SkeletonSpec
from uranus.utils.video import write_h264_video


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
        help="sample dir with meta.json / temporal.json / ref_images",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--teacher-forcing-window-size", type=int, default=4)
    parser.add_argument("--step-length", type=int, default=4, help="frames per step call")
    parser.add_argument("--num-chunks", type=int, default=10, help="step calls (= chunks)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--fps", type=int, default=10)
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
    """Load one XML-environment sample into runner-ready data."""
    sample_dir = Path(sample_dir)
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (sample_dir / "temporal.json").open("r", encoding="utf-8") as f:
        temporal = json.load(f)

    cameras = []
    for cam in meta["cameras"]:
        if isinstance(cam, str):
            cameras.append(CameraSpec(name=cam))
        else:
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
        )
        for override in skel.get("gripper_keypoint_overrides", [])
    )
    skeleton = SkeletonSpec(
        mode=str(skel.get("mode", "full_tree")),
        chains=tuple(tuple(chain) for chain in skel.get("chains", [])),
        skip_bodies=tuple(skel.get("skip_bodies", [])),
        gripper_keypoint_overrides=overrides,
    )

    states = temporal.get("states", temporal.get("step_qpos"))
    if states is None:
        raise ValueError("temporal.json must contain 'states'")
    total_step_frames = num_chunks * step_length
    if len(states) < total_step_frames + 1:
        raise ValueError(
            f"Not enough qpos frames for num_chunks={num_chunks} * step_length={step_length}: "
            f"need >= {total_step_frames + 1}, got {len(states)}"
        )

    height = int(meta.get("height", 384))
    width = int(meta.get("width", 640))
    create = {
        "prompt": str(meta.get("prompt", "")),
        "mjcf_path": str(_resolve_path(meta["mjcf_path"], sample_dir)),
        "ref_cam_images": tuple(
            _resolve_path(meta["ref_cam_images"][name], sample_dir).read_bytes()
            for name in camera_names
        ),
        "ref_qpos": states[0],
        "cameras": cameras,
        "end_effectors": [_build_ee_spec(ee) for ee in meta.get("end_effectors", [])],
        "skeleton": skeleton,
        "camera_names": camera_names,
    }
    steps = []
    for chunk_index in range(num_chunks):
        start = chunk_index * step_length + 1
        steps.append(
            {
                "qpos": tuple(states[start : start + step_length]),
                "num_step": step_length,
            }
        )
    return {
        "camera_names": camera_names,
        "robot_type": (meta.get("migrated_from") or {}).get("robot_type"),
        "gt_videos": {
            name: _resolve_path(meta["gt_videos"][name], sample_dir)
            for name in camera_names
            if name in meta.get("gt_videos", {})
        },
        "height": height,
        "width": width,
        "create": create,
        "steps": steps,
    }


def _read_gt_frames(
    paths: dict[str, Path],
    frame_counts: dict[str, int],
) -> dict[str, list[np.ndarray]]:
    missing = sorted(set(frame_counts) - set(paths))
    if missing:
        raise ValueError(f"meta.json is missing gt_videos for cameras: {missing}")
    result = {}
    for camera_name, frame_count in frame_counts.items():
        capture = cv2.VideoCapture(str(paths[camera_name]))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open GT video: {paths[camera_name]}")
        frames = []
        try:
            while len(frames) < frame_count:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if len(frames) != frame_count:
            raise ValueError(
                f"GT video {paths[camera_name]} has only {len(frames)} frames; "
                f"need {frame_count}"
            )
        result[camera_name] = frames
    return result


def _save_frames(
    frames: dict[str, list[np.ndarray]],
    output_dir: Path,
    *,
    fps: int,
    save_images: bool,
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
        video_path = output_dir / f"{camera_name}.mp4"
        write_h264_video(cam_frames, video_path, fps=fps)
        saved.append(str(video_path))
    return saved



def _preview_cell(
    frame: np.ndarray,
    *,
    size: tuple[int, int],
    label: str,
) -> np.ndarray:
    """Letterbox one RGB frame into a labelled preview cell."""
    target_height, target_width = size
    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    cell = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    cell[top : top + resized_height, left : left + resized_width] = resized
    cv2.rectangle(cell, (0, 0), (min(target_width, 240), 34), (0, 0, 0), -1)
    cv2.putText(
        cell,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cell


def _save_preview(
    *,
    gt: dict[str, list[np.ndarray]],
    gen: dict[str, list[np.ndarray]],
    skeleton: dict[str, list[np.ndarray]],
    plucker: dict[str, list[np.ndarray]],
    camera_names: tuple[str, ...],
    path: Path,
    fps: int,
) -> str:
    """Write one camera-row x GT/GEN/SKELETON/PLUCKER-column preview."""
    streams = {
        "GT": gt,
        "GEN": gen,
        "SKELETON": skeleton,
        "PLUCKER": plucker,
    }
    for stream_name, camera_frames in streams.items():
        missing = sorted(set(camera_names) - set(camera_frames))
        if missing:
            raise ValueError(f"{stream_name} is missing cameras: {missing}")
    frame_count = min(
        len(streams[stream_name][camera])
        for stream_name in streams
        for camera in camera_names
    )
    if frame_count <= 0:
        raise ValueError("cannot create preview without frames")

    target_height, target_width = gen[camera_names[0]][0].shape[:2]
    preview_frames = []
    for index in range(frame_count):
        rows = []
        for camera in camera_names:
            cells = [
                _preview_cell(
                    streams[stream_name][camera][index],
                    size=(target_height, target_width),
                    label=f"{stream_name} / {camera}",
                )
                for stream_name in streams
            ]
            rows.append(np.concatenate(cells, axis=1))
        preview_frames.append(np.concatenate(rows, axis=0))
    write_h264_video(preview_frames, path, fps=fps)
    return str(path)


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
                f"{required} not found in {sample_dir} — expected the XML environment format "
                f"(meta.json with camera names/mjcf_path, temporal.json with states)"
            )
    if not Path(weights_dir).is_dir():
        raise SystemExit(f"weights dir not found: {weights_dir}")

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
        aggregated_plucker: dict[str, list[np.ndarray]] = {}
        total_step_s = 0.0
        for i, step_data in enumerate(sample["steps"]):
            t0 = perf_counter()
            frames, skeleton_frames, plucker_frames = runner.step(
                **step_data,
                return_skeleton=True,
                return_plucker=True,
            )
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
            for camera, cam_frames in skeleton_frames.items():
                aggregated_skeleton.setdefault(camera, []).extend(cam_frames)
            for camera, cam_frames in plucker_frames.items():
                aggregated_plucker.setdefault(camera, []).extend(cam_frames)
    finally:
        runner.close()

    output_dir = Path(args.output_dir)
    frame_counts = {camera: len(frames) for camera, frames in aggregated.items()}
    gt_frames = _read_gt_frames(sample["gt_videos"], frame_counts)
    saved = {
        "gen": _save_frames(
            aggregated,
            output_dir / "gen",
            fps=args.fps,
            save_images=args.save_images,
        ),
        "gt": _save_frames(
            gt_frames,
            output_dir / "gt",
            fps=args.fps,
            save_images=args.save_images,
        ),
        "skeleton": _save_frames(
            aggregated_skeleton,
            output_dir / "skeleton",
            fps=args.fps,
            save_images=args.save_images,
        ),
        "plucker": _save_frames(
            aggregated_plucker,
            output_dir / "plucker",
            fps=args.fps,
            save_images=args.save_images,
        ),
    }
    saved["preview"] = _save_preview(
        gt=gt_frames,
        gen=aggregated,
        skeleton=aggregated_skeleton,
        plucker=aggregated_plucker,
        camera_names=sample["camera_names"],
        path=output_dir / "preview.mp4",
        fps=args.fps,
    )
    meta["rollout"] = {c: len(f) for c, f in aggregated.items()}
    meta["outputs"] = saved
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[uranus-cli] done: create={create_s:.1f} s, steps={total_step_s:.1f} s over "
        f"{len(sample['steps'])} chunks. Artifacts in {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
