"""CLI driver for the single-GPU Uranus streaming runner.

Loads one sample directory holding ``meta.json`` (metadata: camera_names /
robot_type / prompt / ref_cam_images) and ``temporal.json`` (time series:
step_qpos / step_ext / step_int, produced by ``scripts/convert_samples.py``),
drives ``uranus.runner.UranusRunner`` through ``create → N x step`` (models
are assembled lazily inside the first ``create``), and writes per-camera mp4
files plus a horizontally-concatenated ``preview.mp4``.

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
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--teacher-forcing-window-size", type=int, default=2)
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


def _as_numpy_tree(value):
    """Restore JSON lists back to numpy where the pipeline expects arrays.

    Numeric (possibly nested) lists become ndarrays; dicts and anything
    non-convertible (e.g. ragged or mixed structures) are kept as JSON trees.
    """
    if isinstance(value, dict):
        return {key: _as_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):  # ragged / non-numeric content
            return [_as_numpy_tree(item) for item in value]
    return value


def load_sample(sample_dir: Path, *, step_length: int, num_chunks: int) -> dict:
    """Load a sample dir (meta.json + temporal.json) into runner-ready data.

    Frame entries of ``step_qpos`` are telemetry dicts
    (``{"observation.state": ..., "observation.robot": ...}``) — passed
    through as-is; ``UranusRunner.step`` splits them internally.
    """
    sample_dir = Path(sample_dir)
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (sample_dir / "temporal.json").open("r", encoding="utf-8") as f:
        temporal = json.load(f)

    camera_names = list(meta["camera_names"])
    step_qpos = temporal["step_qpos"]
    step_ext = temporal["step_ext"]
    step_int = temporal["step_int"]

    total_step_frames = num_chunks * step_length
    if len(step_qpos) < total_step_frames + 1:
        raise ValueError(
            f"Not enough qpos frames for num_chunks={num_chunks} * step_length={step_length}: "
            f"need >= {total_step_frames + 1}, got {len(step_qpos)}"
        )

    height = int(meta.get("height", 384))
    width = int(meta.get("width", 640))
    create = {
        "prompt": str(meta.get("prompt", "")),
        "ref_cam_images": tuple(
            _resolve_path(meta["ref_cam_images"][name], sample_dir).read_bytes()
            for name in camera_names
        ),
        "ref_cam_extrinsics": {
            name: _as_numpy_tree(step_ext[name][0]) for name in camera_names
        },
        "ref_cam_intrinsics": {
            name: _as_numpy_tree(step_int[name][0]) for name in camera_names
        },
        "ref_qpos": _as_numpy_tree(step_qpos[0]),
        "robot_type": str(meta["robot_type"]),
        "camera_names": tuple(camera_names),
    }
    steps = []
    for chunk_index in range(num_chunks):
        start = chunk_index * step_length + 1
        steps.append(
            {
                "qpos": tuple(_as_numpy_tree(f) for f in step_qpos[start : start + step_length]),
                "cam_extrinsics": {
                    name: [_as_numpy_tree(x) for x in step_ext[name][start : start + step_length]]
                    for name in camera_names
                },
                "cam_intrinsics": {
                    name: [_as_numpy_tree(x) for x in step_int[name][start : start + step_length]]
                    for name in camera_names
                },
                "num_step": step_length,
            }
        )
    return {"camera_names": camera_names, "robot_type": create["robot_type"], "height": height, "width": width,
            "create": create, "steps": steps}


def _save_frames(frames: dict[str, list[np.ndarray]], output_dir: Path, *, fps: int, save_images: bool) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    if not frames:
        return saved

    for camera_name, cam_frames in frames.items():
        if not cam_frames:
            continue
        height, width = int(cam_frames[0].shape[0]), int(cam_frames[0].shape[1])
        if save_images:
            cam_dir = output_dir / "obs" / camera_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(cam_frames):
                cv2.imwrite(str(cam_dir / f"{index:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer = cv2.VideoWriter(
            str(output_dir / f"{camera_name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_dir / f'{camera_name}.mp4'}")
        try:
            for frame in cam_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        saved.append(str(output_dir / f"{camera_name}.mp4"))

    frame_count = min(len(f) for f in frames.values() if f)
    if frame_count > 0:
        cameras = list(frames)
        preview_frames = [
            np.concatenate([frames[c][i] for c in cameras], axis=1) for i in range(frame_count)
        ]
        ph, pw = int(preview_frames[0].shape[0]), int(preview_frames[0].shape[1])
        writer = cv2.VideoWriter(
            str(output_dir / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_dir / 'preview.mp4'}")
        try:
            for frame in preview_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        saved.append(str(output_dir / "preview.mp4"))
    return saved


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
                f"{required} not found in {sample_dir} — only the meta.json/temporal.json "
                f"format is supported (convert with scripts/convert_samples.py)"
            )
    if not Path(weights_dir).is_dir():
        raise SystemExit(f"weights dir not found: {weights_dir}")

    sample = load_sample(sample_dir, step_length=args.step_length, num_chunks=args.num_chunks)
    print(
        f"[uranus-cli] sample={sample_dir} cameras={sample['camera_names']} "
        f"robot_type={sample['robot_type']} chunks={len(sample['steps'])}",
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
    finally:
        runner.close()

    output_dir = Path(args.output_dir)
    saved = _save_frames(aggregated, output_dir, fps=args.fps, save_images=args.save_images)
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
