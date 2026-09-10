"""Download complete camera MP4s for the selected interaction episodes.

The Lance exporter stores calibration, reference images, and robot states, but
it intentionally does not copy the source videos.  This companion exporter
copies the original per-camera MP4 files byte-for-byte from ``uranus-data``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uranus_dataset import UranusDataset
from utils.fsspec_util import get_tosfs


DATASET_ROOT = "tos://uranus-data/las/lance-dataset/robotics/"

# Output order matches the existing interaction_11 XML/state export.
SELECTIONS = (
    ("agibot", "agibot", 0),
    ("agibot-challenge-2026", "agibot-world-2026", 204),
    ("droid", "droid", 6),
    ("rc-table30-v1/rc-aloha", "rc-table30-v1/rc-aloha", 0),
    ("rc-table30-v1/rc-arx5", "rc-table30-v1/rc-arx5", 0),
    ("rc-table30-v1/rc-franka", "rc-table30-v1/rc-franka", 0),
    ("rc-table30-v1/rc-ur5", "rc-table30-v1/rc-ur5", 0),
    ("rc-table30-v2/rc-aloha", "rc-table30-v2/rc-aloha", 0),
    ("rc-table30-v2/rc-arx5", "rc-table30-v2/rc-arx5", 0),
    ("rc-table30-v2/rc-dos-w1", "rc-table30-v2/rc-dos-w1", 0),
    ("rc-table30-v2/rc-ur5", "rc-table30-v2/rc-ur5", 0),
)


def _download_one(fs, source: str, target: Path) -> int:
    expected = int(fs.size(source))
    if target.is_file() and target.stat().st_size == expected:
        return expected

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    partial.unlink(missing_ok=True)
    fs.get_file(source, str(partial))
    actual = partial.stat().st_size
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise IOError(
            f"incomplete download for {source}: got {actual} bytes, "
            f"expected {expected}"
        )
    partial.replace(target)
    return actual


def _sequence_name(repo_id: str, episode_index: int) -> str:
    dataset_name = repo_id.replace("/", "__")
    return f"{dataset_name}_ep{episode_index:06d}"


def export_videos(output_root: Path) -> list[dict]:
    fs = get_tosfs()
    if fs is None:
        raise RuntimeError("VOLC_ACCESSKEY and VOLC_SECRETKEY are required")

    manifest = []
    for output_index, (repo_id, source_repo_id, episode_index) in enumerate(SELECTIONS):
        dataset = UranusDataset(
            repo_id=source_repo_id,
            root=f"{DATASET_ROOT.rstrip('/')}/{source_repo_id}",
        )
        if episode_index < 0 or episode_index >= dataset.num_episodes:
            raise IndexError(
                f"{source_repo_id}: episode {episode_index} is out of range "
                f"(num_episodes={dataset.num_episodes})"
            )

        row = dataset.meta.episodes.iloc[episode_index]
        cameras = list(dataset.camera_names)
        if len(cameras) < 2:
            raise ValueError(
                f"{repo_id}: expected at least 2 cameras, got {cameras}"
            )
        if len(cameras) != 3:
            print(
                f"WARNING: {repo_id} has {len(cameras)} native views: {cameras}",
                flush=True,
            )
        sequence_name = _sequence_name(repo_id, episode_index)
        sample_dir = output_root / sequence_name
        video_dir = sample_dir / "videos"
        sample_dir.mkdir(parents=True, exist_ok=True)

        video_files = {}
        video_sources = {}
        video_sizes = {}
        for camera in cameras:
            source = row.get(f"video_path_{camera}")
            if not isinstance(source, str) or not source:
                raise ValueError(f"{repo_id}: missing video_path_{camera}")
            target = video_dir / f"{camera}.mp4"
            print(
                f"[{output_index + 1}/11] {repo_id} episode={episode_index} "
                f"camera={camera} -> {target}",
                flush=True,
            )
            size = _download_one(fs, source, target)
            video_files[camera] = str(target.relative_to(sample_dir))
            video_sources[camera] = source
            video_sizes[camera] = size

        meta_path = sample_dir / "meta.json"
        meta = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "sequence_name": sequence_name,
                "repo_id": repo_id,
                "source_repo_id": source_repo_id,
                "episode_index": episode_index,
                "robot": str(row.get("robot", "")),
                "task": str(row.get("task", "")),
                "fps": float(row.get("fps", dataset.fps)),
                "length": int(row.get("length", 0)),
                "cameras": cameras,
                "videos": video_files,
                "video_sources": video_sources,
                "video_sizes_bytes": video_sizes,
            }
        )
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        manifest.append(
            {
                "output_index": output_index,
                "sequence_name": sequence_name,
                "repo_id": repo_id,
                "source_repo_id": source_repo_id,
                "episode_index": episode_index,
                "robot": str(row.get("robot", "")),
                "task": str(row.get("task", "")),
                "cameras": cameras,
                "videos": video_files,
                "video_sources": video_sources,
                "video_sizes_bytes": video_sizes,
            }
        )

    (output_root / "selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/interaction_11"),
    )
    args = parser.parse_args()
    export_videos(args.output_root)


if __name__ == "__main__":
    main()
