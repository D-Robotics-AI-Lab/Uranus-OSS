import sys
import time
from pathlib import Path
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from examples.load_dataset import export_multiview_grid
from uranus_dataset import UranusMultiDataset


def _rgb_uint8(value, temporal_index: int) -> np.ndarray:
    """Convert a [T, C, H, W] or [C, H, W] tensor to an RGB uint8 image."""
    if value.ndim == 4:
        value = value[temporal_index]
    if value.ndim != 3 or value.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected [T, C, H, W] or [C, H, W], got {value.shape}")

    image = value.detach().cpu().numpy()
    image = np.moveaxis(image[:3], 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def _label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.putText(
        image,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image

def export_multiview_grid(
    frame: dict,
    item_index: int,
    temporal_index: int,
    delta_s: float,
    output_dir: Path,
    cameras: list[str],
) -> Path:
    """Export rows of original, skeleton, and overlay; cameras are columns."""
    originals = [
        _rgb_uint8(frame[f"observation.images.{camera}"], temporal_index)
        for camera in cameras
    ]
    skeletons = [
        _rgb_uint8(frame[f"observation.skeleton.{camera}"], temporal_index)
        for camera in cameras
    ]

    # Normalize tile sizes so cameras with different source resolutions can be joined.
    height, width = originals[0].shape[:2]

    def resize(image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (height, width):
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    originals = [resize(image) for image in originals]
    skeletons = [resize(image) for image in skeletons]
    overlays = []
    for original, skeleton in zip(originals, skeletons, strict=True):
        overlay = original.copy()
        mask = np.any(skeleton != 0, axis=-1)
        overlay[mask] = (
            0.35 * original[mask] + 0.65 * skeleton[mask]
        ).astype(np.uint8)
        overlays.append(overlay)

    rows = []
    for row_name, images in (
        ("original", originals),
        ("skeleton", skeletons),
        ("overlay", overlays),
    ):
        labeled = [
            _label(image, f"{row_name} | {camera} | {delta_s:+.3f}s")
            for camera, image in zip(cameras, images, strict=True)
        ]
        rows.append(np.concatenate(labeled, axis=1))
    grid = np.concatenate(rows, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_index = int(frame["episode_index"].item())
    frame_index = int(frame["frame_index"].item())
    sample_dir = output_dir / (
        f"item_{item_index:06d}_episode_{episode_index:06d}_"
        f"anchor_frame_{frame_index:06d}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    filename = f"timestep_{temporal_index:02d}_delta_{delta_s:+.3f}s.png"
    output_path = sample_dir / filename
    if not cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {output_path}")
    return output_path



DATASET_ROOT = "tos://uranus-data/las/lance-dataset/robotics/"
DATASET_CONFIGS = {
    "agibot": {"camera_names": ["head", "hand_left", "hand_right"]},
    "agibot-challenge-2026": {
        "camera_names": ["head", "hand_left", "hand_right"]
    },
    "droid": {"camera_names": ["ext1", "ext2", "wrist"]},
    "rc-table30-v1/rc-aloha": {
        "camera_names": ["global", "left_wrist", "right_wrist"]
    },
    "rc-table30-v1/rc-arx5": {
        "camera_names": ["global", "side", "wrist"]
    },
    "rc-table30-v1/rc-franka": {
        "camera_names": ["main", "side", "wrist"]
    },
    "rc-table30-v1/rc-ur5": {"camera_names": ["global", "wrist"]},
    "rc-table30-v2/rc-aloha": {
        "camera_names": ["global", "left_wrist", "right_wrist"]
    },
    "rc-table30-v2/rc-arx5": {
        "camera_names": ["global", "side", "wrist"]
    },
    "rc-table30-v2/rc-dos-w1": {
        "camera_names": ["global", "left_wrist", "right_wrist"]
    },
    "rc-table30-v2/rc-ur5": {
        "camera_names": ["global", "wrist", "pad"]
    },
}
OUTPUT_DIR = Path("outputs/multidataset_skeleton_grids")
MAX_EPISODES = 8
OUTPUT_HZ = 3
NUM_TEMPORAL_SAMPLES = 34


def build_deltas(source_fps: int) -> list[float]:
    frame_offsets = [
        round(sample_index * source_fps / OUTPUT_HZ)
        for sample_index in range(NUM_TEMPORAL_SAMPLES - 1, -1, -1)
    ]
    return [-frame_offset / source_fps for frame_offset in frame_offsets]


def main() -> None:
    repo_ids = list(DATASET_CONFIGS)
    cameras_by_repo = {
        repo_id: config["camera_names"]
        for repo_id, config in DATASET_CONFIGS.items()
    }
    metadata_ds = UranusMultiDataset(
        repo_ids,
        root=DATASET_ROOT,
        camera_names=cameras_by_repo,
        shared_features_only=False,
    )
    fps_by_repo = metadata_ds.fps
    deltas_by_repo = {
        repo_id: build_deltas(source_fps)
        for repo_id, source_fps in fps_by_repo.items()
    }
    delta_timestamps_by_repo = {
        repo_id: {
            "observation.images.*": deltas,
            "observation.state": deltas,
        }
        for repo_id, deltas in deltas_by_repo.items()
    }

    dataset = UranusMultiDataset(
        repo_ids,
        root=DATASET_ROOT,
        camera_names=cameras_by_repo,
        delta_timestamps=delta_timestamps_by_repo,
        render_skeleton=True,
        episode_level=True,
        shared_features_only=False,
    )

    print(f"Dataset configs: {DATASET_CONFIGS}")
    print(f"Episodes: all ({len(dataset)})")
    print(f"Source FPS: {fps_by_repo}")
    print(f"Output FPS: {OUTPUT_HZ}")

    item_indices = []
    for episode_offset in range(MAX_EPISODES):
        for start, size in zip(
            dataset.dataset_start_indices,
            dataset.dataset_sizes,
        ):
            if episode_offset < size:
                item_indices.append(start + episode_offset)
            if len(item_indices) == MAX_EPISODES:
                break
        if len(item_indices) == MAX_EPISODES:
            break
    total = 0.0
    exported = 0
    for item_index in item_indices:
        start = time.perf_counter()
        item = dataset[item_index]
        elapsed = time.perf_counter() - start
        total += elapsed

        dataset_index = int(item["dataset_index"].item())
        repo_id = repo_ids[dataset_index]
        cameras = cameras_by_repo[repo_id]
        deltas = deltas_by_repo[repo_id]
        episode_index = int(item["episode_index"].item())
        anchor_frame = int(item["frame_index"].item())
        output_paths = [
            export_multiview_grid(
                item,
                item_index,
                temporal_index,
                delta_s,
                OUTPUT_DIR / repo_id,
                cameras,
            )
            for temporal_index, delta_s in enumerate(deltas)
        ]
        exported += len(output_paths)
        print(
            f"[{item_index}] repo={repo_id} episode={episode_index} "
            f"anchor_frame={anchor_frame} {elapsed:.4f}s "
            f"exported={len(output_paths)} dir={output_paths[0].parent}"
        )

    print(f"\nAvg: {total / len(item_indices):.4f}s per episode")
    print(f"Exported {exported} grids to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
