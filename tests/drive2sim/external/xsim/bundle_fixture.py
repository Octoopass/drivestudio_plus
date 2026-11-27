"""Dựng gói quan sát nhỏ trên đĩa cho test của adapter XSIM (CONTRACTS mục 5.4)."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def write_bundle(root: Path, eval_mask: bool = True) -> Path:
    """Hai khung (3, 4), một camera mắt cá chỉ số 1, LiDAR ở cả hai khung."""
    root.mkdir(parents=True, exist_ok=True)
    width, height = 16, 12
    for frame in (3, 4):
        (root / "images").mkdir(exist_ok=True)
        pixels = np.full((height, width, 3), 40 * frame, dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"{frame:03d}_1.png")
        (root / "lidar").mkdir(exist_ok=True)
        pose = np.eye(4)
        pose[0, 3] = frame
        np.savez(
            root / "lidar" / f"{frame:03d}.npz",
            points=np.array([[frame, 5, 0], [frame, 5, -1]], dtype=np.float32),
            origins=np.array([[frame, 0, 0]] * 2, dtype=np.float32),
            sensor_to_world=pose,
        )
    (root / "masks").mkdir(exist_ok=True)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, : width // 2] = 255
    if eval_mask:
        Image.fromarray(mask).save(root / "masks" / "eval_1.png")
    Image.fromarray(mask).save(root / "masks" / "train_1.png")
    np.savez(root / "cameras.npz", **{"1": np.stack([np.eye(4), np.eye(4)])})
    index = {
        "format": "d2s-observations",
        "version": 1,
        "scene_id": "kitti360_x",
        "split": {
            "schema_version": "d2s-split/0.1",
            "split_id": "every2_fish",
            "scene_id": "kitti360_x",
            "strategy": "every_nth_frame:2",
            "frame_range": [3, 4],
            "cameras": ["image_02"],
            "train": [[3, "image_02"]],
            "test": [[4, "image_02"]],
        },
        "camera_indices": {"image_02": 1},
        "frames": [3, 4],
        "source_frames": [100, 102],
        "timestamps_us": [1600000000000000, 1600000000100000],
        "cameras": [
            {
                "index": 1,
                "camera_id": "image_02",
                "model": "fisheye",
                "width": width,
                "height": height,
                "intrinsics": [8.0, 8.0, 8.0, 6.0],
                "distortion": [0.1, 0.0, 0.0, 0.0],
                "train_mask": True,
                "eval_mask": eval_mask,
            }
        ],
        "lidar_frames": [3, 4],
        "instances": False,
        "smpl": False,
    }
    (root / "observations.json").write_text(json.dumps(index))
    return root
