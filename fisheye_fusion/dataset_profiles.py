"""Which driving datasets the lab can import, how to recognise them and how DriveStudio preprocesses them.

Every dataset except KITTI-360 follows the DriveStudio convention (docs/<Name>.md):
    data/<id>/raw/        original files, kept in the dataset's own folder structure
    data/<id>/processed*/ output of datasets/preprocess.py, one NNN/ folder per scene (what training reads)
KITTI-360 keeps its archive layout under kitti360/ and is prepared by PipelineManager.prepare_drivestudio.
Adding a dataset = one entry in PROFILES (+ its datasets/<id>/ loader in DriveStudio).
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

PROFILES = {
    "kitti360": {
        "label": "KITTI-360", "doc": "docs/drivestudio-kitti360.md", "raw": "kitti360",
        "hint": "data_poses.zip, calibration, data_2d_raw/<seq>_image_0X.tar, data_3d_raw/<seq>.tar",
    },
    "waymo": {
        "label": "Waymo Open", "doc": "docs/Waymo.md", "raw": "data/waymo/raw",
        "hint": "các file segment-*.tfrecord",
        "preprocess": {"split": "training", "target": "data/waymo/processed/training", "scene_ids": "0"},
    },
    "nuscenes": {
        "label": "nuScenes", "doc": "docs/NuScenes.md", "raw": "data/nuscenes/raw",
        "hint": "thư mục v1.0-mini / v1.0-trainval, samples/, sweeps/, maps/",
        "preprocess": {"split": "v1.0-mini", "target": "data/nuscenes/processed_10Hz/mini", "start_idx": 0,
                       "num_scenes": 10, "interpolate_N": 4},
    },
    "pandaset": {
        "label": "PandaSet", "doc": "docs/Pandaset.md", "raw": "data/pandaset/raw",
        "hint": "các thư mục scene 001/, 002/… (mỗi cái có camera/, lidar/, annotations/)",
        "preprocess": {"target": "data/pandaset/processed", "split_file": "data/pandaset_example_scenes.txt"},
    },
    "argoverse": {
        "label": "Argoverse 2", "doc": "docs/ArgoVerse.md", "raw": "data/argoverse/raw",
        "hint": "thư mục train/ chứa các log dạng UUID",
        "preprocess": {"target": "data/argoverse/processed/training", "split_file": "data/argoverse_example_scenes.txt"},
    },
    "kitti": {
        "label": "KITTI (raw)", "doc": "docs/KITTI.md", "raw": "data/kitti/raw",
        "hint": "thư mục ngày 2011_09_26/ chứa *_drive_*_sync và calib_*.txt",
        "preprocess": {"split": "2011_09_26", "target": "data/kitti/processed", "split_file": "data/kitti_example_scenes.txt"},
    },
    "nuplan": {
        "label": "nuPlan", "doc": "docs/Nuplan.md", "raw": "data/nuplan/raw",
        "hint": "maps/ và nuplan-v1.1/ (splits/, sensor_blobs/)",
        "preprocess": {"split": "mini", "target": "data/nuplan/processed", "split_file": "data/nuplan_example_scenes.txt",
                       "start_frame_idx": 1000, "max_frame_limit": 300},
    },
    "physicalai": {
        "label": "NVIDIA PhysicalAI (NCore)", "doc": "docs/PhysicalAI.md", "raw": "data/physicalai/raw",
        "hint": "clips/<uuid>/pai_<uuid>.json và *.zarr.itar",
        "preprocess": {"target": "data/physicalai/processed", "scene_ids": "0"},
    },
}

KITTI360_SEQ = re.compile(r"2013_05_28_drive_\d{4}_sync")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SAFE_PART = re.compile(r"[^/\\:*?\"<>|\x00-\x1f]{1,200}")
# (dataset, test on a set of relative paths) - first match wins, most specific first
RULES = [
    ("kitti360", lambda names, parts: any(n in names for n in ("data_poses.zip", "data_2d_raw", "data_3d_raw"))
        or any(KITTI360_SEQ.match(n) for n in names)),
    ("physicalai", lambda names, parts: "clips" in names or any(n.startswith("pai_") for n in names)),
    ("nuplan", lambda names, parts: "nuplan-v1.1" in names or any(n.endswith(".db") for n in names)),
    ("nuscenes", lambda names, parts: any(n.startswith("v1.0-") for n in names) or {"samples", "sweeps"} <= names),
    ("waymo", lambda names, parts: any(n.endswith(".tfrecord") for n in names)),
    ("kitti", lambda names, parts: any(re.fullmatch(r"20\d\d_\d\d_\d\d", n) for n in names)
        or any(n.startswith("calib_cam_to_cam") for n in names)),
    ("argoverse", lambda names, parts: any(UUID.fullmatch(n) for n in names)),
    ("pandaset", lambda names, parts: "pandaset-dataset.zip" in names
        or (any(re.fullmatch(r"\d{3}", n) for n in names) and bool({"camera", "lidar", "annotations"} & names))),
]


def detect(paths) -> str | None:
    """Dataset whose layout matches these relative paths (a folder listing or picked files), else None."""
    names, parts = set(), []
    for path in paths:
        bits = [b for b in str(path).replace("\\", "/").split("/") if b]
        parts.append(bits)
        names.update(bits[:3])          # look at the first levels only: cheap on huge trees
    for dataset, test in RULES:
        if test(names, parts):
            return dataset
    return None


NESTED = {"argoverse", "pandaset"}   # markers sit one level down (train/<uuid>, 001/camera)


def detect_folder(names, children=()) -> str | None:
    """Dataset rooted exactly at a folder: its own entries decide; children only for nested layouts."""
    direct = detect(names)
    if direct:
        return direct
    nested = detect([*names, *children])
    return nested if nested in NESTED else None


def safe_relative(path: str, max_depth: int = 12) -> str:
    """A relative POSIX path with ordinary file names only (no '..', no drive letters, no hidden roots)."""
    bits = [b for b in str(path or "").replace("\\", "/").split("/") if b not in ("", ".")]
    if not bits or len(bits) > max_depth or any(b == ".." or not SAFE_PART.fullmatch(b) for b in bits) \
            or bits[0].startswith("."):
        raise ValueError(f"đường dẫn không hợp lệ: {path}")
    return str(PurePosixPath(*bits))


# dependency (fisheye_fusion/deps.py) each dataset's preprocessing needs
SDK = {"nuscenes": "sdk-nuscenes", "waymo": "sdk-waymo", "pandaset": "sdk-pandaset", "argoverse": "sdk-argoverse",
       "kitti": "sdk-kitti", "nuplan": "sdk-nuplan", "physicalai": "sdk-physicalai", "kitti360": "ds-ncore"}


def public() -> list[dict]:
    """Profile list for the UI."""
    return [{"id": k, "label": v["label"], "raw": v["raw"], "hint": v["hint"], "doc": v["doc"],
             "preprocess": v.get("preprocess"), "sdk": SDK.get(k)} for k, v in PROFILES.items()]
