"""Phép so của nghiệm thu tầng dữ liệu (``scripts/data_acceptance.py``) bắt đúng khác biệt."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from drive2sim.scripts.data_acceptance import compare  # noqa: E402


def _write(path: Path, pose: "np.ndarray", depth: "np.ndarray") -> None:
    nonzero = np.flatnonzero(depth)
    np.savez(
        path,
        **{
            "pixel.cam_to_worlds": pose,
            "pixel.cam_to_worlds::shape": np.array(pose.shape),
            "pixel.camera_data[0].lidar_depth_maps": depth,
            "pixel.camera_data[0].lidar_depth_maps::shape": np.array(depth.shape),
            "sparse_depth[0]::idx": nonzero,
            "sparse_depth[0]::val": depth.ravel()[nonzero],
        },
    )


def test_identical_paths_pass(tmp_path: Path) -> None:
    pose, depth = np.eye(4)[None], np.array([[0.0, 5.0], [9.0, 0.0]])
    _write(tmp_path / "a.npz", pose, depth)
    _write(tmp_path / "b.npz", pose, depth)
    assert compare(tmp_path / "a.npz", tmp_path / "b.npz") == []


def test_changed_pose_is_reported(tmp_path: Path) -> None:
    depth = np.zeros((2, 2))
    moved = np.eye(4)[None].copy()
    moved[0, 0, 3] = 1e-6  # lệch 1 micromet vẫn bị bắt: so từng bit
    _write(tmp_path / "a.npz", np.eye(4)[None], depth)
    _write(tmp_path / "b.npz", moved, depth)
    assert compare(tmp_path / "a.npz", tmp_path / "b.npz") == ["pixel.cam_to_worlds: khác"]


def test_nearer_depth_is_known_but_farther_is_not(tmp_path: Path) -> None:
    pose = np.eye(4)[None]
    _write(tmp_path / "old.npz", pose, np.array([[0.0, 20.0], [9.0, 0.0]]))
    _write(tmp_path / "near.npz", pose, np.array([[0.0, 6.0], [9.0, 0.0]]))
    _write(tmp_path / "far.npz", pose, np.array([[0.0, 30.0], [9.0, 0.0]]))
    _write(tmp_path / "moved.npz", pose, np.array([[3.0, 20.0], [9.0, 0.0]]))
    assert compare(tmp_path / "old.npz", tmp_path / "near.npz") == []
    assert compare(tmp_path / "old.npz", tmp_path / "far.npz") == [
        "sparse_depth[0]: có pixel đường mới không gần hơn"
    ]
    assert (
        compare(tmp_path / "old.npz", tmp_path / "moved.npz")[-1] == "sparse_depth[0]: tập pixel"
    )
