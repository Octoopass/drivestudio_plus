"""Kiểm tra trục lên ghi vào PLY (``drive2sim/recon/tools/export_ply.py``).

Hệ world của DriveStudio lấy theo camera trước ở khung đầu: nuScenes (extrinsic OpenCV) là -y lên,
Waymo là z lên. Trục lên phải suy từ pose camera, không ghi cứng.
"""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
export_ply = pytest.importorskip("drive2sim.recon.tools.export_ply")


def write_cameras(path: Path, rotations: list) -> Path:
    frames = []
    for rotation in rotations:
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        frames.append(c2w.flatten().tolist())
    path.write_text(json.dumps({"convention": "opencv", "cameras": [{"c2w": frames}]}))
    return path


def test_up_axis_follows_camera_poses(tmp_path: Path) -> None:
    # nuScenes: world = camera trước khung đầu (OpenCV, y xuống)
    assert export_ply.up_axis_from_cameras(write_cameras(tmp_path / "a.json", [np.eye(3)])) == "-y"
    # Waymo: camera nhìn theo +x, y của camera hướng -z của world
    waymo = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    assert export_ply.up_axis_from_cameras(write_cameras(tmp_path / "b.json", [waymo])) == "+z"
    assert export_ply.up_axis_from_cameras(tmp_path / "none.json") is None
    assert "up=-y" in export_ply.ply_comment("-y")
    assert "up=" not in export_ply.ply_comment(None)
