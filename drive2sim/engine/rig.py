"""Sơ đồ camera và quỹ đạo xe của một cảnh (``reports/rig.json``), phía không cần PyTorch.

``reports/rig.json`` do ``drive2sim/recon/tools/scene_rig.py`` ghi: nó dựng loader DriveStudio của
dataset ở chế độ viewer để lấy đúng pose camera mà trainer dùng, nên cần PyTorch. File này đọc kết
quả cho Studio, và gọi script đó ở một tiến trình con bằng Python đang chạy tool (môi trường
``drive2sim``) khi cần tạo: sau ``scene ingest``, ``d2s scene rig`` và nút tạo của Studio.

File này chứa:
- ``RIG_FILE``, ``read_rig``: đọc file.
- ``RigError``, ``generate_rig``: tạo file.
- ``read_lidar_check``, ``generate_lidar_check``: ảnh chồng LiDAR và ``reports/lidar_check.json``
  (``recon/tools/lidar_overlay.py``), cùng cách chạy module GPU ở tiến trình con
  (``run_scene_tool``).
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

RIG_FILE = "rig.json"
MODULE = "drive2sim.recon.tools.scene_rig"
TIMEOUT_S = 900


class RigError(RuntimeError):
    """Không tạo được ``rig.json``; thông báo gồm phần cuối log của script."""


def rig_path(scene_dir: Path) -> Path:
    """Đường dẫn ``reports/rig.json`` của một thư mục cảnh."""
    return scene_dir / "reports" / RIG_FILE


def read_rig(scene_dir: Path) -> Optional[dict[str, Any]]:
    """Nội dung ``reports/rig.json``; ``None`` nếu chưa có hoặc hỏng."""
    try:
        data = json.loads(rig_path(scene_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
        return None
    return data


def run_scene_tool(module: str, scene_dir: Path, what: str, python: str = sys.executable) -> None:
    """Chạy một module GPU (``python -m <module> <scene_dir>``) ở tiến trình con.

    Raises:
        RigError: Nếu module lỗi hoặc quá ``TIMEOUT_S``; thông báo gồm phần cuối log.
    """
    try:
        done = subprocess.run(
            [python, "-m", module, str(scene_dir)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RigError(f"{what} quá {TIMEOUT_S} giây") from None
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-6:])
        raise RigError(f"không {what} được ({python} -m {module}):\n{tail}")


def generate_rig(scene_dir: Path, python: str = sys.executable) -> dict[str, Any]:
    """Chạy ``scene_rig`` cho một cảnh rồi đọc kết quả.

    Args:
        scene_dir: ``data/<scene_id>/``.
        python: Python của môi trường có PyTorch.

    Raises:
        RigError: Nếu script lỗi, quá ``TIMEOUT_S`` hoặc không ghi ra file đọc được.
    """
    run_scene_tool(MODULE, scene_dir, "tạo sơ đồ camera", python)
    rig = read_rig(scene_dir)
    if rig is None:
        raise RigError(f"{MODULE} chạy xong nhưng không có {rig_path(scene_dir)}")
    return rig


# Chiếu LiDAR lên ảnh (recon/tools/lidar_overlay.py): ảnh chồng và số liệu kiểm tra thô.
LIDAR_MODULE = "drive2sim.recon.tools.lidar_overlay"
LIDAR_CHECK_FILE = "lidar_check.json"
LIDAR_OVERLAY_DIR = "lidar_overlay"


def read_lidar_check(scene_dir: Path) -> Optional[dict[str, Any]]:
    """Nội dung ``reports/lidar_check.json``; ``None`` nếu chưa có hoặc hỏng."""
    try:
        data = json.loads((scene_dir / "reports" / LIDAR_CHECK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("cameras"), list) else None


def generate_lidar_check(scene_dir: Path, python: str = sys.executable) -> dict[str, Any]:
    """Chạy ``lidar_overlay`` cho một cảnh (cần ảnh trên máy) rồi đọc kết quả.

    Raises:
        RigError: Nếu script lỗi hoặc không ghi ra file đọc được.
    """
    run_scene_tool(LIDAR_MODULE, scene_dir, "chiếu LiDAR lên ảnh", python)
    data = read_lidar_check(scene_dir)
    if data is None:
        raise RigError(f"{LIDAR_MODULE} chạy xong nhưng không có reports/{LIDAR_CHECK_FILE}")
    return data
