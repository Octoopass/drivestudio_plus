"""Sơ đồ camera và quỹ đạo xe của một cảnh (``reports/rig.json``), phía không cần PyTorch.

``reports/rig.json`` do ``drive2sim/recon/tools/scene_rig.py`` ghi: nó dựng loader DriveStudio của
dataset ở chế độ viewer để lấy đúng pose camera mà trainer dùng, nên cần PyTorch. File này đọc kết
quả cho Studio, và gọi script đó ở một tiến trình con bằng Python đang chạy tool (môi trường
``drive2sim``) khi cần tạo: sau ``scene ingest``, ``d2s scene rig`` và nút tạo của Studio.

File này chứa:
- ``RIG_FILE``, ``read_rig``: đọc file.
- ``RigError``, ``generate_rig``: tạo file.
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


def generate_rig(scene_dir: Path, python: str = sys.executable) -> dict[str, Any]:
    """Chạy ``scene_rig`` cho một cảnh rồi đọc kết quả.

    Args:
        scene_dir: ``data/<scene_id>/``.
        python: Python của môi trường có PyTorch.

    Raises:
        RigError: Nếu script lỗi, quá ``TIMEOUT_S`` hoặc không ghi ra file đọc được.
    """
    try:
        done = subprocess.run(
            [python, "-m", MODULE, str(scene_dir)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RigError(f"tạo sơ đồ camera quá {TIMEOUT_S} giây") from None
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-6:])
        raise RigError(f"không tạo được sơ đồ camera ({python} -m {MODULE}):\n{tail}")
    rig = read_rig(scene_dir)
    if rig is None:
        raise RigError(f"{MODULE} chạy xong nhưng không có {rig_path(scene_dir)}")
    return rig
