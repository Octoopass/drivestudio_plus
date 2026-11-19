"""Khung bao cảnh (``scene_aabb``) lúc train, để viewer dựng trainer giống hệt lúc train.

Trainer nhận ``scene_aabb`` từ dataset; nó không nằm trong checkpoint mà ảnh hưởng tới ảnh vẽ ra:
``DeformableNodes`` chuẩn hoá vị trí Gaussian theo khung này trước khi đưa vào mạng biến dạng
(``models/gaussians/deformgs.py``, ``contract``). Lúc train dataset có LiDAR nên khung tính từ
LiDAR; viewer, frame server tắt LiDAR (máy có thể không có LiDAR của cảnh) nên dataset tính lại từ
quỹ đạo camera và ra khung khác (nuScenes ``scene-0061``: ``[-69.7, -10.2, 9.6, 32.8, 1.3, 104.0]``
lúc train, ``[-76.2, -41.9, -5, 40.0, 40, 20]`` lúc xem). Vì vậy train ghi khung vào
``scene_aabb.json`` trong thư mục log, viewer đọc lại; run train trước khi có file này thì đọc dòng
``Auto AABB from ...: tensor([...])`` cuối cùng trong ``log.txt`` (DriveStudio in khung mỗi lần
tính, lần cuối là lần đưa vào trainer).

Không import PyTorch: trả danh sách số, nơi gọi tự đổi sang tensor.

File này chứa ``save_scene_aabb``, ``training_aabb``.
"""

import json
import re
from pathlib import Path
from typing import Optional, Sequence

AABB_FILE = "scene_aabb.json"
LOG_FILE = "log.txt"
_LOG_PATTERN = re.compile(r"Auto AABB from [A-Za-z]+: tensor\(\[([^\]]+)\]\)")


def save_scene_aabb(log_dir: Path, aabb: Sequence[float]) -> Path:
    """Ghi khung bao cảnh lúc train vào ``<log_dir>/scene_aabb.json``.

    Args:
        log_dir: Thư mục log của lần train (chứa checkpoint).
        aabb: Sáu số ``[x_min, y_min, z_min, x_max, y_max, z_max]``.

    Returns:
        Đường dẫn file vừa ghi.
    """
    values = [float(value) for value in aabb]
    if len(values) != 6:
        raise ValueError(f"scene_aabb cần 6 số, nhận {len(values)}")
    path = Path(log_dir) / AABB_FILE
    path.write_text(json.dumps({"scene_aabb": values}), encoding="utf-8")
    return path


def training_aabb(log_dir: Path) -> Optional[list[float]]:
    """Khung bao cảnh mà trainer đã nhận lúc train.

    Args:
        log_dir: Thư mục log của lần train (thư mục chứa checkpoint).

    Returns:
        Sáu số; ``None`` nếu không có ``scene_aabb.json`` và ``log.txt`` không ghi khung.
    """
    path = Path(log_dir) / AABB_FILE
    if path.is_file():
        values = json.loads(path.read_text(encoding="utf-8"))["scene_aabb"]
        return [float(value) for value in values]
    log = Path(log_dir) / LOG_FILE
    if not log.is_file():
        return None
    found = _LOG_PATTERN.findall(log.read_text(encoding="utf-8", errors="replace"))
    if not found:
        return None
    values = [float(value) for value in found[-1].split(",")]
    return values if len(values) == 6 else None
