"""Chuyển một lượt quét Velodyne thành các ô đo trên ảnh cầu cho worker XSIM.

Hàm ``read_sweep_grid`` đọc XYZI do ingest KITTI-360 ghi và giữ phép đo gần nhất
trong từng ô. Worker dùng mặt nạ trả về để tính loss LiDAR trên điểm có đo.
Xem CONTRACTS mục 5.4 về quy ước của chế độ LiDAR theo khung.
"""

import math
from pathlib import Path

import numpy as np


def read_sweep_grid(
    path: Path,
    height: int,
    width: int,
    elevation_top: float,
    elevation_bottom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Chiếu các điểm XYZI trong hệ Velodyne theo quy ước SphericalCamera của xsimgs.

    Args:
        path: File gồm các bộ bốn số float32 little-endian của một sweep.
        height: Số hàng của lưới góc đều.
        width: Số cột phủ đủ 2π.
        elevation_top: Góc nâng ở biên trên, radian.
        elevation_bottom: Góc nâng ở biên dưới, radian.

    Returns:
        Mảng XYZI gốc theo ô dạng H×W×4 và mặt nạ bool H×W. Ô trống bằng 0.
        Hai return cùng ô giữ điểm gần gốc Velodyne nhất, hoà thì giữ điểm đến trước.

    Raises:
        ValueError: Kích thước, góc hoặc byte XYZI không hợp lệ.
    """
    if height <= 0 or width <= 0:
        raise ValueError("kích thước lưới LiDAR phải dương")
    if not -math.pi / 2 <= elevation_bottom < elevation_top <= math.pi / 2:
        raise ValueError("giới hạn góc nâng LiDAR không hợp lệ")
    if path.stat().st_size % 16:
        raise ValueError(f"file XYZI bị thiếu byte: {path}")
    points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    points = points[np.isfinite(points).all(axis=1)]
    xyz = points[:, :3].astype(np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    elevation = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    keep = (radius > 0) & (elevation >= elevation_bottom) & (elevation <= elevation_top)
    # xsimgs lấy +y làm phương vị 0, góc dương quay về +x.
    phi = np.mod(np.arctan2(xyz[keep, 0], xyz[keep, 1]), 2 * math.pi)
    cols = (phi * (width / (2 * math.pi))).astype(np.int64) % width
    rows = (elevation_top - elevation[keep]) * height / (elevation_top - elevation_bottom)
    cells = np.minimum(rows.astype(np.int64), height - 1) * width + cols
    order = np.argsort(radius[keep], kind="stable")
    unique, first = np.unique(cells[order], return_index=True)
    grid = np.zeros((height * width, 4), dtype=np.float32)
    valid = np.zeros(height * width, dtype=bool)
    grid[unique] = points[keep][order[first]]
    valid[unique] = True
    return grid.reshape(height, width, 4), valid.reshape(height, width)
