"""Chuyển một lượt quét LiDAR của gói quan sát thành ảnh cầu cho ``SphericalCamera`` của xsimgs.

Lưới phủ đúng dải góc nâng mà cảm biến đo được (đo trên các sweep của cảnh), không phải ±90°, nên
mỗi hàng ứng với khoảng một tia của LiDAR. Ô có nhiều điểm thì giữ điểm gần cảm biến nhất; worker
ghi tỉ lệ điểm giữ được vào log. Loss LiDAR của XSIM lấy hướng tia từ chính điểm đo
(``rays_from_sweep``), nên lượng tử hoá góc của lưới không làm lệch tia. Xem CONTRACTS mục 5.4.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# Bước góc nâng mặc định: HDL-64E của KITTI-360 có tia cách nhau khoảng 0,4°, nuScenes (32 tia)
# khoảng 1,3°; 0,2° giữ hầu hết tia trên một hàng riêng.
DEFAULT_ELEVATION_STEP = math.radians(0.2)
DEFAULT_AZIMUTH_BINS = 2048


@dataclass(frozen=True)
class GridSpec:
    """Lưới góc của một LiDAR.

    Attributes:
        top: Góc nâng ở mép trên của hàng đầu (rad).
        bottom: Góc nâng ở mép dưới của hàng cuối (rad).
        rows: Số hàng.
        cols: Số cột phủ đủ 2π.
    """

    top: float
    bottom: float
    rows: int
    cols: int

    @property
    def limits(self) -> tuple[float, float, float, float]:
        """``[phi_min, phi_range, theta_min, theta_range]`` theo quy ước SphericalCamera."""
        return (0.0, 2 * math.pi, self.top, self.bottom - self.top)


def local_points(points: np.ndarray, sensor_to_world: np.ndarray) -> np.ndarray:
    """Điểm world (N×3) → hệ cảm biến."""
    rotation, translation = sensor_to_world[:3, :3], sensor_to_world[:3, 3]
    return (points.astype(np.float64) - translation) @ rotation


def elevations(local: np.ndarray) -> np.ndarray:
    """Góc nâng (rad) của điểm trong hệ cảm biến."""
    return np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))


def grid_for(
    elevation_ranges: Iterable[tuple[float, float]],
    step: float = DEFAULT_ELEVATION_STEP,
    cols: int = DEFAULT_AZIMUTH_BINS,
) -> GridSpec:
    """Lưới phủ dải góc nâng của mọi sweep, nới nửa bước ở hai mép.

    Args:
        elevation_ranges: ``(thấp nhất, cao nhất)`` của từng sweep.
        step: Bước góc nâng (rad).
        cols: Số cột.

    Raises:
        ValueError: Nếu không có sweep hoặc tham số không dương.
    """
    ranges = list(elevation_ranges)
    if not ranges or step <= 0 or cols <= 0:
        raise ValueError("cần ít nhất một sweep và bước, số cột dương")
    low = max(min(r[0] for r in ranges) - step / 2, -math.pi / 2)
    high = min(max(r[1] for r in ranges) + step / 2, math.pi / 2)
    rows = max(1, math.ceil((high - low) / step))
    # Giữ đúng bước ở mép dưới; mép trên chạm π/2 thì bước nhỏ đi một chút.
    return GridSpec(top=min(low + rows * step, math.pi / 2), bottom=low, rows=rows, cols=cols)


def sweep_grid(
    points: np.ndarray, origins: np.ndarray, sensor_to_world: np.ndarray, spec: GridSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Đặt các điểm của một sweep vào lưới.

    Args:
        points: Điểm world (N×3).
        origins: Gốc tia world của từng điểm (N×3).
        sensor_to_world: Pose cảm biến (4×4).
        spec: Lưới của LiDAR.

    Returns:
        ``(xyz, origin, valid, kept)``: điểm và gốc tia world theo ô (H×W×3, ô trống bằng 0), mặt
        nạ ô có đo (H×W) và số điểm được giữ. Hai điểm cùng ô giữ điểm gần cảm biến nhất, hoà thì
        giữ điểm đến trước.
    """
    local = local_points(points, sensor_to_world)
    radius = np.linalg.norm(local, axis=1)
    elevation = elevations(local)
    keep = (radius > 0) & (elevation >= spec.bottom) & (elevation <= spec.top)
    # xsimgs: phương vị 0 là +y, tăng về phía +x (cartesian_to_spherical).
    phi = np.mod(np.arctan2(local[keep, 0], local[keep, 1]), 2 * math.pi)
    cols = (phi * (spec.cols / (2 * math.pi))).astype(np.int64) % spec.cols
    rows = (spec.top - elevation[keep]) * spec.rows / (spec.top - spec.bottom)
    cells = np.minimum(rows.astype(np.int64), spec.rows - 1) * spec.cols + cols
    order = np.argsort(radius[keep], kind="stable")
    unique, first = np.unique(cells[order], return_index=True)
    size = spec.rows * spec.cols
    xyz = np.zeros((size, 3), dtype=np.float32)
    origin = np.zeros((size, 3), dtype=np.float32)
    valid = np.zeros(size, dtype=bool)
    chosen = np.flatnonzero(keep)[order[first]]
    xyz[unique] = points[chosen]
    origin[unique] = origins[chosen]
    valid[unique] = True
    shape = (spec.rows, spec.cols)
    return xyz.reshape(*shape, 3), origin.reshape(*shape, 3), valid.reshape(shape), len(unique)
