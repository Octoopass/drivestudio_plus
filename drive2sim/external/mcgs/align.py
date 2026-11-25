"""Căn quỹ đạo SLAM về hệ toạ độ của cảnh và đo sai số quỹ đạo (ATE).

SLAM ước lượng pose trong hệ riêng (gốc là camera đầu tiên, tỉ lệ do nó tự chọn), nên trước khi
so với pose thật hay render ảnh test phải tìm phép đồng dạng (xoay, dịch, tỉ lệ) đưa quỹ đạo của
nó về hệ của cảnh.

File này chứa:
- ``umeyama``: phép đồng dạng tốt nhất giữa hai tập điểm (Umeyama, 1991).
- ``read_tum``: đọc file quỹ đạo định dạng TUM của MCGS-SLAM.
- ``match_by_time``: ghép pose của SLAM với pose thật theo timestamp.
- ``ate_rmse``: sai số quỹ đạo tuyệt đối sau khi căn.

``drive2sim/stages/eval/mcgs.py`` dùng file này (căn pose khung test để render), và
``d2s benchmark`` ghi ATE vào ghi chú kết quả của track B. Xem ARCHITECTURE mục 2.19.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> tuple:
    """Phép đồng dạng ``target ≈ scale · R · source + t`` tốt nhất theo bình phương tối thiểu.

    Args:
        source: Điểm gốc ``(N, 3)``.
        target: Điểm đích ``(N, 3)`` tương ứng từng dòng, ``N >= 3`` và không thẳng hàng.
        with_scale: Có ước lượng tỉ lệ không; ``False`` là phép cứng (``scale = 1``).

    Returns:
        ``(scale, R, t)``: ``R`` là ma trận xoay ``(3, 3)`` định thức +1, ``t`` dạng ``(3,)``.

    Raises:
        ValueError: Nếu hai tập khác cỡ, ít hơn 3 điểm, hoặc điểm gốc suy biến.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source và target phải cùng dạng (N, 3)")
    count = source.shape[0]
    if count < 3:
        raise ValueError("cần ít nhất 3 điểm để căn quỹ đạo")
    mean_source, mean_target = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - mean_source, target - mean_target
    variance = (centered_source**2).sum() / count
    if variance < 1e-12:
        raise ValueError("các điểm gốc trùng nhau, không căn được")
    covariance = centered_target.T @ centered_source / count
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0  # tránh phép phản xạ: định thức của R phải là +1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(singular) @ sign) / variance) if with_scale else 1.0
    translation = mean_target - scale * rotation @ mean_source
    return scale, rotation, translation


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Đọc quỹ đạo TUM: mỗi dòng ``timestamp tx ty tz qx qy qz qw``.

    Args:
        path: File quỹ đạo, ví dụ ``traj_mcgs.txt``.

    Returns:
        ``(timestamp (N,), pose (N, 7))`` theo thứ tự trong file.

    Raises:
        ValueError: Nếu một dòng không đúng 8 số.
    """
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 8:
            raise ValueError(f"{path}:{number}: cần 8 số, có {len(values)}")
        rows.append([float(v) for v in values])
    data = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    return data[:, 0], data[:, 1:]


def match_by_time(
    stamps: np.ndarray, reference_stamps: np.ndarray, tolerance: float = 1e-3
) -> list[tuple[int, int]]:
    """Ghép mỗi timestamp của SLAM với timestamp gần nhất của pose thật trong ``tolerance``.

    Args:
        stamps: Timestamp của pose SLAM ``(N,)``.
        reference_stamps: Timestamp của pose thật ``(M,)``.
        tolerance: Chênh lệch tối đa, cùng đơn vị với timestamp.

    Returns:
        Các cặp ``(chỉ số trong stamps, chỉ số trong reference_stamps)``; timestamp không có cặp
        bị bỏ.
    """
    reference = np.asarray(reference_stamps, dtype=np.float64)
    pairs = []
    for index, stamp in enumerate(np.asarray(stamps, dtype=np.float64)):
        nearest = int(np.abs(reference - stamp).argmin())
        if abs(reference[nearest] - stamp) <= tolerance:
            pairs.append((index, nearest))
    return pairs


def ate_rmse(
    estimated: np.ndarray, truth: np.ndarray, with_scale: bool = True
) -> tuple[float, tuple]:
    """ATE RMSE (mét) của vị trí camera sau khi căn đồng dạng ``estimated`` vào ``truth``.

    Args:
        estimated: Vị trí camera của SLAM ``(N, 3)``.
        truth: Vị trí thật ``(N, 3)`` tương ứng từng dòng.
        with_scale: Căn cả tỉ lệ (SLAM đơn mắt); ``False`` với stereo, đa camera đã có tỉ lệ mét.

    Returns:
        ``(rmse, (scale, R, t))``.
    """
    transform = umeyama(estimated, truth, with_scale)
    scale, rotation, translation = transform
    aligned = scale * (rotation @ np.asarray(estimated, dtype=np.float64).T).T + translation
    error = np.linalg.norm(aligned - np.asarray(truth, dtype=np.float64), axis=1)
    return float(np.sqrt((error**2).mean())), transform


def pose_to_matrix(pose: np.ndarray, quaternion_last: bool = True) -> np.ndarray:
    """Đổi một pose ``(tx, ty, tz, qx, qy, qz, qw)`` thành ma trận 4×4.

    Args:
        pose: 7 số; ``quaternion_last=False`` là ``(qw, qx, qy, qz)`` sau phần dịch.
        quaternion_last: Thứ tự quaternion; TUM và lietorch để ``qw`` cuối.

    Returns:
        Ma trận ``(4, 4)`` dạng camera → world nếu ``pose`` là pose camera trong world.
    """
    pose = np.asarray(pose, dtype=np.float64)
    x, y, z, w = pose[3:7] if quaternion_last else (*pose[4:7], pose[3])
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = pose[:3]
    return matrix


def positions(poses: np.ndarray, indices: Optional[list[int]] = None) -> np.ndarray:
    """Vị trí (3 số đầu) của các pose ``(N, 7)``; ``indices`` chọn một phần."""
    array = np.asarray(poses, dtype=np.float64)
    return array[indices, :3] if indices is not None else array[:, :3]
