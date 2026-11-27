"""Đo độ sâu render so với độ sâu LiDAR trên ảnh test, cho benchmark (yêu cầu R4).

File này chỉ chứa phần tính toán thuần numpy, không nạp PyTorch, để test được ở môi trường nhẹ:
- ``compute_depth_metrics``: AbsRel, RMSE và tỉ lệ ``delta < 1.25`` của một ảnh, chỉ trên điểm ảnh
  có tia LiDAR trúng.
- ``mean_depth_metrics``: trung bình các chỉ số đó trên nhiều ảnh.

``drive2sim/recon/rendering/videos.py::render`` gọi ``compute_depth_metrics`` cho từng ảnh test, và
``drive2sim/recon/tools/eval.py`` ghi kết quả vào file metric của run; ``eval.standard`` đưa nó
vào ``summary.json``. Xem CONTRACTS mục 6.3 và báo cáo tuần 03 mục 4.2 (yêu cầu R4).
"""

from typing import Optional

import numpy as np

# Dưới ngưỡng này thì ảnh không đủ điểm LiDAR để đo; bỏ ảnh đó thay vì lấy trung bình nhiễu.
MIN_LIDAR_POINTS = 50
DELTA_THRESHOLD = 1.25


def compute_depth_metrics(
    pred_depth: np.ndarray,
    lidar_depth: np.ndarray,
    valid: Optional[np.ndarray] = None,
    min_points: int = MIN_LIDAR_POINTS,
) -> Optional[dict[str, float]]:
    """Tính AbsRel, RMSE và ``delta < 1.25`` của độ sâu render so với độ sâu LiDAR.

    Độ sâu LiDAR là bản đồ thưa (``lidar_depth_map`` của loader, đơn vị mét, 0 là không có điểm
    LiDAR); cùng đại lượng với độ sâu mà loss depth của trainer so sánh. Chỉ điểm ảnh có LiDAR,
    hợp lệ (ngoài vùng che của fisheye, thân xe) và có độ sâu render dương mới được tính.

    Args:
        pred_depth: Độ sâu render, ``(H, W)`` hoặc ``(H, W, 1)``, mét.
        lidar_depth: Độ sâu LiDAR, ``(H, W)``, mét; 0 hoặc âm là không có điểm.
        valid: Mặt nạ điểm ảnh hợp lệ ``(H, W)``; ``None`` là mọi điểm ảnh hợp lệ.
        min_points: Số điểm LiDAR tối thiểu để đo.

    Returns:
        ``{"abs_rel", "rmse", "delta_1_25", "num_points"}``; ``None`` nếu có ít hơn
        ``min_points`` điểm đo được.
    """
    pred = np.asarray(pred_depth, dtype=np.float64).reshape(np.shape(lidar_depth))
    lidar = np.asarray(lidar_depth, dtype=np.float64)
    keep = (lidar > 0) & (pred > 0)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    count = int(keep.sum())
    if count < min_points:
        return None
    pred_values, lidar_values = pred[keep], lidar[keep]
    ratio = np.maximum(pred_values / lidar_values, lidar_values / pred_values)
    return {
        "abs_rel": float(np.mean(np.abs(pred_values - lidar_values) / lidar_values)),
        "rmse": float(np.sqrt(np.mean((pred_values - lidar_values) ** 2))),
        "delta_1_25": float(np.mean(ratio < DELTA_THRESHOLD)),
        "num_points": count,
    }


def mean_depth_metrics(frames: list[dict[str, float]]) -> Optional[dict[str, float]]:
    """Trung bình ``abs_rel``, ``rmse``, ``delta_1_25`` trên các ảnh đã đo.

    Mỗi ảnh có trọng số bằng nhau, giống cách PSNR, SSIM, LPIPS được lấy trung bình.

    Args:
        frames: Kết quả của ``compute_depth_metrics`` cho từng ảnh (đã bỏ ảnh ``None``).

    Returns:
        ``{"abs_rel", "rmse", "delta_1_25"}``; ``None`` nếu không có ảnh nào.
    """
    if not frames:
        return None
    return {
        name: float(np.mean([frame[name] for frame in frames]))
        for name in ("abs_rel", "rmse", "delta_1_25")
    }
