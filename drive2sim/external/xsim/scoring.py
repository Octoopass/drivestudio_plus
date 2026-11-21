"""Chấm ảnh RGB của XSIM bằng protocol dùng chung với DriveStudio.

score_pairs kiểm tra danh sách quan sát, áp mask và lượng tử hoá ảnh render
trước khi lấy trung bình theo ảnh và camera. Xem SPEC mục 9.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from statistics import mean

import torch

from drive2sim.core.schema import Metrics
from drive2sim.recon.rendering.image_metrics import compute_image_metrics


def score_pairs(
    pairs: Iterable[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]],
    expected: set[tuple[int, str]],
    camera_names: dict[int, str],
    lpips: Callable,
) -> tuple[Metrics, dict[str, Metrics]]:
    """Đo metric trên đúng các cặp test và báo lỗi khi renderer bỏ hoặc lặp ảnh.

    Args:
        pairs: Iterator trả frame, sensor, RGB render, ground truth và mask của từng ảnh.
        expected: Các cặp frame ingest và tên camera trong split test.
        camera_names: Chỉ số sensor sang tên camera của manifest.
        lpips: Module LPIPS nhận hai tensor BCHW trong miền [0, 1].

    Returns:
        Metric chung theo ảnh và metric riêng cho mỗi camera có pixel hợp lệ.
    """
    seen = set()
    values = defaultdict(list)
    for frame, sensor, rgb, gt, valid in pairs:
        name = camera_names[sensor]
        key = (frame, name)
        if key in seen or key not in expected:
            raise ValueError(f"Cặp ảnh XSIM ngoài split hoặc bị lặp: {key}")
        seen.add(key)
        mask = valid.bool()[..., None]
        result = compute_image_metrics((rgb * 255).round() / 255 * mask, gt * mask, valid, lpips)
        if result is not None:
            values[name].append(result)
    if seen != expected or not values:
        raise ValueError("XSIM thiếu cặp ảnh test hoặc toàn bộ ảnh không có pixel hợp lệ")

    def average(rows: list[tuple]) -> Metrics:
        """Lấy trung bình từng metric với trọng số bằng nhau cho mỗi ảnh."""
        psnr, ssim, perceptual = map(mean, zip(*rows))
        return Metrics(psnr=psnr, ssim=ssim, lpips=perceptual)

    return average([row for rows in values.values() for row in rows]), {
        name: average(rows) for name, rows in values.items()
    }
