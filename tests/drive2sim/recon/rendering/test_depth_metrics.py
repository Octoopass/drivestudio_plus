"""Kiểm tra ``compute_depth_metrics`` và ``mean_depth_metrics`` trong ``depth_metrics.py``.

Độ sâu render chỉ được so với LiDAR ở điểm ảnh có tia LiDAR trúng, hợp lệ và có độ sâu render
dương; ảnh có quá ít điểm bị bỏ qua thay vì làm trung bình nhiễu.
"""

import pytest

np = pytest.importorskip("numpy")

from drive2sim.recon.rendering.depth_metrics import (  # noqa: E402
    compute_depth_metrics,
    mean_depth_metrics,
)


def test_perfect_depth_has_zero_error() -> None:
    lidar = np.full((20, 20), 10.0)
    result = compute_depth_metrics(lidar[..., None], lidar)
    assert result is not None
    assert result["abs_rel"] == 0.0 and result["rmse"] == 0.0
    assert result["delta_1_25"] == 1.0 and result["num_points"] == 400


def test_known_error_values() -> None:
    lidar = np.full((10, 10), 10.0)
    pred = lidar.copy()
    pred[:5] = 20.0  # nửa trên sai gấp đôi: AbsRel 1.0, tỉ lệ 2.0 > 1.25
    result = compute_depth_metrics(pred, lidar)
    assert result is not None
    assert result["abs_rel"] == pytest.approx(0.5)
    assert result["rmse"] == pytest.approx(np.sqrt(0.5 * 100.0))
    assert result["delta_1_25"] == pytest.approx(0.5)


def test_pixels_without_lidar_or_invalid_are_ignored() -> None:
    lidar = np.full((10, 10), 10.0)
    lidar[:, 5:] = 0.0  # nửa phải không có LiDAR
    pred = np.full((10, 10), 10.0)
    pred[:, :5] = 40.0  # nửa trái sai nhiều
    valid = np.ones((10, 10), dtype=bool)
    valid[:, :5] = False  # nhưng nửa trái không hợp lệ
    assert compute_depth_metrics(pred, lidar, valid) is None  # không còn điểm nào
    assert compute_depth_metrics(pred, lidar, np.ones((10, 10), dtype=bool)) is not None


def test_pixels_with_no_rendered_depth_are_ignored() -> None:
    lidar = np.full((10, 10), 10.0)
    pred = lidar.copy()
    pred[:5] = 0.0  # chưa có Gaussian nào phủ
    result = compute_depth_metrics(pred, lidar, min_points=10)
    assert result is not None and result["num_points"] == 50 and result["abs_rel"] == 0.0


def test_too_few_points_is_skipped() -> None:
    lidar = np.zeros((10, 10))
    lidar[0, :3] = 5.0
    assert compute_depth_metrics(np.full((10, 10), 5.0), lidar) is None


def test_mean_over_frames() -> None:
    frames = [
        {"abs_rel": 0.1, "rmse": 1.0, "delta_1_25": 1.0, "num_points": 100},
        {"abs_rel": 0.3, "rmse": 3.0, "delta_1_25": 0.5, "num_points": 5000},
    ]
    assert mean_depth_metrics(frames) == pytest.approx(
        {"abs_rel": 0.2, "rmse": 2.0, "delta_1_25": 0.75}
    )
    assert mean_depth_metrics([]) is None
