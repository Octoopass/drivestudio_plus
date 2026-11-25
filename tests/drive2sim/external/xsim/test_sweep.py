"""Kiểm tra lưới cầu LiDAR: dải góc nâng theo dữ liệu, ô trùng, seam và ô trống."""

import math

import pytest

np = pytest.importorskip("numpy")

from drive2sim.external.xsim.sweep import grid_for, sweep_grid  # noqa: E402


def test_grid_covers_measured_elevations_not_whole_sphere() -> None:
    spec = grid_for([(math.radians(-24.9), math.radians(2.0))], math.radians(0.2), 2048)
    assert spec.rows == math.ceil(math.radians(27.1) / math.radians(0.2))
    assert spec.bottom == pytest.approx(math.radians(-25.0))
    assert spec.top >= math.radians(2.0)
    assert spec.limits[3] == pytest.approx(spec.bottom - spec.top)


def test_nearest_return_seam_and_empty_cells_in_sensor_frame() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [10, 20, 0]
    local = np.array(
        [
            [0, 3, 0],  # phương vị 0
            [0, 2, 0],  # cùng ô, gần hơn: được giữ
            [-0.01, 2, 0],  # ngay trước seam 2π: cột cuối
        ]
    )
    points = local + pose[:3, 3]
    origins = np.repeat(pose[None, :3, 3], len(points), axis=0)
    spec = grid_for([(-0.1, 0.1)], 0.05, 64)
    xyz, origin, valid, kept = sweep_grid(points, origins, pose, spec)
    row = int((spec.top - 0.0) * spec.rows / (spec.top - spec.bottom))
    assert kept == 2 and valid.sum() == 2
    assert np.allclose(xyz[row, 0], [10, 22, 0])
    assert valid[row, spec.cols - 1]
    # Ô trống có gốc và điểm bằng 0, nên XSIM không coi là có đo (ray_dirs = 0).
    assert not origin[~valid].any() and not xyz[~valid].any()
