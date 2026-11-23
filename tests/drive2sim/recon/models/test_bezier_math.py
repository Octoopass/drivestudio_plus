"""Kiểm tra toán Bézier thuần torch của BezierGS (``bezier_math.py``) trên CPU.

Gồm: hệ số Bernstein và đạo hàm, fit control point bằng bình phương tối thiểu, tham số hoá theo
độ dài dây cung, fit đường đi lặp và hàm map thời gian thật sang tham số Bézier.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")

from drive2sim.recon.models.nodes import bezier_math as bm  # noqa: E402


def test_bernstein_partition_of_unity() -> None:
    ts = torch.linspace(0, 1, 11)[:, None]
    assert torch.allclose(bm.bernstein(ts, 3).sum(-1), torch.ones(11), atol=1e-6)


def test_bernstein_derivative_matches_finite_difference() -> None:
    ts = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)[:, None]
    h = 1e-6
    fd = (bm.bernstein(ts + h, 3) - bm.bernstein(ts - h, 3)) / (2 * h)
    assert torch.allclose(bm.bernstein_derivative(ts, 3), fd, atol=1e-5)


def test_fit_control_points_recovers_cubic() -> None:
    cp = torch.randn(5, 4, 3, dtype=torch.float64)
    ts = torch.linspace(0, 1, 20, dtype=torch.float64)[:, None]
    pts = torch.einsum("tk,nkd->ntd", bm.bernstein(ts, 3), cp)
    assert torch.allclose(bm.fit_control_points(pts, ts, 3), cp, atol=1e-6)


def test_chord_length_params_edge_cases() -> None:
    assert bm.chord_length_params(torch.zeros(1, 3)).tolist() == [0.0]
    assert torch.allclose(bm.chord_length_params(torch.zeros(4, 3)), torch.linspace(0, 1, 4))
    line = torch.tensor([[0.0, 0, 0], [1, 0, 0], [3, 0, 0]])
    assert torch.allclose(bm.chord_length_params(line), torch.tensor([0, 1 / 3, 1.0]), atol=1e-6)


def test_fit_trajectory_reproduces_curved_path() -> None:
    t = torch.linspace(0, 1, 30)
    traj = torch.stack([10 * t, 3 * t**2, torch.zeros_like(t)], -1)
    cp, bez_t = bm.fit_trajectory(traj)
    recon = bm.bernstein(bez_t[:, None], 3) @ cp
    assert (recon - traj).norm(dim=-1).max() < 1e-2
    assert torch.all(bez_t[1:] >= bez_t[:-1])


def test_fit_time_map_maps_times_to_bezier_t() -> None:
    times = torch.linspace(0.2, 0.6, 15)
    bez_t = torch.linspace(0, 1, 15)
    tcp = bm.fit_time_map(times, bez_t)
    assert torch.allclose((bm.bernstein(times[:, None], 3) @ tcp)[:, 0], bez_t, atol=1e-4)
