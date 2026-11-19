"""Khớp mô hình Mei sang OpenCV fisheye (``drive2sim/core/camera.py``).

Dùng tham số Mei thật của camera ``image_02`` của KITTI-360 (``calibration/image_02.yaml``).
"""

import math

import pytest

from drive2sim.core.camera import (
    FIT_SAMPLES,
    MEI_FIT_MAX_ANGLE,
    fisheye_radius,
    fit_mei_to_fisheye,
    mei_fit_error_px,
    mei_radius,
)

MEI_02 = {"xi": 2.2134047507854890, "k1": 0.016798235660113681, "k2": 1.6548773243373522}
GAMMA_02 = (1336.3220825849971, 1335.7883350012958)


def fit() -> tuple[float, float, list[float]]:
    return fit_mei_to_fisheye(MEI_02["xi"], MEI_02["k1"], MEI_02["k2"], *GAMMA_02)


def test_fit_is_sub_pixel_accurate_on_kitti360() -> None:
    fx, fy, coeffs = fit()
    error = mei_fit_error_px(MEI_02["xi"], MEI_02["k1"], MEI_02["k2"], GAMMA_02[0], fx, coeffs)
    assert error < 0.01  # pixel, tới MEI_FIT_MAX_ANGLE
    # gần trục quang Mei cho bán kính ≈ θ/(1+ξ), nên tiêu cự fisheye ≈ γ/(1+ξ)
    assert fx == pytest.approx(GAMMA_02[0] / (1 + MEI_02["xi"]), rel=1e-3)
    assert fy / fx == pytest.approx(GAMMA_02[1] / GAMMA_02[0])
    # ngoài miền khớp sai số lớn dần: không được dùng tia vượt MEI_FIT_MAX_ANGLE
    wide = mei_fit_error_px(
        MEI_02["xi"], MEI_02["k1"], MEI_02["k2"], GAMMA_02[0], fx, coeffs, max_angle=1.55
    )
    assert wide > 10 * error


def test_fit_matches_numpy_least_squares() -> None:
    """Giải bằng khử Gauss thuần Python cho cùng nghiệm với numpy trên cùng hệ phương trình."""
    np = pytest.importorskip("numpy")
    thetas = MEI_FIT_MAX_ANGLE * np.arange(1, FIT_SAMPLES + 1) / FIT_SAMPLES
    design = np.stack([thetas**p for p in (1, 3, 5, 7, 9)], axis=1)
    target = np.array([GAMMA_02[0] * mei_radius(t, **MEI_02) for t in thetas])
    a, *_ = np.linalg.lstsq(design, target, rcond=None)
    fx, _, coeffs = fit()
    assert fx == pytest.approx(a[0], rel=1e-9)
    assert coeffs == pytest.approx(list(a[1:] / a[0]), rel=1e-6, abs=1e-12)


def test_fitted_radius_increases_up_to_max_angle() -> None:
    _, _, coeffs = fit()
    radii = [fisheye_radius(MEI_FIT_MAX_ANGLE * i / 200, coeffs) for i in range(201)]
    assert all(b > a for a, b in zip(radii, radii[1:]))
    assert MEI_FIT_MAX_ANGLE < math.pi / 2


def test_large_xi_recovers_sine_series() -> None:
    """ξ lớn, không méo: Mei cho r ≈ sin θ/(1+ξ), nên k1 ≈ −1/6, k2 ≈ 1/120 (chuỗi của sin θ)."""
    xi = 1e6
    fx, fy, coeffs = fit_mei_to_fisheye(xi, 0.0, 0.0, 1.0 + xi, 1.0 + xi)
    assert fx == pytest.approx(1.0, rel=1e-4) and fy == pytest.approx(fx)
    assert coeffs[0] == pytest.approx(-1 / 6, rel=1e-3)
    assert coeffs[1] == pytest.approx(1 / 120, rel=1e-2)
