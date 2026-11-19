"""Khớp mô hình Mei sang OpenCV fisheye (``drive2sim/core/camera.py``).

Dùng tham số Mei thật của camera ``image_02`` của KITTI-360 (``calibration/image_02.yaml``).
"""

import math

import pytest

from drive2sim.core.camera import (
    FIT_SAMPLES,
    MEI_FIT_MAX_ANGLE,
    field_of_view,
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


def test_field_of_view_pinhole_uses_principal_point() -> None:
    # fx = fy = 800, ảnh 1600×900, tâm lệch: góc hai bên tính riêng.
    horizontal, vertical = field_of_view(
        "opencv_pinhole", [800, 800, 800, 450, 0, 0, 0, 0, 0], 1600, 900
    )
    assert horizontal == pytest.approx(90.0)
    assert vertical == pytest.approx(2 * math.degrees(math.atan(450 / 800)))
    _, shifted = field_of_view("opencv_pinhole", [800, 800, 800, 300], 1600, 900)
    expected = math.degrees(math.atan(300 / 800) + math.atan(600 / 800))
    assert shifted == pytest.approx(expected)


def test_field_of_view_fisheye_inverts_lens_and_clamps() -> None:
    # Không méo (k = 0): r = f·θ. Mép cách tâm 400 px với f = 500 → θ = 0,8 rad mỗi bên.
    horizontal, _ = field_of_view("opencv_fisheye", [500, 500, 400, 400, 0, 0, 0, 0], 800, 800)
    assert horizontal == pytest.approx(math.degrees(1.6), abs=1e-6)
    # Có méo: góc tìm được chiếu đúng tới mép.
    coeffs = [-0.0096, 0.0094, -0.0018, -0.0004]
    _, vertical = field_of_view(
        "opencv_fisheye", [416, 416, 700, 700, *coeffs], 1400, 1400, max_angle=1.9
    )
    half = math.radians(vertical) / 2
    assert 416 * fisheye_radius(half, coeffs) == pytest.approx(700, abs=1e-6)
    # KITTI-360 image_02: mép ảnh vượt 1,45 rad, nên chặn ở góc khớp được (166°).
    _, clamped = field_of_view("opencv_fisheye", [416, 416, 700, 700, *coeffs], 1400, 1400)
    assert clamped == pytest.approx(math.degrees(2 * MEI_FIT_MAX_ANGLE))


def test_field_of_view_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="ftheta"):
        field_of_view("ftheta", [1, 1, 1, 1], 10, 10)
