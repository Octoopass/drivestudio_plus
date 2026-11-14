"""Kiểm tra phép khớp mô hình Mei sang OpenCV fisheye trong ``drive2sim/core/camera.py``.

Dùng tham số Mei thật của camera ``image_02`` của KITTI-360 (``calibration/image_02.yaml``). Hệ số
mong đợi lấy từ ``numpy.linalg.lstsq`` theo đúng cách của teammate (chạy trong môi trường có
numpy), để chắc bản Python thuần cho cùng kết quả.
"""

import math

import pytest

from drive2sim.core.camera import (
    MEI_FIT_MAX_ANGLE,
    fisheye_radius,
    fit_mei_to_fisheye,
    mei_fit_error_px,
)
from tests.drive2sim.samples import GAMMA_02, MEI_02, NUMPY_COEFFS_02


def test_mei_fit_matches_numpy_lstsq_and_is_accurate() -> None:
    fx, fy, coeffs = fit_mei_to_fisheye(MEI_02["xi"], MEI_02["k1"], MEI_02["k2"], *GAMMA_02)
    assert fx == pytest.approx(GAMMA_02[0] / (1 + MEI_02["xi"]))
    assert fy == pytest.approx(GAMMA_02[1] / (1 + MEI_02["xi"]))
    assert coeffs == pytest.approx(NUMPY_COEFFS_02, abs=1e-9)
    # Ảnh gốc 1400 px: sai số dưới 0,01 px tới 1,45 rad (0,0071 px đo được).
    error = mei_fit_error_px(MEI_02["xi"], MEI_02["k1"], MEI_02["k2"], GAMMA_02[0], coeffs)
    assert error < 0.01
    # Ngoài miền khớp sai số lớn dần: không được dùng tia vượt MEI_FIT_MAX_ANGLE.
    wide = mei_fit_error_px(
        MEI_02["xi"], MEI_02["k1"], MEI_02["k2"], GAMMA_02[0], coeffs, max_angle=math.pi / 2
    )
    assert wide > error


def test_fitted_radius_increases_up_to_max_angle() -> None:
    _, _, coeffs = fit_mei_to_fisheye(MEI_02["xi"], MEI_02["k1"], MEI_02["k2"], *GAMMA_02)
    radii = [fisheye_radius(MEI_FIT_MAX_ANGLE * i / 200, coeffs) for i in range(201)]
    assert all(b > a for a, b in zip(radii, radii[1:]))
    assert MEI_FIT_MAX_ANGLE < math.pi / 2


def test_mei_fit_identity_when_no_distortion() -> None:
    # xi = 0, không méo: Mei là pinhole tan θ; đa thức chỉ xấp xỉ, nhưng tiêu cự giữ nguyên.
    fx, _, _ = fit_mei_to_fisheye(0.0, 0.0, 0.0, 500.0, 500.0)
    assert fx == 500.0
