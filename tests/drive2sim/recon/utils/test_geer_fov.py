"""Kiểm tra ``recon/utils/geer_fov.py``: cận góc nhìn 3DGEER cho pinhole có điểm chính lệch tâm.

- Điểm chính ở giữa ảnh cho đúng cận cũ ``W/(2fx), H/(2fy)``.
- Điểm chính lệch tâm (KITTI-360 ``image_00``) thì cận tới mép xa điểm chính nhất.
- ``install`` thay hàm trong ``gsplat.geer._preprocess`` một lần; camera méo, fisheye vẫn đi hàm
  gốc.
"""

import sys
import types

import pytest

np = pytest.importorskip("numpy")

from drive2sim.recon.utils import geer_fov  # noqa: E402

KITTI_FX = 552.554261


def intrinsics(fx: float, fy: float, cx: float, cy: float) -> "np.ndarray":
    return np.array([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]])


def test_centred_principal_point_keeps_half_image_bound() -> None:
    tx, ty = geer_fov.pinhole_tanfov(intrinsics(500.0, 400.0, 320.0, 240.0), 640, 480)
    assert tx == pytest.approx(320.0 / 500.0) and ty == pytest.approx(240.0 / 400.0)


def test_off_centre_principal_point_reaches_farthest_edge() -> None:
    K = intrinsics(KITTI_FX, KITTI_FX, 682.049453, 238.769549)
    tx, ty = geer_fov.pinhole_tanfov(K, 1408, 376)
    assert tx == pytest.approx((1408 - 682.049453) / KITTI_FX)  # mép phải xa hơn
    assert ty == pytest.approx(238.769549 / KITTI_FX)  # mép trên xa hơn
    assert tx > 1408 / (2 * KITTI_FX) and ty > 376 / (2 * KITTI_FX)


@pytest.fixture
def fake_gsplat(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    calls = []

    def original(camera_model, Ks, width, height, fov_mod=1, **kwargs):  # noqa: ANN001, ANN003, ANN202
        calls.append(camera_model)
        return width / (2 * Ks[0, 0, 0]), height / (2 * Ks[0, 1, 1])

    gsplat = types.ModuleType("gsplat")
    geer = types.ModuleType("gsplat.geer")
    preprocess = types.ModuleType("gsplat.geer._preprocess")
    preprocess.get_camera_tanfov = original
    preprocess.calls = calls
    geer._preprocess = preprocess
    gsplat.geer = geer
    for name, module in (
        ("gsplat", gsplat),
        ("gsplat.geer", geer),
        ("gsplat.geer._preprocess", preprocess),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return preprocess


def test_install_replaces_the_pinhole_path_once(fake_gsplat: types.ModuleType) -> None:
    assert geer_fov.install() and geer_fov.install()
    patched = fake_gsplat.get_camera_tanfov
    assert getattr(patched, geer_fov.PATCHED_FLAG)
    assert patched.__wrapped__.__name__ == "original"  # không bọc hai lần
    K = intrinsics(KITTI_FX, KITTI_FX, 682.049453, 238.769549)
    assert patched("pinhole", K, 1408, 376)[1] == pytest.approx(238.769549 / KITTI_FX)
    assert fake_gsplat.calls == []


def test_distorted_and_fisheye_cameras_use_the_original(fake_gsplat: types.ModuleType) -> None:
    geer_fov.install()
    K = intrinsics(KITTI_FX, KITTI_FX, 682.049453, 238.769549)
    patched = fake_gsplat.get_camera_tanfov
    assert patched("pinhole", K, 1408, 376, radial_coeffs=[0.1])[1] == pytest.approx(
        376 / (2 * KITTI_FX)
    )
    patched("fisheye", K, 1408, 376)
    assert fake_gsplat.calls == ["pinhole", "fisheye"]


def test_install_without_geer_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gsplat.geer", None)  # import gsplat.geer → ImportError
    assert geer_fov.install() is False
