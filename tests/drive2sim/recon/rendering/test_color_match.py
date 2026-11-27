"""Kiểm tra ``drive2sim/recon/rendering/color_match.py``: khớp màu affine và mask nền tĩnh.

Cần numpy nên bị bỏ qua ở ``.venv`` nhẹ; chạy bằng ``make test-gpu`` hoặc môi trường có numpy.
"""

import pytest

np = pytest.importorskip("numpy")

from drive2sim.recon.rendering.color_match import (  # noqa: E402
    apply_color_affine,
    export_gap,
    fit_color_affine,
    static_region_mask,
)


def test_affine_recovers_gain_and_offset() -> None:
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.1, 0.9, (32, 32, 3))
    pred = (gt - 0.1) / 0.8
    matched = apply_color_affine(pred, fit_color_affine(pred, gt))
    assert np.abs(matched - gt).max() < 1e-6


def test_affine_ignores_invalid_pixels() -> None:
    # nửa ảnh không hợp lệ bị đổi màu bậy; phép khớp chỉ dùng nửa hợp lệ nên vẫn khôi phục đúng
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.1, 0.9, (32, 32, 3))
    pred = (gt - 0.1) / 0.8
    valid = np.zeros((32, 32), bool)
    valid[:, :16] = True
    corrupted = gt.copy()
    corrupted[:, 16:] = rng.uniform(0.0, 1.0, (32, 16, 3))
    affine = fit_color_affine(pred, corrupted, valid)
    matched = apply_color_affine(pred, affine)
    assert np.abs(matched[:, :16] - gt[:, :16]).max() < 1e-6


@pytest.mark.parametrize("case", ["constant", "too_few"])
def test_degenerate_fit_returns_none(case: str) -> None:
    pred = np.full((32, 32, 3), 0.5)
    valid = None
    if case == "too_few":
        pred = np.random.default_rng(1).uniform(size=(32, 32, 3))
        valid = np.zeros((32, 32), bool)
        valid[:5, :5] = True  # 25 < 100 điểm
    assert fit_color_affine(pred, pred, valid) is None


def test_affine_output_is_clipped_to_unit_range() -> None:
    pred = np.full((4, 4, 3), 0.9)
    affine = np.array([[3.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 3.0, -1.0]])
    matched = apply_color_affine(pred, affine)
    assert matched.max() == 1.0 and matched.min() >= 0.0


def test_static_region_mask_combines_masks() -> None:
    valid = np.ones((2, 2), bool)
    valid[0, 0] = False
    sky = np.zeros((2, 2), bool)
    sky[0, 1] = True
    dyn = np.zeros((2, 2), bool)
    dyn[1, 0] = True
    assert static_region_mask((2, 2), valid, sky, dyn).tolist() == [[False, False], [False, True]]
    assert static_region_mask((2, 2)).all()


def test_export_gap_uses_common_images_only() -> None:
    native = {(8, 0): 26.0, (16, 0): 24.0, (24, 0): 30.0}
    ply = {(8, 0): 25.0, (16, 0): 23.0}
    result = export_gap(native, ply)
    assert result["num_images"] == 2 and result["export_gap_db"] == pytest.approx(1.0)
    assert result["native_psnr"] == pytest.approx(25.0) and result["ply_psnr"] == pytest.approx(
        24.0
    )


def test_export_gap_without_common_images_is_none() -> None:
    assert export_gap({(8, 0): 26.0}, {(16, 0): 25.0}) is None


def test_export_gap_can_be_negative() -> None:
    # PLY tốt hơn render gốc (ví dụ render gốc có lỗi ở nền không bị mask): giữ dấu, không cắt về 0
    assert export_gap({(8, 0): 24.0}, {(8, 0): 25.0})["export_gap_db"] == pytest.approx(-1.0)
