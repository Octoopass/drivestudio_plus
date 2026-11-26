"""Kiểm tra metric từng ảnh của vòng render: ``color_matched_metrics`` và ``static_psnr``.

Hai hàm này cho ``psnr_cc``, ``lpips_cc`` (sau chỉnh màu affine) và ``psnr_static`` (vùng nền
tĩnh) trong mỗi bản ghi của ``frames.jsonl``. Cần PyTorch và scikit-image nên bị bỏ qua ở
``.venv`` nhẹ; chạy bằng ``make test-gpu``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from drive2sim.recon.rendering.videos import color_matched_metrics, static_psnr  # noqa: E402


def _zero_lpips(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """LPIPS giả luôn bằng 0, để test chỉ nhìn PSNR."""
    return torch.tensor(0.0)


def test_color_matched_metrics_undo_gain() -> None:
    torch.manual_seed(0)
    gt = torch.rand(16, 16, 3) * 0.8 + 0.1
    rgb = (gt - 0.1) / 0.8
    psnr_cc, lpips_cc = color_matched_metrics(rgb, gt, None, _zero_lpips)
    assert psnr_cc > 60 and lpips_cc == 0.0


def test_color_matched_metrics_degenerate_is_none() -> None:
    flat = torch.full((16, 16, 3), 0.5)
    assert color_matched_metrics(flat, flat, None, _zero_lpips) == (None, None)


def test_color_matched_metrics_use_valid_pixels_only() -> None:
    torch.manual_seed(0)
    gt = torch.rand(16, 16, 3) * 0.8 + 0.1
    rgb = (gt - 0.1) / 0.8
    valid = torch.zeros(16, 16, dtype=torch.bool)
    valid[:, :8] = True
    rgb[:, 8:] = torch.rand(16, 8, 3)  # nửa không hợp lệ render sai hoàn toàn
    psnr_cc, _ = color_matched_metrics(rgb, gt, valid, _zero_lpips)
    assert psnr_cc > 60


def test_static_psnr_excludes_sky_dynamic_and_invalid() -> None:
    gt = torch.full((4, 4, 3), 0.5)
    rgb = gt.clone()
    rgb[0] = 0.0  # hàng 0 là trời, render sai
    rgb[1] = 0.0  # hàng 1 là vật động, render sai
    sky = torch.zeros(4, 4, dtype=torch.bool)
    sky[0] = True
    dynamic = torch.zeros(4, 4, dtype=torch.bool)
    dynamic[1] = True
    assert static_psnr(rgb, gt, None, sky, dynamic) == float("inf")
    assert static_psnr(rgb, gt, None, None, None) < 10  # không loại gì thì thấy lỗi


def test_static_psnr_is_none_when_nothing_is_left() -> None:
    gt = torch.zeros(2, 2, 3)
    assert static_psnr(gt, gt, None, torch.ones(2, 2, dtype=torch.bool), None) is None
