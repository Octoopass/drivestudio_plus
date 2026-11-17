"""Kiểm tra ``compute_image_metrics`` (``recon/rendering/videos.py``).

Metric chỉ tính trên điểm ảnh hợp lệ khi có ``valid_masks``, để camera mắt cá so được với camera
pinhole; ảnh không có điểm hợp lệ nào bị bỏ qua thay vì làm trung bình thành NaN.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from drive2sim.recon.rendering.videos import compute_image_metrics  # noqa: E402


def _mean_abs_lpips(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """LPIPS giả: trung bình sai khác tuyệt đối, đủ để thấy vùng nào được đưa vào."""
    return (prediction - target).abs().mean()


def _images() -> tuple:
    torch.manual_seed(0)
    gt = torch.rand(64, 64, 3)
    rgb = gt.clone()
    rgb[:, 32:] = torch.rand(64, 32, 3)  # nửa phải render sai
    valid = torch.zeros(64, 64, dtype=torch.bool)
    valid[:, 32:] = True  # chỉ nửa phải hợp lệ
    return rgb, gt, valid


def test_metrics_ignore_invalid_pixels() -> None:
    rgb, gt, valid = _images()
    masked = compute_image_metrics(
        rgb * valid[..., None], gt * valid[..., None], valid, _mean_abs_lpips
    )
    full = compute_image_metrics(
        rgb * valid[..., None], gt * valid[..., None], None, _mean_abs_lpips
    )
    psnr, ssim_score, lpips = masked
    # vùng đen khớp hoàn hảo không được tính: SSIM thấp hơn, LPIPS cao hơn so với đo cả ảnh
    assert ssim_score < full[1] - 0.2
    assert lpips == pytest.approx(full[2] * 2, rel=1e-5)
    assert psnr < 15


def test_metrics_without_mask_use_whole_image() -> None:
    rgb, gt, _ = _images()
    psnr, ssim_score, lpips = compute_image_metrics(rgb, gt, None, _mean_abs_lpips)
    assert lpips == pytest.approx(float((rgb - gt).abs().mean()), rel=1e-5)
    assert 0 < ssim_score < 1 and psnr > 0


def test_image_without_valid_pixel_is_skipped() -> None:
    rgb, gt, valid = _images()
    assert compute_image_metrics(rgb, gt, torch.zeros_like(valid), _mean_abs_lpips) is None
