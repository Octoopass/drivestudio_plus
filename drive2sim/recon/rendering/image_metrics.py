"""Đo PSNR, SSIM và LPIPS của từng ảnh trên vùng pixel hợp lệ.

Renderer DriveStudio và worker XSIM dùng cùng compute_image_metrics khi đánh giá
các cặp render và ground truth. Xem SPEC mục 9 và CONTRACTS mục 6.3.
"""

from typing import Callable, Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.nn import functional as F


def get_numpy(x: Tensor) -> np.ndarray:
    return x.squeeze().cpu().numpy()

def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def compute_image_metrics(
    rgb: Tensor,
    gt: Tensor,
    valid: Optional[Tensor],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
) -> Optional[tuple]:
    """Đo PSNR, SSIM và LPIPS của một ảnh trên các pixel hợp lệ.

    DriveStudio, XSIM và NuRec gọi chung hàm này khi đánh giá. PSNR chỉ lấy pixel hợp lệ;
    SSIM lấy trung bình bản đồ trên các pixel đó. Pixel không hợp lệ được đặt về 0 ở cả hai
    ảnh trước bản đồ SSIM, nên ảnh thô và ảnh đã che cho cùng kết quả ở biên (protocol 0.6).
    LPIPS dùng hộp bao vùng hợp lệ, thay pixel không hợp lệ bằng ground truth để sai khác ở
    đó bằng 0. Các lỗ trong hộp bao vẫn làm metric LPIPS bị pha loãng một phần.

    Args:
        rgb: Ảnh render dạng (H, W, 3), giá trị trong [0, 1].
        gt: Ảnh thật cùng kích thước và miền giá trị.
        valid: Mask (H, W), True là pixel được chấm; None chấm toàn ảnh.
        lpips_fn: Hàm LPIPS nhận hai tensor (1, 3, H, W).

    Returns:
        Bộ ba (psnr, ssim, lpips) dạng float, hoặc None nếu ảnh không có pixel hợp lệ.
    """
    if valid is None:
        psnr = compute_psnr(rgb, gt)
        ssim_score = ssim(get_numpy(rgb), get_numpy(gt), data_range=1.0, channel_axis=-1)
        rgb_for_lpips = rgb
    else:
        valid = valid.bool()
        if not valid.any():
            return None
        rgb = rgb * valid[..., None]
        gt = gt * valid[..., None]
        psnr = compute_psnr(rgb[valid], gt[valid])
        ssim_map = ssim(
            get_numpy(rgb), get_numpy(gt), data_range=1.0, channel_axis=-1, full=True
        )[1]
        ssim_score = ssim_map[get_numpy(valid).astype(bool)].mean()
        rows = torch.nonzero(valid.any(dim=1)).flatten()
        cols = torch.nonzero(valid.any(dim=0)).flatten()
        box = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1))
        rgb_for_lpips = torch.where(valid[..., None], rgb, gt)[box]
        gt = gt[box]
    lpips = lpips_fn(
        rgb_for_lpips[None, ...].permute(0, 3, 1, 2), gt[None, ...].permute(0, 3, 1, 2)
    ).item()
    return float(psnr), float(ssim_score), float(lpips)
