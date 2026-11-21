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
    """drive2sim: PSNR, SSIM and LPIPS of one rendered image, restricted to valid pixels.

    When the loader marks invalid pixels (``valid_masks``: fisheye outside the usable field of
    view, ego-car body), all three metrics use only valid pixels, so a fisheye camera is measured
    on the same footing as a pinhole one: PSNR uses the valid pixels and SSIM averages its
    per-pixel map over them. LPIPS has no per-pixel map in torchmetrics, so it is measured on the
    bounding box of the valid pixels with the ground truth pasted into the invalid ones (zero
    difference there); invalid pixels inside that box (fisheye corners, ego-car body) still dilute
    it a little. Without a mask the whole image is used, as in DriveStudio.

    Before this, render_images zeroed invalid pixels in both images and measured SSIM and LPIPS on
    the full frame, so the black region counted as a perfect match and inflated both scores.

    Args:
        rgb: Rendered image, (H, W, 3) in [0, 1].
        gt: Ground-truth image, (H, W, 3) in [0, 1].
        valid: Boolean mask (H, W), or None.
        lpips_fn: LPIPS module taking two (1, 3, H, W) tensors.

    Returns:
        (psnr, ssim, lpips) as floats, or None when the mask has no valid pixel (the image is
        then left out of the averages instead of turning them into NaN).
    """
    if valid is None:
        psnr = compute_psnr(rgb, gt)
        ssim_score = ssim(get_numpy(rgb), get_numpy(gt), data_range=1.0, channel_axis=-1)
        rgb_for_lpips = rgb
    else:
        valid = valid.bool()
        if not valid.any():
            return None
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
