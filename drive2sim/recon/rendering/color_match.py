"""Khớp màu affine giữa ảnh render và ảnh thật, và mask vùng nền tĩnh, cho các metric benchmark.

File này chứa (numpy thuần, không cần PyTorch):
- ``fit_color_affine``: tìm phép affine màu 3x4 bình phương tối thiểu từ ảnh render sang ảnh thật
  trên các điểm ảnh hợp lệ; trả ``None`` khi phép khớp suy biến.
- ``apply_color_affine``: áp phép affine đó lên ảnh render và kẹp về ``[0, 1]``.
- ``static_region_mask``: mask vùng nền tĩnh = hợp lệ, không phải trời, ngoài vật động.
- ``export_gap``: PSNR render gốc trừ PSNR render từ ``.ply`` trên các ảnh chung (R3).

``drive2sim/recon/rendering/videos.py`` dùng hai hàm đầu để ghi ``psnr_cc``, ``lpips_cc`` cho từng
ảnh test, và dùng ``static_region_mask`` để ghi ``psnr_static``; ``drive2sim/recon/tools/
eval_ply.py`` dùng ``export_gap`` để ghi ``ply_gap.json``. Lý do khớp màu: camera chỉ test
(R2) chưa từng train nên không có ma trận chỉnh màu (``AffineTransform``) học được; không khớp màu
thì PSNR của nó lẫn sai lệch phơi sáng giữa hai camera vào chất lượng hình học. Xem
``docs/superpowers/specs/2026-11-25-benchmark-requirements.md`` mục R2, R3 và ARCHITECTURE mục 2.17.
"""

from typing import Optional

import numpy as np

# Dưới số điểm hợp lệ này, 12 hệ số affine không được xác định đủ chắc (khớp thừa): bỏ qua ảnh.
MIN_COLOR_PIXELS = 100


def fit_color_affine(
    pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Tìm phép affine màu 3x4 đưa ảnh render về gần ảnh thật nhất.

    Giải ``gt ≈ A @ [r, g, b, 1]`` theo bình phương tối thiểu, mỗi điểm ảnh hợp lệ là một phương
    trình; ``A`` có 12 hệ số (ma trận 3x3 trộn kênh cộng với độ lệch từng kênh). Phép khớp suy
    biến (ảnh gần như một màu, hoặc quá ít điểm) cho ``A`` vô nghĩa nên trả ``None`` và người
    gọi bỏ ảnh đó khỏi trung bình sau chỉnh màu, thay vì sinh NaN hay phép biến đổi bậy.

    Args:
        pred: Ảnh render, shape ``(H, W, 3)``, giá trị trong ``[0, 1]``.
        gt: Ảnh thật, cùng shape và thang giá trị.
        valid: Mask điểm ảnh hợp lệ, shape ``(H, W)`` kiểu bool; ``None`` là mọi điểm hợp lệ.

    Returns:
        Ma trận ``(3, 4)``, hoặc ``None`` khi có dưới ``MIN_COLOR_PIXELS`` điểm hợp lệ, ma trận
        thiết kế hạng dưới 4, hoặc dữ liệu không hữu hạn.
    """
    pred_pixels = pred.reshape(-1, 3).astype(np.float64)
    gt_pixels = gt.reshape(-1, 3).astype(np.float64)
    if valid is not None:
        keep = valid.reshape(-1).astype(bool)
        pred_pixels, gt_pixels = pred_pixels[keep], gt_pixels[keep]
    if len(pred_pixels) < MIN_COLOR_PIXELS:
        return None
    if not (np.isfinite(pred_pixels).all() and np.isfinite(gt_pixels).all()):
        return None
    design = np.concatenate([pred_pixels, np.ones((len(pred_pixels), 1))], axis=1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, gt_pixels, rcond=None)
    if rank < 4:
        return None
    return coefficients.T


def apply_color_affine(pred: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Áp phép affine màu lên ảnh render rồi kẹp về ``[0, 1]``.

    Args:
        pred: Ảnh render, shape ``(H, W, 3)``.
        affine: Ma trận ``(3, 4)`` từ ``fit_color_affine``.

    Returns:
        Ảnh đã chỉnh màu, shape ``(H, W, 3)``, giá trị trong ``[0, 1]``.
    """
    matched = pred @ affine[:, :3].T + affine[:, 3]
    return np.clip(matched, 0.0, 1.0)


def static_region_mask(
    shape: tuple[int, int],
    valid: Optional[np.ndarray] = None,
    sky: Optional[np.ndarray] = None,
    dynamic: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tạo mask vùng nền tĩnh: điểm hợp lệ, không phải trời, ngoài vật động.

    Mask thiếu được coi là không loại điểm nào (cảnh không có box thì không có ``dynamic``).

    Args:
        shape: ``(H, W)`` của ảnh.
        valid: Mask điểm ảnh hợp lệ (bool), hoặc ``None``.
        sky: Mask trời (bool, ``True`` là trời), hoặc ``None``.
        dynamic: Mask vật động (bool, ``True`` là vật động), hoặc ``None``.

    Returns:
        Mask bool shape ``(H, W)``, ``True`` ở điểm thuộc nền tĩnh.
    """
    mask = np.ones(shape, dtype=bool)
    if valid is not None:
        mask &= valid.astype(bool)
    if sky is not None:
        mask &= ~sky.astype(bool)
    if dynamic is not None:
        mask &= ~dynamic.astype(bool)
    return mask


def export_gap(
    native: dict[tuple[int, int], float], ply: dict[tuple[int, int], float]
) -> Optional[dict[str, float]]:
    """Tính export gap: PSNR render gốc trừ PSNR render từ ``.ply``, trên các ảnh chung.

    Cả hai PSNR phải đo trên cùng mask nền tĩnh (``static_region_mask``) thì phép trừ mới nói về
    độ mất mát khi xuất PLY chứ không về vùng đo. Chỉ lấy ảnh có ở cả hai bên, nên một bên thiếu
    ảnh (render lỗi, mask rỗng) không làm lệch trung bình của bên kia. Gap âm là hợp lệ (PLY khớp
    ảnh thật hơn model gốc) và được giữ nguyên dấu.

    Args:
        native: ``(chỉ số khung, chỉ số camera)`` → PSNR vùng nền tĩnh của render gốc, dB.
        ply: Cùng khoá → PSNR vùng nền tĩnh của render từ ``.ply``, dB.

    Returns:
        ``{"export_gap_db", "native_psnr", "ply_psnr", "num_images"}`` với PSNR là trung bình theo
        ảnh trên các ảnh chung; ``None`` khi không có ảnh chung nào.
    """
    common = sorted(set(native) & set(ply))
    if not common:
        return None
    native_mean = sum(native[key] for key in common) / len(common)
    ply_mean = sum(ply[key] for key in common) / len(common)
    return {
        "export_gap_db": native_mean - ply_mean,
        "native_psnr": native_mean,
        "ply_psnr": ply_mean,
        "num_images": len(common),
    }
