"""Cấu hình Faster-GS không cần PyTorch: giá trị mặc định và danh sách kỹ thuật đang bật.

Tách khỏi ``fastergs.py`` để test được trên máy không có GPU (``docs/methods/fastergs.md``).

File này chứa ``FASTERGS_DEFAULTS``, ``TECHNIQUES``, ``KERNEL_TECHNIQUES``, ``parse_config``,
``active_techniques``, ``kernel_techniques``, ``morton_due``.
"""

from types import SimpleNamespace
from typing import Any, Mapping, Optional

# Kỹ thuật phần Python, theo thứ tự ghi vào log và train_stats.json.
TECHNIQUES = ("fused_adam", "fused_ssim", "fewer_syncs", "morton_order")
# Kỹ thuật trong kernel gsplat (bản fork third_party/gsplat-geer). Chỉ bật khi bản gsplat đang cài
# tự khai là hỗ trợ cho render mode đang dùng (gsplat.fastergs_support()).
KERNEL_TECHNIQUES = ("exact_tiles", "two_stage_sort", "per_gaussian_backward")

# Giá trị mặc định của khối trainer.fastergs. `enabled` là công tắc chung; mỗi kỹ thuật chỉ chạy
# khi công tắc chung bật và cờ riêng của nó bật.
FASTERGS_DEFAULTS = dict(
    enabled=False,
    fused_adam=True,  # torch.optim.Adam(fused=True)
    fused_ssim=True,  # gói fused-ssim thay pytorch_msssim trong loss (cùng định nghĩa, padding valid)
    fewer_syncs=True,  # gộp các lần chờ GPU khi kiểm NaN/Inf và khi ghi log mỗi bước
    # Thử nghiệm, mặc định tắt: sắp Gaussian nền theo đường cong Z (Morton) để truy cập bộ nhớ liền
    # mạch hơn. Kết quả chỉ tương đương về thống kê (đổi thứ tự khi sort trùng độ sâu và số ngẫu
    # nhiên khi split), không trùng tới mức làm tròn như ba kỹ thuật trên.
    morton_order=False,
    morton_interval=5000,  # sắp lại mỗi bấy nhiêu bước, sau khi model refine
    # Kernel: chỉ ghép Gaussian với tile có pixel thật sự nhận màu (ảnh, gradient không đổi).
    exact_tiles=True,
    # Kernel: sort theo độ sâu một lần rồi theo tile (thứ tự trong tile, ảnh không đổi).
    two_stage_sort=True,
    # Kernel: backward chạy theo từng Gaussian thay vì từng pixel (ảnh không đổi, gradient lệch
    # trong mức dao động của bản cũ). Tốn thêm bộ nhớ GPU theo số cặp Gaussian–tile (~300 MB với
    # 300k Gaussian ở 1600×900).
    per_gaussian_backward=True,
)


def parse_config(cfg: Optional[Mapping[str, Any]]) -> SimpleNamespace:
    """Gộp khối ``trainer.fastergs`` với giá trị mặc định.

    Raises:
        AssertionError: Nếu có khoá lạ.
    """
    values = dict(cfg) if cfg is not None else {}
    unknown = set(values) - set(FASTERGS_DEFAULTS)
    assert not unknown, f"unknown trainer.fastergs keys: {sorted(unknown)}"
    return SimpleNamespace(**{**FASTERGS_DEFAULTS, **values})


def active_techniques(cfg: SimpleNamespace) -> list[str]:
    """Các kỹ thuật phần Python được bật, theo thứ tự ``TECHNIQUES``; rỗng khi công tắc chung tắt."""
    if not cfg.enabled:
        return []
    return [name for name in TECHNIQUES if getattr(cfg, name)]


def kernel_techniques(
    cfg: SimpleNamespace, support: Optional[Mapping[str, Any]], render_mode: str
) -> tuple[list[str], list[str]]:
    """Kỹ thuật kernel được bật và kỹ thuật được yêu cầu nhưng bản gsplat không có cho mode này.

    Args:
        cfg: Cấu hình Faster-GS.
        support: Bảng ``gsplat.fastergs_support()``; ``None`` khi bản gsplat không có bảng này
            (bản gốc của Bosch).
        render_mode: ``default``, ``ut`` hoặc ``geer``.

    Returns:
        ``(bật, không có)``, theo thứ tự ``KERNEL_TECHNIQUES``.
    """
    if not cfg.enabled:
        return [], []
    wanted = [name for name in KERNEL_TECHNIQUES if getattr(cfg, name)]
    available = set((support or {}).get(render_mode, ()))
    return [n for n in wanted if n in available], [n for n in wanted if n not in available]


def morton_due(step: int, cfg: SimpleNamespace) -> bool:
    """Bước ``step`` có sắp lại Gaussian theo Z-order không."""
    return "morton_order" in active_techniques(cfg) and step > 0 and step % cfg.morton_interval == 0
