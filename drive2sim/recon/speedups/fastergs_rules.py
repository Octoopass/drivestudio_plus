"""Cấu hình Faster-GS không cần PyTorch: giá trị mặc định và danh sách kỹ thuật đang bật.

Tách khỏi ``fastergs.py`` để test được trên máy không có GPU (``docs/methods/fastergs.md``).

File này chứa ``FASTERGS_DEFAULTS``, ``TECHNIQUES``, ``parse_config``, ``active_techniques``.
"""

from types import SimpleNamespace
from typing import Any, Mapping, Optional

# Kỹ thuật phần Python, theo thứ tự ghi vào log và train_stats.json.
TECHNIQUES = ("fused_adam", "fused_ssim", "fewer_syncs")

# Giá trị mặc định của khối trainer.fastergs. `enabled` là công tắc chung; mỗi kỹ thuật chỉ chạy
# khi công tắc chung bật và cờ riêng của nó bật.
FASTERGS_DEFAULTS = dict(
    enabled=False,
    fused_adam=True,  # torch.optim.Adam(fused=True)
    fused_ssim=True,  # gói fused-ssim thay pytorch_msssim trong loss (cùng định nghĩa, padding valid)
    fewer_syncs=True,  # gộp các lần chờ GPU khi kiểm NaN/Inf và khi ghi log mỗi bước
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
    """Các kỹ thuật thật sự được bật, theo thứ tự ``TECHNIQUES``; rỗng khi công tắc chung tắt."""
    if not cfg.enabled:
        return []
    return [name for name in TECHNIQUES if getattr(cfg, name)]
