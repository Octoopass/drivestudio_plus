"""Test cấu hình Faster-GS không cần PyTorch (``drive2sim/recon/speedups/fastergs_rules.py``).

Phần chạy trên GPU (fused Adam, fused SSIM) đã được so với bản thường bằng script chạy tay trên
GPU (C1 trong plan Faster-GS); không nằm trong bộ test này.
"""

import pytest

from drive2sim.recon.speedups.fastergs_rules import (
    FASTERGS_DEFAULTS,
    TECHNIQUES,
    active_techniques,
    kernel_techniques,
    morton_due,
    parse_config,
)


def test_off_by_default() -> None:
    cfg = parse_config(None)
    assert vars(cfg) == FASTERGS_DEFAULTS
    assert active_techniques(cfg) == []


def test_active_techniques_follow_flags() -> None:
    # Z-order là thử nghiệm: không bật theo công tắc chung
    assert active_techniques(parse_config({"enabled": True})) == [
        "fused_adam",
        "fused_ssim",
        "fewer_syncs",
    ]
    cfg = parse_config({"enabled": True, "morton_order": True})
    assert active_techniques(cfg) == list(TECHNIQUES)
    cfg = parse_config({"enabled": True, "fused_ssim": False})
    assert active_techniques(cfg) == ["fused_adam", "fewer_syncs"]
    # cờ riêng bật nhưng công tắc chung tắt: không có gì chạy
    assert active_techniques(parse_config({"fused_adam": True})) == []


def test_unknown_key() -> None:
    with pytest.raises(AssertionError, match="unknown trainer.fastergs keys"):
        parse_config({"enabled": True, "fused_lamb": True})


def test_morton_due() -> None:
    on = parse_config({"enabled": True, "morton_order": True, "morton_interval": 5000})
    assert [step for step in (0, 4900, 5000, 7000, 10000) if morton_due(step, on)] == [5000, 10000]
    assert not morton_due(5000, parse_config({"enabled": True}))  # cờ riêng tắt
    assert not morton_due(5000, parse_config({"morton_order": True}))  # công tắc chung tắt


def test_kernel_techniques_follow_gsplat_support() -> None:
    on = parse_config({"enabled": True})
    table = {"default": {"exact_tiles"}, "ut": set(), "geer": set()}
    assert kernel_techniques(on, table, "default") == (["exact_tiles"], [])
    assert kernel_techniques(on, table, "ut") == ([], ["exact_tiles"])
    # bản gsplat gốc của Bosch không có bảng khả năng: không bật gì
    assert kernel_techniques(on, None, "default") == ([], ["exact_tiles"])
    off = parse_config({"enabled": True, "exact_tiles": False})
    assert kernel_techniques(off, table, "default") == ([], [])
    assert kernel_techniques(parse_config(None), table, "default") == ([], [])
