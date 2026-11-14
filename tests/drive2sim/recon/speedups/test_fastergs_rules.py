"""Test cấu hình Faster-GS không cần PyTorch (``drive2sim/recon/speedups/fastergs_rules.py``).

Phần chạy trên GPU (fused Adam, fused SSIM) đã được so với bản thường bằng script chạy tay trên
GPU (C1 trong plan Faster-GS); không nằm trong bộ test này.
"""

import pytest

from drive2sim.recon.speedups.fastergs_rules import (
    FASTERGS_DEFAULTS,
    TECHNIQUES,
    active_techniques,
    parse_config,
)


def test_off_by_default() -> None:
    cfg = parse_config(None)
    assert vars(cfg) == FASTERGS_DEFAULTS
    assert active_techniques(cfg) == []


def test_active_techniques_follow_flags() -> None:
    assert active_techniques(parse_config({"enabled": True})) == list(TECHNIQUES)
    cfg = parse_config({"enabled": True, "fused_ssim": False})
    assert active_techniques(cfg) == ["fused_adam", "fewer_syncs"]
    # cờ riêng bật nhưng công tắc chung tắt: không có gì chạy
    assert active_techniques(parse_config({"fused_adam": True})) == []


def test_unknown_key() -> None:
    with pytest.raises(AssertionError, match="unknown trainer.fastergs keys"):
        parse_config({"enabled": True, "fused_lamb": True})
