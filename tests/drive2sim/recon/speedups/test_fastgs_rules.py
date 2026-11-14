"""Test quy tắc của FastGS không cần PyTorch: cấu hình, lịch chạy, cách chọn ảnh.

Phần chấm điểm cần GPU; nó được so với bản cũ trong trainer bằng một script chạy tay trên GPU khi
dời code (B1 trong plan Faster-GS), không nằm trong bộ test này.
"""

from types import SimpleNamespace

import pytest

from drive2sim.recon.speedups.fastgs_rules import (
    FASTGS_DEFAULTS,
    model_schedule,
    parse_config,
    sample_view_ids,
)

CTRL = SimpleNamespace(
    warmup_steps=500, refine_interval=100, reset_alpha_interval=3000, stop_split_at=15000
)


def test_parse_config() -> None:
    assert vars(parse_config(None)) == FASTGS_DEFAULTS
    cfg = parse_config({"enabled": True, "score_thresh": 2.0})
    assert cfg.enabled and cfg.score_thresh == 2.0 and cfg.num_views == 10
    with pytest.raises(AssertionError, match="unknown trainer.fastgs keys"):
        parse_config({"bogus": 1})


@pytest.mark.parametrize(
    "step,expected",
    [
        (500, (False, False)),  # chưa qua warmup
        (650, (False, False)),  # không phải bước refine
        (3100, (False, False)),  # ngay sau reset opacity: phase 100 <= refine_interval
        (1200, (True, False)),  # trong khoảng densify
        (14900, (True, False)),  # phase 2900 nhưng chưa qua prune_from
        (15100, (False, False)),  # đã dừng densify, chưa tới lần refine cuối trước reset
        (17900, (False, True)),  # lần refine cuối trước reset, sau prune_from: prune
        (29900, (False, True)),
    ],
)
def test_model_schedule(step: int, expected: tuple[bool, bool]) -> None:
    cfg = parse_config({"enabled": True})
    assert model_schedule(step, CTRL, 5, cfg, num_iters=30000) == expected


def test_model_schedule_limits() -> None:
    cfg = parse_config({"enabled": True})
    assert model_schedule(29900, CTRL, 5, cfg, num_iters=29900) == (False, False)  # bước cuối
    assert model_schedule(17900, CTRL, 5, parse_config({"prune": False}), 30000) == (False, False)
    # nhiều ảnh train: chờ thấy hết ảnh sau reset mới refine
    assert model_schedule(1200, CTRL, 1500, cfg, 30000) == (False, False)


def test_sample_view_ids() -> None:
    assert sample_view_ids(100, 10, 0.0) == list(range(0, 100, 10))
    ids = sample_view_ids(100, 10, 0.99)
    assert ids == sorted(set(ids)) and len(ids) == 10 and max(ids) <= 99
    assert sample_view_ids(4, 10, 0.5) == [0, 1, 2, 3]  # ít ảnh hơn số cần lấy
