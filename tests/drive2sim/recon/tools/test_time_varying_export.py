"""Kiểm tra ``drive2sim/recon/tools/time_varying_export.py``: chọn Gaussian nền tĩnh của PVG.

- Độ dài đời tính bằng khung là ``exp(_betas) / time_interval``.
- Gaussian sống ít hơn ``min_life_frames`` khung (vệt xe chạy) bị bỏ; ``0`` giữ mọi Gaussian.
- ``time_interval`` đọc từ ``config.yaml`` của lần train, chỉ khi node nền là PVG.
"""

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from drive2sim.recon.tools import time_varying_export as tv  # noqa: E402


def test_static_mask_keeps_long_lived_gaussians() -> None:
    interval = 0.02
    # độ dài đời 5, 19,9, 20, 200 khung
    betas = [[math.log(frames * interval)] for frames in (5.0, 19.9, 20.0, 200.0)]
    mask, stats = tv.static_mask(betas, interval, min_life_frames=20.0)
    assert mask.tolist() == [False, False, True, True]
    assert stats["kept"] == 2 and stats["dropped"] == 2
    assert tv.life_frames(betas, interval) == pytest.approx([5.0, 19.9, 20.0, 200.0])
    keep_all, _ = tv.static_mask(betas, interval, min_life_frames=0.0)
    assert keep_all.all()


def test_is_pvg_state() -> None:
    assert tv.is_pvg_state({"_means": 0, "_taus": 0, "_betas": 0})
    assert not tv.is_pvg_state({"_means": 0, "_scales": 0})


def test_time_interval_from_training_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  Background:\n"
        "    type: drive2sim.recon.models.gaussians.PeriodicVibrationGaussians\n"
        "    ctrl:\n      time_interval: 0.05\n"
    )
    assert tv.pvg_time_interval(config) == pytest.approx(0.05)
    config.write_text("model:\n  Background:\n    type: x.VanillaGaussians\n")
    assert tv.pvg_time_interval(config) is None
    assert tv.pvg_time_interval(tmp_path / "missing.yaml") is None
