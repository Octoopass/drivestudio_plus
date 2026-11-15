"""Kiểm tra thống kê densify (``recon/utils/densify_stats.py``): chỉ đọc, đúng phân vị."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from drive2sim.recon.utils.densify_stats import STATS_FILE, DensifyStats  # noqa: E402


class FakeModel:
    """Lớp Gaussian giả: chỉ các trường mà ``record`` đọc."""

    densify_stats = None

    def __init__(self) -> None:
        self.class_prefix = "Background#"
        self.num_points = 5
        self.ctrl_cfg = SimpleNamespace(densify_grad_thresh=0.25)
        # trung bình gradient: 0, 0.1, 0.2, 0.3, 0.4; Gaussian cuối chưa ai thấy (vis 0)
        self.xys_grad_norm = torch.tensor([0.0, 0.2, 0.6, 1.2, 9.0])
        self.vis_counts = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0])


def test_record_writes_quantiles_without_touching_model(tmp_path: Path) -> None:
    stats = DensifyStats({"enabled": True})
    model = FakeModel()
    trainer = SimpleNamespace(render_cfg={"render_mode": "ut"}, models={"Background": model})
    stats.attach(trainer, str(tmp_path))
    assert model.densify_stats is stats
    before = model.xys_grad_norm.clone()
    row = stats.record(model, 300)
    assert torch.equal(model.xys_grad_norm, before)
    assert row["num_seen"] == 4 and row["num_above"] == 1  # chỉ 0.3 > 0.25
    assert row["frac_above"] == pytest.approx(0.25)
    assert row["quantiles"]["0.5"] == pytest.approx(0.15)
    assert row["render_mode"] == "ut"
    lines = (tmp_path / STATS_FILE).read_text().splitlines()
    assert json.loads(lines[0])["step"] == 300


def test_disabled_stats_attach_nothing(tmp_path: Path) -> None:
    model = FakeModel()
    trainer = SimpleNamespace(render_cfg={}, models={"Background": model})
    DensifyStats(None).attach(trainer, str(tmp_path))
    assert model.densify_stats is None
    assert not (tmp_path / STATS_FILE).exists()
