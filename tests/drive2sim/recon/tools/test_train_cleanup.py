"""Dọn checkpoint theo bước sau khi train xong (``recon/tools/train.py``)."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from drive2sim.recon.tools.train import remove_step_checkpoints  # noqa: E402


def test_only_step_checkpoints_are_removed(tmp_path: Path) -> None:
    for name in (
        "checkpoint_15000.pth",
        "checkpoint_00500.pth",
        "checkpoint_final.pth",
        "checkpoint_resume.pth",
        "checkpoint_15000.pth.tmp",
        "model.pth",
        "metrics.json",
    ):
        (tmp_path / name).write_text("x")
    removed = remove_step_checkpoints(str(tmp_path))
    assert sorted(Path(p).name for p in removed) == [
        "checkpoint_00500.pth",
        "checkpoint_15000.pth",
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint_15000.pth.tmp",
        "checkpoint_final.pth",
        "checkpoint_resume.pth",
        "metrics.json",
        "model.pth",
    ]
