"""Kiểm tra ``drive2sim/engine/render_check.py`` với GPU, PyTorch và gsplat giả."""

from pathlib import Path

import pytest

from drive2sim.engine import render_check


def test_ready_when_everything_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_check, "_gpus", lambda: [("Quadro T2000", 4096, "580.1")])
    monkeypatch.setattr(
        render_check,
        "package_versions",
        lambda: {"pytorch": "2.4.1+cu121", "cuda": "12.1", "gsplat": "1.5.3"},
    )
    monkeypatch.setattr(render_check, "_torch_cuda", lambda: (True, "Quadro T2000"))
    checkpoint = tmp_path / "d2s" / "stages" / "train.drivestudio" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    result = render_check.render_readiness(tmp_path)
    assert result["ready"] is True
    assert [c["id"] for c in result["checks"]] == [
        "gpu",
        "pytorch",
        "cuda",
        "gsplat",
        "checkpoint",
    ]
    assert result["checks"][-1]["value"] == "d2s/stages/train.drivestudio/checkpoint_final.pth"


def test_checkpoint_with_people_requires_body_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_ready_when_everything_present(tmp_path, monkeypatch)
    train = tmp_path / "d2s/stages/train.drivestudio"
    config = train / "config.yaml"
    config.write_text("model: {SMPLNodes: {}}\ndata: {pixel_source: {load_smpl: true}}\n")
    monkeypatch.setattr(render_check, "find_smpl_model", lambda root: None)
    result = render_check.render_readiness(tmp_path)
    assert not result["ready"]
    assert result["checks"][-1] == {"id": "smpl_model", "ok": False, "value": "SMPL_NEUTRAL.pkl"}
    monkeypatch.setattr(
        render_check, "find_smpl_model", lambda root: tmp_path / "SMPL_NEUTRAL.pkl"
    )
    assert render_check.render_readiness(tmp_path)["ready"]
    config.write_text("model: {SMPLNodes: {}}\ndata: {pixel_source: {load_smpl: false}}\n")
    monkeypatch.setattr(render_check, "find_smpl_model", lambda root: None)
    assert render_check.render_readiness(tmp_path)["ready"]


def test_missing_pytorch_skips_cuda_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_check, "_gpus", lambda: [])
    monkeypatch.setattr(
        render_check, "package_versions", lambda: {"pytorch": None, "cuda": None, "gsplat": None}
    )
    monkeypatch.setattr(render_check, "_torch_cuda", lambda: pytest.fail("không được hỏi CUDA"))
    result = render_check.render_readiness(tmp_path)
    assert result["ready"] is False
    assert all(not check["ok"] for check in result["checks"])
