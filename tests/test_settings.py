"""Deployment settings: defaults, JSON file and environment overrides, public view without paths."""
import json
from pathlib import Path

from fisheye_fusion.settings import Settings, display_path


def test_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("FFLAB_SETTINGS", raising=False)
    monkeypatch.delenv("FFLAB_DS_VENV", raising=False)
    (tmp_path / "fflab.settings.json").write_text(json.dumps({
        "app": {"name": "Acme Recon"},
        "colab": {"gpus": [{"id": "L4", "label": "L4", "memory_gb": 24, "tier": "paid"}], "default_gpu": "T4"}}))
    s = Settings(tmp_path)
    assert s.data["app"]["name"] == "Acme Recon" and s.data["app"]["tagline"]      # merged, not replaced
    assert [g["id"] for g in s.gpus] == ["L4"] and s.data["colab"]["default_gpu"] == "L4"
    monkeypatch.setenv("FFLAB_DS_VENV", str(tmp_path / "env"))
    assert Settings(tmp_path).path("ds_venv") == tmp_path / "env"
    pub = s.public()
    assert "paths" not in pub and "drive_cache" not in pub["colab"]


def test_display_path_hides_home_and_workspace(tmp_path):
    assert display_path(tmp_path / "a" / "b", tmp_path) == "./a/b"
    assert display_path(Path.home() / "venvs" / "ds") == "~/venvs/ds"
    assert display_path(None) == ""
