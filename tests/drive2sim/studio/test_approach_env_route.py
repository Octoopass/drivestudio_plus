"""Test ``GET /api/approaches/{id}/environment``: file SMPL theo stage thật của run."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.app import create_app


def smpl_level(client: TestClient, query: str) -> object:
    data = client.get(f"/api/approaches/omnire/environment{query}").json()
    return {c["id"]: c["level"] for c in data["checks"]}.get("smpl_model")


def test_smpl_follows_resolved_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from drive2sim.engine import environment

    monkeypatch.setattr(environment, "find_smpl_model", lambda root=None: None)
    app = create_app(Workdir(tmp_path / "work"), registry=StageRegistry(), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert smpl_level(client, "") == "warn"  # chưa giải stage: Tự động có thể bỏ SMPL
        assert smpl_level(client, "?stages=prep.smpl,train.drivestudio") == "missing"
        assert smpl_level(client, "?stages=train.drivestudio") is None  # không dựng người
