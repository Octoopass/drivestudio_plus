"""Test cài môi trường từ Studio (``drive2sim/engine/env_install.py`` và route).

Script cài giả là một file bash thật in dòng bước như ``envs/install_main.sh``, nên test chạy cả
tiến trình con, đọc bước, ghi log và dừng nhóm tiến trình.
"""

import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.engine import env_install
from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.app import create_app

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="cần bash")

SCRIPT = """#!/usr/bin/env bash
FROM=1
while [ $# -gt 0 ]; do case "$1" in --python) shift 2;; --from) FROM="$2"; shift 2;; esac; done
step() { echo; echo "=== 12:00:00 $*"; }
[ "$FROM" -le 1 ] && step "1/2 công cụ" && echo cài a
[ "$FROM" -le 2 ] && step "2/2 gói" && echo cài b
[ -n "$SLOW" ] && sleep 30
[ -n "$FAIL" ] && [ "$FROM" -le 1 ] && exit 3
exit 0
"""


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "envs").mkdir(parents=True, exist_ok=True)
    (repo / "envs" / "install_main.sh").write_text(SCRIPT)
    return repo


def flow(tmp_path: Path, **kwargs: object) -> env_install.InstallFlow:
    options = {"gpus": lambda: [("GPU", 1, "1")], "python_version": (3, 10)}
    options.update(kwargs)
    return env_install.InstallFlow(tmp_path / "logs", repo=make_repo(tmp_path), **options)


def wait(installer: env_install.InstallFlow) -> dict:
    deadline = time.monotonic() + 20
    while installer.view()["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    return installer.view()


def test_preflight_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GPU"):
        flow(tmp_path, gpus=lambda: []).start()
    with pytest.raises(ValueError, match="Python 3.10"):
        flow(tmp_path, python_version=(3, 12)).start()
    with pytest.raises(ValueError, match="không có môi trường"):
        flow(tmp_path).start("no-such-env")
    missing = env_install.InstallFlow(
        tmp_path / "logs", repo=tmp_path / "none", python_version=(3, 10)
    )
    with pytest.raises(ValueError, match="mã nguồn"):
        missing.preflight("main")


def test_install_steps_log_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installer = flow(tmp_path)
    assert installer.view()["state"] == "idle"
    monkeypatch.setenv("FAIL", "1")
    installer.start()
    failed = wait(installer)
    assert failed["state"] == "failed" and failed["step"] == 2 and failed["steps"] == 2
    assert failed["label"] == "gói" and "cài b" in failed["log"]
    assert "cài a" in (tmp_path / "logs" / "install-main.log").read_text(encoding="utf-8")
    installer.start(from_step=2)
    done = wait(installer)
    assert done["state"] == "succeeded" and done["from_step"] == 2 and "cài a" not in done["log"]


def test_cancel_stops_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOW", "1")
    installer = flow(tmp_path)
    installer.start()
    assert installer.start()["state"] == "running"  # đang cài thì không chạy lần thứ hai
    installer.cancel()
    assert wait(installer)["state"] == "cancelled"


def test_routes(tmp_path: Path) -> None:
    app = create_app(Workdir(tmp_path / "work"), registry=StageRegistry(), frontend_dir=None)
    app.state.env_install = flow(tmp_path, gpus=lambda: [])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/environment/install").json()["state"] == "idle"
        refused = client.post("/api/environment/install", json={"env": "main"})
        assert refused.status_code == 400 and "GPU" in refused.json()["error"]["message"]
        assert client.post("/api/environment/install", json={"from_step": 0}).status_code == 400
        assert client.post("/api/environment/install/cancel").json()["state"] == "idle"
