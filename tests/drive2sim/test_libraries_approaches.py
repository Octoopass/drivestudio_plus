"""Test approach PVG, DeformableGS, OmniRe nhiều camera và bảng thư viện của Studio."""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.core.schema import Manifest, Split
from drive2sim.engine import env_install, libraries
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.registry import StageRegistry, load_builtin_stages
from drive2sim.engine.resolve import resolve
from drive2sim.engine.workdir import Workdir
from drive2sim.stages.train.drivestudio import TrainDrivestudioParams, has_dynamic_masks
from drive2sim.studio.api.app import create_app
from tests.drive2sim.samples import MANIFEST, SPLIT

load_builtin_stages()


@pytest.mark.parametrize("approach_id", ["pvg", "deformablegs"])
def test_box_free_approaches(approach_id: str) -> None:
    approach = load_approach(approach_id)
    assert approach.requires.cuboids == "none"
    plan = resolve(approach, Manifest.model_validate(MANIFEST), Split.model_validate(SPLIT))
    train = next(stage for stage in plan.stages if stage.stage_id == "train.drivestudio")
    params = TrainDrivestudioParams(**train.params)
    assert params.config_file.name == f"{approach_id}.yaml"
    masks = next(s for s in approach.stages if s.stage == "prep.masks_segformer")
    assert masks.params == {"dynamic": False}


def test_omnire_extended_cam() -> None:
    assert "extended_cam" in load_approach("omnire").options
    params = TrainDrivestudioParams(method="omnire", campose=True, extended_cam=True)
    assert params.config_file.name == "omnire_extended_cam.yaml"
    with pytest.raises(ValueError, match="extended_cam"):
        TrainDrivestudioParams(method="omnire", renderer="3dgut", extended_cam=True)
    assert TrainDrivestudioParams(method="omnire", campose=True).config_file.name == "omnire.yaml"


def test_dynamic_masks_detection(tmp_path: Path) -> None:
    assert not has_dynamic_masks(tmp_path, {})
    assert has_dynamic_masks(tmp_path, {"fine_dynamic_masks": tmp_path})
    (tmp_path / "dynamic_masks").mkdir()
    assert has_dynamic_masks(tmp_path, {})


def test_catalog_and_install_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = {item["id"] for item in libraries.CATALOG}
    assert {"env-main", "env-nurec", "renderers", "segformer", "sdk-nuscenes", "uv"} <= ids
    assert all(item["group"] in libraries.GROUPS for item in libraries.CATALOG)
    assert libraries.install_plan("renderers") == ("env", "main", 4)
    assert libraries.install_plan("env-nurec") == ("env", "nurec", 1)
    monkeypatch.setattr(libraries, "pip_constraints", lambda python, path=None: Path("c.txt"))
    monkeypatch.setattr(libraries, "_has_pip", lambda python: True)
    # pip không đổi phiên bản gói đã có: chạy với file ràng buộc
    assert libraries.install_plan("sdk-kitti", "py")[1] == [
        "py", "-m", "pip", "install", "-c", "c.txt", "pykitti",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="cài tay"):
        libraries.install_plan("node")
    with pytest.raises(ValueError):
        libraries.install_plan("nope")
    monkeypatch.setattr(libraries, "_which", lambda name: None)
    with pytest.raises(ValueError, match="uv"):
        libraries.install_plan("colab-cli")


def test_item_status_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {"check": ("import", ["json", "no_such_module_xyz"])}
    assert libraries.item_status(item) == {"status": "partial", "found": "json"}
    monkeypatch.setattr(libraries, "_which", lambda name: "/bin/" + name)
    assert libraries.item_status({"check": ("which", "uv")})["status"] == "ok"
    monkeypatch.setattr(libraries, "_hf_cached", lambda repo: False)
    assert libraries.item_status({"check": ("hf", "x/y")})["status"] == "missing"


def test_start_command_runs_and_reads_steps(tmp_path: Path) -> None:
    flow = env_install.InstallFlow(tmp_path / "logs", gpus=lambda: [], python_version=(3, 10))
    code = "print('=== 12:00:00 1/1 cài thử'); print('xong')"
    flow.start_command("sdk-test", [sys.executable, "-c", code])
    deadline = time.monotonic() + 20
    while flow.view()["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    view = flow.view()
    assert view["state"] == "succeeded" and view["env"] == "sdk-test" and view["step"] == 1
    assert "xong" in (tmp_path / "logs" / "install-sdk-test.log").read_text(encoding="utf-8")


def test_libraries_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = {}

    class Flow:
        python = "py"

        def view(self) -> dict:
            return {"state": "idle"}

        def start(self, env: str, from_step: int) -> dict:
            started["env"] = (env, from_step)
            return {"state": "running"}

        def start_command(self, name: str, command: list[str]) -> dict:
            started["command"] = (name, command)
            return {"state": "running"}

        def cancel(self) -> dict:
            return {"state": "idle"}

    monkeypatch.setattr(
        libraries, "item_status", lambda item: {"status": "missing", "found": None}
    )
    monkeypatch.setattr(libraries, "pip_constraints", lambda python, path=None: Path("c.txt"))
    monkeypatch.setattr(libraries, "_has_pip", lambda python: True)
    app = create_app(Workdir(tmp_path), registry=StageRegistry(), frontend_dir=None)
    app.state.env_install = Flow()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        listed = client.get("/api/libraries").json()
        assert listed["install"] == {"state": "idle"} and listed["items"][0]["status"] == "missing"
        assert "check" not in listed["items"][0] and "installable" in listed["items"][0]
        assert client.post("/api/libraries/renderers/install").json()["state"] == "running"
        assert started["env"] == ("main", 4)
        client.post("/api/libraries/sdk-kitti/install")
        assert started["command"] == (
            "sdk-kitti", ["py", "-m", "pip", "install", "-c", "c.txt", "pykitti"],
        )  # fmt: skip
        assert client.post("/api/libraries/node/install").status_code == 400
