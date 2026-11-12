"""Kiểm tra API của Studio (CONTRACTS mục 11) bằng ``fastapi.testclient``.

File này dựng một thư mục làm việc tạm có cảnh mẫu (manifest, ảnh giả, báo cáo kiểm tra), một
approach giả ``toy`` dùng ba stage giả (``init.toy`` có cache và cần LiDAR, ``train.toy`` ghi một
video, ``eval.standard`` ghi ``summary.json``), rồi kiểm tra:
- route cảnh và split: liệt kê, xem, ảnh theo khung, tạo split, các lỗi 400, 404, 409;
- route approach: xem, ``check`` (stage dùng lại cache, lựa chọn bị khoá), lỗi 422 khi bị khoá,
  400 khi sai lựa chọn hoặc có trường lạ;
- route run: tạo run chạy nền (``202``), chờ xong, xem ``run.json`` kèm ``summary``, liệt kê,
  log, video, chạy tiếp, sự kiện SSE;
- bảng so sánh JSON và CSV;
- ``/files/`` phục vụ file trong thư mục làm việc và chặn đường dẫn ra ngoài;
- ``EventBroker`` gửi lại sự kiện bị lỡ theo ``Last-Event-ID`` và gửi dòng giữ kết nối.
"""

import asyncio
import json
import struct
import subprocess
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, Manifest, Summary
from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle import client as kaggle_client
from drive2sim.studio.api.app import create_app
from drive2sim.studio.api.events import EventBroker, stream_events
from tests.drive2sim.samples import MANIFEST

SCENE = MANIFEST["scene_id"]
TOY = """
schema_version: d2s-approach/0.1
approach_id: toy
name: Toy
version: 0.1.0
codebase: drivestudio
track: pinhole
level: L1
requires: {camera_models: [opencv_pinhole], cuboids: optional, split_strategies: [every_nth_frame]}
options:
  init: {choices: [init.toy, init.plain], default: auto}
stages:
  - {stage: "${options.init}"}
  - {stage: train.toy}
  - {stage: eval.standard}
"""


def write_jpeg(path: Path, width: int = 64, height: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x11\0" * 3
    path.write_bytes(b"\xff\xd8" + sof0 + b"\xff\xd9")


def _points(self: Stage, ctx: StageContext, inputs: dict, params: BaseModel) -> dict:
    out = ctx.output_dir / "points.npz"
    out.write_text("p")
    return {"points": out}


def _train(self: Stage, ctx: StageContext, inputs: dict, params: BaseModel) -> dict:
    ctx.progress(0.5, "nửa đường")
    (ctx.output_dir / "videos").mkdir()
    (ctx.output_dir / "videos" / "test_set.mp4").write_bytes(b"mp4")
    (ctx.output_dir / "log.txt").write_text("train log\n")
    return {"model": ctx.output_dir}


def _eval(self: Stage, ctx: StageContext, inputs: dict, params: BaseModel) -> dict:
    summary = Summary.model_validate(
        {
            "schema_version": SCHEMA_VERSIONS["summary"],
            "protocol": {
                "version": "0.1",
                "image_dtype": "uint8",
                "resolution": "native",
                "lpips": {"net": "alex", "package": "torchmetrics", "package_version": "0.10.3"},
            },
            "split_id": ctx.split.split_id,
            "overall": {"psnr": 27.5, "ssim": 0.85, "lpips": 0.2},
        }
    )
    files.dump(summary, ctx.output_dir / "summary.json")
    return {"report": ctx.output_dir}


def make_registry() -> StageRegistry:
    registry = StageRegistry()
    stages = [
        (
            "init.toy",
            "InitToyStage",
            {
                "cacheable": True,
                "uses_lidar": True,
                "inputs": {"s": "scene"},
                "outputs": {"points": "points_init"},
                "run": _points,
            },
        ),
        (
            "init.plain",
            "InitPlainStage",
            {
                "cacheable": True,
                "inputs": {"s": "scene"},
                "outputs": {"points": "points_init"},
                "run": _points,
            },
        ),
        (
            "train.toy",
            "TrainToyStage",
            {
                "inputs": {"p": "points_init", "split": "split"},
                "outputs": {"model": "model"},
                "run": _train,
            },
        ),
        (
            "eval.standard",
            "EvalStandardStage",
            {
                "inputs": {"m": "model", "split": "split"},
                "outputs": {"report": "eval_report"},
                "run": _eval,
            },
        ),
    ]
    for stage_id, name, attrs in stages:
        registry.register(stage_id)(type(name, (Stage,), {"version": "0.1.0", **attrs}))
    return registry


def no_kaggle_cli(command: list[str]) -> subprocess.CompletedProcess:
    """Thay lệnh kaggle thật: mọi lệnh đều báo chưa đăng nhập."""
    return subprocess.CompletedProcess(command, 1, "", "401 Unauthorized")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    workdir = Workdir(tmp_path / "work")
    scene_dir = workdir.scene_dir(SCENE)
    files.dump(Manifest.model_validate(MANIFEST), scene_dir / "manifest.yaml")
    for t in range(10):
        write_jpeg(scene_dir / "images" / f"{t:03d}_0.jpg")
    (scene_dir / "reports").mkdir()
    (scene_dir / "reports" / "validation.json").write_text(
        json.dumps({"summary": {"errors": 0, "warnings": 0}, "checks": []})
    )
    approaches = tmp_path / "approaches"
    approaches.mkdir()
    (approaches / "toy.yaml").write_text(TOY)
    monkeypatch.setattr(paths, "APPROACHES_DIR", approaches)
    # Không bao giờ gọi Kaggle thật trong test, kể cả khi máy có khoá API.
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "no-kaggle"))
    monkeypatch.setattr(kaggle_client, "_run_cli", no_kaggle_cli)
    app = create_app(workdir, registry=make_registry(), frontend_dir=None)
    with TestClient(app) as test_client:
        yield test_client


def create_split(client: TestClient) -> str:
    response = client.post(
        f"/api/scenes/{SCENE}/splits",
        json={"strategy": "every_nth_frame:4", "cameras": ["CAM_FRONT"]},
    )
    assert response.status_code == 201, response.text
    return response.json()["split_id"]


def test_scenes_and_splits(client: TestClient) -> None:
    assert client.get("/api/scenes").json()["total"] == 1
    scene = client.get(f"/api/scenes/{SCENE}").json()
    assert scene["manifest"]["scene_id"] == SCENE
    assert scene["validation"]["summary"]["errors"] == 0
    frame = client.get(f"/api/scenes/{SCENE}/frames/3").json()
    assert frame["images"][0] == {
        "camera_id": "CAM_FRONT",
        "url": f"/files/data/{SCENE}/images/003_0.jpg",
    }
    assert frame["images"][1]["url"] is None  # camera thứ hai không có ảnh
    assert client.get(frame["images"][0]["url"]).status_code == 200

    split_id = create_split(client)
    assert split_id == "every4_1cams"
    assert client.get(f"/api/scenes/{SCENE}/splits").json()["total"] == 1
    assert client.get(f"/api/scenes/{SCENE}/splits/{split_id}").json()["strategy"]

    again = client.post(
        f"/api/scenes/{SCENE}/splits",
        json={"strategy": "every_nth_frame:4", "cameras": ["CAM_FRONT"]},
    )
    assert again.status_code == 409 and again.json()["error"]["code"] == "conflict"
    bad = client.post(
        f"/api/scenes/{SCENE}/splits", json={"strategy": "every_nth_frame:4", "cameras": ["CAM_X"]}
    )
    assert bad.status_code == 400 and "CAM_X" in bad.json()["error"]["message"]
    extra = client.post(f"/api/scenes/{SCENE}/splits", json={"strategy": "x", "bogus": 1})
    assert extra.status_code == 400
    missing = client.get("/api/scenes/nuscenes_scene-9999")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_approach_check(client: TestClient) -> None:
    split_id = create_split(client)
    assert client.get("/api/approaches").json()["items"][0]["approach_id"] == "toy"
    body = {"scene_id": SCENE, "split_id": split_id}
    check = client.post("/api/approaches/toy/check", json=body).json()
    assert check["options"] == {"init": "init.toy"}
    assert [s["stage_id"] for s in check["stages"]] == ["init.toy", "train.toy", "eval.standard"]
    assert check["blocked_choices"] == {}
    bad = client.post("/api/approaches/toy/check", json={**body, "options": {"init": "x"}})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_request"


def test_approach_blocked(client: TestClient, tmp_path: Path) -> None:
    split_id = create_split(client)
    manifest = dict(MANIFEST, lidars=[], available={"lidar": False, "cuboids": "gt"})
    files.dump(
        Manifest.model_validate(manifest), tmp_path / "work" / "data" / SCENE / "manifest.yaml"
    )
    body = {"scene_id": SCENE, "split_id": split_id}
    check = client.post("/api/approaches/toy/check", json=body).json()
    assert check["options"] == {"init": "init.plain"}
    assert check["blocked_choices"]["init"][0]["choice"] == "init.toy"
    blocked = client.post(
        "/api/approaches/toy/check", json={**body, "options": {"init": "init.toy"}}
    )
    assert blocked.status_code == 422
    assert "LiDAR" in blocked.json()["error"]["details"]["reasons"][0]["message"]


def test_run_lifecycle(client: TestClient) -> None:
    split_id = create_split(client)
    created = client.post(
        "/api/runs", json={"approach_id": "toy", "scene_id": SCENE, "split_id": split_id}
    )
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    assert created.json()["status"] == "pending"
    client.app.state.studio.queue.wait_idle()

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "succeeded", [s.get("error") for s in run["stages"]]
    assert run["summary"]["overall"]["psnr"] == 27.5
    assert client.get("/api/runs", params={"scene_id": SCENE}).json()["total"] == 1
    assert client.get("/api/runs", params={"status": "failed"}).json()["total"] == 0
    log = client.get(f"/api/runs/{run_id}/log").json()
    assert log["stages"]["train.toy"] == "train log\n"
    assert client.get(f"/api/runs/{run_id}/splats").json() == {"preview": None, "ply": None}
    preview_dir = (
        Path(client.app.state.studio.workdir.find_run(run_id)) / "d2s/stages/export.preview"
    )
    (preview_dir / "actors").mkdir(parents=True)
    for name in ("background.ply", "actors/rigid_0000.ply", "actor_poses.json", "done"):
        (preview_dir / name).write_text("x")
    (preview_dir / "preview.json").write_text(
        json.dumps(
            {
                "schema_version": "d2s-preview/0.1",
                "source_run": "x",
                "sh_degree": 3,
                "rigid_actors": ["rigid_0000"],
            }
        )
    )
    splats = client.get(f"/api/runs/{run_id}/splats").json()["preview"]
    assert splats["background_url"].endswith("export.preview/background.ply")
    assert client.get(splats["actor_urls"]["rigid_0000"]).status_code == 200
    videos = client.get(f"/api/runs/{run_id}/videos").json()["items"]
    assert videos[0]["stage_id"] == "train.toy" and videos[0]["name"] == "videos/test_set.mp4"
    assert client.get(videos[0]["url"]).content == b"mp4"

    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        text = "".join(stream.iter_text())
    assert "event: status" in text and "event: done" in text

    report = client.get("/api/report", params={"scene_id": SCENE}).json()["items"]
    assert report[0]["run_id"] == run_id and report[0]["psnr"] == 27.5
    csv_text = client.get("/api/report", params={"format": "csv"}).text
    assert csv_text.splitlines()[0].startswith("run_id,") and "init=init.toy" in csv_text

    resumed = client.post(f"/api/runs/{run_id}/resume")
    assert resumed.status_code == 202
    client.app.state.studio.queue.wait_idle()
    assert client.get("/api/runs/20260101-000000-toy-ffff").status_code == 404
    kaggle = client.post(
        "/api/runs",
        json={"approach_id": "toy", "scene_id": SCENE, "split_id": split_id, "executor": "kaggle"},
    )
    assert kaggle.status_code == 202 and kaggle.json()["executor"] == "kaggle"
    client.app.state.studio.jobs.wait_idle()
    push_job = client.get("/api/jobs").json()["items"][0]
    assert push_job["kind"] == "kaggle_push" and push_job["status"] == "failed"
    status = client.get(f"/api/kaggle/{kaggle.json()['run_id']}")
    assert status.status_code == 502 and status.json()["error"]["code"] == "kaggle_failed"


def test_files_block_traversal(client: TestClient) -> None:
    assert client.get("/files/../../etc/passwd").status_code == 404
    assert client.get("/files/data/nope.jpg").status_code == 404


def test_broker_replays_missed_events_and_keepalive() -> None:
    async def scenario() -> list[str]:
        broker = EventBroker()
        thread = threading.Thread(
            target=lambda: [
                broker.publish({"event": "progress", "run_id": "r", "n": i}) for i in range(3)
            ]
        )
        thread.start()
        thread.join()
        chunks: list[str] = []
        stream = stream_events(broker, "r", [], last_event_id=1, keepalive_s=0.05)
        chunks.append(await stream.__anext__())  # sự kiện 2
        chunks.append(await stream.__anext__())  # sự kiện 3
        chunks.append(await stream.__anext__())  # giữ kết nối
        broker.publish({"event": "done", "run_id": "r", "status": "succeeded"})
        chunks.append(await stream.__anext__())
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        return chunks

    chunks = asyncio.run(scenario())
    assert chunks[0].startswith("event: progress\nid: 2")
    assert chunks[1].startswith("event: progress\nid: 3")
    assert chunks[2] == ": keepalive\n\n"
    assert chunks[3].startswith("event: done\nid: 4")


def test_frontend_served_at_root(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>drive2sim Studio</title>")
    app = create_app(Workdir(tmp_path / "work"), registry=make_registry(), frontend_dir=dist)
    with TestClient(app) as test_client:
        root = test_client.get("/")
        assert "drive2sim Studio" in root.text and root.headers["cache-control"] == "no-cache"
        assert test_client.get("/api/scenes").json() == {"items": [], "total": 0}


def test_built_frontend_is_committed() -> None:
    # Người dùng chạy Studio không cần Node.js: bản build phải có sẵn (ARCHITECTURE mục 2.11).
    from drive2sim.studio.api.app import FRONTEND_DIST

    assert (FRONTEND_DIST / "index.html").is_file()


def test_housekeeping_routes(client: TestClient) -> None:
    split_id = create_split(client)
    run_id = client.post(
        "/api/runs", json={"approach_id": "toy", "scene_id": SCENE, "split_id": split_id}
    ).json()["run_id"]
    client.app.state.studio.queue.wait_idle()

    cache = client.get("/api/cache").json()
    assert cache["total"] == 1 and cache["items"][0]["used_by"] == [run_id]
    assert client.post("/api/cache/gc", json={"dry_run": True}).json()["removed"] == []
    validated = client.post(f"/api/scenes/{SCENE}/validate").json()
    assert "summary" in validated and validated["checks"]

    export = client.post(f"/api/runs/{run_id}/export", json={"kind": "preview"})
    assert export.status_code == 400 and "export.preview" in export.json()["error"]["message"]
    cancelled = client.post(f"/api/runs/{run_id}/cancel")
    assert cancelled.status_code == 202 and cancelled.json()["status"] == "succeeded"

    in_use = client.delete(f"/api/scenes/{SCENE}/splits/{split_id}")
    assert in_use.status_code == 409 and in_use.json()["error"]["code"] == "conflict"
    assert client.delete(f"/api/runs/{run_id}").json()["removed"][0].endswith(run_id)
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    gc = client.post("/api/cache/gc", json={}).json()
    assert len(gc["removed"]) == 1 and client.get("/api/cache").json()["total"] == 0
    assert client.delete(f"/api/scenes/{SCENE}/splits/{split_id}").status_code == 200
    assert f"data/{SCENE}" in client.delete(f"/api/scenes/{SCENE}").json()["removed"]
    assert client.get("/api/scenes").json()["total"] == 0


def test_source_inspect_and_ingest_job(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drive2sim.ingest import drivestudio
    from tests.drive2sim.ingest.test_ingest import fake_preprocess, make_raw_nuscenes

    monkeypatch.setattr(drivestudio, "run_logged", fake_preprocess([]))
    raw = make_raw_nuscenes(tmp_path / "raw")
    report = client.post("/api/sources/inspect", json={"path": str(raw)}).json()
    assert report["dataset"] == "nuscenes" and report["ingest_supported"]
    missing = client.post("/api/sources/inspect", json={"path": str(tmp_path / "nope")})
    assert missing.status_code == 404
    assert client.post("/api/sources/inspect", json={"path": ""}).status_code == 400

    ambiguous = client.post("/api/sources/ingest", json={"path": str(raw)})
    assert (
        ambiguous.status_code == 400 and "chọn một chuỗi" in ambiguous.json()["error"]["message"]
    )
    job = client.post("/api/sources/ingest", json={"path": str(raw), "sequence": "scene-0103"})
    assert job.status_code == 202 and job.json()["status"] in ("queued", "running")
    client.app.state.studio.jobs.wait_idle()
    done = client.get(f"/api/jobs/{job.json()['job_id']}").json()
    assert done["status"] == "succeeded" and done["result"] == "nuscenes_scene-0103"
    again = client.post("/api/sources/ingest", json={"path": str(raw), "sequence": "scene-0103"})
    client.app.state.studio.jobs.wait_idle()
    failed = client.get(f"/api/jobs/{again.json()['job_id']}").json()
    assert failed["status"] == "failed" and "đã có" in failed["error"]
    assert client.get("/api/jobs").json()["total"] == 2
    assert client.get("/api/jobs/nope").status_code == 404
