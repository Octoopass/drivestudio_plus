"""KITTI-360 -> NCore -> NuRec -> eval pipeline: discovery, validation and the read-only outputs route."""
import json
from pathlib import Path

import pytest

from fisheye_fusion.pipeline import Discovery, PipelineManager


@pytest.fixture
def ws(tmp_path):
    """A workspace whose NuRec venv python is a symlink to an interpreter outside it (as uv makes)."""
    outside = tmp_path / "uv-python" / "python3.11"
    outside.parent.mkdir()
    outside.write_text("")
    root = tmp_path / "ws"
    venv_bin = root / "backends" / "instant-nurec" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(outside)
    (root / "backends" / "instant-nurec" / "run_inference.py").write_text("")
    ckpt = root / "backends" / "instant-nurec" / "checkpoints" / "pth"
    ckpt.mkdir(parents=True)
    (ckpt / "instant_nurec_pa_multiview_1.1.0.pth").write_text("")
    data = root / "kitti360"
    (data / "data_2d_raw").mkdir(parents=True)
    (data / "data_poses.zip").write_text("")
    for cam in ("image_00", "image_02"):
        (data / "data_2d_raw" / f"2013_05_28_drive_0003_sync_{cam}.tar").write_text("")
    return root


def test_report_handles_symlinked_venv_python(ws):
    report = Discovery(ws).report()
    assert report["nurec"]["ok"]
    assert report["nurec"]["python"] == "backends/instant-nurec/.venv/bin/python"
    assert report["nurec"]["checkpoints"] == {
        "pa-multiview": "backends/instant-nurec/checkpoints/pth/instant_nurec_pa_multiview_1.1.0.pth"}
    (seq,) = report["datasets"][0]["sequences"].items()
    assert seq == ("2013_05_28_drive_0003_sync", {"cameras": ["image_00", "image_02"], "lidar": False})


def test_evaluations_are_listed_with_their_inputs(ws):
    run = ws / "outputs" / "eval" / "abc"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"camera_front_left_00": {"psnr_mean": 16.0, "views": 3}}))
    (run / "job.json").write_text(json.dumps({"ply": "a.ply", "clip": "c.json"}))
    (run / "camera_front_left_00_0.png").write_bytes(b"png")
    (ev,) = Discovery(ws).evaluations()
    assert ev["path"] == "outputs/eval/abc" and ev["ply"] == "a.ply" and ev["clip"] == "c.json"
    assert ev["images"] == ["camera_front_left_00_0.png"]


@pytest.mark.parametrize("bad", ["../secret.png", "/etc/passwd", "eval\\x.png", "c:x.png", "eval/abc/notes.txt",
                                 "eval/abc/missing.png"])
def test_output_file_rejects_escapes_and_other_types(ws, bad):
    run = ws / "outputs" / "eval" / "abc"
    run.mkdir(parents=True)
    (run / "notes.txt").write_text("x")
    (ws / "secret.png").write_bytes(b"x")
    with pytest.raises(ValueError):
        Discovery(ws).output_file(bad)


def test_output_file_serves_results(ws):
    run = ws / "outputs" / "reconstruction" / "r" / "ply"
    run.mkdir(parents=True)
    (run / "scene.ply").write_bytes(b"ply")
    assert Discovery(ws).output_file("reconstruction/r/ply/scene.ply") == (run / "scene.ply").resolve()


def test_output_file_rejects_symlink_out_of_outputs(ws, tmp_path):
    (ws / "outputs").mkdir()
    (tmp_path / "elsewhere.png").write_bytes(b"x")
    (ws / "outputs" / "link.png").symlink_to(tmp_path / "elsewhere.png")
    with pytest.raises(ValueError):
        Discovery(ws).output_file("link.png")


def _captured_convert(ws, monkeypatch, **cfg):
    seen = {}
    manager = PipelineManager(ws)
    monkeypatch.setattr(manager, "_start", lambda kind, out, steps, env, info: seen.update(steps=steps, info=info) or {})
    manager.convert({"dataset": "kitti360", "sequence": "2013_05_28_drive_0003_sync", **cfg})
    return seen


def test_convert_passes_fisheye_model_and_defaults_to_rectify120(ws, monkeypatch):
    seen = _captured_convert(ws, monkeypatch, frames=40)
    argv = seen["steps"][0][1]
    assert argv[argv.index("--fisheye-model") + 1] == "rectify120"
    assert seen["info"]["fisheye_model"] == "rectify120"
    assert [label for label, _ in seen["steps"]] == ["convert", "projection check"]
    seen = _captured_convert(ws, monkeypatch, frames=40, fisheye_model="opencv", lidar=False, name="kb")
    argv = seen["steps"][0][1]
    assert argv[argv.index("--fisheye-model") + 1] == "opencv" and "--no-lidar" in argv
    assert len(seen["steps"]) == 1


@pytest.mark.parametrize("cfg", [{"fisheye_model": "pinhole"}, {"sequence": "drive_3"}, {"frames": 1},
                                 {"cameras": ["image_09"]}, {"lidar": "yes"}, {"name": "../x"}, {"unknown": 1}])
def test_convert_rejects_bad_config_before_creating_output(ws, monkeypatch, cfg):
    manager = PipelineManager(ws)
    monkeypatch.setattr(manager, "_start", lambda *a: pytest.fail("must not start"))
    with pytest.raises(ValueError):
        manager.convert({"dataset": "kitti360", "sequence": "2013_05_28_drive_0003_sync", **cfg})
    assert not (ws / "outputs" / "ncore").exists() or not any((ws / "outputs" / "ncore").iterdir())


def test_outputs_route(ws, monkeypatch):
    from fastapi.testclient import TestClient

    import fisheye_fusion.server as server

    monkeypatch.setattr(server, "WORKSPACE", Path(ws).resolve())
    run = ws / "outputs" / "eval" / "abc"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text("{}")
    with TestClient(server.create_app()) as client:
        assert client.get("/api/outputs/eval/abc/metrics.json").status_code == 200
        assert client.get("/api/outputs/eval/abc/nope.png").status_code == 400
        setup = client.get("/api/setup").json()
        assert setup["nurec"]["ok"] and "evaluations" in setup


def test_sync_drive_validates_and_launches(ws, monkeypatch):
    fake_colab = ws / "fake-colab"
    fake_colab.write_text("")
    manager = PipelineManager(ws)
    monkeypatch.setattr(Discovery, "colab_bin", property(lambda self: str(fake_colab)))
    seen = {}
    monkeypatch.setattr(manager, "_start", lambda kind, out, steps, env, info: seen.update(kind=kind, steps=steps, info=info) or {"job_id": "0" * 32})
    manager.sync_drive({"mode": "dataset", "drive_path": "/content/drive/MyDrive/datasets/kitti360_full",
                        "sequence": "2013_05_28_drive_0003_sync", "session": "fisheye-lab"})
    assert seen["kind"] == "drive_sync"
    assert seen["info"]["dest"] == "kitti360"
    for bad in ({"drive_path": "/etc/passwd"}, {"drive_path": "/content/drive/../x"}, {"dest": "../escape"}, {"sequence": "bad"}):
        with pytest.raises(ValueError):
            manager.sync_drive({"mode": "dataset", "drive_path": "/content/drive/MyDrive/datasets/kitti360_full", **bad})
