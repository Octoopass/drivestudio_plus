"""DriveStudio+ panel backend: log parsing, run discovery, file access and launch validation."""
import subprocess
from pathlib import Path

import os

import pytest
from fastapi.testclient import TestClient

from fisheye_fusion import drivestudio as ds

# Lines as train.py writes them (taken from a real KITTI-360 run on the lab machine).
LOG = b"""[drivestudio] /venv/bin/python tools/train.py --config_file configs/streetgs_ut.yaml --output_root /w/fflab-runs/r1 --project drivestudio --run_name r1 dataset=kitti360/ncore_3cams trainer.optim.num_iters=2000
I20261117 07:56:28 root logging.py:111]   [   0/2001]  eta: 1:45:31  elapsed: 0:00:03  train_metrics/psnr: 9.690170 (9.690170)
I20261117 07:57:07 root logging.py:111]   [ 500/2001]  eta: 0:02:05  elapsed: 0:00:41  train_metrics/psnr: 11.147514 (12.016101)
I20261117 07:58:40 root logging.py:111]   [2000/2001]  eta: 0:00:00  elapsed: 0:02:30  train_metrics/psnr: 17.1 (16.0)
rendering test: 100%|##########| 57/57 [00:18<00:00,  3.60it/s]\rI20261117 07:59:59 root video_utils.py:73] Eval over 57 images:
I20261117 07:59:59 root video_utils.py:74] \t Full Image  PSNR: 17.9769
I20261117 07:59:59 root video_utils.py:75] \t Full Image  SSIM: 0.7111
I20261117 07:59:59 root video_utils.py:76] \t Full Image LPIPS: 0.4201
I20261117 07:59:59 root video_utils.py:77] \t     Non-Sky PSNR: 17.6709
I20261117 07:59:59 root video_utils.py:79] \tDynamic-Only PSNR: -1.0000
"""


def make_run(ws: Path, name: str, log: bytes, images=("step_2000_rgbs.png",)) -> Path:
    root = ws / "fflab-runs" / name
    (root / "drivestudio" / name / "images").mkdir(parents=True)
    (root / "fflab-train.log").write_bytes(log)
    for img in images:
        (root / "drivestudio" / name / "images" / img).write_bytes(b"\x89PNG")
    return root


def test_parse_log_reads_progress_and_last_evaluation():
    info = ds.parse_log(LOG)
    assert (info["step"], info["total"], info["eta"]) == (2000, 2000, "0:00:00")
    assert info["eval_images"] == 57
    assert info["metrics"] == {"Full Image PSNR": 17.9769, "Full Image SSIM": 0.7111, "Full Image LPIPS": 0.4201,
                               "Non-Sky PSNR": 17.6709}  # -1 (no dynamic mask) is dropped
    assert info["error"] is None


def test_parse_log_reports_the_exception_line():
    info = ds.parse_log(LOG[:400] + b"\nTraceback (most recent call last):\n  File x\nTypeError: bad thing\n")
    assert info["error"] == "TypeError: bad thing"


def test_runs_lists_finished_run_with_config(tmp_path):
    make_run(tmp_path, "r1", LOG)
    log = tmp_path / "fflab-runs" / "r1" / "fflab-train.log"
    old = log.stat().st_mtime - 3600                       # finished an hour ago (a fresh log means still working)
    os.utime(log, (old, old))
    runs = ds.DriveStudioRuns(tmp_path, venv=tmp_path / "venv").runs()
    assert [r["name"] for r in runs] == ["r1"]
    r = runs[0]
    assert r["state"] == "succeeded"
    assert (r["method"], r["dataset"], r["steps_cfg"]) == ("configs/streetgs_ut.yaml", "kitti360/ncore_3cams", "2000")
    assert r["images"] == ["drivestudio/r1/images/step_2000_rgbs.png"]

    os.utime(log, None)                                     # log written just now: still rendering videos
    assert ds.DriveStudioRuns(tmp_path, venv=tmp_path / "venv").runs()[0]["state"] == "running"


def test_video_preview_waits_for_complete_mp4(tmp_path):
    make_run(tmp_path, "r1", LOG)
    vdir = tmp_path / "fflab-runs" / "r1" / "drivestudio" / "r1" / "videos"
    vdir.mkdir(parents=True)
    (vdir / "half.mp4").write_bytes(bytes.fromhex("0000002066747970") + b"isom" + bytes(20) + b"mdat" + bytes([1]) * 200)
    runs = ds.DriveStudioRuns(tmp_path, venv=tmp_path / "venv")
    assert not runs.video_complete(vdir / "half.mp4")
    with pytest.raises(ValueError, match="đang được ghi"):
        runs.preview("r1", "drivestudio/r1/videos/half.mp4")
    item = next(v for v in runs.videos() if v["id"].endswith("half.mp4"))
    assert item["ready"] is False and "/preview/" in item["url"] and "/files/" in item["download"]


def test_file_access_stays_inside_the_run(tmp_path):
    make_run(tmp_path, "r1", LOG)
    (tmp_path / "secret.png").write_bytes(b"x")
    runs = ds.DriveStudioRuns(tmp_path, venv=tmp_path / "venv")
    assert runs.file("r1", "drivestudio/r1/images/step_2000_rgbs.png").name == "step_2000_rgbs.png"
    for bad in ("../../secret.png", "drivestudio/r1/../../../secret.png", "fflab-train.py"):
        with pytest.raises(ValueError):
            runs.file("r1", bad)
    with pytest.raises(ValueError):
        runs.file("../x", "a.png")


def test_train_validates_and_launches_fflab(tmp_path, monkeypatch):
    (tmp_path / "configs" / "datasets" / "kitti360").mkdir(parents=True)
    (tmp_path / "configs" / "streetgs_ut.yaml").write_text("x")
    (tmp_path / "configs" / "datasets" / "kitti360" / "ncore_3cams.yaml").write_text("x")
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").write_text("")
    calls = []

    class FakePopen:
        def __init__(self, argv, **kw):
            calls.append((argv, kw))

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(ds.shutil, "which", lambda _: "/usr/bin/node")
    runs = ds.DriveStudioRuns(tmp_path, venv=tmp_path / "venv")
    base = {"name": "k1", "method": "configs/streetgs_ut.yaml", "dataset": "kitti360/ncore_3cams",
            "render_mode": "ut", "fastgs": True, "steps": 30000, "gpu": 2}
    runs.train(base)
    argv, kw = calls[0]
    assert argv[2:5] == ["train", "--backend", "drivestudio"]
    assert ["--render-mode", "ut"] == argv[argv.index("--render-mode"):argv.index("--render-mode") + 2]
    assert argv[argv.index("--fastgs") + 1] == "true" and kw["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert kw["shell"] is False
    with pytest.raises(ValueError, match="already exists"):
        runs.train(base)
    for bad in ({"method": "../evil.yaml"}, {"dataset": "nope/x"}, {"render_mode": "fast"}, {"name": "../x"}):
        with pytest.raises(ValueError):
            runs.train({**base, "name": "k2", **bad})


def test_api_lists_runs(tmp_path, monkeypatch):
    make_run(tmp_path, "r1", LOG)
    monkeypatch.setattr("fisheye_fusion.server.WORKSPACE", tmp_path)
    from fisheye_fusion.server import create_app
    client = TestClient(create_app())
    data = client.get("/api/drivestudio").json()
    assert data["runs"][0]["name"] == "r1" and data["runs"][0]["metrics"]["Full Image PSNR"] == 17.9769
    assert client.get("/api/drivestudio/runs/r1/files/drivestudio/r1/images/step_2000_rgbs.png").status_code == 200
    assert client.get("/api/drivestudio/runs/r1/files/..%2F..%2Fx.png").status_code in (400, 404)
    assert client.get("/api/drivestudio/runs/missing").status_code == 404


def test_viewer_starts_on_localhost_with_the_final_checkpoint(tmp_path, monkeypatch):
    root = make_run(tmp_path, "r1", LOG)
    (root / "drivestudio" / "r1" / "checkpoint_15000.pth").write_bytes(b"x")
    (root / "drivestudio" / "r1" / "checkpoint_final.pth").write_bytes(b"x")
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").write_text("")
    calls = []

    class FakePopen:
        returncode = None

        def __init__(self, argv, **kw):
            calls.append((argv, kw))

        def poll(self):
            return None

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    runs = ds.DriveStudioRuns(tmp_path, venv=tmp_path / "venv")
    st = runs.viewer_start("r1", gpu=0)
    argv, kw = calls[0]
    assert argv[1] == "tools/viewer.py" and argv[argv.index("--ckpt") + 1].endswith("checkpoint_final.pth")
    assert kw["env"]["DRIVESTUDIO_VIEWER_HOST"] == "127.0.0.1" and kw["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert st["run"] == "r1" and st["state"] in ("loading", "ready")
    assert runs.viewer_stop()["state"] == "stopped"
    with pytest.raises(ValueError, match="no checkpoint"):
        make_run(tmp_path, "r2", LOG)
        runs.viewer_start("r2")


def _write_tiny_mp4(path: Path, color: tuple[int, int, int]):
    import cv2, numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(4):
        fr = np.full((48, 64, 3), color, dtype=np.uint8)
        fr[:, i * 8:(i + 1) * 8] = 255
        w.write(fr)
    w.release()


def test_metrics_and_video_compare_across_methods(tmp_path, monkeypatch):
    import json, cv2
    ext_log = LOG + (
        b"I20261117 08:00:00 root video_utils.py:85] \t  Vehicle-Only PSNR: 21.4000\n"
        b"I20261117 08:00:00 root video_utils.py:87] \t       Depth RMSE: 1.8500\n"
        b"I20261117 08:00:00 root video_utils.py:88] \t     Depth AbsRel: 0.0920\n"
        b"I20261117 08:00:00 root video_utils.py:89] \t     Depth Delta1: 0.8910\n"
        b"I20261117 08:00:00 root video_utils.py:90] \t       Render FPS: 42.5000\n"
        b"I20261117 08:00:00 root video_utils.py:91] \t    Peak VRAM GiB: 6.2500\n"
    )
    r1 = make_run(tmp_path, "ds_ut", ext_log)
    _write_tiny_mp4(r1 / "drivestudio" / "ds_ut" / "videos" / "test_set_2000.mp4", (40, 120, 200))

    g_dir = tmp_path / "fflab-runs" / "gut_road1"
    g_dir.mkdir(parents=True)
    (g_dir / "metrics.json").write_text(json.dumps({
        "psnr_per_camera": {"image_00": 18.2, "image_01": 18.2, "image_02": 16.0, "image_03": 16.0},
        "gaussians": 950000, "steps": 10000, "train_seconds": 215,
        "config": {"sequence": "2013_05_28_drive_0003_sync"},
    }))
    (g_dir / "train.log").write_text("step 10000/10000 loss=0.05 peak_vram=4.50GiB\n")
    _write_tiny_mp4(g_dir / "fly.mp4", (200, 80, 40))

    monkeypatch.setattr("fisheye_fusion.server.WORKSPACE", tmp_path)
    from fisheye_fusion.server import create_app
    client = TestClient(create_app())
    ov = client.get("/api/drivestudio").json()
    by_name = {r["name"]: r for r in ov["runs"]}
    assert by_name["ds_ut"]["metrics"]["Depth RMSE"] == 1.85
    assert by_name["ds_ut"]["metrics"]["Depth AbsRel"] == 0.092
    assert by_name["ds_ut"]["metrics"]["Depth Delta1"] == 0.891
    assert by_name["ds_ut"]["metrics"]["Render FPS"] == 42.5
    assert by_name["gut_road1"]["backend"] == "3dgut-gsplat"
    assert by_name["gut_road1"]["metrics"]["Pinhole PSNR"] == pytest.approx(18.2)
    assert by_name["gut_road1"]["metrics"]["Fisheye PSNR"] == pytest.approx(16.0)
    assert by_name["gut_road1"]["metrics"]["Peak VRAM GiB"] == pytest.approx(4.5)

    vids = ov["videos"]
    assert len(vids) == 2
    comp = client.post("/api/videos/compare", json={"video_a": vids[0]["id"], "video_b": vids[1]["id"], "layout": "hstack"})
    assert comp.status_code == 200
    mp4_resp = client.get(comp.json()["url"])
    assert mp4_resp.status_code == 200 and len(mp4_resp.content) > 200
    out_mp4 = tmp_path / comp.json()["url"].replace("/api/", "")
    cap = cv2.VideoCapture(str(out_mp4))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    cap.release()
