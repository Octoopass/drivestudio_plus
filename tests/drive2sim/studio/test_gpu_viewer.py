"""Kiểm tra ``drive2sim/studio/api/gpu_viewer.py`` trên dữ liệu mẫu, không cần GPU.

Viewer thật cần PyTorch và GPU, nên test thay trình thông dịch bằng một chương trình giả: nó
đọc ``--port`` rồi nghe ở cổng đó, giống viewer đã nạp xong. Test kiểm tra:
- dựng lại cảnh ảo và config trỏ vào thư mục viewer, kèm mask và SMPL từ cache;
- từ chối khi máy chưa đủ điều kiện hoặc run chưa có checkpoint;
- bật, báo ``running`` kèm địa chỉ trên 127.0.0.1, rồi tắt.
"""

import sys
import time
from pathlib import Path

import pytest
import yaml

from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine.workdir import Workdir
from drive2sim.scripts import make_demo
from drive2sim.studio.api import gpu_viewer

RUN_ID = "20261112-081000-omnire-a1b2"
FAKE_VIEWER = """\
import socket, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
host = sys.argv[sys.argv.index("--host") + 1]
server = socket.socket()
server.bind((host, port))
server.listen()
time.sleep(60)
"""


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Workdir:
    """Thư mục mẫu có checkpoint và config giả trong stage train của ``RUN_ID``."""
    root = tmp_path_factory.mktemp("gpu") / "demo"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(make_demo, "write_video", lambda path, seconds=3: False)
        assert make_demo.main([str(root)]) == 0
    work = Workdir(root)
    run_dir = work.find_run(RUN_ID)
    # Cache mẫu chỉ có payload giả; thêm thư mục mask và SMPL như stage prep thật ghi ra.
    for stage in files.load(run_dir / "d2s" / "run.json", RunRecord).stages:
        name = {"prep.masks_segformer": "sky_masks", "prep.smpl": "humanpose"}.get(stage.stage_id)
        if name and stage.cache_key:
            (work.cache_dir / stage.cache_key / name).mkdir()
    train = run_dir / "d2s" / "stages" / "train.drivestudio"
    (train / "checkpoint_final.pth").write_bytes(b"x")
    config = {"data": {"data_root": "/kaggle/working/view", "scene_idx": 0}, "trainer": {}}
    (train / "config.yaml").write_text(yaml.safe_dump(config))
    return work


def ready(run_dir: Path) -> dict:
    """Thay ``render_readiness``: máy đủ điều kiện."""
    return {"ready": True, "checks": []}


def test_prepare_rebuilds_view_and_config(demo: Workdir) -> None:
    checkpoint, config, out = gpu_viewer.prepare(demo, demo.find_run(RUN_ID))
    assert checkpoint.name == "checkpoint_final.pth"
    data = yaml.safe_load(config.read_text())["data"]
    assert data["data_root"] == str(out / "scene_view") and data["scene_idx"] == 0
    view = out / "scene_view" / "000"
    assert (view / "images").is_dir()
    assert (view / "sky_masks").is_symlink() and (view / "humanpose").is_symlink()


def test_start_refuses_when_not_ready(demo: Workdir, monkeypatch: pytest.MonkeyPatch) -> None:
    viewers = gpu_viewer.GpuViewers(demo)
    with pytest.raises(gpu_viewer.GpuViewerError, match="còn thiếu: pytorch"):
        monkeypatch.setattr(
            gpu_viewer,
            "render_readiness",
            lambda run_dir: {"ready": False, "checks": [{"id": "pytorch", "ok": False}]},
        )
        viewers.start(RUN_ID)
    monkeypatch.setattr(gpu_viewer, "render_readiness", ready)
    with pytest.raises(gpu_viewer.GpuViewerError, match="chưa có checkpoint"):
        viewers.start("20261112-143000-streetgs-e5f6")  # run lỗi giữa train
    assert viewers.status()["state"] == "stopped"


def test_start_status_stop(demo: Workdir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "fake_viewer.py"
    script.write_text(FAKE_VIEWER)
    fake_python = tmp_path / "python"
    fake_python.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    fake_python.chmod(0o755)
    monkeypatch.setattr(gpu_viewer, "render_readiness", ready)
    viewers = gpu_viewer.GpuViewers(demo, python=str(fake_python))
    viewers.start(RUN_ID)
    deadline = time.time() + 10
    while viewers.status()["state"] == "starting" and time.time() < deadline:
        time.sleep(0.1)
    status = viewers.status()
    assert status["state"] == "running" and status["run_id"] == RUN_ID
    assert status["url"].startswith("http://127.0.0.1:")
    viewers.stop()
    assert viewers.status()["state"] == "stopped"
