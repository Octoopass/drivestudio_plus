"""Colab GPU training pipeline end to end against a fake `colab` CLI that maps /content to a temp folder."""
import json
import sys
import time

import pytest

import fisheye_fusion.colab_train as ct
from fisheye_fusion.colab import ColabHelper
from fisheye_fusion.drivestudio import DriveStudioRuns

FAKE_COLAB = r'''#!__PY__
import os, shutil, subprocess, sys
ROOT = os.environ["FAKE_COLAB_ROOT"]
args = sys.argv[1:]
cmd, rest, pos = args[0], args[1:], []
i = 0
while i < len(rest):
    if rest[i] in ("-s", "--session", "--timeout", "--gpu"):
        i += 2
        continue
    pos.append(rest[i]); i += 1
m = lambda p: ROOT + p if p.startswith("/content") else p
if cmd == "sessions":
    print("[gpu1] m-abc | Hardware: T4 | Shape: Standard")
    print("[cpu1] m-def | Hardware: CPU | Shape: Standard")
elif cmd == "exec":
    code = sys.stdin.read()
    if "nvidia-smi" in code:
        print("Tesla T4, 15360 MiB"); sys.exit(0)
    code = code.replace("'/content", "'" + ROOT + "/content").replace('"/content', '"' + ROOT + "/content")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)
elif cmd == "upload":
    os.makedirs(os.path.dirname(m(pos[1])), exist_ok=True); shutil.copy(pos[0], m(pos[1])); print("Uploaded")
elif cmd == "download":
    shutil.copy(m(pos[0]), pos[1]); print("Downloaded")
'''

FAKE_RUN = r'''import os, pathlib, time
out = pathlib.Path(os.environ["DRIVESTUDIO_OUT"]) / "drivestudio" / os.environ["DRIVESTUDIO_RUN_NAME"]
assert pathlib.Path(os.environ["DRIVESTUDIO_OVERRIDES"].split("=", 1)[1], "000", "metadata.json").exists(), "data missing"
print("[drivestudio] python tools/train.py --config_file " + os.environ["DRIVESTUDIO_CONFIG"] + " dataset=" + os.environ["DRIVESTUDIO_DATASET"], flush=True)
for s in range(0, 101, 50):
    print(f"[{s}/101] eta: 0:00:01 elapsed: 0:00:0{s // 50}", flush=True)
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "images" / "a.png").write_bytes(b"png")
(out / "checkpoint_final.pth").write_bytes(b"x" * 10)
(out / "checkpoint_00050.pth").write_bytes(b"x" * 10)
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "configs" / "datasets" / "kitti360").mkdir(parents=True)
    (ws / "configs" / "streetgs_ut.yaml").write_text("x: 1\n")
    (ws / "configs" / "datasets" / "kitti360" / "ncore_3cams.yaml").write_text("data:\n  data_root: data/kitti360/processed # root\n")
    scene = ws / "data" / "kitti360" / "processed" / "000"
    (scene / "images").mkdir(parents=True)
    (scene / "metadata.json").write_text("{}")
    (scene / "images" / "0.jpg").write_bytes(b"j" * 1000)
    (ws / "tools").mkdir()
    (ws / "tools" / "drivestudio_run.py").write_text(FAKE_RUN)
    remote = tmp_path / "remote"
    (remote / "content" / "fflab").mkdir(parents=True)
    (remote / "content" / "fflab" / ".setup_ok").write_text("tag")       # skip the real installer
    fake = tmp_path / "colab"
    fake.write_text(FAKE_COLAB.replace("__PY__", sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(remote))
    monkeypatch.setattr(ct, "POLL", 0.2)
    colab = ColabHelper(lambda: str(fake))
    ds = DriveStudioRuns(ws)
    trainer = ct.ColabTrainer(ws, colab, ds)
    yield ws, remote, trainer, ds
    trainer.shutdown()


def wait(trainer, name, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = trainer.status(name)
        if s["state"] != "running":
            return s
        time.sleep(0.3)
    raise AssertionError(trainer.status(name))


CFG = {"method": "configs/streetgs_ut.yaml", "dataset": "kitti360/ncore_3cams", "render_mode": "ut",
       "fastgs": None, "steps": 100, "data_root": "", "session": "gpu1", "colab_gpu": "T4"}


def test_full_run(env):
    ws, remote, trainer, ds = env
    trainer.start({**CFG, "name": "r1"})
    s = wait(trainer, "r1")
    assert s["state"] == "succeeded", s
    assert all(v["state"] == "done" for v in s["stages"].values())
    run = ws / "fflab-runs" / "r1"
    assert (run / "drivestudio" / "r1" / "checkpoint_final.pth").exists()
    assert not (run / "drivestudio" / "r1" / "checkpoint_00050.pth").exists()
    info = ds.run("r1")
    assert info["method"] == "configs/streetgs_ut.yaml" and info["step"] == 100 and info["colab"]["session"] == "gpu1"
    assert (remote / "content/fflab/repo/data/kitti360/processed/000/.fflab_data").exists()
    # a second run in the same session reuses the uploaded code and data
    trainer.start({**CFG, "name": "r2"})
    s2 = wait(trainer, "r2")
    assert s2["state"] == "succeeded"
    assert "bỏ qua" in s2["stages"]["code"]["note"] and "đã có" in s2["stages"]["data"]["note"]
    assert [j["name"] for j in trainer.jobs()["jobs"]] == ["r2", "r1"]


def test_cpu_session_rejected(env):
    _, _, trainer, _ = env
    trainer.start({**CFG, "name": "c1", "session": "cpu1"})
    s = wait(trainer, "c1")
    assert s["state"] == "failed" and "CPU" in s["error"] and s["stages"]["session"]["state"] == "error"


def test_validation(env):
    ws, _, trainer, _ = env
    with pytest.raises(ValueError):
        trainer.start({**CFG, "name": "v1", "method": "configs/nope.yaml"})
    with pytest.raises(ValueError):
        trainer.start({**CFG, "name": "v2", "data_root": "data/kitti360/other"})
    assert not (ws / "fflab-runs" / "v1").exists()
    assert ct.dataset_data_root(ws, "kitti360/ncore_3cams") == "data/kitti360/processed"
    assert json.dumps(trainer.jobs())
