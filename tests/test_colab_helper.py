"""Guided Colab / Drive helper: sessions, consent-URL mount flow, Drive browsing (fake colab CLI)."""
import sys
import time

import pytest
from fastapi.testclient import TestClient

from fisheye_fusion.colab import ColabHelper

SEQ = "2013_05_28_drive_0003_sync"
FAKE = r'''#!__PY__
import json, os, subprocess, sys
state = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.json")
sessions = json.load(open(state)) if os.path.exists(state) else ["fisheye-lab"]
args = sys.argv[1:]
cmd = args[0]
name = args[args.index("-s") + 1] if "-s" in args else None
drive = os.environ["FAKE_DRIVE"]
if cmd == "sessions":
    print("\n".join(f"[{s}] m-s-{s} | Hardware: CPU | Shape: Standard" for s in sessions))
elif cmd == "new":
    sessions.append(name); json.dump(sessions, open(state, "w")); print("[colab] Session READY.")
elif cmd == "url":
    print(f"https://colab.research.google.com/notebooks/empty.ipynb?dbu=x#{name}")
elif cmd == "exec":
    r = subprocess.run([sys.executable, "-"], input=sys.stdin.read(), capture_output=True, text=True)
    sys.stdout.write(r.stdout + r.stderr); sys.exit(r.returncode)
elif cmd == "drivemount":
    print("[colab] REQUIRED: Google Drive Authorization needed.\nPlease visit:\n\nhttps://accounts.google.com/o/consent?x=1\n", flush=True)
    sys.stdin.readline()
    os.makedirs(os.path.join(drive, "MyDrive"), exist_ok=True)
    print("[colab] Credentials propagated. Resuming mount...")
'''


@pytest.fixture
def helper(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "colab"
    fake.parent.mkdir()
    fake.write_text(FAKE.replace("__PY__", sys.executable))
    fake.chmod(0o755)
    drive = tmp_path / "drive"
    drive.mkdir()
    monkeypatch.setenv("FAKE_DRIVE", str(drive))
    h = ColabHelper(lambda: str(fake))
    h.drive_root = str(drive)
    yield h
    h.shutdown()


def make_kitti(root):
    (root / "data_2d_raw").mkdir(parents=True)
    (root / "data_3d_raw").mkdir()
    (root / "data_poses.zip").write_bytes(b"p")
    for i in range(4):
        (root / "data_2d_raw" / f"{SEQ}_image_0{i}.tar").write_bytes(b"x" * 10)
    (root / "data_3d_raw" / f"{SEQ}.tar").write_bytes(b"y" * 5)


def test_sessions_create_and_url(helper):
    assert [s["name"] for s in helper.sessions()["sessions"]] == ["fisheye-lab"]
    assert helper.sessions()["sessions"][0]["hardware"] == "CPU"
    names = [s["name"] for s in helper.create("lab2")["sessions"]]
    assert names == ["fisheye-lab", "lab2"]
    with pytest.raises(ValueError):
        helper.create("lab2")
    with pytest.raises(ValueError):
        helper.create("bad name;rm")
    assert helper.url("lab2")["url"].startswith("https://colab.research.google.com/")


def test_mount_shows_consent_url_then_finishes_after_confirm(helper):
    assert helper.drive_status("fisheye-lab")["mounted"] is False
    view = helper.mount("fisheye-lab")
    assert view["state"] == "needs_consent"
    assert view["consent_url"] == "https://accounts.google.com/o/consent?x=1"
    helper.confirm_mount("fisheye-lab")
    for _ in range(40):
        if helper.mount_status("fisheye-lab")["state"] == "succeeded":
            break
        time.sleep(0.1)
    assert helper.mount_status("fisheye-lab")["state"] == "succeeded"
    assert helper.drive_status("fisheye-lab")["mounted"] is True


def test_browse_and_find_detect_kitti_roots(helper):
    my = helper.drive_root + "/MyDrive"
    from pathlib import Path
    make_kitti(Path(my) / "datasets" / "kitti360_full")
    (Path(my) / "Photos").mkdir()
    top = helper.browse("fisheye-lab")
    assert top["mounted"] and top["dirs"] == ["datasets", "Photos"] and top["kitti"] is None and top["dataset"] is None
    root = helper.browse("fisheye-lab", f"{my}/datasets/kitti360_full")
    assert root["kitti"]["sequences"][SEQ] == {"cameras": 4, "lidar": True, "bytes": 45}
    assert root["kitti"]["common"]["data_poses.zip"] is True
    (Path(my) / "waymo_raw").mkdir()
    (Path(my) / "waymo_raw" / "segment-1.tfrecord").write_bytes(b"w")
    found = {r["path"]: r for r in helper.find("fisheye-lab")["roots"]}
    assert set(found) == {f"{my}/datasets/kitti360_full", f"{my}/waymo_raw"}
    assert SEQ in found[f"{my}/datasets/kitti360_full"]["kitti"]["sequences"]
    assert found[f"{my}/waymo_raw"]["dataset"] == "waymo"
    assert helper.browse("fisheye-lab", f"{my}/waymo_raw")["dataset"] == "waymo"
    assert helper.browse("fisheye-lab", f"{my}/datasets")["dataset"] is None


@pytest.mark.parametrize("path", ["/etc", "/content/drive/../../etc", "/content/drive/My" + chr(92) + "x"])
def test_browse_refuses_paths_outside_drive(helper, path):
    with pytest.raises(ValueError):
        helper.browse("fisheye-lab", path)


def test_colab_routes(tmp_path, monkeypatch):
    monkeypatch.setattr("fisheye_fusion.server.WORKSPACE", tmp_path)
    from fisheye_fusion.server import create_app
    client = TestClient(create_app())
    assert client.get("/api/colab/browse", params={"session": "x", "path": "/etc"}).status_code == 400
    assert client.post("/api/colab/sessions", json={"session": "bad name"}).status_code == 422
