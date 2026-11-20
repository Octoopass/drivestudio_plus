"""Test lấy dữ liệu gốc từ Google Drive qua Colab (``drive2sim/colab/drive.py`` và route).

VM giả chạy thật các đoạn code gửi lên (``exec``) trên một thư mục tạm đóng vai ``MyDrive``, nên
test kiểm tra cả đoạn code chạy trên Colab lẫn phần ghép file trên máy.
"""

import contextlib
import io
import shutil
import subprocess
from collections import namedtuple
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from drive2sim.colab import drive
from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.app import create_app


class FakeVM:
    """Client Colab giả: ``exec_python`` chạy code ngay trên máy, ``download`` chép file."""

    def __init__(self) -> None:
        self.downloads = 0

    def exec_python(self, session: str, code: str, timeout_s: float = 60) -> str:
        """Chạy code ngay trên máy, trả stdout."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, "<vm>", "exec"), {})  # noqa: S102
        return out.getvalue()

    def download(self, session: str, remote: str, local: Path) -> None:
        """Chép file như tải từ VM."""
        self.downloads += 1
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote, local)


@pytest.fixture
def my_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "vm" / "MyDrive"
    (root / "kitti" / "seq").mkdir(parents=True)
    (root / "kitti" / "seq" / "a.bin").write_bytes(bytes(range(256)) * 40)
    (root / "kitti" / "poses.txt").write_text("1 2 3\n")
    (root / "kitti" / ".DS_Store").write_text("x")
    monkeypatch.setattr(drive, "MY_DRIVE", str(root))
    monkeypatch.setattr(drive, "VM_TMP", str(tmp_path / "vm" / "tmp"))
    monkeypatch.setattr(drive, "CHUNK_BYTES", 4096)
    return root


def test_safe_drive_path() -> None:
    assert drive.safe_drive_path("/a\\b/./c/") == "a/b/c"
    assert drive.safe_drive_path("") == ""
    with pytest.raises(ValueError):
        drive.safe_drive_path("a/../../etc")


def test_browse(my_drive: Path) -> None:
    root = drive.browse(FakeVM(), "s", "")
    assert root["mounted"] and root["dirs"] == ["kitti"] and root["files"] == []
    inner = drive.browse(FakeVM(), "s", "kitti")
    assert inner["dirs"] == ["seq"] and ["poses.txt", 6] in inner["files"]
    assert drive.browse(FakeVM(), "s", "missing")["exists"] is False


def test_pull_folder_resumes(my_drive: Path, tmp_path: Path) -> None:
    target = tmp_path / "work" / "uploads" / "kitti"
    seen = []
    vm = FakeVM()
    result = drive.pull_folder(vm, "s", "kitti", target, progress=lambda d, t, f: seen.append(d))
    data = (my_drive / "kitti" / "seq" / "a.bin").read_bytes()
    assert (target / "seq" / "a.bin").read_bytes() == data
    assert (target / "poses.txt").read_text() == "1 2 3\n"
    assert not (target / ".DS_Store").exists() and not (target / ".d2s_drive_tmp").exists()
    assert result["files"] == 2 and result["bytes"] == len(data) + 6
    assert seen[-1] == result["bytes"]
    assert vm.downloads == 4  # 10240 byte theo phần 4096 là 3 phần, cộng poses.txt
    # tải lại: file đủ thì bỏ qua; file dở đi tiếp từ phần đã có
    (target / "seq" / "a.bin").unlink()
    (target / "seq" / "a.bin.part").write_bytes(data[:4096])
    again = FakeVM()
    result = drive.pull_folder(again, "s", "kitti", target)
    assert result["skipped"] == 1 and again.downloads == 2
    assert (target / "seq" / "a.bin").read_bytes() == data


def test_pull_refuses_unmounted_and_full_disk(
    my_drive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="không có file"):
        drive.pull_folder(FakeVM(), "s", "missing", tmp_path / "t")
    usage = namedtuple("usage", "total used free")(10, 10, 1024)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage)
    with pytest.raises(ValueError, match="không đủ chỗ"):
        drive.pull_folder(FakeVM(), "s", "kitti", tmp_path / "t")
    monkeypatch.setattr(drive, "MY_DRIVE", str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="chưa mount"):
        drive.pull_folder(FakeVM(), "s", "", tmp_path / "t")


class FakeProcess:
    """Tiến trình ``colab drivemount`` giả: in đường dẫn đồng ý, chờ Enter rồi thoát."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stdout = io.StringIO(
            "[colab] Connecting to https://colab.research.google.com...\n"
            "Please visit https://accounts.google.com/o/oauth2/auth?x=1 to authorize\n"
        )
        self.stdin = io.StringIO()
        self.code = None

    def poll(self) -> Optional[int]:
        """Thoát với mã 0 sau khi nhận Enter."""
        if self.code is None and self.stdin.getvalue():
            self.code = 0
        return self.code

    def wait(self, timeout: float = 0) -> int:
        """Chờ như ``Popen.wait``."""
        if self.poll() is None:
            raise subprocess.TimeoutExpired("colab", timeout)
        return self.code

    def kill(self) -> None:
        """Dừng tiến trình."""
        self.code = -9


def test_mount_flow_consent() -> None:
    flow = drive.MountFlow(popen=FakeProcess)
    assert flow.view()["state"] == "idle"
    with pytest.raises(ValueError):
        flow.start("bad name")
    started = flow.start("d2s-drive", wait_s=2)
    assert started["state"] == "needs_consent"
    assert started["url"].startswith("https://accounts.google.com/")
    assert flow.confirm(wait_s=0)["state"] == "succeeded"
    with pytest.raises(ValueError):
        flow.confirm()


def test_drive_routes_and_job_progress(
    my_drive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drive2sim.studio.api.routes import colab as colab_routes

    monkeypatch.setattr(colab_routes, "ColabClient", FakeVM)
    app = create_app(Workdir(tmp_path / "work"), registry=StageRegistry(), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        listed = client.get("/api/colab/drive/browse", params={"path": "kitti"})
        assert listed.status_code == 200 and listed.json()["dirs"] == ["seq"]
        assert client.get("/api/colab/drive/browse", params={"path": "../x"}).status_code == 400
        bad = client.post("/api/colab/drive/pull", json={"path": "kitti", "name": "a/b"})
        assert bad.status_code == 400
        job = client.post("/api/colab/drive/pull", json={"path": "kitti", "name": "kitti"})
        assert job.status_code == 202 and job.json()["kind"] == "drive_pull"
        app.state.studio.jobs.wait_idle()
        done = client.get(f"/api/jobs/{job.json()['job_id']}").json()
        assert done["status"] == "succeeded", done
        assert done["result"].endswith("uploads/kitti")
        assert done["progress"]["done"] == done["progress"]["total"] == 10246
        assert done["started_at"] is not None
