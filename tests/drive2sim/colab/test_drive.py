"""Test lấy dữ liệu gốc từ Google Drive qua Colab (``drive2sim/colab/drive.py`` và route).

VM giả chạy thật các đoạn code gửi lên (``exec``) trên một thư mục tạm đóng vai ``MyDrive``, nên
test kiểm tra cả đoạn code chạy trên Colab lẫn phần ghép file trên máy.
"""

import contextlib
import io
import shutil
import subprocess
import threading
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


class GatedVM(FakeVM):
    """VM giả dừng ở khối tải đầu tiên tới khi test mở cổng (để huỷ giữa chừng)."""

    started = threading.Event()
    gate = threading.Event()

    def download(self, session: str, remote: str, local: Path) -> None:
        """Chờ cổng mở rồi chép như ``FakeVM``."""
        GatedVM.started.set()
        GatedVM.gate.wait(10)
        super().download(session, remote, local)


def test_drive_pull_cancel_resume_and_delete(
    my_drive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drive2sim.studio.api.routes import colab as colab_routes

    monkeypatch.setattr(colab_routes, "ColabClient", GatedVM)
    workdir = Workdir(tmp_path / "work")
    app = create_app(workdir, registry=StageRegistry(), frontend_dir=None)
    jobs = app.state.studio.jobs
    with TestClient(app, base_url="http://127.0.0.1") as client:
        body = {"path": "kitti", "name": "kitti"}
        job = client.post("/api/colab/drive/pull", json=body).json()
        assert job["cancellable"] and job["target"] == "kitti"
        # cùng thư mục đang tải: trả việc cũ, không tải hai lần
        assert client.post("/api/colab/drive/pull", json=body).json()["job_id"] == job["job_id"]
        assert GatedVM.started.wait(10)
        assert client.post(f"/api/jobs/{job['job_id']}/cancel").status_code == 200
        GatedVM.gate.set()
        jobs.wait_idle()
        state = client.get(f"/api/jobs/{job['job_id']}").json()
        assert state["status"] == "cancelled" and state["error"] is None
        [item] = client.get("/api/sources/uploads").json()["items"]
        assert item["origin"] == "drive" and item["partial"] == 1
        resumed = client.post("/api/colab/drive/pull/kitti/resume")
        assert resumed.status_code == 202
        jobs.wait_idle()
        assert client.get(f"/api/jobs/{resumed.json()['job_id']}").json()["status"] == "succeeded"
        [item] = client.get("/api/sources/uploads").json()["items"]
        assert item["partial"] == 0 and item["files"] == 2
        assert client.post(f"/api/jobs/{job['job_id']}/retry").status_code == 202
        jobs.wait_idle()
        assert client.post("/api/colab/drive/pull/other/resume").status_code == 400
        assert client.post("/api/jobs/nope/cancel").status_code == 404
        assert client.delete("/api/sources/uploads/kitti").json() == {"removed": ["uploads/kitti"]}
        assert not (workdir.uploads_dir / ".origins" / "kitti.json").exists()
        assert client.get("/api/sources/uploads").json()["items"] == []


def make_kitti360(root: Path) -> None:
    """Gốc KITTI-360 giả trên Drive: calib, pose, hai chuỗi, kèm thư mục không liên quan."""
    (root / "calibration").mkdir(parents=True)
    (root / "calibration" / "calib_cam_to_pose.txt").write_text("calib")
    (root / "data_poses.zip").write_bytes(b"z" * 10)
    (root / "ImageSets-rosa-6class").mkdir()
    (root / "ImageSets-rosa-6class" / "kitti360_rosa_infos_val.pkl").write_bytes(b"m" * 3)
    for seq in ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync"):
        for cam in range(4):
            (root / "data_2d_raw").mkdir(exist_ok=True)
            (root / "data_2d_raw" / f"{seq}_image_0{cam}.tar").write_bytes(b"i" * 100)
        (root / "data_3d_raw").mkdir(exist_ok=True)
        (root / "data_3d_raw" / f"{seq}.tar").write_bytes(b"l" * 50)
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "big.pth").write_bytes(b"c" * 70)


def test_inspect_detects_dataset_and_subfolders(my_drive: Path) -> None:
    make_kitti360(my_drive / "datasets" / "k360")
    inner = drive.inspect(FakeVM(), "s", "datasets/k360")
    report = inner["report"]
    assert report["dataset"] == "kitti360" and report["has_lidar"] is True
    assert report["versions"][""] == ["2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync"]
    assert report["path"] == "datasets/k360" and report["inventory"] == []
    assert inner["sequence_bytes"]["2013_05_28_drive_0005_sync"] == 4 * 100 + 50
    assert "calibration/" in inner["pull_sets"]["2013_05_28_drive_0005_sync"]
    # thư mục cha: không phải dataset, nhưng báo thư mục con có dataset bằng đường dẫn trên Drive
    outer = drive.inspect(FakeVM(), "s", "")
    assert outer["report"]["dataset"] is None
    child = {"path": "datasets/k360", "dataset": "kitti360", "relation": "child"}
    assert child in outer["report"]["candidates"]
    assert all("d2s_drive_" not in note for note in outer["report"]["notes"])
    assert drive.inspect(FakeVM(), "s", "missing")["report"] is None


def test_inspect_reads_nuscenes_tables(my_drive: Path) -> None:
    root = my_drive / "nus"
    (root / "v1.0-mini").mkdir(parents=True)
    (root / "v1.0-mini" / "scene.json").write_text('[{"name": "scene-0061"}]')
    (root / "v1.0-mini" / "sensor.json").write_text(
        '[{"channel": "CAM_FRONT", "modality": "camera"},'
        ' {"channel": "LIDAR_TOP", "modality": "lidar"}]'
    )
    (root / "v1.0-mini" / "sample_annotation.json").write_text('[{"token": "a"}]')
    report = drive.inspect(FakeVM(), "s", "nus")["report"]
    assert report["dataset"] == "nuscenes" and report["versions"] == {"v1.0-mini": ["scene-0061"]}
    assert report["cameras"] == ["CAM_FRONT"] and report["has_cuboids"] is True
    # thiếu sweeps/; lỗi ghi đường dẫn trên Drive, không phải thư mục tạm
    assert report["errors"] and "MyDrive/nus" in report["errors"][0]


def test_pull_one_kitti360_sequence(my_drive: Path, tmp_path: Path) -> None:
    make_kitti360(my_drive / "k360")
    sets = drive.inspect(FakeVM(), "s", "k360")["pull_sets"]
    target = tmp_path / "work" / "uploads" / "k360"
    result = drive.pull_folder(
        FakeVM(), "s", "k360", target, include=sets["2013_05_28_drive_0005_sync"]
    )
    got = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert got == [
        "ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl",
        "calibration/calib_cam_to_pose.txt",
        "data_2d_raw/2013_05_28_drive_0005_sync_image_00.tar",
        "data_2d_raw/2013_05_28_drive_0005_sync_image_01.tar",
        "data_2d_raw/2013_05_28_drive_0005_sync_image_02.tar",
        "data_2d_raw/2013_05_28_drive_0005_sync_image_03.tar",
        "data_3d_raw/2013_05_28_drive_0005_sync.tar",
        "data_poses.zip",
    ]
    assert result["bytes"] == 5 + 3 + 10 + 4 * 100 + 50  # ROSA: Instant NuRec cần
    picked = drive.pull_folder(FakeVM(), "s", "k360", tmp_path / "p", include=["checkpoints/"])
    assert picked["files"] == 1
    with pytest.raises(ValueError, match="không file nào"):
        drive.pull_folder(FakeVM(), "s", "k360", tmp_path / "q", include=["nothing/"])
    with pytest.raises(ValueError):
        drive.pull_folder(FakeVM(), "s", "k360", tmp_path / "q", include=["../x"])


def test_inspect_route(my_drive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from drive2sim.studio.api.routes import colab as colab_routes

    make_kitti360(my_drive / "k360")
    monkeypatch.setattr(colab_routes, "ColabClient", FakeVM)
    app = create_app(Workdir(tmp_path / "work"), registry=StageRegistry(), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        found = client.get("/api/colab/drive/inspect", params={"path": "k360"})
        assert found.status_code == 200 and found.json()["report"]["dataset"] == "kitti360"
        bad = client.post(
            "/api/colab/drive/pull", json={"path": "k360", "name": "k", "include": ["../etc"]}
        )
        assert bad.status_code == 400
        job = client.post(
            "/api/colab/drive/pull",
            json={"path": "k360", "name": "k", "include": ["checkpoints/"]},
        )
        assert job.status_code == 202
        app.state.studio.jobs.wait_idle()
        done = client.get(f"/api/jobs/{job.json()['job_id']}").json()
        assert done["status"] == "succeeded" and done["progress"]["total"] == 70
