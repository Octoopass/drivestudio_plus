"""Kiểm tra executor Colab mà không gọi Colab thật.

``FakeColab`` thay ``ColabClient``: "VM" là một thư mục tạm (``REMOTE_ROOT`` trỏ vào đó), mã Python
mà executor gửi lên được chạy thật bằng Python của máy test, upload và download là chép file. Nhờ
vậy các đoạn mã chạy trên VM (ghép phần upload, giải nén, chạy job tách khỏi kết nối, đọc log, gói
output) được chạy thật. Job thật (clone, cài môi trường, ``d2s kaggle run``) được thay bằng một
script giả chỉ dựng thư mục làm việc từ gói job (``materialize``) và ghi kết quả.

File này kiểm tra:
- đẩy lần đầu: tạo phiên, upload cảnh và file SMPL, chạy job; đọc trạng thái và log; tải về và
  ghép như Kaggle; đẩy lần hai trên cùng phiên chạy tiếp từ bản trên VM; run xong thì tắt phiên;
- mất phiên: báo ``lost``, ngừng theo dõi; đẩy lại tạo phiên mới và upload bản run trên máy;
- từ chối khi VM đang chạy job khác, khi thiếu file SMPL, khi chưa đẩy;
- theo dõi log khi kết nối rớt vài lần;
- client đọc đúng output của lệnh ``colab``.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pytest

from drive2sim import paths
from drive2sim.cli.main import main
from drive2sim.colab import client as client_module
from drive2sim.colab import executor
from drive2sim.colab.client import ColabClient, ColabError, ColabSessionLostError
from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, Manifest, RunRecord, Split
from drive2sim.engine.run import D2S_DIR, RUN_FILE
from drive2sim.engine.workdir import Workdir
from tests.drive2sim.samples import MANIFEST, SPLIT

SCENE = MANIFEST["scene_id"]
RUN_ID = "20261113-010203-omnire-ab12"
CACHE_KEY = f"{SCENE}/prep.smpl@0.1.0-0123abcd"
REPO = Path(__file__).resolve().parents[3]
REAL_RUN_CLI = client_module._run_cli  # lấy lúc nạp file, trước khi conftest chặn lệnh colab

# Job giả thay templates/job.sh: cùng tham số, nhưng chỉ dựng thư mục làm việc từ gói job như
# "d2s kaggle run" rồi ghi kết quả. Lần đầu run lỗi; lần sau (cùng VM, hoặc VM mới đã nhận bản run
# upload từ máy) thì xong.
FAKE_JOB = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"; BUNDLE="$5"; SCENE_INPUT="$6"
echo "fake job $4"
PYTHONPATH={REPO} {sys.executable} - "$BUNDLE" "$ROOT/work" "$SCENE_INPUT" <<'PY'
import sys
from pathlib import Path
from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle.bundle import materialize, read_bundle
work = Workdir(Path(sys.argv[2]))
run_dir = materialize(read_bundle(Path(sys.argv[1])), work, Path(sys.argv[3]))
stage = run_dir / "d2s" / "stages" / "prep.smpl"
second = (stage / "first").is_file()
stage.mkdir(parents=True, exist_ok=True)
(stage / ("done" if second else "first")).write_text("x")
assert (work.root / "smpl_models" / "SMPL_NEUTRAL.pkl").is_file()
record = files.load(run_dir / "d2s" / "run.json", RunRecord)
record.status = "succeeded" if second else "failed"
record.stages[0].cache_key = "{CACHE_KEY}"
files.dump(record, run_dir / "d2s" / "run.json")
cache = work.cache_dir / "{CACHE_KEY}"
cache.mkdir(parents=True, exist_ok=True)
(cache / "artifact.json").write_text("{{}}")
print("status", record.status)
sys.exit(0 if second else 1)
PY
"""


class FakeColab(ColabClient):
    """Colab giả: một "VM" là thư mục tạm, mã gửi lên chạy bằng Python của máy test."""

    def __init__(self, vm_root: Path) -> None:
        super().__init__(runner=self._unused)
        self.vm_root = vm_root
        self.sessions_on_server: dict[str, str] = {}
        self.created: list[tuple[str, str]] = []
        self.uploads: list[str] = []
        self.stopped: list[str] = []
        self.drop_exec = 0  # số lần exec tiếp theo bị "rớt kết nối"

    @staticmethod
    def _unused(command: object, timeout_s: float) -> None:
        raise AssertionError("FakeColab không chạy lệnh")

    def _check(self, session: str) -> None:
        if session not in self.sessions_on_server:
            raise ColabSessionLostError(f"phiên Colab {session} đã mất")

    def sessions(self) -> list[dict[str, str]]:
        """Các phiên đang có."""
        return [{"name": n, "hardware": h} for n, h in self.sessions_on_server.items()]

    def create_session(self, session: str, gpu: str) -> None:
        """Tạo phiên: VM mới, trống."""
        self.created.append((session, gpu))
        self.sessions_on_server[session] = gpu
        shutil.rmtree(self.vm_root, ignore_errors=True)
        self.vm_root.mkdir(parents=True)

    def lose(self) -> None:
        """Giả Colab thu hồi VM: phiên và mọi file trên VM mất."""
        self.sessions_on_server.clear()
        shutil.rmtree(self.vm_root, ignore_errors=True)

    def stop(self, session: str) -> None:
        """Tắt phiên."""
        self.stopped.append(session)
        self.sessions_on_server.pop(session, None)

    def exec_python(self, session: str, code: str, timeout_s: float = 60, **_: object) -> str:
        """Chạy mã bằng Python của máy test, trong thư mục của VM."""
        self._check(session)
        if self.drop_exec:
            self.drop_exec -= 1
            raise ColabError("colab exec lỗi: Connection was lost")
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=self.vm_root
        )
        if result.returncode != 0:
            raise ColabError(f"lỗi khi chạy trên VM: {result.stderr[-500:]}")
        return result.stdout

    def upload(self, session: str, local: Path, remote: str) -> None:
        """Chép file; như Colab thật, thư mục đích phải có sẵn."""
        self._check(session)
        if not Path(remote).parent.is_dir():
            raise ColabError("Upload failed: 500 Server Error")
        self.uploads.append(remote)
        shutil.copyfile(local, remote)

    def download(self, session: str, remote: str, local: Path) -> None:
        """Chép file về."""
        self._check(session)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote, local)


def make_workdir(root: Path) -> Workdir:
    workdir = Workdir(root)
    scene_dir = workdir.scene_dir(SCENE)
    files.dump(Manifest.model_validate(MANIFEST), scene_dir / "manifest.yaml")
    (scene_dir / "images").mkdir()
    (scene_dir / "images" / "000_0.jpg").write_bytes(b"jpg" * 1000)
    files.dump(Split.model_validate(SPLIT), workdir.split_path(SCENE, SPLIT["split_id"]))
    run_dir = workdir.run_dir(SCENE, "omnire", RUN_ID)
    record = RunRecord(
        schema_version=SCHEMA_VERSIONS["run"],
        run_id=RUN_ID,
        scene_id=SCENE,
        split_id=SPLIT["split_id"],
        approach={"approach_id": "omnire", "version": "0.1.0", "options": {}},
        level="L1",
        executor="colab",
        status="pending",
        stages=[{"stage_id": "prep.smpl", "status": "pending", "params": {}}],
    )
    files.dump(record, run_dir / D2S_DIR / RUN_FILE)
    return workdir


def wait_job(workdir: Workdir, fake: FakeColab, timeout_s: float = 60) -> dict:
    """Chờ job giả chạy xong trên "VM"."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = executor.job_state(workdir, RUN_ID, client=fake)
        if state["status"] != "running":
            return state
        time.sleep(0.2)
    raise AssertionError("job giả chạy quá lâu")


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Workdir, FakeColab, Path]:
    vm_root = tmp_path / "vm" / "d2s"
    monkeypatch.setattr(executor, "REMOTE_ROOT", str(vm_root))
    monkeypatch.setattr(executor, "CHUNK_BYTES", 1024)  # nhiều phần, để thử ghép
    monkeypatch.setattr(executor, "code_source", lambda branch: ("c0ffee" * 6 + "abcd", "u"))
    job = tmp_path / "job.sh"
    job.write_text(FAKE_JOB)
    monkeypatch.setattr(executor, "TEMPLATE", job)
    smpl = tmp_path / "SMPL_NEUTRAL.pkl"
    smpl.write_bytes(b"smpl")
    monkeypatch.setattr(paths, "find_smpl_model", lambda root=None: smpl)
    return make_workdir(tmp_path / "work"), FakeColab(vm_root), vm_root


def test_push_poll_pull_resume_and_stop(setup: tuple[Workdir, FakeColab, Path]) -> None:
    workdir, fake, vm = setup
    state = executor.push_run(workdir, RUN_ID, client=fake)
    assert fake.created == [("d2s", "T4")]
    assert state.attempt == 1 and state.job == f"{RUN_ID}-1" and state.gpu == "T4"
    scene_dirs = list((vm / "input").glob(f"*/{SCENE}/manifest.yaml"))
    assert len(scene_dirs) == 1
    assert (vm / "work" / "smpl_models" / "SMPL_NEUTRAL.pkl").read_bytes() == b"smpl"
    assert sum("/upload/" in path for path in fake.uploads) > 2  # cảnh cắt thành nhiều phần
    assert not any((vm / "upload").iterdir())  # phần đã ghép xong được xoá
    bundle = json.loads((vm / "bundles" / state.job / "scene_ref.json").read_text())
    assert bundle == {"scene_id": SCENE, "converter_version": "0.1.0+test", "kaggle_dataset": None}

    first = wait_job(workdir, fake)
    assert first["status"] == "failed" and first["exit_code"] == 1
    assert "fake job c0ffee" in "\n".join(first["log_tail"])
    assert [d for d, _ in executor.watched_runs(workdir)] == [workdir.find_run(RUN_ID)]

    record = executor.pull_run(workdir, RUN_ID, client=fake)
    assert record.status == "failed"
    run_dir = workdir.find_run(RUN_ID)
    assert (run_dir / D2S_DIR / "stages" / "prep.smpl" / "first").is_file()
    assert (workdir.cache_dir / CACHE_KEY / "artifact.json").is_file()
    assert executor.load_state(run_dir).pulled_utc is not None
    assert executor.watched_runs(workdir) == []
    assert fake.stopped == []  # run lỗi: giữ VM để chạy tiếp nhanh

    uploads_before = len(fake.uploads)
    second = executor.push_run(workdir, RUN_ID, client=fake)
    assert second.attempt == 2 and fake.created == [("d2s", "T4")]  # dùng lại phiên
    assert len(fake.uploads) == uploads_before  # cảnh, SMPL, run đã có trên VM
    assert wait_job(workdir, fake)["status"] == "succeeded"
    record = executor.pull_run(workdir, RUN_ID, client=fake)
    assert record.status == "succeeded"
    assert (run_dir / D2S_DIR / "stages" / "prep.smpl" / "done").is_file()
    assert (run_dir / D2S_DIR / "colab.json").is_file()  # bản trên máy giữ nguyên
    assert fake.stopped == ["d2s"] and executor.load_state(run_dir).stopped_utc
    with pytest.raises(ColabError, match="đã xong"):
        executor.push_run(workdir, RUN_ID, client=fake)


def test_lost_session_resumes_from_local_copy(setup: tuple[Workdir, FakeColab, Path]) -> None:
    workdir, fake, vm = setup
    executor.push_run(workdir, RUN_ID, client=fake)
    wait_job(workdir, fake)
    executor.pull_run(workdir, RUN_ID, client=fake)
    executor.push_run(workdir, RUN_ID, client=fake)
    fake.lose()
    state = executor.job_state(workdir, RUN_ID, client=fake)
    assert state["status"] == "lost" and "d2s colab push" in state["status_error"]
    assert executor.watched_runs(workdir) == []
    with pytest.raises(ColabError, match="đã mất"):
        executor.pull_run(workdir, RUN_ID, client=fake)

    third = executor.push_run(workdir, RUN_ID, client=fake)
    assert third.attempt == 3 and len(fake.created) == 2
    remote_run = vm / "work" / workdir.find_run(RUN_ID).relative_to(workdir.root)
    assert (remote_run / D2S_DIR / "stages" / "prep.smpl" / "first").is_file()
    assert (vm / "work" / "cache" / CACHE_KEY / "artifact.json").is_file()
    assert wait_job(workdir, fake)["status"] == "succeeded"


def test_push_refuses_busy_vm_and_missing_smpl(
    setup: tuple[Workdir, FakeColab, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir, fake, vm = setup
    fake.create_session("d2s", "T4")
    jobs = vm / "jobs"
    jobs.mkdir()
    (jobs / "other-1.pid").write_text(str(os.getpid()))  # tiến trình còn sống
    with pytest.raises(ColabError, match="đang chạy job other-1"):
        executor.push_run(workdir, RUN_ID, client=fake)
    with pytest.raises(ColabError, match="GPU X1"):
        executor.push_run(workdir, RUN_ID, client=fake, gpu="X1")
    monkeypatch.setattr(paths, "find_smpl_model", lambda root=None: None)
    with pytest.raises(ColabError, match="SMPL_NEUTRAL.pkl"):
        executor.push_run(workdir, RUN_ID, client=fake)
    fake.sessions_on_server["d2s"] = "CPU"
    monkeypatch.setattr(paths, "find_smpl_model", lambda root=None: vm / "x")
    with pytest.raises(ColabError, match="không có GPU"):
        executor.push_run(workdir, RUN_ID, client=fake)


def test_follow_log_survives_dropped_connections(
    setup: tuple[Workdir, FakeColab, Path],
) -> None:
    workdir, fake, _ = setup
    executor.push_run(workdir, RUN_ID, client=fake)
    wait_job(workdir, fake)
    fake.drop_exec = 3
    lines: list[str] = []
    final = executor.follow_log(workdir, RUN_ID, lines.append, client=fake, sleep=lambda s: None)
    assert final == "failed"
    assert lines[0].startswith("fake job")
    assert "status failed" in lines
    fake.drop_exec = 5
    with pytest.raises(ColabError, match="không đọc được log"):
        executor.follow_log(
            workdir, RUN_ID, lines.append, client=fake, max_failures=3, sleep=lambda s: None
        )


def test_cli_status_without_push(
    setup: tuple[Workdir, FakeColab, Path], capsys: pytest.CaptureFixture
) -> None:
    workdir, _, _ = setup
    code = main(["colab", "status", RUN_ID, "--workdir", str(workdir.root), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "colab_failed"


class Done:
    """Kết quả giả của một lệnh ``colab``."""

    def __init__(self, code: int, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = code, out, err


def test_client_parses_cli_output(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    answers: dict[str, Done] = {
        "version": Done(0, "Version: 0.7.5.dev2+ga84e094c6\n"),
        "sessions": Done(
            0,
            "[d2s] gpu-t4-s-abc | Hardware: T4 | Shape: Standard | Variant: GPU\n"
            "[?] cpu-x | Hardware: CPU | Shape: Standard | Variant: DEFAULT\n",
        ),
        "usage": Done(0, "Current balance: 12.50 compute units\nUsage rate: 1.76/hr\n"),
        "new": Done(0, "[colab] Creating session 'd2s'...\n[colab] Session READY.\n"),
        "upload": Done(1, "[colab] Session 'd2s' appears to be lost (404/401). Cleaning up.\n"),
    }
    execs = iter([Done(1, "", "Connection was lost"), Done(0, "@@OK\n")])

    def runner(command: Sequence[str], timeout_s: float) -> Done:
        calls.append(list(command))
        if command[1] == "exec":
            assert Path(command[-1]).read_text() == "print(1)"
            return next(execs)
        return answers.get(command[1], Done(1, "", "lỗi"))

    client = ColabClient(runner=runner)  # type: ignore[arg-type]
    assert client.version() == "0.7.5"
    assert client.sessions() == [
        {"name": "d2s", "hardware": "T4"},
        {"name": "?", "hardware": "CPU"},
    ]
    assert client.usage() == {"balance": 12.5, "rate_per_h": 1.76, "assignments": None}
    client.create_session("d2s", "T4")
    assert ["colab", "new", "-s", "d2s", "--gpu", "T4"] in calls
    with pytest.raises(ColabError, match="không có trên Colab"):
        client.create_session("d2s", "K80")
    assert client.exec_python("d2s", "print(1)", sleep=lambda s: None) == "@@OK\n"
    with pytest.raises(ColabSessionLostError):
        client.upload("d2s", tmp_path / "x", "/content/x")
    with pytest.raises(ColabError, match="colab stop lỗi"):
        client.stop("d2s")


def test_missing_cli_reports_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(command: Sequence[str], **kwargs: object) -> Optional[Done]:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(client_module.subprocess, "run", missing)
    with pytest.raises(ColabError, match="uv tool install google-colab-cli"):
        REAL_RUN_CLI(["colab", "sessions"], 10)
    assert ColabClient(runner=REAL_RUN_CLI).version() is None


def test_upload_assemble_is_safe_to_repeat(setup: tuple[Workdir, FakeColab, Path]) -> None:
    """Kết nối rớt sau khi VM đã ghép xong: gọi lại báo xong, không báo lỗi."""
    _, fake, vm = setup
    fake.create_session("d2s", "T4")
    local = vm.parent / "model.bin"
    local.write_bytes(b"m" * 3000)
    target = f"{vm}/work/smpl_models/model.bin"
    executor.upload_file(fake, "d2s", local, target)
    assert Path(target).read_bytes() == b"m" * 3000
    config = {"parts": f"{vm}/upload/gone", "target": target, "count": 3, "sha256": "x"}
    again = executor._run_on_vm(fake, "d2s", executor._ASSEMBLE, {**config, "extract_to": None})
    assert "@@OK" in again


def test_stopped_session_is_not_polled(setup: tuple[Workdir, FakeColab, Path]) -> None:
    workdir, fake, _ = setup
    executor.push_run(workdir, RUN_ID, client=fake)
    wait_job(workdir, fake)
    executor.pull_run(workdir, RUN_ID, client=fake)
    executor.stop_session(workdir, RUN_ID, client=fake)
    fake.lose()  # mọi lệnh tới VM giờ sẽ báo mất phiên
    assert executor.job_state(workdir, RUN_ID, client=fake)["status"] == "stopped"
