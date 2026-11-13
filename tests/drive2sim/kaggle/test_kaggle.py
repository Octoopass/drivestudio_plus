"""Kiểm tra executor Kaggle mà không gọi Kaggle thật.

``FakeKaggle`` thay ``KaggleClient``: nhớ dataset đã tạo, chép notebook đã đẩy, trả trạng thái
đặt sẵn, và "chạy" notebook bằng cách dựng output giống notebook thật ghi ra (``runs/``,
``cache/``). File này kiểm tra:
- đẩy lần đầu: upload cảnh một lần, notebook có gói job và đúng commit, ghi ``kaggle.json``;
- dựng thư mục làm việc trên Kaggle từ gói job (``materialize``), kể cả chạy tiếp từ output cũ;
- tải về: ghép ``d2s/`` và cache, giữ ``kaggle.json`` và stage chỉ có trên máy;
- đẩy lần hai: chạy tiếp từ notebook trước; từ chối khi notebook trước còn chạy;
- lệnh CLI ``d2s kaggle status`` và lỗi ``kaggle_failed``.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from drive2sim.cli.main import main
from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, Manifest, RunRecord, Split
from drive2sim.engine.run import D2S_DIR, RUN_FILE
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle import executor
from drive2sim.kaggle.bundle import materialize, read_bundle
from drive2sim.kaggle.client import KaggleClient, KaggleError
from tests.drive2sim.samples import MANIFEST, SPLIT

SCENE = MANIFEST["scene_id"]
RUN_ID = "20261113-010203-omnire-ab12"
CACHE_KEY = "prep.smpl@0.1.0-0123abcd"


class FakeKaggle(KaggleClient):
    """Kaggle giả, giữ mọi thứ trong bộ nhớ và một thư mục tạm."""

    def __init__(self, root: Path) -> None:
        super().__init__(runner=self._unused)
        self.root = root
        self.datasets: dict[str, list[str]] = {}
        self.kernels: dict[str, dict] = {}
        self.status = "complete"

    @staticmethod
    def _unused(command: object) -> None:
        raise AssertionError("FakeKaggle không chạy lệnh")

    def username(self) -> str:
        """Người dùng cố định."""
        return "alice"

    def dataset_exists(self, slug: str) -> bool:
        """Dataset đã tạo trong bộ nhớ chưa."""
        return slug in self.datasets

    def dataset_status(self, slug: str) -> str:
        """Dataset giả xử lý xong ngay."""
        return "ready"

    def create_dataset(self, folder: Path) -> None:
        """Nhớ dataset và chép nó vào thư mục input giả."""
        slug = json.loads((folder / "dataset-metadata.json").read_text())["id"]
        self.datasets[slug] = sorted(p.name for p in folder.iterdir())
        shutil.copytree(folder, self.root / "input" / slug.split("/")[1], symlinks=False)

    def push_kernel(self, folder: Path) -> None:
        """Nhớ notebook và metadata đã đẩy."""
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        notebook = json.loads((folder / "run.ipynb").read_text())
        self.kernels[metadata["id"]] = {"metadata": metadata, "notebook": notebook}

    def kernel_status(self, slug: str) -> str:
        """Trạng thái đặt sẵn."""
        return self.status

    def kernel_output(self, slug: str, destination: Path) -> None:
        """Chép output giả của notebook."""
        shutil.copytree(self.root / "output" / slug.split("/")[1], destination, dirs_exist_ok=True)


def make_workdir(root: Path) -> Workdir:
    workdir = Workdir(root)
    scene_dir = workdir.scene_dir(SCENE)
    files.dump(Manifest.model_validate(MANIFEST), scene_dir / "manifest.yaml")
    (scene_dir / "images").mkdir()
    (scene_dir / "images" / "000_0.jpg").write_bytes(b"jpg")
    files.dump(Split.model_validate(SPLIT), workdir.split_path(SCENE, SPLIT["split_id"]))
    run_dir = workdir.run_dir(SCENE, "omnire", RUN_ID)
    record = RunRecord(
        schema_version=SCHEMA_VERSIONS["run"],
        run_id=RUN_ID,
        scene_id=SCENE,
        split_id=SPLIT["split_id"],
        approach={"approach_id": "omnire", "version": "0.1.0", "options": {}},
        level="L1",
        executor="kaggle",
        status="pending",
        stages=[{"stage_id": "prep.smpl", "status": "pending", "params": {}}],
    )
    files.dump(record, run_dir / D2S_DIR / RUN_FILE)
    return workdir


def notebook_bundle(fake: FakeKaggle, kernel: str) -> dict:
    """Lấy gói job nhúng trong notebook đã đẩy."""
    for cell in fake.kernels[kernel]["notebook"]["cells"]:
        text = "".join(cell["source"])
        if "json.loads(r" in text:
            return json.loads(text.split('r"""')[1].split('"""')[0])
    raise AssertionError("notebook không có gói job")


def fake_kaggle_run(fake: FakeKaggle, kernel: str, status: str, prev: Optional[str]) -> Path:
    """Giả notebook chạy: dựng thư mục làm việc từ gói, ghi kết quả vào output của notebook."""
    bundle_dir = fake.root / "bundle"
    shutil.rmtree(bundle_dir, ignore_errors=True)
    bundle_dir.mkdir()
    for name, content in notebook_bundle(fake, kernel).items():
        (bundle_dir / name).write_text(json.dumps(content))
    input_root = fake.root / "input"
    if prev:
        shutil.copytree(fake.root / "output" / prev.split("/")[1], input_root / prev.split("/")[1])
    working = Workdir(fake.root / "working" / kernel.split("/")[1])
    run_dir = materialize(read_bundle(bundle_dir), working, input_root)
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    record.status = status
    files.dump(record, run_dir / D2S_DIR / RUN_FILE)
    (working.cache_dir / SCENE / CACHE_KEY).mkdir(parents=True, exist_ok=True)
    (working.cache_dir / SCENE / CACHE_KEY / "artifact.json").write_text("{}")
    output = fake.root / "output" / kernel.split("/")[1]
    shutil.rmtree(output, ignore_errors=True)
    for part in ("runs", "cache"):
        shutil.copytree(working.root / part, output / part)
    return run_dir


def no_kaggle_cli(command: list[str]) -> subprocess.CompletedProcess:
    """Thay lệnh kaggle thật: mọi lệnh đều báo chưa đăng nhập."""
    return subprocess.CompletedProcess(command, 1, "", "401 Unauthorized")


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Workdir, FakeKaggle]:
    monkeypatch.setattr(
        executor,
        "code_source",
        lambda branch="drive2sim": ("c0ffee" * 6 + "abcd", "https://x/r.git"),
    )
    return make_workdir(tmp_path / "work"), FakeKaggle(tmp_path / "kaggle")


def test_push_materialize_pull_and_resume(
    setup: tuple[Workdir, FakeKaggle], monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir_, fake_ = setup
    with pytest.raises(KaggleError, match="D2S_KAGGLE_SMPL_DATASET"):
        executor.push_run(workdir_, RUN_ID, client=fake_)
    monkeypatch.setenv("D2S_KAGGLE_SMPL_DATASET", "alice/smpl-model")
    workdir, fake = setup
    state = executor.push_run(workdir, RUN_ID, client=fake)
    assert state.attempt == 1 and state.kernel.startswith("alice/d2s-")
    assert list(fake.datasets.values()) == [["dataset-metadata.json", SCENE]]
    pushed = fake.kernels[state.kernel]
    assert pushed["metadata"]["dataset_sources"] == [state.dataset, "alice/smpl-model"]
    assert pushed["metadata"]["kernel_sources"] == []
    assert pushed["metadata"]["enable_gpu"] and pushed["metadata"]["is_private"]
    assert pushed["metadata"]["machine_shape"] == "NvidiaTeslaT4"
    text = json.dumps(pushed["notebook"])
    assert "git checkout --quiet c0ffee" in text and "https://x/r.git" in text
    # mọi chỗ giữ chỗ của template đã được điền ($rc, $(nvidia-smi ...) là của bash, giữ nguyên)
    for name in ("run_id", "commit", "repo_url", "branch", "bundle_json"):
        assert f"${name}" not in text and f"${{{name}}}" not in text
    assert notebook_bundle(fake, state.kernel)["scene_ref.json"]["kaggle_dataset"] == state.dataset
    assert not (workdir.root / ".kaggle" / "upload").exists() or not any(
        (workdir.root / ".kaggle" / "upload").iterdir()
    )

    remote_run = fake_kaggle_run(fake, state.kernel, "failed", None)
    working = remote_run.parents[3]
    scene_link = working / "data" / SCENE / "images"
    assert scene_link.is_symlink() and (working / "data" / SCENE / "splits").is_dir()
    assert not (working / "data" / SCENE / "splits").is_symlink()

    record = executor.pull_run(workdir, RUN_ID, client=fake)
    assert record.status == "failed"
    run_dir = workdir.find_run(RUN_ID)
    assert executor.load_state(run_dir).pulled_utc is not None
    assert (workdir.cache_dir / SCENE / CACHE_KEY / "artifact.json").is_file()

    fake.status = "running"
    with pytest.raises(KaggleError, match="còn đang running"):
        executor.push_run(workdir, RUN_ID, client=fake)
    fake.status = "complete"
    second = executor.push_run(workdir, RUN_ID, client=fake)
    assert second.attempt == 2 and second.kernel.endswith("-2")
    assert fake.kernels[second.kernel]["metadata"]["kernel_sources"] == [state.kernel]
    assert len(fake.datasets) == 1  # cảnh không upload lại

    resumed = fake_kaggle_run(fake, second.kernel, "succeeded", state.kernel)
    assert files.load(resumed / D2S_DIR / RUN_FILE, RunRecord).status == "succeeded"
    assert (resumed.parents[3] / "cache" / SCENE / CACHE_KEY).is_dir()
    # Giữa hai lần tải, run được xuất thêm trên máy: stage này không có trên Kaggle, tải lần nữa
    # không được xoá nó.
    local_d2s = run_dir / D2S_DIR
    local_record = files.load(local_d2s / RUN_FILE, RunRecord)
    extra = local_record.stages[-1].model_copy(update={"stage_id": "export.preview"})
    local_record.stages.append(extra)
    files.dump(local_record, local_d2s / RUN_FILE)
    (local_d2s / "stages" / "export.preview").mkdir(parents=True)
    (local_d2s / "stages" / "export.preview" / "preview.json").write_text("{}")
    pulled = executor.pull_run(workdir, RUN_ID, client=fake)
    assert pulled.status == "succeeded"
    assert pulled.stages[-1].stage_id == "export.preview"
    assert (local_d2s / "stages" / "export.preview" / "preview.json").is_file()
    with pytest.raises(KaggleError, match="đã xong"):
        executor.push_run(workdir, RUN_ID, client=fake)


def test_pull_without_push_and_cli_errors(
    setup: tuple[Workdir, FakeKaggle], capsys: pytest.CaptureFixture
) -> None:
    workdir, fake = setup
    with pytest.raises(KaggleError, match="chưa đẩy"):
        executor.pull_run(workdir, RUN_ID, client=fake)
    code = main(["kaggle", "status", RUN_ID, "--workdir", str(workdir.root), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "kaggle_failed"


def test_client_parses_cli_output(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class Done:
        def __init__(self, code: int, out: str, err: str = "") -> None:
            self.returncode, self.stdout, self.stderr = code, out, err

    def runner(command: list[str]) -> Done:
        calls.append(command)
        if command[1:3] == ["kernels", "status"]:
            return Done(0, 'alice/d2s-1 has status "KernelWorkerStatus.RUNNING"\n')
        if command[1:3] == ["config", "view"]:
            return Done(0, "Configuration values from x\n- username: None\n")
        if command[1:3] == ["datasets", "status"]:
            return Done(1, "", "404 Not Found")
        return Done(1, "", "401 Unauthorized")

    client = KaggleClient(runner=runner, config_dir=tmp_path)  # type: ignore[arg-type]
    assert client.kernel_status("alice/d2s-1") == "running"
    assert not client.dataset_exists("alice/d2s-x")
    with pytest.raises(KaggleError, match="401"):
        client.push_kernel(tmp_path)
    with pytest.raises(KaggleError, match="khoá API"):
        client.username()
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "bob", "key": "k"}))
    assert client.username() == "bob"


def test_resume_picks_newest_attempt_that_has_the_run(tmp_path: Path) -> None:
    """Lần đẩy gần nhất lỗi trước khi ghi output: chạy tiếp từ lần cũ hơn còn kết quả."""
    from types import SimpleNamespace

    from drive2sim.kaggle.bundle import _find_previous_run

    record = SimpleNamespace(
        scene_id=SCENE, run_id=RUN_ID, approach=SimpleNamespace(approach_id="omnire")
    )
    for attempt in (1, 2):
        run = tmp_path / f"d2s-abc-{attempt}" / "runs" / SCENE / "omnire" / RUN_ID / D2S_DIR
        run.mkdir(parents=True)
        (run / RUN_FILE).write_text("{}")
    (tmp_path / "d2s-abc-3").mkdir()  # lần 3 lỗi lúc cài, output trống
    kernels = ["alice/d2s-abc-3", "alice/d2s-abc-2", "alice/d2s-abc-1"]
    found = _find_previous_run(tmp_path, record, kernels)  # type: ignore[arg-type]
    assert found is not None and found.relative_to(tmp_path).parts[0] == "d2s-abc-2"


def test_pull_does_not_copy_process_files(tmp_path: Path) -> None:
    """Pid và cancel của phiên Kaggle không được chép về: run sẽ bị coi là đang chạy mãi."""
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    for name in ("pid", "cancel", "log.txt"):
        (remote / name).write_text("42 kaggle-host\n")
    executor._merge_d2s(local, remote)
    assert (local / "log.txt").is_file()
    assert not (local / "pid").exists() and not (local / "cancel").exists()


def test_code_source_requires_commit_on_the_cloned_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commit chỉ có trên remote khác (ví dụ fork) thì notebook không checkout được."""
    url = "https://github.com/team/repo.git"
    answers = {
        ("rev-parse", "HEAD"): "abc1234def",
        ("remote", "get-url", "origin"): url,
        ("remote", "-v"): f"fork\thttps://github.com/me/repo.git (fetch)\norigin\t{url} (fetch)",
        ("rev-parse", "--verify", "--quiet", "refs/remotes/origin/drive2sim"): "abc",
        ("branch", "-r", "--contains", "abc1234def", "--list", "origin/*"): "",
    }
    monkeypatch.delenv("D2S_KAGGLE_REPO", raising=False)
    monkeypatch.setattr(executor, "_git", lambda *args: answers.get(args, ""))
    with pytest.raises(KaggleError, match="chưa có trên remote origin"):
        executor.code_source("drive2sim")
    answers[("branch", "-r", "--contains", "abc1234def", "--list", "origin/*")] = (
        "origin/drive2sim"
    )
    assert executor.code_source("drive2sim") == ("abc1234def", url)
    with pytest.raises(KaggleError, match="không có nhánh main"):
        executor.code_source("main")


def test_wait_dataset_ready_polls_until_ready() -> None:
    states = iter(["pending", "pending", "ready"])
    client = FakeKaggle.__new__(FakeKaggle)
    client.dataset_status = lambda slug: next(states)  # type: ignore[method-assign]
    sleeps: list[float] = []
    executor.wait_dataset_ready(client, "alice/d2s-x", sleep=sleeps.append)
    assert len(sleeps) == 2
    client.dataset_status = lambda slug: "pending"  # type: ignore[method-assign]
    with pytest.raises(KaggleError, match="vẫn đang xử lý"):
        executor.wait_dataset_ready(client, "alice/d2s-x", timeout_s=20, sleep=lambda s: None)
