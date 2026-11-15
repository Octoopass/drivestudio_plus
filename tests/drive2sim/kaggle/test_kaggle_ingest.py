"""Kiểm tra ingest trên Kaggle (``drive2sim/kaggle/ingest.py``) mà không gọi Kaggle thật.

File này kiểm tra:
- ``d2s kaggle ingest`` đẩy notebook CPU gắn đúng nguồn (dataset hoặc output notebook), chờ xong,
  chỉ ghi file nhỏ của cảnh về máy cùng ``kaggle_scene.json``; gọi lại thì không đẩy lại;
- ``d2s kaggle push`` của run trên cảnh đó không upload cảnh mà gắn output notebook ingest, và
  notebook train tìm được cảnh trong output đó;
- cảnh chỉ có trên Kaggle không chạy được trên máy, không kiểm tra lại được trên máy.
"""

import base64
import json
import re
import shutil
from pathlib import Path
from typing import Optional

import pytest

from drive2sim.cli.main import main
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.workdir import KAGGLE_SCENE_FILE, Workdir
from drive2sim.ingest.validate import validate_scene
from drive2sim.kaggle import executor, ingest
from drive2sim.kaggle.bundle import materialize, read_bundle
from drive2sim.kaggle.client import KaggleError
from tests.drive2sim.kaggle.test_kaggle import (
    RUN_ID,
    SCENE,
    FakeKaggle,
    make_workdir,
    notebook_bundle,
)
from tests.drive2sim.samples import MANIFEST

SOURCE = "alice/nuscenes-raw"
REQUEST = ingest.IngestRequest(SOURCE, "", ("--sequence", "scene-0061"))
COMMIT = "c0ffee" * 6 + "abcd"


class FakeIngestKaggle(FakeKaggle):
    """Kaggle giả có notebook ingest: đẩy thì ghi output giống notebook thật."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.datasets[SOURCE] = ["v1.0-mini"]
        self.patterns: list[Optional[str]] = []
        self.logs: dict[str, str] = {}

    def kernel_logs(self, slug: str) -> str:
        """Log của notebook (notebook ingest in file nhỏ của cảnh vào đây)."""
        return self.logs.get(slug, "")

    def push_kernel(self, folder: Path) -> None:
        """Nhớ notebook; notebook ingest thì ghi luôn output của nó."""
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        name = metadata["code_file"]
        notebook = json.loads((folder / name).read_text())
        self.kernels[metadata["id"]] = {"metadata": metadata, "notebook": notebook}
        if name == "ingest.ipynb":  # notebook ingest "chạy": output có cảnh đầy đủ
            out = self.root / "output" / metadata["id"].split("/")[1]
            scene = out / "data" / SCENE
            files.dump(Manifest.model_validate(MANIFEST), scene / "manifest.yaml")
            (scene / "images").mkdir(exist_ok=True)
            (scene / "images" / "000_0.jpg").write_bytes(b"jpg")
            (scene / "reports").mkdir(exist_ok=True)
            report = {"summary": {"errors": 0, "warnings": 1}, "checks": []}
            (scene / "reports" / "validation.json").write_text(json.dumps(report))
            text = "".join(notebook["cells"][-1]["source"])
            signature = json.loads(text.split('result.update(json.loads(r"""')[1].split('"""')[0])
            result = {"scene_id": SCENE, "commit": COMMIT, "exit_code": 0, **signature}
            (out / "ingest_result.json").write_text(json.dumps(result))
            light = {
                "result": result,
                "files": {
                    "manifest.yaml": (scene / "manifest.yaml").read_text(),
                    "reports/validation.json": json.dumps(report),
                },
            }
            encoded = base64.b64encode(json.dumps(light).encode()).decode()
            self.logs[metadata["id"]] = f"pip ...\n{ingest.LIGHT_MARKER} {encoded}\n"

    def kernel_status(self, slug: str) -> str:
        """Notebook chưa đẩy thì báo 404 như Kaggle."""
        if slug not in self.kernels:
            raise KaggleError("404 Not Found")
        return self.status

    def kernel_output(
        self, slug: str, destination: Path, file_pattern: Optional[str] = None
    ) -> None:
        """Như Kaggle CLI: chỉ chép file khớp ``file_pattern``."""
        self.patterns.append(file_pattern)
        source = self.root / "output" / slug.split("/")[1]
        for path in source.rglob("*"):
            relative = path.relative_to(source).as_posix()
            if path.is_file() and (file_pattern is None or re.search(file_pattern, relative)):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())


@pytest.fixture()
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeIngestKaggle:
    for module in (executor, ingest):
        monkeypatch.setattr(
            module, "code_source", lambda branch="drive2sim": (COMMIT, "https://x/r.git")
        )
    return FakeIngestKaggle(tmp_path / "kaggle")


def test_ingest_writes_light_files_and_reuses_notebook(
    tmp_path: Path, fake: FakeIngestKaggle
) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    scene_dir = ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    kernel = f"alice/{REQUEST.kernel_name()}"
    pushed = fake.kernels[kernel]
    assert pushed["metadata"]["dataset_sources"] == [SOURCE]
    assert pushed["metadata"]["kernel_sources"] == []
    assert not pushed["metadata"]["enable_gpu"] and pushed["metadata"]["is_private"]
    text = json.dumps(pushed["notebook"])
    for name in ("source", "source_name", "source_path", "ingest_args", "commit", "repo_url"):
        assert f"${name}" not in text
    assert "scene-0061" in text and f"git checkout --quiet {COMMIT}" in text
    # Chỉ tải file nhỏ: không có ảnh trên máy.
    assert sorted(p.name for p in scene_dir.iterdir()) == [
        KAGGLE_SCENE_FILE,
        "manifest.yaml",
        "reports",
    ]
    ref = ingest.load_scene_ref(scene_dir, "alice")
    assert ref is not None and ref.kernel == kernel and ref.commit == COMMIT
    assert workdir.load_scene(SCENE).manifest.scene_id == SCENE
    with pytest.raises(ValueError, match="chỉ có trên Kaggle"):
        validate_scene(scene_dir, workdir.load_scene(SCENE).manifest)

    # Gọi lại cùng tham số: dùng lại notebook đã xong, không đẩy lại.
    pushes = len(fake.kernels)
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    assert len(fake.kernels) == pushes
    assert [r.username for r in ingest.load_scene_refs(scene_dir)] == ["alice"]

    # Tài khoản thứ hai ingest cùng cảnh: thêm notebook của nó, giữ của alice.
    fake.user = "bob"
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    refs = ingest.load_scene_refs(scene_dir)
    assert [(r.username, r.kernel) for r in refs] == [
        ("alice", kernel),
        ("bob", f"bob/{REQUEST.kernel_name()}"),
    ]

    # Cùng tên notebook nhưng tham số khác: không ghi đè output notebook cũ.
    other = ingest.IngestRequest(SOURCE, "", ("--sequence", "scene-0061", "--num-frames", "5"))
    assert other.kernel_name() == REQUEST.kernel_name()
    with pytest.raises(KaggleError, match="tham số khác"):
        ingest.ingest_on_kaggle(workdir, other, client=fake, sleep=lambda s: None)

    # Cảnh cùng tên mà ingest trên máy (không có kaggle_scene.json): không đè.
    (scene_dir / KAGGLE_SCENE_FILE).unlink()
    with pytest.raises(KaggleError, match="đã có trên máy"):
        ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)


def test_ingest_from_notebook_output_and_failed_notebook(
    tmp_path: Path, fake: FakeIngestKaggle
) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    request = ingest.IngestRequest("alice/d2s-kitti360-0003-raw", "kitti360", ())
    fake.status = "error"
    with pytest.raises(KaggleError, match="trạng thái error"):
        ingest.ingest_on_kaggle(workdir, request, client=fake, sleep=lambda s: None)
    metadata = fake.kernels[f"alice/{request.kernel_name()}"]["metadata"]
    assert metadata["dataset_sources"] == [] and metadata["kernel_sources"] == [request.source]
    cells = fake.kernels[f"alice/{request.kernel_name()}"]["notebook"]["cells"]
    assert 'name, sub = "d2s-kitti360-0003-raw", "kitti360"\n' in cells[-1]["source"]


def test_push_run_attaches_ingest_notebook(
    tmp_path: Path, fake: FakeIngestKaggle, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = make_workdir(tmp_path / "work")
    # Cảnh trên máy chỉ còn file nhỏ: bỏ ảnh, thêm kaggle_scene.json như sau d2s kaggle ingest.
    scene_dir = workdir.scene_dir(SCENE)
    (scene_dir / "images" / "000_0.jpg").unlink()
    (scene_dir / "images").rmdir()
    kernel = f"alice/{REQUEST.kernel_name()}"
    ref = ingest.KaggleSceneRef("alice", kernel, SOURCE, COMMIT, "2026-11-15T00:00:00+00:00")
    (scene_dir / KAGGLE_SCENE_FILE).write_text(json.dumps({"kernels": [ref.__dict__]}))
    # Output của notebook ingest trên Kaggle giả.
    out = fake.root / "output" / kernel.split("/")[1] / "data" / SCENE
    files.dump(Manifest.model_validate(MANIFEST), out / "manifest.yaml")
    (out / "images").mkdir()
    (out / "images" / "000_0.jpg").write_bytes(b"jpg")

    monkeypatch.setenv("D2S_KAGGLE_SMPL_DATASET", "alice/smpl-model")
    state = executor.push_run(workdir, RUN_ID, client=fake)
    assert fake.datasets.keys() == {SOURCE}  # không upload cảnh
    metadata = fake.kernels[state.kernel]["metadata"]
    assert metadata["dataset_sources"] == ["alice/smpl-model"]
    assert metadata["kernel_sources"] == [kernel]
    assert state.dataset == kernel

    # Notebook train: Kaggle gắn output notebook ingest dưới /kaggle/input/<tên>/.
    input_root = fake.root / "input"
    shutil.copytree(fake.root / "output" / kernel.split("/")[1], input_root / kernel.split("/")[1])
    bundle_dir = fake.root / "bundle"
    bundle_dir.mkdir()
    for name, content in notebook_bundle(fake, state.kernel).items():
        (bundle_dir / name).write_text(json.dumps(content))
    working = Workdir(fake.root / "working")
    materialize(read_bundle(bundle_dir), working, input_root)
    assert (working.scene_dir(SCENE) / "images" / "000_0.jpg").is_file()

    # Tài khoản khác không đọc được notebook ingest private của alice: phải tự ingest.
    (workdir.find_run(RUN_ID) / "d2s" / executor.STATE_FILE).unlink()
    fake.user = "bob"
    with pytest.raises(KaggleError, match="tài khoản alice.*chạy d2s kaggle ingest bằng bob"):
        executor.push_run(workdir, RUN_ID, client=fake)


def test_local_run_on_kaggle_scene_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    workdir = make_workdir(tmp_path / "work")
    (workdir.scene_dir(SCENE) / KAGGLE_SCENE_FILE).write_text('{"kernels": []}')
    code = main(
        [
            "run",
            "start",
            "streetgs",
            "--scene",
            SCENE,
            "--split",
            "every8_front1",
            "--workdir",
            str(workdir.root),
            "--json",
        ]
    )
    assert code == 1
    assert "chỉ có trên Kaggle" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_complete_notebook_without_log_result_is_rerun(
    tmp_path: Path, fake: FakeIngestKaggle
) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    kernel = f"alice/{REQUEST.kernel_name()}"
    fake.logs[kernel] = "log của template cũ, không có kết quả"
    (workdir.scene_dir(SCENE) / KAGGLE_SCENE_FILE).unlink()
    (workdir.scene_dir(SCENE) / "manifest.yaml").unlink()
    pushes = []
    original = fake.push_kernel
    fake.push_kernel = lambda folder: (pushes.append(folder), original(folder))  # type: ignore
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    assert len(pushes) == 1
    assert fake.patterns == []  # không liệt kê output
