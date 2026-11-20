"""Kiểm tra ingest trên Kaggle (``drive2sim/kaggle/ingest.py``) mà không gọi Kaggle thật.

File này kiểm tra:
- ``d2s kaggle ingest`` đẩy notebook CPU gắn đúng nguồn (dataset hoặc output notebook), chờ xong,
  chỉ ghi file nhỏ của cảnh về máy cùng ``kaggle_scene.json``; gọi lại thì không đẩy lại;
- ``d2s kaggle push`` của run trên cảnh đó không upload cảnh mà gắn output notebook ingest, và
  notebook train tìm được cảnh trong output đó;
- cảnh chỉ có trên Kaggle không chạy được trên máy, không kiểm tra lại được trên máy;
- converter đổi đầu ra: notebook cũ không bị đẩy lại, lệnh dùng notebook ``-c<phiên bản>`` mới;
  cảnh cũ trên máy không bị thay âm thầm.
"""

import copy
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import pytest

from drive2sim.cli.main import main
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.workdir import KAGGLE_SCENE_FILE, Workdir
from drive2sim.ingest import manifest as manifest_module
from drive2sim.ingest.validate import validate_scene
from drive2sim.kaggle import executor, ingest
from drive2sim.kaggle.bundle import materialize, read_bundle
from drive2sim.kaggle.client import KaggleError
from drive2sim.kaggle.scene_files import light_scene_files, write_log_lines
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


def log_text(folder: Path, marker: str, payload: dict) -> str:
    """Log như notebook thật ghi bằng ``write_log_lines``, có dòng stdout chen giữa."""
    path = folder / f"{marker}.log"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        write_log_lines(marker, payload, fd=fd)
    finally:
        os.close(fd)
    lines = path.read_text().splitlines(keepends=True)
    return "pip ...\n" + lines[0] + "stdout của notebook\n" + "".join(lines[1:])


class FakeIngestKaggle(FakeKaggle):
    """Kaggle giả có notebook ingest: đẩy thì ghi output giống notebook thật."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.datasets[SOURCE] = ["v1.0-mini"]
        self.patterns: list[Optional[str]] = []
        self.logs: dict[str, str] = {}
        self.manifest: dict = MANIFEST  # manifest mà notebook ingest "ghi ra"
        self.pushed: list[str] = []
        self.with_pose = True  # output ingest có pose, calib
        self.light_pose = True  # notebook ingest in cả pose, calib vào log (template mới)

    def kernel_logs(self, slug: str) -> str:
        """Log của notebook (notebook ingest in file nhỏ của cảnh vào đây)."""
        return self.logs.get(slug, "")

    def push_kernel(self, folder: Path) -> None:
        """Nhớ notebook; notebook ingest thì ghi luôn output của nó."""
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        name = metadata["code_file"]
        notebook = json.loads((folder / name).read_text())
        self.kernels[metadata["id"]] = {"metadata": metadata, "notebook": notebook}
        self.pushed.append(metadata["id"])
        if name == "ingest.ipynb":  # notebook ingest "chạy": output có cảnh đầy đủ
            out = self.root / "output" / metadata["id"].split("/")[1]
            scene = out / "data" / SCENE
            files.dump(Manifest.model_validate(self.manifest), scene / "manifest.yaml")
            (scene / "images").mkdir(exist_ok=True)
            (scene / "images" / "000_0.jpg").write_bytes(b"jpg")
            (scene / "reports").mkdir(exist_ok=True)
            report = {"summary": {"errors": 0, "warnings": 1}, "checks": []}
            (scene / "reports" / "validation.json").write_text(json.dumps(report))
            if self.with_pose:  # pose, calib như converter ghi (template cũ không in chúng)
                for name in ("intrinsics/0.txt", "extrinsics/000_0.txt", "lidar_pose/000.txt"):
                    (scene / name).parent.mkdir(exist_ok=True)
                    (scene / name).write_text("1 0 0 0\n")
            text = "".join(notebook["cells"][-1]["source"])
            signature = json.loads(text.split('result.update(json.loads(r"""')[1].split('"""')[0])
            result = {"scene_id": SCENE, "commit": COMMIT, "exit_code": 0, **signature}
            (out / "ingest_result.json").write_text(json.dumps(result))
            # Notebook ingest in file nhỏ bằng light_scene_files; notebook theo template cũ
            # (light_pose=False) chỉ in manifest và reports.
            light_files = light_scene_files(scene)
            if not self.light_pose:
                light_files = {
                    name: text
                    for name, text in light_files.items()
                    if name == "manifest.yaml" or name.startswith("reports/")
                }
            light = {"result": result, "files": light_files}
            self.logs[metadata["id"]] = log_text(self.root, ingest.LIGHT_MARKER, light)
        if name == "scene_files.ipynb":  # notebook lấy bù: đọc cảnh trong output ingest đã gắn
            source = metadata["kernel_sources"][0]
            scene = self.root / "output" / source.split("/")[1] / "data" / SCENE
            light = {"scene_id": SCENE, "files": light_scene_files(scene)}
            self.logs[metadata["id"]] = log_text(self.root, ingest.SCENE_FILES_MARKER, light)

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
    # Chỉ tải file nhỏ: pose, calib để xem model, không có ảnh trên máy.
    assert sorted(p.name for p in scene_dir.iterdir()) == [
        "extrinsics",
        "intrinsics",
        KAGGLE_SCENE_FILE,
        "lidar_pose",
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


def kitti360_manifest(output_version: Optional[int]) -> dict:
    """Manifest của cảnh ingest bằng converter KITTI-360 với phiên bản đầu ra cho trước."""
    data = copy.deepcopy(MANIFEST)
    data["source"]["converter"] = "kitti360"
    if output_version is not None:
        data["source"]["output_version"] = output_version
    return data


def test_outdated_converter_uses_new_notebook_and_keeps_old_one(
    tmp_path: Path, fake: FakeIngestKaggle, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    old_kernel = f"alice/{REQUEST.kernel_name()}"
    new_kernel = f"alice/{REQUEST.kernel_name(manifest_module.KITTI360_OUTPUT_VERSION)}"
    assert new_kernel == old_kernel + f"-c{manifest_module.KITTI360_OUTPUT_VERSION}"

    # Notebook cũ: ingest khi converter KITTI-360 còn ở phiên bản 1 (manifest chưa có trường).
    with monkeypatch.context() as patch:
        patch.setattr(manifest_module, "KITTI360_OUTPUT_VERSION", 1)
        fake.manifest = kitti360_manifest(None)
        ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    assert fake.pushed == [old_kernel]
    old_notebook = fake.kernels[old_kernel]

    # Converter đổi đầu ra: đẩy notebook mới, không đẩy lại notebook cũ (run cũ còn gắn output).
    fake.manifest = kitti360_manifest(manifest_module.KITTI360_OUTPUT_VERSION)
    messages: list[str] = []
    with pytest.raises(KaggleError, match="converter cũ.*d2s scene delete"):
        ingest.ingest_on_kaggle(
            workdir, REQUEST, client=fake, sleep=lambda s: None, notify=messages.append
        )
    assert fake.pushed == [old_kernel, new_kernel]
    assert fake.kernels[old_kernel] is old_notebook
    assert any("converter cũ" in m and new_kernel in m for m in messages)
    # Cảnh cũ trên máy không bị thay âm thầm.
    scene_dir = workdir.scene_dir(SCENE)
    assert workdir.load_scene(SCENE).manifest.source.output_version is None
    assert [r.kernel for r in ingest.load_scene_refs(scene_dir)] == [old_kernel]

    # Xoá cảnh cũ rồi chạy lại cùng lệnh: dùng lại notebook mới, không đẩy gì thêm.
    shutil.rmtree(scene_dir)
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    assert fake.pushed == [old_kernel, new_kernel]
    assert (
        workdir.load_scene(SCENE).manifest.source.output_version
        == manifest_module.KITTI360_OUTPUT_VERSION
    )
    assert [r.kernel for r in ingest.load_scene_refs(scene_dir)] == [new_kernel]


def test_fresh_notebook_is_reused_without_suffix(tmp_path: Path, fake: FakeIngestKaggle) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    fake.manifest = kitti360_manifest(manifest_module.KITTI360_OUTPUT_VERSION)
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    assert fake.pushed == [f"alice/{REQUEST.kernel_name()}"]


def test_new_notebook_still_outdated_is_reported(tmp_path: Path, fake: FakeIngestKaggle) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    fake.manifest = kitti360_manifest(None)  # code đẩy lên chưa có converter mới
    with pytest.raises(KaggleError, match="vẫn ingest bằng converter cũ"):
        ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    assert fake.pushed == [
        f"alice/{REQUEST.kernel_name()}",
        f"alice/{REQUEST.kernel_name(manifest_module.KITTI360_OUTPUT_VERSION)}",
    ]


def test_fetch_scene_files_backfills_pose_for_old_ingest(
    tmp_path: Path, fake: FakeIngestKaggle
) -> None:
    # Cảnh ingest bằng template cũ: log chỉ có manifest, reports; máy thiếu pose, calib.
    fake.light_pose = False
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    scene_dir = ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    assert not (scene_dir / "lidar_pose").exists()
    written = ingest.fetch_scene_files(workdir, SCENE, client=fake, sleep=lambda s: None)
    assert sorted(written) == ["extrinsics/000_0.txt", "intrinsics/0.txt", "lidar_pose/000.txt"]
    assert (scene_dir / "lidar_pose" / "000.txt").is_file()
    ingest_kernel = f"alice/{REQUEST.kernel_name()}"
    files_kernel = ingest_kernel.replace("/d2s-ingest-", "/d2s-files-")
    pushed = fake.kernels[files_kernel]
    assert pushed["metadata"]["kernel_sources"] == [ingest_kernel]
    assert pushed["metadata"]["dataset_sources"] == []
    assert not pushed["metadata"]["enable_gpu"] and pushed["metadata"]["is_private"]
    text = json.dumps(pushed["notebook"])
    assert SCENE in text and f"git checkout --quiet {COMMIT}" in text
    for name in ("scene_id", "ingest_kernel", "commit", "repo_url", "branch"):
        assert f"${name}" not in text
    # Đã đủ: gọi lại không đẩy notebook nào.
    count = len(fake.pushed)
    assert ingest.fetch_scene_files(workdir, SCENE, client=fake, sleep=lambda s: None) == []
    assert len(fake.pushed) == count


def test_fetch_scene_files_needs_ingest_of_current_account(
    tmp_path: Path, fake: FakeIngestKaggle
) -> None:
    fake.light_pose = False
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    fake.user = "bob"
    with pytest.raises(KaggleError, match="alice"):
        ingest.fetch_scene_files(workdir, SCENE, client=fake, sleep=lambda s: None)


def test_fetch_scene_files_refuses_local_scene(tmp_path: Path, fake: FakeIngestKaggle) -> None:
    workdir = Workdir(tmp_path / "work")
    scene_dir = workdir.scene_dir(SCENE)
    files.dump(Manifest.model_validate(MANIFEST), scene_dir / "manifest.yaml")
    with pytest.raises(KaggleError, match="không phải cảnh ingest trên Kaggle"):
        ingest.fetch_scene_files(workdir, SCENE, client=fake, sleep=lambda s: None)


def test_fetch_scene_files_repushes_when_log_is_corrupted(
    tmp_path: Path, fake: FakeIngestKaggle
) -> None:
    # Notebook lấy bù theo dạng một dòng cũ: stdout chen vào giữa base64 (đã gặp với KITTI-360).
    fake.light_pose = False
    workdir = Workdir(tmp_path / "work")
    workdir.root.mkdir()
    scene_dir = ingest.ingest_on_kaggle(workdir, REQUEST, client=fake, sleep=lambda s: None)
    files_kernel = f"alice/{REQUEST.kernel_name()}".replace("/d2s-ingest-", "/d2s-files-")
    fake.kernels[files_kernel] = {"metadata": {}, "notebook": {}}
    fake.logs[files_kernel] = f"{ingest.SCENE_FILES_MARKER} eyJzY2VuZV9pZCI6kitti360_drive0003\n"
    count = len(fake.pushed)
    written = ingest.fetch_scene_files(workdir, SCENE, client=fake, sleep=lambda s: None)
    assert fake.pushed[count:] == [files_kernel]
    assert "lidar_pose/000.txt" in written and (scene_dir / "lidar_pose").is_dir()
