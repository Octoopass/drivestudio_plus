"""Kiểm tra gói đầy đủ, cache, symlink và mở lại run trong workdir khác."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from drive2sim.cli.main import main
from drive2sim.core import files
from drive2sim.core.schema import Manifest, RunRecord, Split
from drive2sim.engine import pack as pack_module
from drive2sim.engine.pack import plan_pack, write_pack
from drive2sim.engine.run import execute_run
from drive2sim.engine.workdir import Workdir
from tests.drive2sim.engine.test_run import BEHAVIOUR, make_registry, new_run
from tests.drive2sim.samples import MANIFEST, SPLIT


@pytest.fixture
def packed_run(tmp_path: Path) -> tuple[Workdir, Path]:
    BEHAVIOUR.clear()
    work = Workdir(tmp_path / "source")
    files.dump(
        Manifest.model_validate(MANIFEST), work.scene_dir(MANIFEST["scene_id"]) / "manifest.yaml"
    )
    files.dump(
        Split.model_validate(SPLIT), work.split_path(MANIFEST["scene_id"], SPLIT["split_id"])
    )
    images = work.scene_dir(MANIFEST["scene_id"]) / "images"
    images.mkdir()
    (images / "000_0.jpg").write_bytes(b"fixture image")
    registry = make_registry()
    folder = new_run(work, registry)
    execute_run(folder, work, registry)
    train = folder / "d2s/stages/train.toy"
    for name in ["checkpoint_final.pth", "checkpoint_15000.pth"]:
        (train / name).write_bytes(b"fixture checkpoint")
    view = train / "scene_view/000"
    view.mkdir(parents=True)
    (view / "scene").symlink_to(work.scene_dir(MANIFEST["scene_id"]), target_is_directory=True)
    (train / "latest.pth").symlink_to(train / "checkpoint_final.pth")
    (train / "SMPL_NEUTRAL.pkl").write_bytes(b"private model")
    (train / "access_token").write_text("test credential")
    return work, folder


def test_round_trip_keeps_checkpoints_cache_and_relocates_links(
    packed_run: tuple[Workdir, Path], tmp_path: Path
) -> None:
    work, folder = packed_run
    plan = plan_pack(work, [folder.name, folder.name])
    assert len(plan.run_ids) == 1 and plan.cache_keys
    output = tmp_path / "results.tar"
    write_pack(plan, output)
    destination = tmp_path / "mentor"
    destination.mkdir()
    with tarfile.open(output) as archive:
        assert not any("scene_view" in n for n in archive.getnames())
        assert not any(n.endswith("access_token") for n in archive.getnames())
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        archive.extractall(destination, **options)
    inventory = json.loads((destination / "pack.json").read_text())
    for entry in inventory["files"]:
        path = destination / entry["path"]
        if "sha256" in entry:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    relocated = Workdir(destination)
    record = files.load(relocated.find_run(folder.name) / "d2s/run.json", RunRecord)
    assert record.status == "succeeded"
    assert relocated.load_scene(record.scene_id)
    assert relocated.load_split(record.scene_id, record.split_id)
    train = relocated.find_run(folder.name) / "d2s/stages/train.toy"
    assert (train / "checkpoint_15000.pth").read_bytes() == b"fixture checkpoint"
    shutil.rmtree(work.root)
    assert (train / "latest.pth").read_bytes() == b"fixture checkpoint"
    assert (relocated.cache_dir / record.stages[0].cache_key / "artifact.json").is_file()
    with pytest.raises(FileExistsError):
        write_pack(plan, output)


def test_dry_run_and_missing_dependency(
    packed_run: tuple[Workdir, Path], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    work, folder = packed_run
    output = tmp_path / "missing-parent/out.tar"
    assert (
        main(
            [
                "pack",
                folder.name,
                "--workdir",
                str(work.root),
                "--output",
                str(output),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert not output.parent.exists()
    images = work.scene_dir(MANIFEST["scene_id"]) / "images"
    shutil.rmtree(images)
    with pytest.raises(FileNotFoundError, match="chưa có ảnh"):
        plan_pack(work, [folder.name])
    images.mkdir()
    plan = plan_pack(work, [folder.name])
    shutil.rmtree(work.cache_dir / plan.cache_keys[0])
    with pytest.raises(FileNotFoundError, match="thiếu cache"):
        plan_pack(work, [folder.name])


def test_refuses_active_run_external_link_and_source_changes(
    packed_run: tuple[Workdir, Path], tmp_path: Path
) -> None:
    work, folder = packed_run
    record_path = folder / "d2s/run.json"
    record = files.load(record_path, RunRecord)
    record.status = "running"
    files.dump(record, record_path)
    with pytest.raises(ValueError, match="đang chạy"):
        plan_pack(work, [folder.name])
    record.status = "succeeded"
    files.dump(record, record_path)
    outside = tmp_path / "outside"
    outside.write_text("not selected")
    link = folder / "external"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        plan_pack(work, [folder.name])
    link.unlink()
    plan = plan_pack(work, [folder.name])
    record_path.write_text(record_path.read_text() + "\n")
    output = tmp_path / "changed.tar"
    with pytest.raises(ValueError, match="nguồn đã đổi"):
        write_pack(plan, output)
    assert not output.exists() and not list(tmp_path.glob("*.partial"))


def test_refuses_insufficient_space(
    packed_run: tuple[Workdir, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, folder = packed_run
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type(usage)(100, 100, 0))
    with pytest.raises(ValueError, match="không đủ chỗ"):
        write_pack(plan_pack(work, [folder.name]), tmp_path / "out.tar")


def test_smpl_run_requires_and_includes_real_model_at_portable_path(
    packed_run: tuple[Workdir, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, folder = packed_run
    monkeypatch.setattr(pack_module, "produces_artifact", lambda stages, kind: True)
    monkeypatch.setattr(pack_module.paths, "find_smpl_model", lambda root: None)
    with pytest.raises(FileNotFoundError, match="thiếu SMPL_NEUTRAL.pkl"):
        plan_pack(work, [folder.name])
    model = tmp_path / "external-model.pkl"
    model.write_bytes(b"fixture body model")
    monkeypatch.setattr(pack_module.paths, "find_smpl_model", lambda root: model)
    output = tmp_path / "with-smpl.tar"
    write_pack(plan_pack(work, [folder.name]), output)
    with tarfile.open(output) as archive:
        name = "smpl_models/SMPL_NEUTRAL.pkl"
        assert archive.getnames().count(name) == 1
        assert archive.extractfile(name).read() == model.read_bytes()
        inventory = json.load(archive.extractfile("pack.json"))
    entry = next(item for item in inventory["files"] if item["path"] == name)
    assert entry["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
