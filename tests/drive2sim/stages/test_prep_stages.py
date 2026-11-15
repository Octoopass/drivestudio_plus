"""Kiểm tra thư mục cảnh ảo và hai stage ``prep.masks_segformer``, ``prep.smpl``.

Máy test không có GPU, SegFormer hay 4D-Humans, nên script thật được thay bằng hàm giả ghi đúng
những gì script thật ghi (``sky_masks/``, ``fine_dynamic_masks/``, ``humanpose/smpl.pkl``) vào
thư mục cảnh ảo. File này kiểm tra:
- view chứa symlink tới từng mục của cảnh, bỏ các mục bị loại, thêm các mục từ artifact;
- script ghi vào view không chạm vào thư mục cảnh thật, kể cả khi cảnh có sẵn mục cùng tên;
- hai stage dựng lệnh đúng, chuyển kết quả ra thư mục của stage, xoá view, báo lỗi khi
  thiếu kết quả;
- stage train dựng view có mask và SMPL từ artifact và tắt ``load_smpl`` khi không có SMPL;
- approach ``omnire`` có cả hai stage, ``streetgs`` chỉ có stage mask.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import Approach, Manifest, Split
from drive2sim.engine.stage import NoParams, Scene
from drive2sim.stages.prep import masks_segformer, smpl
from drive2sim.stages.prep.masks_segformer import PrepMasksSegformerParams, PrepMasksSegformerStage
from drive2sim.stages.prep.smpl import PrepSmplStage
from drive2sim.stages.prep.smpl import build_command as build_smpl_command
from drive2sim.stages.scene_view import build_view, collect_outputs
from drive2sim.stages.train import drivestudio
from drive2sim.stages.train.drivestudio import TrainDrivestudioParams, TrainDrivestudioStage
from tests.drive2sim.samples import MANIFEST, SPLIT


def make_scene(root: Path) -> Path:
    scene = root / "data" / "nuscenes_scene-0061"
    (scene / "images").mkdir(parents=True)
    (scene / "images" / "000_0.jpg").write_text("img")
    (scene / "dynamic_masks" / "human").mkdir(parents=True)
    (scene / "humanpose").mkdir()
    (scene / "humanpose" / "smpl.pkl").write_text("original")
    (scene / "manifest.yaml").write_text("x")
    return scene


class Context:
    """Ngữ cảnh tối thiểu cho ``Stage.run``."""

    def __init__(self, scene_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.scene = Scene(scene_dir.name, scene_dir, Manifest.model_validate(MANIFEST))
        self.split = Split.model_validate(SPLIT)

    def progress(self, fraction: float, message: str = "") -> None:
        """Bỏ qua tiến độ."""

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Bỏ qua metric."""

    def should_stop(self) -> bool:
        """Không bao giờ huỷ."""
        return False


def test_build_view_links_excludes_and_extras(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    extra_dir = tmp_path / "cache" / "sky_masks"
    extra_dir.mkdir(parents=True)
    view = build_view(
        scene, tmp_path / "view", exclude=["humanpose"], extra={"sky_masks": extra_dir}
    )
    names = sorted(entry.name for entry in view.iterdir())
    # manifest.yaml có trong view để loader đọc mô hình camera (loader KITTI-360)
    assert names == ["dynamic_masks", "images", "manifest.yaml", "sky_masks"]
    assert all((view / name).is_symlink() for name in names)
    assert (view / "sky_masks").resolve() == extra_dir.resolve()
    # dựng lại thì view cũ bị thay
    view = build_view(scene, tmp_path / "view")
    assert (view / "humanpose").is_symlink()
    with pytest.raises(FileNotFoundError):
        build_view(scene, tmp_path / "view", extra={"x": tmp_path / "missing"})


def test_collect_outputs_never_moves_symlinks(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    view = build_view(scene, tmp_path / "view", exclude=["humanpose"])
    (view / "humanpose").mkdir()
    (view / "humanpose" / "smpl.pkl").write_text("new")
    out = tmp_path / "out"
    out.mkdir()
    assert collect_outputs(view, out, ["humanpose", "images"]) == ["humanpose"]
    assert (out / "humanpose" / "smpl.pkl").read_text() == "new"
    assert (scene / "humanpose" / "smpl.pkl").read_text() == "original"
    assert (scene / "images" / "000_0.jpg").is_file()


def test_masks_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene = make_scene(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        calls.append(command)
        view = Path(command[command.index("--data_root") + 1]) / "000"
        (view / "sky_masks").mkdir()
        (view / "sky_masks" / "000_0.png").write_text("m")
        (view / "fine_dynamic_masks" / "all").mkdir(parents=True)

    monkeypatch.setattr(masks_segformer, "run_logged", fake_run)
    monkeypatch.setenv("D2S_DEVICE", "cpu")
    ctx = Context(scene, tmp_path / "cache" / "masks")
    result = PrepMasksSegformerStage().run(ctx, {}, PrepMasksSegformerParams())
    assert result == {"masks": ctx.output_dir}
    assert (ctx.output_dir / "sky_masks" / "000_0.png").is_file()
    assert (ctx.output_dir / "fine_dynamic_masks").is_dir()
    assert not (ctx.output_dir / "view").exists()
    assert not (scene / "sky_masks").exists()
    command = calls[0]
    assert command[command.index("--device") + 1] == "cpu"
    assert "--process_dynamic_mask" in command

    monkeypatch.setattr(masks_segformer, "run_logged", lambda command, log_path, **kwargs: None)
    with pytest.raises(RuntimeError, match="sky_masks"):
        PrepMasksSegformerStage().run(
            Context(scene, tmp_path / "cache" / "m2"), {}, PrepMasksSegformerParams(dynamic=False)
        )


def test_smpl_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene = make_scene(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        calls.append(command)
        view = Path(command[command.index("--data_root") + 1]) / "000"
        assert not (view / "humanpose").exists()  # mục sắp ghi không có trong view
        (view / "humanpose").mkdir()
        (view / "humanpose" / "smpl.pkl").write_text("new")

    monkeypatch.setattr(smpl, "run_logged", fake_run)
    model = tmp_path / "SMPL_NEUTRAL.pkl"
    model.write_text("smpl")
    monkeypatch.setenv("D2S_SMPL_MODEL", str(model))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ctx = Context(scene, tmp_path / "cache" / "smpl")
    assert PrepSmplStage().run(ctx, {}, NoParams()) == {"smpl": ctx.output_dir}
    # PHALP và 4D-Humans tìm SMPL trong ~/.cache: đã trỏ tới file của người dùng
    for relative in smpl.SMPL_CACHE_PATHS:
        assert (tmp_path / "home" / ".cache" / relative).read_text() == "smpl"
    assert (ctx.output_dir / "humanpose" / "smpl.pkl").read_text() == "new"
    assert (scene / "humanpose" / "smpl.pkl").read_text() == "original"
    assert calls[0][calls[0].index("--dataset") + 1] == "nuscenes"
    assert calls[0][:3] == [sys.executable, "-m", "drive2sim.recon.preprocess.smpl.run"]


def test_smpl_stage_reports_the_logged_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """smpl/run.py bắt lỗi từng cảnh rồi thoát 0: báo lại dòng lỗi trong log."""
    scene = make_scene(tmp_path)

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        log_path.write_text("start\nERROR Error processing scene 0: no such file\n")

    monkeypatch.setattr(smpl, "run_logged", fake_run)
    model = tmp_path / "SMPL_NEUTRAL.pkl"
    model.write_text("smpl")
    monkeypatch.setenv("D2S_SMPL_MODEL", str(model))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.raises(RuntimeError, match="Error processing scene 0"):
        PrepSmplStage().run(Context(scene, tmp_path / "cache" / "smpl"), {}, NoParams())


def test_link_smpl_model_keeps_existing_files(tmp_path: Path) -> None:
    model = tmp_path / "SMPL_NEUTRAL.pkl"
    model.write_text("smpl")
    own = tmp_path / ".cache" / smpl.SMPL_CACHE_PATHS[0]
    own.parent.mkdir(parents=True)
    own.write_text("user's own")
    created = smpl.link_smpl_model(model, tmp_path / ".cache")
    assert created == [tmp_path / ".cache" / smpl.SMPL_CACHE_PATHS[1]]
    assert own.read_text() == "user's own"


def test_train_view_uses_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene = make_scene(tmp_path)
    masks = tmp_path / "cache" / "masks"
    (masks / "sky_masks").mkdir(parents=True)
    commands: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        commands.append(command)
        (log_path.parent / "checkpoint_final.pth").write_text("x")

    monkeypatch.setattr(drivestudio, "run_logged", fake_run)
    ctx = Context(scene, tmp_path / "run" / "train")
    params = TrainDrivestudioParams(method="streetgs")
    TrainDrivestudioStage().run(ctx, {"masks": masks}, params)
    view = ctx.output_dir / "scene_view" / "000"
    assert os.path.realpath(view / "sky_masks") == str(masks.resolve() / "sky_masks")
    assert not (view / "humanpose").exists()  # cảnh có humanpose nhưng run không có SMPL
    command = commands[0]
    assert "data.pixel_source.load_smpl=false" in command
    assert not any(part.startswith("data.pixel_source.load_sky_mask=") for part in command)


def test_approaches_include_prep_stages() -> None:
    stages = {
        name: [
            entry.stage
            for entry in files.load(paths.APPROACHES_DIR / f"{name}.yaml", Approach).stages
        ]
        for name in ("omnire", "streetgs")
    }
    assert stages["omnire"][:2] == ["prep.masks_segformer", "prep.smpl"]
    assert stages["streetgs"][0] == "prep.masks_segformer"
    assert "prep.smpl" not in stages["streetgs"]


def test_smpl_command_for_kitti360_reuses_kitti_branch(tmp_path: Path) -> None:
    # KITTI-360 có nhánh riêng trong smpl/run.py (dùng lại phép chiếu box của KITTI)
    command = build_smpl_command("kitti360", tmp_path)
    assert command[command.index("--dataset") + 1] == "kitti360"
    assert "kitti" in build_smpl_command("kitti", tmp_path)
    # chỉ hai camera pinhole (đọc file: module import OpenCV, không có trong môi trường test nhẹ)
    source = (paths.PACKAGE_DIR / "recon" / "preprocess" / "kitti360" / "human.py").read_text()
    assert "CAMERA_LIST = [0, 1]" in source


def test_smpl_skips_kitti_layout_scene_without_people(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = make_scene(tmp_path)
    ctx = Context(scene, tmp_path / "cache" / "smpl")
    kitti = dict(
        MANIFEST, scene_id="kitti360_0003", source=dict(MANIFEST["source"], dataset="kitti360")
    )
    ctx.scene = Scene(scene.name, scene, Manifest.model_validate(kitti))
    (scene / "instances").mkdir()
    cars = {"0": {"id": 0, "class_name": "Car"}}
    (scene / "instances" / "instances_info.json").write_text(json.dumps(cars))
    assert PrepSmplStage().run(ctx, {}, NoParams()) == {"smpl": ctx.output_dir}
    assert not (ctx.output_dir / "humanpose").exists()
    # Có người: chạy 4D-Humans như mọi dataset (ở đây giả lập lệnh)
    people = {**cars, "1": {"id": 1, "class_name": "Pedestrian"}}
    (scene / "instances" / "instances_info.json").write_text(json.dumps(people))
    calls: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        calls.append(command)
        view = Path(command[command.index("--data_root") + 1]) / "000"
        (view / "humanpose").mkdir(exist_ok=True)
        (view / "humanpose" / "smpl.pkl").write_text("new")

    monkeypatch.setattr(smpl, "run_logged", fake_run)
    model = tmp_path / "SMPL_NEUTRAL.pkl"
    model.write_text("smpl")
    monkeypatch.setenv("D2S_SMPL_MODEL", str(model))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    PrepSmplStage().run(ctx, {}, NoParams())
    assert calls[0][calls[0].index("--dataset") + 1] == "kitti360"
    assert (ctx.output_dir / "humanpose" / "smpl.pkl").read_text() == "new"
