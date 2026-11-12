"""Kiểm tra ba stage bọc mã DriveStudio: train, đánh giá và xuất PLY.

Máy test không có GPU nên không chạy DriveStudio thật. File này kiểm tra:
- ba stage đăng ký được vào bảng chung với khai báo hợp lệ;
- ``train.drivestudio`` chọn đúng file config có sẵn theo tổ hợp tham số, từ chối tổ hợp chưa có
  config, và dựng câu lệnh đúng: split đổi sang ``test_image_stride``, đoạn khung, chỉ số camera
  theo manifest, thư mục log trùng thư mục stage, ``--resume_from`` khi có checkpoint dở;
- ``run`` của stage train trả về thư mục model khi có checkpoint cuối, báo lỗi khi thiếu;
- ``run_logged`` ghi log và báo lỗi kèm phần cuối log khi lệnh thất bại;
- ``eval.standard`` đọc file metric mới nhất của DriveStudio và ghi ``summary.json`` hợp lệ;
- ``export.ply_world`` dựng câu lệnh export đúng.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import Manifest, Split, Summary
from drive2sim.engine.registry import REGISTRY, load_builtin_stages
from drive2sim.engine.stage import Scene
from drive2sim.stages import commands
from drive2sim.stages.eval.standard import make_summary, read_drivestudio_metrics
from drive2sim.stages.export.ply_world import ExportPlyWorldParams
from drive2sim.stages.export.ply_world import build_command as build_export_command
from drive2sim.stages.train import drivestudio
from drive2sim.stages.train.drivestudio import (
    CONFIG_FILES,
    TrainDrivestudioParams,
    TrainDrivestudioStage,
    build_command,
    dataset_config_for,
)
from tests.drive2sim.samples import MANIFEST, SPLIT


def option(command: list[str], key: str) -> str:
    """Lấy giá trị ``key=...`` trong câu lệnh."""
    prefix = key + "="
    return next(part[len(prefix) :] for part in command if part.startswith(prefix))


def test_builtin_stages_register() -> None:
    load_builtin_stages()
    for stage_id in ("train.drivestudio", "eval.standard", "export.ply_world"):
        assert stage_id in REGISTRY


def test_every_config_file_exists() -> None:
    for name in CONFIG_FILES.values():
        assert (paths.CONFIGS_DIR / "methods" / name).is_file(), name


def test_params_pick_existing_config() -> None:
    params = TrainDrivestudioParams(method="streetgs", renderer="3dgeer", fastgs=True)
    assert params.config_file.name == "streetgs_geer_fastgs.yaml"
    assert TrainDrivestudioParams(method="omnire", campose=True).config_file.name == "omnire.yaml"


def test_params_reject_missing_combination() -> None:
    with pytest.raises(ValidationError, match="chưa có config"):
        TrainDrivestudioParams(method="omnire", renderer="3dgut")


def test_build_command_translates_split(tmp_path: Path) -> None:
    manifest = Manifest.model_validate(MANIFEST)
    split = Split.model_validate(
        {
            **SPLIT,
            "frame_range": [2, 9],
            "train": [[2, "CAM_FRONT"]],
            "test": [[8, "CAM_FRONT"]],
            "strategy": "every_nth_frame:6",
        }
    )
    output_dir = tmp_path / "run" / "d2s" / "stages" / "train.drivestudio"
    view_root = output_dir / "scene_view"
    params = TrainDrivestudioParams(method="omnire", num_iters=100, overrides={"a.b": True})
    command = build_command(params, split, manifest, view_root, output_dir, has_smpl=False)

    assert command[:3] == [sys.executable, "-m", "drive2sim.recon.tools.train"]
    assert command[command.index("--config_file") + 1].endswith("omnire_nocampose.yaml")
    root = Path(command[command.index("--output_root") + 1])
    project = command[command.index("--project") + 1]
    run_name = command[command.index("--run_name") + 1]
    assert root / project / run_name == output_dir
    assert "--resume_from" not in command
    assert option(command, "dataset") == "nuscenes/1cams"
    assert option(command, "data.data_root") == str(view_root)
    assert option(command, "data.scene_idx") == "000"
    assert option(command, "data.pixel_source.load_smpl") == "false"
    assert not any(part.startswith("data.pixel_source.load_sky_mask=") for part in command)
    assert option(command, "data.start_timestep") == "2"
    assert option(command, "data.end_timestep") == "9"
    assert option(command, "data.pixel_source.cameras") == "[0]"
    assert option(command, "data.pixel_source.test_image_stride") == "6"
    assert option(command, "trainer.optim.num_iters") == "100"
    assert option(command, "a.b") == "true"


def test_build_command_camera_index_follows_manifest(tmp_path: Path) -> None:
    split = Split.model_validate(
        {**SPLIT, "cameras": ["CAM_FISH"], "train": [[0, "CAM_FISH"]], "test": []}
    )
    params = TrainDrivestudioParams(method="streetgs", dataset="nuscenes/1cams")
    command = build_command(params, split, Manifest.model_validate(MANIFEST), tmp_path, tmp_path)
    assert option(command, "data.pixel_source.cameras") == "[1]"
    assert option(command, "dataset") == "nuscenes/1cams"


def test_dataset_config_for() -> None:
    assert dataset_config_for("nuscenes", 3) == "nuscenes/3cams"
    assert dataset_config_for("argoverse2", 7) == "argoverse/7cams"
    with pytest.raises(ValueError, match="nuscenes/1cams"):
        dataset_config_for("nuscenes", 2)
    with pytest.raises(ValueError, match="chưa có loader"):
        dataset_config_for("colmap", 1)


def test_build_command_resumes(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_resume.pth").write_text("x")
    command = build_command(
        TrainDrivestudioParams(method="omnire"),
        Split.model_validate(SPLIT),
        Manifest.model_validate(MANIFEST),
        tmp_path,
        tmp_path,
    )
    assert command[command.index("--resume_from") + 1] == str(tmp_path / "checkpoint_resume.pth")


class FakeContext:
    """Ngữ cảnh tối thiểu cho ``Stage.run``."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.scene = Scene("nuscenes_scene-0061", output_dir, Manifest.model_validate(MANIFEST))
        self.split = Split.model_validate(SPLIT)
        self.events: list[tuple[float, str]] = []

    def progress(self, fraction: float, message: str = "") -> None:
        """Ghi lại tiến độ để test kiểm tra."""
        self.events.append((fraction, message))

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Bỏ qua metric."""

    def should_stop(self) -> bool:
        """Không bao giờ huỷ."""
        return False


def test_train_stage_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        calls.append(command)
        (tmp_path / "checkpoint_final.pth").write_text("x")

    monkeypatch.setattr(drivestudio, "run_logged", fake_run)
    ctx = FakeContext(tmp_path)
    params = TrainDrivestudioParams(method="omnire")
    assert TrainDrivestudioStage().run(ctx, {}, params) == {"model": tmp_path}
    assert len(calls) == 1
    assert ctx.events[-1][0] == 1.0


def test_train_stage_requires_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(drivestudio, "run_logged", lambda command, log_path, **kwargs: None)
    with pytest.raises(RuntimeError, match="checkpoint_final.pth"):
        TrainDrivestudioStage().run(
            FakeContext(tmp_path), {}, TrainDrivestudioParams(method="omnire")
        )


def test_run_logged_reports_tail(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    commands.run_logged([sys.executable, "-c", "print('hello')"], log)
    assert "hello" in log.read_text()
    with pytest.raises(RuntimeError, match="mã 3") as info:
        commands.run_logged([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], log)
    assert "boom" in str(info.value)
    assert log.read_text().count("$ ") == 2


def write_metrics(model_dir: Path, name: str, values: dict) -> None:
    (model_dir / "metrics").mkdir(parents=True, exist_ok=True)
    data = {f"image_metrics/test/{key}": value for key, value in values.items()}
    (model_dir / "metrics" / name).write_text(json.dumps(data))


def test_eval_reads_latest_metrics_and_writes_summary(tmp_path: Path) -> None:
    write_metrics(
        tmp_path, "images_test_2026-11-12_080000.json", {"psnr": 1, "ssim": 0, "lpips": 1}
    )
    write_metrics(
        tmp_path,
        "images_test_2026-11-12_090000.json",
        {"psnr": 27.6, "ssim": 0.85, "lpips": 0.2, "human_psnr": -1, "vehicle_psnr": 24.0},
    )
    (tmp_path / "config.yaml").write_text(
        "data:\n  pixel_source:\n    downscale_when_loading: [2]\n    downscale: 1\n"
    )
    metrics = read_drivestudio_metrics(tmp_path)
    assert metrics["psnr"] == 27.6
    summary = make_summary(metrics, "every8_front1", tmp_path)
    assert summary.overall.human_psnr is None
    assert summary.overall.vehicle_psnr == 24.0
    assert "downscale_when_loading=[2]" in summary.protocol.resolution
    path = files.dump(summary, tmp_path / "summary.json")
    assert files.load(path, Summary) == summary


def test_eval_without_metrics(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ảnh test"):
        read_drivestudio_metrics(tmp_path)


def test_export_command(tmp_path: Path) -> None:
    command = build_export_command(ExportPlyWorldParams(rigid_frame=0), tmp_path / "m", tmp_path)
    assert command[command.index("--ckpt") + 1] == str(tmp_path / "m" / "checkpoint_final.pth")
    assert command[command.index("--rigid_frame") + 1] == "0"
