"""Kiểm tra ba stage bọc mã DriveStudio: train, đánh giá và xuất PLY.

Máy test không có GPU nên không chạy DriveStudio thật. File này kiểm tra:
- ba stage đăng ký được vào bảng chung với khai báo hợp lệ;
- ``train.drivestudio`` chọn đúng file config có sẵn theo tổ hợp tham số, từ chối tổ hợp chưa có
  config, và dựng câu lệnh đúng: split đổi sang ``test_image_stride``, đoạn khung, chỉ số camera
  theo manifest, thư mục log trùng thư mục stage, ``--resume_from`` khi có checkpoint dở;
- cảnh KITTI-360 chọn config ``kitti360/4cams`` hoặc ``kitti360/2cams`` theo số camera, và hai
  config này gộp ``extends`` ra đúng loader;
- ``run`` của stage train trả về thư mục model khi có checkpoint cuối, báo lỗi khi thiếu;
- ``run_logged`` ghi log và báo lỗi kèm phần cuối log khi lệnh thất bại;
- ``eval.standard`` đọc file metric mới nhất của DriveStudio và ghi ``summary.json`` hợp lệ;
- ``export.ply_world`` dựng câu lệnh export đúng; ``export.preview`` chạy và kiểm tra
  ``preview.json``.
"""

import json
import math
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.config import load_config
from drive2sim.core.schema import Manifest, Split, Summary
from drive2sim.engine.registry import REGISTRY, load_builtin_stages
from drive2sim.engine.stage import Scene
from drive2sim.stages import commands
from drive2sim.stages.eval.standard import (
    make_summary,
    read_drivestudio_metrics,
    read_train_stats,
)
from drive2sim.stages.export import preview as export_preview
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
    assert "train.gsplat_3dgut" not in REGISTRY
    for stage_id in (
        "train.drivestudio",
        "eval.standard",
        "export.ply_world",
        "export.preview",
    ):
        assert stage_id in REGISTRY


def test_every_config_file_exists() -> None:
    for name in CONFIG_FILES.values():
        assert (paths.CONFIGS_DIR / "methods" / name).is_file(), name


def test_params_pick_existing_config() -> None:
    params = TrainDrivestudioParams(method="streetgs", renderer="3dgeer", fastgs=True)
    assert params.config_file.name == "streetgs_geer_fastgs.yaml"
    params = TrainDrivestudioParams(method="streetgs", renderer="3dgut", fastergs=True)
    assert params.config_file.name == "streetgs_ut_fastergs.yaml"
    assert TrainDrivestudioParams(method="omnire", campose=True).config_file.name == "omnire.yaml"


def test_beziergs_config_file() -> None:
    params = drivestudio.TrainDrivestudioParams(method="beziergs")
    assert params.config_file.name == "beziergs.yaml"


def test_beziergs_config_loads_with_bezier_nodes() -> None:
    config = load_config(paths.CONFIGS_DIR / "methods" / "beziergs.yaml")
    assert config["trainer"]["type"].endswith("BezierTrainer")
    assert "BezierNodes" in config["model"] and "RigidNodes" not in config["model"]
    assert "CamPose" not in config["model"]
    assert config["model"]["BezierNodes"]["bezier"]["order"] == 3
    assert config["trainer"]["losses"]["bezier_icc"]["w"] == 0.01
    # BezierGS giữ LR đường đi bằng 1% LR xyz; giá trị tuyệt đối chỉnh theo cảnh nên không ghim.
    xyz_lr = config["trainer"]["gaussian_optim_general_cfg"]["xyz"]["lr"]
    assert config["model"]["BezierNodes"]["optim"]["traj"]["lr"] == pytest.approx(xyz_lr * 0.01)


def test_beziergs_fastgs_config_file() -> None:
    params = drivestudio.TrainDrivestudioParams(method="beziergs", fastgs=True)
    assert params.config_file.name == "beziergs_fastgs.yaml"


def test_beziergs_fastgs_config_keeps_bezier_nodes_and_enables_fastgs() -> None:
    config = load_config(paths.CONFIGS_DIR / "methods" / "beziergs_fastgs.yaml")
    assert config["trainer"]["type"].endswith("BezierTrainer")
    assert "BezierNodes" in config["model"] and "RigidNodes" not in config["model"]
    assert config["trainer"]["fastgs"]["enabled"] is True


def test_params_reject_missing_combination() -> None:
    with pytest.raises(ValidationError, match="chưa có config"):
        TrainDrivestudioParams(method="omnire", renderer="3dgut", campose=True)


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
    assert option(command, "data.preload_device") == "cpu"
    assert option(command, "trainer.optim.num_iters") == "100"
    assert not any(part.startswith("logging.profile_at=") for part in command)
    profiled = TrainDrivestudioParams(method="omnire", profile_at=[2000, 20000], profile_steps=20)
    profiled_command = build_command(profiled, split, manifest, view_root, output_dir)
    assert option(profiled_command, "logging.profile_at") == "[2000,20000]"
    assert option(profiled_command, "logging.profile_steps") == "20"
    assert not any(part.startswith("trainer.densify_stats") for part in command)
    stats = TrainDrivestudioParams(method="omnire", renderer="3dgut", densify_stats=True)
    stats_command = build_command(stats, split, manifest, view_root, output_dir)
    assert option(stats_command, "trainer.densify_stats.enabled") in ("True", "true")
    # checkpoint theo bước: mặc định train.py xoá sau khi xong; tham số giữ lại cho đo bước giữa
    assert not any("keep_step_checkpoints" in part for part in command)
    keep = TrainDrivestudioParams(method="omnire", renderer="3dgut", keep_step_checkpoints=True)
    keep_command = build_command(keep, split, manifest, view_root, output_dir)
    assert option(keep_command, "logging.keep_step_checkpoints") in ("True", "true")
    # đo test giữa chừng: mặc định để train.py chọn nửa số bước; chọn bước hoặc tắt bằng eval_at
    assert not any("logging.eval_at" in part for part in command)
    at = TrainDrivestudioParams(method="omnire", renderer="3dgut", eval_at=[10000, 20000])
    assert (
        option(build_command(at, split, manifest, view_root, output_dir), "logging.eval_at")
        == "[10000,20000]"
    )
    off = TrainDrivestudioParams(method="omnire", renderer="3dgut", eval_at=[])
    assert (
        option(build_command(off, split, manifest, view_root, output_dir), "logging.eval_at")
        == "[]"
    )
    assert not any(part.startswith("trainer.type=") for part in stats_command)  # native
    screen = TrainDrivestudioParams(method="omnire", renderer="3dgut", ut_densify="screen")
    screen_command = build_command(screen, split, manifest, view_root, output_dir)
    assert option(screen_command, "trainer.type") == drivestudio.SCREEN_GRAD_TRAINER
    other = TrainDrivestudioParams(method="omnire", renderer="3dgeer", ut_densify="screen")
    other_command = build_command(other, split, manifest, view_root, output_dir)
    assert not any(part.startswith("trainer.type=") for part in other_command)
    assert not any("refine_interval" in part for part in command)  # mặc định theo config
    every = TrainDrivestudioParams(method="omnire", renderer="3dgeer", refine_interval=100)
    every_command = build_command(every, split, manifest, view_root, output_dir)
    assert option(every_command, "trainer.gaussian_ctrl_general_cfg.refine_interval") == "100"
    # 3DGUT, 3DGEER không vẽ được tia từ 90 độ trở ra: stage đặt giới hạn của renderer.
    ut = TrainDrivestudioParams(method="omnire", renderer="3dgut")
    ut_command = build_command(ut, split, manifest, view_root, output_dir)
    assert option(ut_command, "data.pixel_source.fisheye_renderer_max_angle") == str(
        drivestudio.RENDERER_MAX_ANGLE
    )
    assert drivestudio.RENDERER_MAX_ANGLE < math.pi / 2
    geer = TrainDrivestudioParams(method="omnire", renderer="3dgeer")
    geer_command = build_command(geer, split, manifest, view_root, output_dir)
    assert option(geer_command, "data.pixel_source.fisheye_renderer_max_angle") == str(
        drivestudio.RENDERER_MAX_ANGLE
    )
    assert not any("data.pixel_source.fisheye_max_angle" in part for part in command)
    wide = TrainDrivestudioParams(method="omnire", renderer="3dgut", fisheye_max_angle=1.536)
    wide_command = build_command(wide, split, manifest, view_root, output_dir)
    assert option(wide_command, "data.pixel_source.fisheye_max_angle") == "1.536"
    with pytest.raises(ValueError):
        TrainDrivestudioParams(method="omnire", fisheye_max_angle=3.5)  # quá π
    # --set train.drivestudio.profile_at=2000,20000 cho ra chuỗi; tham số tự đổi sang số
    assert TrainDrivestudioParams(method="omnire", profile_at=["2000", "20000"]).profile_at == [
        2000,
        20000,
    ]
    # một bước (--set train.drivestudio.profile_at=2000) đọc ra số
    assert TrainDrivestudioParams(method="omnire", profile_at=2000).profile_at == [2000]
    assert option(command, "a.b") == "true"


def test_build_command_camera_index_follows_manifest(tmp_path: Path) -> None:
    split = Split.model_validate(
        {**SPLIT, "cameras": ["CAM_FISH"], "train": [[0, "CAM_FISH"]], "test": []}
    )
    params = TrainDrivestudioParams(method="streetgs", dataset="nuscenes/1cams")
    command = build_command(params, split, Manifest.model_validate(MANIFEST), tmp_path, tmp_path)
    assert option(command, "data.pixel_source.cameras") == "[1]"
    assert option(command, "dataset") == "nuscenes/1cams"
    # Camera fisheye: train trên ảnh gốc, không khử méo trong loader.
    assert option(command, "data.pixel_source.undistort") == "false"
    pinhole = build_command(
        params, Split.model_validate(SPLIT), Manifest.model_validate(MANIFEST), tmp_path, tmp_path
    )
    assert not any(part.startswith("data.pixel_source.undistort=") for part in pinhole)


def test_dataset_config_for() -> None:
    assert dataset_config_for("nuscenes", 3) == "nuscenes/3cams"
    assert dataset_config_for("argoverse2", 7) == "argoverse/7cams"
    with pytest.raises(ValueError, match="nuscenes/1cams"):
        dataset_config_for("nuscenes", 2)
    with pytest.raises(ValueError, match="chưa có loader"):
        dataset_config_for("colmap", 1)


def kitti360_manifest() -> Manifest:
    """Manifest của cảnh KITTI-360: hai camera pinhole, hai camera mắt cá."""
    cameras = [
        {"camera_id": camera_id, "model": model, "resolution": size, "shutter": "GLOBAL"}
        for camera_id, model, size in (
            ("image_00", "opencv_pinhole", [1408, 376]),
            ("image_01", "opencv_pinhole", [1408, 376]),
            ("image_02", "opencv_fisheye", [1400, 1400]),
            ("image_03", "opencv_fisheye", [1400, 1400]),
        )
    ]
    return Manifest.model_validate(
        {
            **MANIFEST,
            "scene_id": "kitti360_2013_05_28_drive_0003_sync",
            "source": {**MANIFEST["source"], "dataset": "kitti360", "converter": "kitti360"},
            "track": "fisheye",
            "cameras": cameras,
            "timeline": {"reference_sensor": "image_00", "num_frames": 20},
        }
    )


@pytest.mark.parametrize(
    "cameras,config,undistort",
    [
        (["image_00", "image_01", "image_02", "image_03"], "kitti360/4cams", "false"),
        (["image_00", "image_01"], "kitti360/2cams", None),
    ],
)
def test_kitti360_config_resolves(
    tmp_path: Path, cameras: list[str], config: str, undistort: object
) -> None:
    split = Split.model_validate(
        {
            **SPLIT,
            "scene_id": "kitti360_2013_05_28_drive_0003_sync",
            "frame_range": [0, 19],
            "cameras": cameras,
            "train": [[0, cameras[0]]],
            "test": [],
        }
    )
    params = TrainDrivestudioParams(method="omnire")
    command = build_command(params, split, kitti360_manifest(), tmp_path, tmp_path)
    assert option(command, "dataset") == config
    assert option(command, "data.pixel_source.cameras") == str(list(range(len(cameras)))).replace(
        " ", ""
    )
    if undistort is None:
        assert not any(part.startswith("data.pixel_source.undistort=") for part in command)
    else:
        assert option(command, "data.pixel_source.undistort") == undistort
    data = load_config(paths.CONFIGS_DIR / "datasets" / f"{config}.yaml")["data"]
    assert data["dataset"] == "kitti360"
    pixel = data["pixel_source"]
    assert pixel["type"] == "drive2sim.recon.data.loaders.kitti360.KITTI360PixelSource"
    assert len(pixel["cameras"]) == len(pixel["downscale_when_loading"]) == len(cameras)
    assert pixel["undistort"] is False
    assert data["lidar_source"]["type"] == "drive2sim.recon.data.loaders.kitti.KITTILiDARSource"
    with pytest.raises(ValueError, match="kitti360/2cams"):
        dataset_config_for("kitti360", 3)


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

    def report_peak_vram(self, megabytes: int) -> None:
        """Ghi lại VRAM stage báo."""
        self.peak_vram_mb = megabytes

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
        (tmp_path / "train_stats.json").write_text(json.dumps({"peak_vram_mb": 4727}))

    monkeypatch.setattr(drivestudio, "run_logged", fake_run)
    ctx = FakeContext(tmp_path)
    params = TrainDrivestudioParams(method="omnire")
    assert TrainDrivestudioStage().run(ctx, {}, params) == {"model": tmp_path}
    assert len(calls) == 1
    assert ctx.peak_vram_mb == 4727  # VRAM của train.py (tiến trình con) báo cho engine
    assert ctx.events[-1][0] == 1.0
    # DriveStudio nhận bản config đã gộp extends, ghi trong thư mục stage
    config = Path(calls[0][calls[0].index("--config_file") + 1])
    assert config == tmp_path / drivestudio.METHOD_CONFIG
    assert load_config(config) == load_config(params.config_file)


def test_params_reject_unreadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods" / "omnire_nocampose.yaml").write_text("extends: nope.yaml\n")
    monkeypatch.setattr(drivestudio.paths, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="nope.yaml"):
        TrainDrivestudioParams(method="omnire")


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


def test_eval_adds_train_stats_and_render_speed(tmp_path: Path) -> None:
    write_metrics(
        tmp_path,
        "images_test_2026-11-12_090000.json",
        {
            "psnr": 27.6,
            "ssim": 0.85,
            "lpips": 0.2,
            "masked_psnr": 23.1,
            "render_fps": 41.5,
            "render_width": 800,
            "render_height": 450,
        },
    )
    assert read_train_stats(tmp_path) == {}  # model cũ: không có train_stats.json
    (tmp_path / "train_stats.json").write_text(
        json.dumps(
            {
                "step": 30000,
                "num_iters": 30000,
                "train_time_s": 4100.5,
                "peak_vram_mb": 7400,
                "num_gaussians": 1250000,
                "gaussians_per_node": {"Background": 1000000, "RigidNodes": 250000},
                "speedups": ["fastgs", "fastergs.fused_adam"],
                "eval_time_s": 420.0,
            }
        )
    )
    summary = make_summary(
        read_drivestudio_metrics(tmp_path), "every8_front1", tmp_path, read_train_stats(tmp_path)
    )
    assert summary.overall.dynamic_psnr == 23.1
    assert (summary.train_time_s, summary.peak_vram_mb) == (4100.5, 7400)
    assert (summary.num_gaussians, summary.num_iters) == (1250000, 30000)
    assert summary.render_fps == 41.5 and summary.render_resolution == "800x450"
    assert summary.speedups == ["fastgs", "fastergs.fused_adam"]
    assert summary.eval_time_s == 420.0
    # thiếu train_stats.json thì các trường để trống, không lỗi
    bare = make_summary(read_drivestudio_metrics(tmp_path), "every8_front1", tmp_path)
    assert bare.train_time_s is None and bare.num_gaussians is None and bare.speedups == []


def test_eval_maps_camera_index_to_camera_id(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    values = {
        "image_metrics/test/psnr": 25.0,
        "image_metrics/test/ssim": 0.8,
        "image_metrics/test/lpips": 0.2,
    }
    for index, psnr in (("0", 26.0), ("2", 22.0)):
        for name, value in (("psnr", psnr), ("ssim", 0.8), ("lpips", 0.2)):
            values[f"image_metrics/test/per_camera/{index}/{name}"] = value
    (metrics_dir / "images_test_2026-11-17_120000.json").write_text(json.dumps(values))
    metrics = read_drivestudio_metrics(tmp_path)
    summary = make_summary(
        metrics,
        "every10_mixed4",
        tmp_path,
        camera_ids=["image_00", "image_01", "image_02", "image_03"],
    )
    assert summary.per_camera["image_00"].psnr == 26.0
    assert summary.per_camera["image_02"].psnr == 22.0
    assert set(summary.per_camera) == {"image_00", "image_02"}
    # không có danh sách camera thì giữ nguyên khoá
    assert set(make_summary(metrics, "s", tmp_path).per_camera) == {"0", "2"}


def test_eval_without_metrics(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ảnh test"):
        read_drivestudio_metrics(tmp_path)


def test_export_command(tmp_path: Path) -> None:
    command = build_export_command(ExportPlyWorldParams(rigid_frame=0), tmp_path / "m", tmp_path)
    assert command[command.index("--ckpt") + 1] == str(tmp_path / "m" / "checkpoint_final.pth")
    assert command[command.index("--rigid_frame") + 1] == "0"


def test_preview_stage_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "runs" / "s" / "omnire" / "r1" / "d2s" / "stages" / "export.preview"
    output_dir.mkdir(parents=True)
    commands: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        commands.append(command)
        (output_dir / "background.ply").write_bytes(b"ply")
        preview = {"schema_version": "d2s-preview/0.1", "source_run": "x", "sh_degree": 3}
        if len(commands) > 1:
            del preview["sh_degree"]  # lần hai: preview.json sai schema
        (output_dir / "preview.json").write_text(json.dumps(preview))

    monkeypatch.setattr(export_preview, "run_logged", fake_run)
    # cameras.json của train được chép theo, để viewer mở ở đúng góc camera của video
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "cameras.json").write_text('{"cameras": []}')
    stage = export_preview.ExportPreviewStage()
    params = export_preview.ExportPreviewParams()
    assert stage.run(FakeContext(output_dir), {"model": tmp_path / "m"}, params) == {
        "preview": output_dir
    }
    assert (output_dir / "cameras.json").is_file()
    assert commands[0][commands[0].index("--source_run") + 1] == "runs/s/omnire/r1"
    with pytest.raises(ValueError):
        stage.run(FakeContext(output_dir), {"model": tmp_path / "m"}, params)


def test_omnire_renderer_configs_resolve() -> None:
    """OmniRe có đủ 3DGS, 3DGUT, 3DGEER với mọi tổ hợp FastGS, Faster-GS (không CamPose)."""
    for renderer in ("3dgs", "3dgut", "3dgeer"):
        for fastgs in (False, True):
            for fastergs in (False, True):
                params = TrainDrivestudioParams(
                    method="omnire", renderer=renderer, fastgs=fastgs, fastergs=fastergs
                )
                assert params.config_key in drivestudio.CONFIG_FILES


def test_eval_remeasures_old_metrics_from_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metric từ code đo cũ (không có per_camera) và có checkpoint: render lại tập test."""
    from drive2sim.stages.eval import standard

    model = tmp_path / "model"
    write_metrics(
        model, "images_test_2026-11-12_090000.json", {"psnr": 20.0, "ssim": 0.9, "lpips": 0.1}
    )
    (model / "checkpoint_final.pth").write_bytes(b"x")
    scene = tmp_path / "scene"
    scene.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    commands_run: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        commands_run.append(command)
        new = {
            "image_metrics/test/psnr": 21.0,
            "image_metrics/test/ssim": 0.7,
            "image_metrics/test/lpips": 0.2,
            "image_metrics/test/per_camera/0/psnr": 21.5,
            "image_metrics/test/per_camera/0/ssim": 0.7,
            "image_metrics/test/per_camera/0/lpips": 0.2,
        }
        folder = output_dir / standard.REEVAL_DIR / "metrics_eval"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "images_test_2026-11-18_000000.json").write_text(json.dumps(new))

    monkeypatch.setattr(standard, "run_logged", fake_run)
    ctx = FakeContext(output_dir)
    ctx.scene = Scene("nuscenes_scene-0061", scene, Manifest.model_validate(MANIFEST))
    standard.EvalStandardStage().run(ctx, {"model": model}, standard.NoParams())
    command = commands_run[0]
    assert "drive2sim.recon.tools.eval" in command
    assert command[command.index("--resume_from") + 1] == str(model / "checkpoint_final.pth")
    summary = files.load(output_dir / "summary.json", Summary)
    assert summary.overall.psnr == 21.0 and summary.per_camera

    # Metric đã có per_camera: không đo lại.
    write_metrics(
        model,
        "images_test_2026-11-13_090000.json",
        {"psnr": 22.0, "ssim": 0.8, "lpips": 0.1, "per_camera/0/psnr": 22.0},
    )
    standard.EvalStandardStage().run(ctx, {"model": model}, standard.NoParams())
    assert len(commands_run) == 1
