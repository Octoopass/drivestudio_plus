"""Kiểm tra approach ``gsplat_3dgut``, stage ``train.gsplat_3dgut`` và phần thêm của eval.

File này kiểm tra:
- tham số của stage và câu lệnh train dựng từ tham số (kể cả chạy tiếp từ checkpoint);
- stage chạy trainer qua ``run_logged``, chép split vào thư mục stage, đòi checkpoint cuối;
- engine quyết định đúng stage và tham số cho approach trên cảnh có cả camera pinhole lẫn mắt cá,
  và khoá approach trên cảnh không có LiDAR hoặc có camera F-theta;
- ``eval.standard`` điền ``per_camera`` khi trainer ghi metric theo camera, và lấy độ phân giải
  từ ``train_stats.json`` khi không có ``config.yaml`` của DriveStudio;
- codebase ``gsplat`` dùng môi trường chính.

Trainer thật cần GPU nên không chạy ở đây (đã chạy thật: ``docs/methods/gsplat_3dgut.md``).
"""

import copy
import json
import math
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from drive2sim.core.schema import Approach, Manifest, Split
from drive2sim.engine import environment
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.registry import load_builtin_stages
from drive2sim.engine.resolve import BlockedError, camera_support, resolve
from drive2sim.engine.stage import Scene
from drive2sim.stages.eval.standard import make_summary, read_drivestudio_metrics
from drive2sim.stages.train import gsplat_3dgut
from drive2sim.stages.train.gsplat_3dgut import (
    TrainGsplat3dgutParams,
    TrainGsplat3dgutStage,
    build_command,
)
from tests.drive2sim.samples import MANIFEST, SPLIT

KITTI360_MANIFEST = {
    **MANIFEST,
    "scene_id": "kitti360_2013_05_28_drive_0003_sync",
    "source": {**MANIFEST["source"], "dataset": "kitti360", "converter": "kitti360"},
    "track": "fisheye",
    "cameras": [
        {"camera_id": c, "model": m, "resolution": r, "shutter": "GLOBAL"}
        for c, m, r in (
            ("image_00", "opencv_pinhole", [1408, 376]),
            ("image_01", "opencv_pinhole", [1408, 376]),
            ("image_02", "opencv_fisheye", [1400, 1400]),
            ("image_03", "opencv_fisheye", [1400, 1400]),
        )
    ],
    "lidars": ["velodyne"],
    "timeline": {"reference_sensor": "image_00", "num_frames": 20},
    "available": {"lidar": True, "cuboids": "none"},
}
KITTI360_SPLIT = {
    **SPLIT,
    "split_id": "every10_allcams",
    "scene_id": "kitti360_2013_05_28_drive_0003_sync",
    "strategy": "every_nth_frame:10",
    "frame_range": [0, 19],
    "cameras": ["image_00", "image_01", "image_02", "image_03"],
    "train": [[0, "image_00"], [0, "image_02"]],
    "test": [[10, "image_00"], [10, "image_02"]],
}


@pytest.fixture(autouse=True)
def builtin_stages() -> None:
    load_builtin_stages()


def option(command: list[str], key: str) -> str:
    return command[command.index(key) + 1]


def test_params_defaults_and_limits() -> None:
    params = TrainGsplat3dgutParams()
    assert (params.sky, params.hood_mask, params.ssim, params.optimizer) == (
        True,
        True,
        "fused",
        "selective_adam",
    )
    with pytest.raises(ValidationError):
        TrainGsplat3dgutParams(fisheye_max_angle=math.pi / 2)  # renderer UT không vẽ được
    with pytest.raises(ValidationError):
        TrainGsplat3dgutParams(ssim="off")
    with pytest.raises(ValidationError):
        TrainGsplat3dgutParams(renderer="3dgs")


def test_build_command(tmp_path: Path) -> None:
    params = TrainGsplat3dgutParams(num_iters=300, sky=False, ssim="none", optimizer="adam")
    command = build_command(params, tmp_path / "scene", tmp_path / "split.json", tmp_path)
    assert command[:3] == [sys.executable, "-m", "drive2sim.recon.tools.train_gsplat_3dgut"]
    assert option(command, "--scene_dir") == str(tmp_path / "scene")
    assert option(command, "--split_file") == str(tmp_path / "split.json")
    assert option(command, "--output_dir") == str(tmp_path)
    assert option(command, "--num_iters") == "300"
    assert option(command, "--sky") == "off"
    assert option(command, "--hood_mask") == "on"
    assert option(command, "--ssim") == "none"
    assert option(command, "--optimizer") == "adam"
    assert "--resume_from" not in command
    (tmp_path / "checkpoint_resume.pth").write_text("x")
    command = build_command(params, tmp_path / "scene", tmp_path / "split.json", tmp_path)
    assert option(command, "--resume_from") == str(tmp_path / "checkpoint_resume.pth")


class FakeContext:
    """Ngữ cảnh tối thiểu cho ``Stage.run`` trên một cảnh KITTI-360 giả."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        manifest = Manifest.model_validate(KITTI360_MANIFEST)
        self.scene = Scene(manifest.scene_id, output_dir / "scene", manifest)
        self.split = Split.model_validate(KITTI360_SPLIT)
        self.events: list[tuple[float, str]] = []

    def progress(self, fraction: float, message: str = "") -> None:
        """Ghi lại tiến độ."""
        self.events.append((fraction, message))

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Bỏ qua metric."""

    def should_stop(self) -> bool:
        """Không bao giờ huỷ."""
        return False


def test_stage_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        calls.append(command)
        (tmp_path / "checkpoint_final.pth").write_text("x")

    monkeypatch.setattr(gsplat_3dgut, "run_logged", fake_run)
    ctx = FakeContext(tmp_path)
    result = TrainGsplat3dgutStage().run(ctx, {}, TrainGsplat3dgutParams())
    assert result == {"model": tmp_path}
    split = json.loads((tmp_path / "split.json").read_text())
    assert split["split_id"] == "every10_allcams"
    assert option(calls[0], "--split_file") == str(tmp_path / "split.json")
    assert option(calls[0], "--scene_dir") == str(tmp_path / "scene")


def test_stage_requires_final_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gsplat_3dgut, "run_logged", lambda command, log_path, **kwargs: None)
    with pytest.raises(RuntimeError, match="checkpoint_final.pth"):
        TrainGsplat3dgutStage().run(FakeContext(tmp_path), {}, TrainGsplat3dgutParams())


def test_resolve_on_mixed_pinhole_fisheye_scene() -> None:
    approach = load_approach("gsplat_3dgut")
    assert approach.codebase == "gsplat"
    support = camera_support(approach)
    assert support["tracks"] == ["pinhole", "fisheye"]
    plan = resolve(
        approach,
        Manifest.model_validate(KITTI360_MANIFEST),
        Split.model_validate(KITTI360_SPLIT),
        preset="base",
        stage_params={"train.gsplat_3dgut": {"num_iters": 500}},
    )
    assert [s.stage_id for s in plan.stages] == [
        "train.gsplat_3dgut",
        "eval.standard",
        "export.ply_world",
        "export.preview",
    ]
    train = plan.stages[0]
    assert train.params["sky"] is False and train.params["ssim"] == "none"
    assert train.params["num_iters"] == 500
    assert plan.lidar_usage == ["train.gsplat_3dgut"]
    quick = resolve(
        approach,
        Manifest.model_validate(KITTI360_MANIFEST),
        Split.model_validate(KITTI360_SPLIT),
        quick=True,
    )
    assert quick.stages[0].params["num_iters"] == 3000


def test_resolve_blocks_without_lidar_or_with_ftheta() -> None:
    approach = load_approach("gsplat_3dgut")
    no_lidar = copy.deepcopy(KITTI360_MANIFEST)
    no_lidar["lidars"] = []
    no_lidar["available"]["lidar"] = False
    with pytest.raises(BlockedError):
        resolve(approach, Manifest.model_validate(no_lidar), Split.model_validate(KITTI360_SPLIT))
    ftheta = copy.deepcopy(KITTI360_MANIFEST)
    ftheta["cameras"][2]["model"] = "ftheta"
    with pytest.raises(BlockedError):
        resolve(approach, Manifest.model_validate(ftheta), Split.model_validate(KITTI360_SPLIT))


def test_approach_codebase_values() -> None:
    data = load_approach("gsplat_3dgut").model_dump(mode="json")
    for codebase in ("gsplat", "drivestudio", "external:nurec"):
        Approach.model_validate({**data, "codebase": codebase})
    with pytest.raises(ValidationError):
        Approach.model_validate({**data, "codebase": "gsplat3"})


def test_gsplat_codebase_uses_main_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, "_gpus", lambda: [("Quadro T2000", 4096, "580")])
    monkeypatch.setattr(
        environment,
        "package_versions",
        lambda: {"pytorch": "2.4.1+cu121", "cuda": "12.1", "gsplat": "1.5.3"},
    )
    monkeypatch.setattr(environment, "_cached_torch_cuda", lambda: (True, "Quadro T2000"))
    result = environment.environment_check("gsplat")
    assert result["env"] == "main" and result["level"] == "ok"


def test_eval_reads_per_camera_and_resolution_from_train_stats(tmp_path: Path) -> None:
    values = {"psnr": 20.0, "ssim": 0.7, "lpips": 0.3, "render_fps": 100.0}
    for camera, psnr in (("image_00", 21.0), ("image_02", 19.0)):
        values.update(
            {
                f"per_camera/{camera}/psnr": psnr,
                f"per_camera/{camera}/ssim": 0.7,
                f"per_camera/{camera}/lpips": 0.3,
            }
        )
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "images_test_2026-11-14_100000.json").write_text(
        json.dumps({f"image_metrics/test/{k}": v for k, v in values.items()})
    )
    (tmp_path / "train_stats.json").write_text(
        json.dumps({"train_time_s": 80.5, "num_gaussians": 1000, "resolution": "downscale=2"})
    )
    metrics = read_drivestudio_metrics(tmp_path)
    summary = make_summary(metrics, "every10_allcams", tmp_path, {"train_time_s": 80.5})
    assert summary.overall.psnr == 20.0
    assert {c: m.psnr for c, m in summary.per_camera.items()} == {
        "image_00": 21.0,
        "image_02": 19.0,
    }
    assert summary.protocol.resolution == "downscale=2"


def test_eval_without_per_camera_keeps_drivestudio_behaviour(tmp_path: Path) -> None:
    summary = make_summary({"psnr": 20.0, "ssim": 0.7, "lpips": 0.3}, "s", tmp_path)
    assert summary.per_camera == {}
    assert summary.protocol.resolution == "unknown"
