"""Kiểm tra stage ``eval.ply`` (R3, export gap của ``.ply`` nền tĩnh) trên máy không có GPU.

File kiểm tra:
- stage đăng ký được, và mọi approach xuất ``.ply`` đặt ``eval.ply`` ngay sau ``export.ply_world``;
- ``build_eval_command`` dựng đúng câu lệnh ``recon/tools/eval_ply.py``;
- ``run`` chạy hai lệnh theo thứ tự (xuất PLY chỉ nền, rồi đo), từ chối ``frames.jsonl`` chưa có
  ``psnr_static`` (protocol dưới 0.5) bằng thông báo tiếng Việt, và báo lỗi khi không ra
  ``ply_gap.json``.
Phần vẽ PLY bằng GPU (``eval_ply.py``) chưa chạy thật ở đây (ARCHITECTURE mục 7.1).
"""

import json
import sys
from pathlib import Path

import pytest

from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.registry import REGISTRY, load_builtin_stages
from drive2sim.engine.stage import Scene
from drive2sim.stages.eval import ply
from drive2sim.stages.eval.ply import (
    EvalPlyStage,
    build_eval_command,
    check_frames,
)
from drive2sim.stages.eval.standard import FRAMES_FILE
from tests.drive2sim.samples import MANIFEST, SPLIT

EXPORT_APPROACHES = ("omnire", "streetgs", "pvg", "deformablegs", "beziergs")


class FakeContext:
    """Ngữ cảnh tối thiểu cho ``Stage.run``."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.scene = Scene("nuscenes_scene-0061", output_dir, Manifest.model_validate(MANIFEST))
        self.split = Split.model_validate(SPLIT)

    def progress(self, fraction: float, message: str = "") -> None:
        """Bỏ qua tiến độ."""

    def report_peak_vram(self, megabytes: int) -> None:
        """Bỏ qua VRAM."""

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Bỏ qua metric."""

    def should_stop(self) -> bool:
        """Không bao giờ huỷ."""
        return False


def test_stage_registers_with_report_input() -> None:
    load_builtin_stages()
    stage = REGISTRY.get("eval.ply")
    assert stage is EvalPlyStage and stage.version == "0.1.0"
    assert stage.inputs["model"] == "model" and stage.inputs["report"] == "eval_report"


@pytest.mark.parametrize("approach_id", EXPORT_APPROACHES)
def test_eval_ply_runs_last(approach_id: str) -> None:
    # sau eval.standard (cần frames.jsonl của nó) và export.ply_world, và là stage cuối: stage mới
    # chưa chạy thật trên GPU nên nếu lỗi thì không làm mất export.preview
    stages = [entry.stage for entry in load_approach(approach_id).stages]
    assert stages[-1] == "eval.ply"
    assert stages.index("eval.standard") < stages.index("export.ply_world") < len(stages) - 1


def test_nurec_approach_has_no_eval_ply() -> None:
    assert "eval.ply" not in [entry.stage for entry in load_approach("nurec").stages]


def test_build_eval_command(tmp_path: Path) -> None:
    command = build_eval_command(
        tmp_path / "model",
        tmp_path / "out" / "background" / "scene.ply",
        tmp_path / "report" / FRAMES_FILE,
        tmp_path / "out",
        tmp_path / "out" / "scene_view",
    )
    assert command[:3] == [sys.executable, "-m", "drive2sim.recon.tools.eval_ply"]
    assert command[command.index("--model_dir") + 1] == str(tmp_path / "model")
    assert command[command.index("--ply") + 1].endswith("background/scene.ply")
    assert command[command.index("--frames") + 1].endswith("report/frames.jsonl")
    assert command[command.index("--out") + 1] == str(tmp_path / "out")
    assert command[-1] == f"data.data_root={tmp_path / 'out' / 'scene_view'}"


def write_frames(directory: Path, record: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FRAMES_FILE
    path.write_text(json.dumps(record) + "\n")
    return path


def test_check_frames_rejects_frames_without_static_psnr(tmp_path: Path) -> None:
    old = write_frames(tmp_path / "old", {"frame_index": 8, "camera_index": 0, "psnr": 25.0})
    with pytest.raises(ValueError, match="psnr_static.*0.5"):
        check_frames(old)
    with pytest.raises(ValueError, match="frames.jsonl"):
        check_frames(tmp_path / "missing" / FRAMES_FILE)
    new = write_frames(
        tmp_path / "new", {"frame_index": 8, "camera_index": 0, "psnr_static": None}
    )
    check_frames(new)  # khoá có mặt (giá trị null khi không còn điểm nền tĩnh) là đủ


def make_inputs(tmp_path: Path, record: dict) -> dict[str, Path]:
    model = tmp_path / "model"
    model.mkdir()
    report = tmp_path / "report"
    write_frames(report, record)
    return {"model": model, "report": report}


def test_run_exports_background_then_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = make_inputs(tmp_path, {"frame_index": 8, "camera_index": 0, "psnr_static": 25.0})
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    commands: list[list[str]] = []

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        commands.append(command)
        if "drive2sim.recon.tools.export_ply" in command:
            (output_dir / ply.BACKGROUND_DIR).mkdir()
            (output_dir / ply.BACKGROUND_DIR / "scene.ply").write_bytes(b"ply")
        else:
            (output_dir / ply.PLY_GAP_FILE).write_text(json.dumps({"export_gap_db": 0.4}))

    monkeypatch.setattr(ply, "run_logged", fake_run)
    monkeypatch.setattr(ply, "build_view", lambda *args, **kwargs: None)
    result = EvalPlyStage().run(FakeContext(output_dir), inputs, EvalPlyStage.Params())
    assert result == {"gap": output_dir}
    export, measure = commands
    assert export[export.index("--rigid_frame") + 1] == "-1"  # chỉ nền, không xe
    assert export[export.index("--out") + 1] == str(output_dir / ply.BACKGROUND_DIR)
    assert measure[measure.index("--ply") + 1] == str(
        output_dir / ply.BACKGROUND_DIR / "scene.ply"
    )
    assert measure[measure.index("--frames") + 1] == str(inputs["report"] / FRAMES_FILE)


def test_run_does_not_start_gpu_work_on_stale_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = make_inputs(tmp_path, {"frame_index": 8, "camera_index": 0, "psnr": 25.0})
    called: list[list[str]] = []
    monkeypatch.setattr(ply, "run_logged", lambda command, *args, **kwargs: called.append(command))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="psnr_static"):
        EvalPlyStage().run(FakeContext(output_dir), inputs, EvalPlyStage.Params())
    assert called == []


def test_run_fails_without_gap_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = make_inputs(tmp_path, {"frame_index": 8, "camera_index": 0, "psnr_static": 25.0})
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run(command: list[str], log_path: Path, **kwargs: object) -> None:
        if "drive2sim.recon.tools.export_ply" in command:  # xuất PLY được nhưng đo thì không ra
            (output_dir / ply.BACKGROUND_DIR).mkdir()
            (output_dir / ply.BACKGROUND_DIR / "scene.ply").write_bytes(b"ply")

    monkeypatch.setattr(ply, "run_logged", fake_run)
    monkeypatch.setattr(ply, "build_view", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="ply_gap.json"):
        EvalPlyStage().run(FakeContext(output_dir), inputs, EvalPlyStage.Params())
