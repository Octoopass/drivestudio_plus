"""Kiểm tra eval.xsim chấm điểm bằng tiến trình con và ghi summary đúng split."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from drive2sim.core import files  # noqa: E402
from drive2sim.core.schema import Summary  # noqa: E402
from drive2sim.engine.stage import NoParams  # noqa: E402
from drive2sim.stages.eval import xsim  # noqa: E402


def test_eval_scores_in_subprocess_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "eval.xsim"
    out.mkdir()
    observations = tmp_path / "obs"
    observations.mkdir()
    camera = {
        "index": 0,
        "camera_id": "CAM_FRONT",
        "model": "pinhole",
        "width": 16,
        "height": 12,
        "intrinsics": [8, 8, 8, 6],
        "distortion": [0, 0, 0, 0, 0],
        "train_mask": False,
        "eval_mask": False,
    }
    (observations / "observations.json").write_text(
        json.dumps(
            {
                "format": "d2s-observations",
                "version": 1,
                "frames": [0],
                "timestamps_us": [1],
                "cameras": [camera],
            }
        )
    )

    def render(*args: object) -> dict:
        (out / "renders.json").write_text(json.dumps({"step": 40, "split_id": "every8_front1"}))
        return {"report": out}

    def score(command: list[str], log: Path, **kwargs: object) -> None:
        assert command[1:3] == ["-m", "drive2sim.external.xsim.scoring"]
        metrics = {"psnr": 20.0, "ssim": 0.5, "lpips": 0.2}
        Path(command[command.index("--out") + 1]).write_text(
            json.dumps({"overall": metrics, "per_camera": {"CAM_FRONT": metrics}, "count": 1})
        )

    monkeypatch.setattr(xsim, "run_xsim_worker", render)
    monkeypatch.setattr(xsim, "run_logged", score)
    monkeypatch.setattr(xsim, "version", lambda name: "1.0")
    ctx = SimpleNamespace(
        output_dir=out,
        split=SimpleNamespace(split_id="every8_front1"),
        should_stop=lambda: False,
        progress=lambda *args: None,
    )
    stage = xsim.EvalXsimStage()
    assert stage.run(ctx, {"observations": observations}, NoParams()) == {"report": out}
    summary = files.load(out / "summary.json", Summary)
    assert summary.num_iters == 40 and summary.overall.psnr == 20.0
    assert summary.protocol.resolution == "16x12"
    ctx.split.split_id = "other"
    with pytest.raises(ValueError, match="không thuộc split"):
        stage.run(ctx, {"observations": observations}, NoParams())
