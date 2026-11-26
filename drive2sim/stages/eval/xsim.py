"""Stage đánh giá checkpoint XSIM trên đúng các cặp test của split.

Worker XSIM render ảnh test và video trong môi trường riêng; stage này chấm điểm trong môi trường
chính bằng ``compute_image_metrics``, ảnh thật và mask đánh giá của gói quan sát, cùng LPIPS
torchmetrics với trainer DriveStudio, rồi ghi ``summary.json`` cho Studio và CLI. Xem CONTRACTS
mục 6.3 và 9.
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, EvalProtocol, LpipsSetting, Summary
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.commands import run_logged
from drive2sim.stages.eval.standard import PROTOCOL_VERSION, SUMMARY_FILE
from drive2sim.stages.xsim_worker import run_xsim_worker

RENDERS_DIR = "renders"
RENDERS_INDEX = "renders.json"
SCORES_FILE = "scores.json"
SCORE_LOG = "score.log"


@register_stage("eval.xsim")
class EvalXsimStage(Stage):
    """Render bằng XSIM, chấm bằng scorer của tool, xuất báo cáo đánh giá."""

    version: ClassVar[str] = "0.3.0"
    inputs: ClassVar[dict[str, str]] = {"model": "model", "observations": "observations"}
    outputs: ClassVar[dict[str, str]] = {"report": "eval_report"}

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy worker render rồi chấm các cặp test của split bằng tiến trình riêng.

        Chấm điểm chạy tiến trình con ở môi trường chính (như ``train.py`` của DriveStudio), nên
        tiến trình engine, Studio không nạp PyTorch hay giữ bộ nhớ GPU, và huỷ được giữa chừng.

        Raises:
            ValueError: Nếu ảnh render không thuộc đúng split của run.
        """
        # bundle cần numpy, Pillow: chỉ nạp khi stage chạy, để engine nạp stage trên máy nhẹ.
        from drive2sim.external.xsim.bundle import Observations

        outputs = run_xsim_worker(ctx, inputs, params, self.stage_id, "report", RENDERS_INDEX)
        index = json.loads((ctx.output_dir / RENDERS_INDEX).read_text("utf-8"))
        if index["split_id"] != ctx.split.split_id:
            raise ValueError("ảnh render XSIM không thuộc split của run")
        ctx.progress(0.9, "chấm ảnh test")
        scores_path = ctx.output_dir / SCORES_FILE
        scores_path.unlink(missing_ok=True)
        run_logged(
            [
                sys.executable,
                "-m",
                "drive2sim.external.xsim.scoring",
                "--observations",
                str(inputs["observations"]),
                "--renders",
                str(ctx.output_dir / RENDERS_DIR),
                "--out",
                str(scores_path),
            ],
            ctx.output_dir / SCORE_LOG,
            should_stop=ctx.should_stop,
        )
        scores = json.loads(scores_path.read_text("utf-8"))
        observations = Observations(inputs["observations"])
        size = {(c.width, c.height) for c in observations.cameras.values()}
        files.dump(
            Summary(
                schema_version=SCHEMA_VERSIONS["summary"],
                protocol=EvalProtocol(
                    version=PROTOCOL_VERSION,
                    image_dtype="uint8",
                    resolution=", ".join(f"{w}x{h}" for w, h in sorted(size)),
                    lpips=LpipsSetting(
                        net="alex",
                        package="torchmetrics",
                        package_version=version("torchmetrics"),
                    ),
                ),
                split_id=ctx.split.split_id,
                num_iters=int(index["step"]),
                overall=scores["overall"],
                per_camera=scores["per_camera"],
                shifted=scores.get("shifted"),
            ),
            ctx.output_dir / SUMMARY_FILE,
        )
        ctx.progress(1.0, f"đã chấm {scores['count']} ảnh test")
        return outputs
