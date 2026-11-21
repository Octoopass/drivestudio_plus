"""Stage đánh giá checkpoint XSIM trên đúng các cặp test của split.

Engine gọi EvalXsimStage để worker native vẽ ảnh và ghi báo cáo metric dùng
chung cho Studio và CLI. Xem CONTRACTS mục 6.3 và 9.
"""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import Summary
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.xsim_worker import run_xsim_worker


@register_stage("eval.xsim")
class EvalXsimStage(Stage):
    """Dùng renderer XSIM và scorer RGB chung để xuất báo cáo đánh giá."""

    version: ClassVar[str] = "0.1.0"
    uses_lidar: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {
        "scene": "scene",
        "model": "model",
        "clip": "xsim_scene",
        "masks": "masks",
    }
    outputs: ClassVar[dict[str, str]] = {"report": "eval_report"}

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy worker rồi kiểm tra báo cáo thuộc đúng split trước khi engine ghi done."""
        outputs = run_xsim_worker(ctx, inputs, params, self.stage_id, "report", "summary.json")
        summary = files.load(ctx.output_dir / "summary.json", Summary)
        if summary.split_id != ctx.split.split_id:
            raise ValueError("Báo cáo XSIM không khớp split của run")
        return outputs
