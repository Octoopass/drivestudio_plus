"""Stage train XSIM bằng interpreter riêng và hợp đồng job/result ở CONTRACTS mục 9.

Engine gọi TrainXsimStage với clip đã chuẩn bị và mask của cảnh. Stage trao đổi
đường dẫn tương đối với worker để cùng job có thể chạy ở workdir khác.
"""

import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, JobResult, JobSpec
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.commands import run_logged


class TrainXsimParams(BaseModel):
    """Tham số cho baseline camera và LiDAR với cấu hình native waymo_default."""

    model_config = ConfigDict(extra="forbid")
    num_iters: int = Field(default=40000, ge=1)
    save_every: int = Field(default=1000, ge=1)
    downscale: float = Field(default=1.0, ge=1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


@register_stage("train.xsim")
class TrainXsimStage(Stage):
    """Khởi chạy worker XSIM và xác nhận checkpoint của lần train trước khi báo xong."""

    version: ClassVar[str] = "0.1.0"
    uses_lidar: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {"scene": "scene", "clip": "xsim_scene", "masks": "masks"}
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = TrainXsimParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Ghi job, chuyển yêu cầu huỷ tới subprocess và đối chiếu kết quả."""
        root = ctx.scene.root.parent.parent.absolute()
        out = ctx.output_dir.absolute()
        relative = out.relative_to(root).as_posix()
        job = JobSpec(
            schema_version=SCHEMA_VERSIONS["job"],
            job_id=f"{ctx.run_id}/{self.stage_id}",
            kind="stage",
            stage_id=self.stage_id,
            params=params.model_dump(),
            inputs={
                k: v.absolute().relative_to(root).as_posix()
                for k, v in {**inputs, "scene": ctx.scene.root}.items()
            },
            output_dir=relative,
            env="xsim",
        )
        job_path, result_path = out / "job.json", out / "result.json"
        files.dump(job, job_path)
        result_path.unlink(missing_ok=True)
        python = os.environ.get("D2S_XSIM_PYTHON")
        command = [python] if python else ["conda", "run", "-n", "drive2sim-xsim", "python"]
        run_logged(
            command
            + [
                "-m",
                "drive2sim.external.xsim.worker",
                "--job",
                str(job_path),
                "--result",
                str(result_path),
            ],
            out / "worker.log",
            env={"D2S_WORKDIR": str(root)},
            should_stop=ctx.should_stop,
        )
        result = files.load(result_path, JobResult)
        if (
            result.job_id != job.job_id
            or result.status != "succeeded"
            or result.outputs != {"model": relative}
            or not (out / "checkpoint.pt").is_file()
        ):
            raise ValueError("Worker XSIM chưa trả về đúng model của job")
        ctx.report_peak_vram(result.peak_vram_mb)
        return {"model": out}
