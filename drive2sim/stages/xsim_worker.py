"""Giao tiếp job/result giữa engine và interpreter XSIM theo CONTRACTS mục 9."""

import os
from pathlib import Path

from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, JobResult, JobSpec
from drive2sim.engine.stage import StageContext
from drive2sim.stages.commands import run_logged


def run_xsim_worker(
    ctx: StageContext,
    inputs: dict[str, Path],
    params: BaseModel,
    stage_id: str,
    output_name: str,
    output_file: str,
) -> dict[str, Path]:
    """Chạy job ở interpreter riêng và xác nhận kết quả của đúng lần gọi."""
    root = ctx.scene.root.parent.parent.absolute()
    out = ctx.output_dir.absolute()
    relative = out.relative_to(root).as_posix()
    job = JobSpec(
        schema_version=SCHEMA_VERSIONS["job"],
        job_id=f"{ctx.run_id}/{stage_id}",
        kind="stage",
        stage_id=stage_id,
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
        or result.outputs != {output_name: relative}
        or not (out / output_file).is_file()
    ):
        raise ValueError("Worker XSIM chưa trả về đúng artifact của job")
    ctx.report_peak_vram(result.peak_vram_mb)
    return {output_name: out}
