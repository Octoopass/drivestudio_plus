"""Giao tiếp job/result giữa engine và interpreter XSIM theo CONTRACTS mục 9."""

import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, JobResult, JobSpec
from drive2sim.engine.stage import StageContext
from drive2sim.external.xsim import OBSERVATIONS_INDEX, xsim_dir, xsim_python
from drive2sim.paths import SMPL_MODEL_NAME, find_smpl_model
from drive2sim.stages.commands import run_logged

WORKER_LOG = "worker.log"
_STEP = re.compile(rb"XSIM step=(\d+)/(\d+)")


def _progress_watcher(ctx: StageContext, log: Path, total: Optional[int]) -> Callable[[], bool]:
    """Hàm ``should_stop`` cho ``run_logged`` kèm báo tiến độ từ dòng ``XSIM step=`` của log."""
    state = {"offset": 0, "step": -1}

    def should_stop() -> bool:
        if total:
            try:
                with log.open("rb") as stream:
                    stream.seek(state["offset"])
                    chunk = stream.read()
                    state["offset"] = stream.tell()
                found = _STEP.findall(chunk)
                if found:
                    step = int(found[-1][0])
                    if step != state["step"]:
                        state["step"] = step
                        ctx.progress(step / total, f"XSIM bước {step}/{total}")
            except OSError:
                pass
        return ctx.should_stop()

    return should_stop


def smpl_environment(ctx: StageContext, observations: Path, root: Path) -> dict[str, str]:
    """Biến ``SMPL_PATH`` cho worker khi gói quan sát có người dựng bằng SMPL.

    Mô hình SMPL chuyển sang định dạng XSIM đọc ở môi trường chính
    (``recon/tools/convert_smpl.py``), lưu trong thư mục cài XSIM, ngoài ``cache/`` và ``runs/``.

    Returns:
        ``{"SMPL_PATH": file}``, hoặc ``{}`` khi gói không có SMPL.

    Raises:
        RuntimeError: Nếu gói có SMPL mà không thấy file mô hình SMPL.
    """
    index = json.loads((observations / OBSERVATIONS_INDEX).read_text("utf-8"))
    if not index.get("smpl"):
        return {}
    source = find_smpl_model(root)
    if source is None:
        raise RuntimeError(f"cảnh có người (SMPL) nhưng không thấy file mô hình {SMPL_MODEL_NAME}")
    target = (xsim_dir() / "smpl" / "SMPL_NEUTRAL.pth").absolute()
    if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
        command = [sys.executable, "-m", "drive2sim.recon.tools.convert_smpl"]
        run_logged(
            command + ["--source", str(source), "--target", str(target)],
            ctx.output_dir / "smpl.log",
            should_stop=ctx.should_stop,
        )
    return {"SMPL_PATH": str(target)}


def run_xsim_worker(
    ctx: StageContext,
    inputs: dict[str, Path],
    params: BaseModel,
    stage_id: str,
    output_name: str,
    output_file: str,
    total_steps: Optional[int] = None,
) -> dict[str, Path]:
    """Chạy job ở interpreter riêng và xác nhận kết quả của đúng lần gọi.

    Args:
        ctx: Ngữ cảnh stage.
        inputs: Đầu vào của stage (đường dẫn tuyệt đối trong workdir).
        params: Tham số của stage, ghi nguyên vào job.
        stage_id: ``train.xsim`` hoặc ``eval.xsim``.
        output_name: Tên đầu ra worker phải báo.
        output_file: File phải có trong thư mục stage sau khi worker xong.
        total_steps: Tổng số bước train để báo tiến độ; ``None`` thì không báo.

    Raises:
        ValueError: Nếu worker không trả đúng kết quả của job này.
    """
    # Cảnh ở <workdir>/data/<cảnh>: mọi đường dẫn của job tính từ workdir.
    root = ctx.scene.root.parent.parent.absolute()
    out = ctx.output_dir.absolute()
    relative = out.relative_to(root).as_posix()
    job = JobSpec(
        schema_version=SCHEMA_VERSIONS["job"],
        job_id=f"{ctx.run_id}/{stage_id}",
        kind="stage",
        stage_id=stage_id,
        params=params.model_dump(),
        inputs={k: v.absolute().relative_to(root).as_posix() for k, v in inputs.items()},
        output_dir=relative,
        env="xsim",
    )
    job_path, result_path = out / "job.json", out / "result.json"
    files.dump(job, job_path)
    result_path.unlink(missing_ok=True)
    log = out / WORKER_LOG
    command = [
        str(xsim_python()),
        "-m",
        "drive2sim.external.xsim.worker",
        "--job",
        str(job_path),
        "--result",
        str(result_path),
    ]
    env = {"D2S_WORKDIR": str(root), "PYTHONUNBUFFERED": "1"}
    env.update(smpl_environment(ctx, inputs["observations"], root))
    run_logged(
        command,
        log,
        env=env,
        should_stop=_progress_watcher(ctx, log, total_steps),
    )
    result = files.load(result_path, JobResult)
    if (
        result.job_id != job.job_id
        or result.status != "succeeded"
        or result.outputs != {output_name: relative}
        or not (out / output_file).is_file()
    ):
        raise ValueError("worker XSIM chưa trả về đúng artifact của job")
    ctx.report_peak_vram(result.peak_vram_mb)
    return {output_name: out}
