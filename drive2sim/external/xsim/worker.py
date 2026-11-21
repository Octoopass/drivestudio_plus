"""Điểm vào CLI cho worker XSIM chạy trong môi trường native.

Worker đọc JobSpec, thực thi stage rồi ghi JobResult theo CONTRACTS mục 9.1.
Phụ thuộc CUDA chỉ được nạp khi thực hiện job train.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, ErrorInfo, JobResult, JobSpec
from drive2sim.stages.train.xsim import TrainXsimParams


def _train(job: JobSpec, root: Path, out: Path) -> int:
    """Dựng pipeline train native từ job rồi trả về số megabyte VRAM đỉnh."""
    import numpy as np
    import torch
    import xsim

    from drive2sim.external.xsim.provider import KittiScene
    from drive2sim.external.xsim.training import build_training_pipeline, train

    params = TrainXsimParams.model_validate(job.params)
    inputs = {key: root / value for key, value in job.inputs.items()}
    source = Path(xsim.__file__).resolve().parents[2]
    native = source / "configs" / "waymo_default.yaml"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    digest = hashlib.sha256()
    digest.update(revision.strip().encode())
    digest.update(subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"]))
    for key in ("clip", "masks"):
        digest.update(job.inputs[key].encode())
        digest.update((inputs[key] / "artifact.json").read_bytes())
    digest.update((inputs["clip"] / "scene.json").read_bytes())
    digest.update(json.dumps(job.params, sort_keys=True).encode())
    for module in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(module.read_bytes())
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.reset_peak_memory_stats()
    provider = KittiScene(str(inputs["scene"]), str(inputs["clip"]), str(inputs["masks"]))
    subset, nodes = build_training_pipeline(provider, native, "cuda", params.downscale)
    train(
        subset,
        nodes,
        "cuda",
        checkpoint=out / "checkpoint.pt",
        fingerprint=digest.hexdigest(),
        max_steps=params.num_iters,
        save_every=params.save_every,
    )
    return torch.cuda.max_memory_allocated() // (1024 * 1024)


def execute(job_path: Path, result_path: Path, root: Path) -> None:
    """Kiểm job trước khi chạy và ghi kết quả kể cả khi native runtime gặp lỗi."""
    job = files.load(job_path, JobSpec)
    start = time.monotonic()
    shared = {"schema_version": SCHEMA_VERSIONS["result"], "job_id": job.job_id}
    try:
        if job.kind != "stage" or job.stage_id != "train.xsim" or job.env != "xsim":
            raise ValueError("Loại job XSIM chưa được hỗ trợ")
        for path in [job.output_dir, *job.inputs.values()]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("Đường dẫn job phải tương đối với workdir")
        out = root / job.output_dir
        if out.resolve() != result_path.resolve().parent:
            raise ValueError("Thư mục kết quả khác thư mục stage")
        peak = _train(job, root, out)
    except Exception as exc:
        files.dump(
            JobResult(
                **shared,
                status="failed",
                duration_s=time.monotonic() - start,
                peak_vram_mb=0,
                error=ErrorInfo(code="stage_failed", message=f"Worker XSIM gặp lỗi: {exc}"),
            ),
            result_path,
        )
        raise
    files.dump(
        JobResult(
            **shared,
            status="succeeded",
            outputs={"model": job.output_dir},
            duration_s=time.monotonic() - start,
            peak_vram_mb=peak,
        ),
        result_path,
    )


def main() -> None:
    """Đọc đường dẫn job/result và chạy worker trong workdir của executor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    execute(args.job, args.result, Path(os.environ["D2S_WORKDIR"]))


if __name__ == "__main__":
    main()
