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


def _pipeline(job: JobSpec, root: Path) -> tuple:
    """Dựng pipeline từ job train và tính fingerprint để kiểm tra model."""
    import numpy as np
    import torch
    import xsim

    from drive2sim.external.xsim.provider import KittiScene
    from drive2sim.external.xsim.training import build_training_pipeline

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
    return subset, nodes, params, digest.hexdigest()


def _train(job: JobSpec, root: Path, out: Path) -> int:
    """Chạy vòng tối ưu trên GPU và trả về VRAM đỉnh."""
    import torch

    from drive2sim.external.xsim.training import train

    subset, nodes, params, fingerprint = _pipeline(job, root)
    train(
        subset,
        nodes,
        "cuda",
        checkpoint=out / "checkpoint.pt",
        fingerprint=fingerprint,
        max_steps=params.num_iters,
        save_every=params.save_every,
    )
    return torch.cuda.max_memory_allocated() // (1024 * 1024)


def _evaluate(job: JobSpec, root: Path, out: Path) -> int:
    """Nạp đúng checkpoint của run, vẽ test và xuất summary dùng cho Studio."""
    from importlib.metadata import version

    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from drive2sim.core.schema import EvalProtocol, LpipsSetting, Summary
    from drive2sim.external.xsim.evaluation import render_pairs
    from drive2sim.external.xsim.provider import KittiScene
    from drive2sim.external.xsim.scoring import score_pairs

    model = root / job.inputs["model"]
    original = files.load(model / "job.json", JobSpec)
    if (
        original.stage_id != "train.xsim"
        or original.job_id.rpartition("/")[0] != job.job_id.rpartition("/")[0]
        or original.inputs != {k: v for k, v in job.inputs.items() if k != "model"}
    ):
        raise ValueError("Checkpoint XSIM khác run hoặc khác dữ liệu đầu vào")
    subset, nodes, params, fingerprint = _pipeline(original, root)
    state = torch.load(model / "checkpoint.pt", map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["fingerprint"] != fingerprint:
        raise ValueError("Fingerprint của checkpoint XSIM khác cấu hình khi đánh giá")
    nodes.load_state_dict(state["weights"], strict=True)
    del state
    if int(nodes.step.item()) != params.num_iters:
        raise ValueError("Checkpoint XSIM chưa đạt số bước train của run")
    inputs = {k: root / v for k, v in original.inputs.items()}
    clip = json.loads((inputs["clip"] / "scene.json").read_text())
    provider = KittiScene(
        str(inputs["scene"]), str(inputs["clip"]), str(inputs["masks"]), train=False
    )
    names = {index: name for name, index in clip["camera_indices"].items()}
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).eval().cuda()
    with torch.no_grad():
        overall, per_camera = score_pairs(
            render_pairs(provider, subset, nodes, params.downscale, "cuda"),
            {tuple(pair) for pair in clip["split"]["test"]},
            names,
            metric,
        )
    files.dump(
        Summary(
            schema_version=SCHEMA_VERSIONS["summary"],
            protocol=EvalProtocol(
                version="0.3",
                image_dtype="uint8",
                resolution=f"downscale={params.downscale:g}",
                lpips=LpipsSetting(
                    net="alex", package="torchmetrics", package_version=version("torchmetrics")
                ),
            ),
            split_id=clip["split"]["split_id"],
            num_iters=params.num_iters,
            overall=overall,
            per_camera=per_camera,
        ),
        out / "summary.json",
    )
    return torch.cuda.max_memory_allocated() // (1024 * 1024)


def execute(job_path: Path, result_path: Path, root: Path) -> None:
    """Kiểm job trước khi chạy và ghi kết quả kể cả khi native runtime gặp lỗi."""
    job = files.load(job_path, JobSpec)
    start = time.monotonic()
    shared = {"schema_version": SCHEMA_VERSIONS["result"], "job_id": job.job_id}
    try:
        if (
            job.kind != "stage"
            or job.stage_id not in {"train.xsim", "eval.xsim"}
            or job.env != "xsim"
        ):
            raise ValueError("Loại job XSIM chưa được hỗ trợ")
        for path in [job.output_dir, *job.inputs.values()]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("Đường dẫn job phải tương đối với workdir")
        out = root / job.output_dir
        if out.resolve() != result_path.resolve().parent:
            raise ValueError("Thư mục kết quả khác thư mục stage")
        if job.stage_id == "train.xsim":
            peak, output = _train(job, root, out), "model"
        else:
            peak, output = _evaluate(job, root, out), "report"
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
            outputs={output: job.output_dir},
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
