"""Điểm vào CLI cho worker XSIM chạy trong môi trường native.

Worker đọc JobSpec, thực thi job ``train.xsim`` (tối ưu, checkpoint) hoặc ``eval.xsim`` (render ảnh
và video), rồi ghi JobResult theo CONTRACTS mục 9.1. Việc chấm điểm làm ở môi trường chính
(``drive2sim/stages/eval/xsim.py``). Phụ thuộc CUDA chỉ được nạp khi thực hiện job.
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, ErrorInfo, JobResult, JobSpec
from drive2sim.stages.train.xsim import TrainXsimParams

# Mã của adapter quyết định model học được gì. Không gồm render, video, worker: sửa các file đó
# không làm checkpoint cũ mất hiệu lực.
MODEL_MODULES = (
    "bundle.py",
    "checkpoint.py",
    "observations.py",
    "optimization.py",
    "provider.py",
    "sensors.py",
    "sweep.py",
    "training.py",
)
# Tham số chỉ đổi tốc độ hoặc độ dài train, không đổi model ở mỗi bước.
RUNTIME_PARAMS = ("num_iters", "save_every", "preload_images", "num_workers")
RENDERS_DIR = "renders"
RENDERS_INDEX = "renders.json"


def native_config(source: Path) -> Path:
    """Config gốc của XSIM: camera + LiDAR, CNN decoder (``configs/waymo_default.yaml``)."""
    return source / "configs" / "waymo_default.yaml"


def fingerprint(job: JobSpec, root: Path, source: Path) -> str:
    """Định danh dữ liệu, cấu hình và mã quyết định model của một job train.

    Gồm revision và bản vá của XSIM, config native, các module ``MODEL_MODULES``, ``artifact.json``
    và ``observations.json`` của gói, và tham số train trừ ``RUNTIME_PARAMS``.
    """
    digest = hashlib.sha256()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    digest.update(revision.strip().encode())
    digest.update(subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"]))
    # Bản vá submodule pytoast (patches/pytoast.patch): diff của XSIM chỉ ghi submodule "-dirty".
    pytoast = source / "third_party" / "pytoast"
    if (pytoast / ".git").exists():
        digest.update(subprocess.check_output(["git", "-C", str(pytoast), "diff", "HEAD"]))
    digest.update(native_config(source).read_bytes())
    here = Path(__file__).parent
    for name in MODEL_MODULES:
        digest.update(name.encode())
        digest.update((here / name).read_bytes())
    observations = root / job.inputs["observations"]
    digest.update(job.inputs["observations"].encode())
    for name in ("artifact.json", "observations.json"):
        digest.update((observations / name).read_bytes())
    params = {k: v for k, v in job.params.items() if k not in RUNTIME_PARAMS}
    digest.update(json.dumps(params, sort_keys=True).encode())
    return digest.hexdigest()


def _pipeline(job: JobSpec, root: Path) -> tuple:
    """Dựng pipeline từ job train và tính fingerprint để kiểm tra model."""
    import numpy as np
    import torch
    import xsim

    from drive2sim.external.xsim.provider import BundleScene
    from drive2sim.external.xsim.training import build_training_pipeline

    params = TrainXsimParams.model_validate(job.params)
    source = Path(xsim.__file__).resolve().parents[2]
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.reset_peak_memory_stats()
    provider = BundleScene(
        str(root / job.inputs["observations"]),
        lidar_policy=params.lidar_policy,
        lidar_step=math.radians(params.lidar_elevation_step_deg),
        lidar_cols=params.lidar_azimuth_bins,
    )
    spec = provider.lidar_spec
    print(
        f"XSIM lidar: {len(provider.index.lidar_frames)} sweep, lưới {spec.rows}x{spec.cols}, "
        f"góc nâng {math.degrees(spec.bottom):.2f}..{math.degrees(spec.top):.2f} độ, "
        f"giữ {provider.lidar_retention:.1%} điểm",
        flush=True,
    )
    subset, nodes = build_training_pipeline(
        provider,
        native_config(source),
        "cuda",
        preload_images=params.preload_images,
        num_workers=params.num_workers,
    )
    return subset, nodes, params, fingerprint(job, root, source), provider


def _train(job: JobSpec, root: Path, out: Path) -> int:
    """Chạy vòng tối ưu trên GPU và trả về VRAM đỉnh."""
    import torch

    from drive2sim.external.xsim.training import train

    subset, nodes, params, digest, _ = _pipeline(job, root)
    train(
        subset,
        nodes,
        "cuda",
        checkpoint=out / "checkpoint.pt",
        fingerprint=digest,
        max_steps=params.num_iters,
        save_every=params.save_every,
    )
    return torch.cuda.max_memory_allocated() // (1024 * 1024)


def _evaluate(job: JobSpec, root: Path, out: Path) -> int:
    """Nạp đúng checkpoint của run, vẽ mọi cặp của split, ghi ảnh test và video."""
    from heapq import merge
    from tempfile import TemporaryDirectory

    import torch

    from drive2sim.external.xsim.evaluation import render_pairs, save_render
    from drive2sim.external.xsim.provider import BundleScene
    from drive2sim.external.xsim.videos import write_videos

    model = root / job.inputs["model"]
    original = files.load(model / "job.json", JobSpec)
    if (
        original.stage_id != "train.xsim"
        or original.job_id.rpartition("/")[0] != job.job_id.rpartition("/")[0]
        or original.inputs != {k: v for k, v in job.inputs.items() if k != "model"}
    ):
        raise ValueError("checkpoint XSIM khác run hoặc khác dữ liệu đầu vào")
    subset, nodes, params, digest, provider = _pipeline(original, root)
    state = torch.load(model / "checkpoint.pt", map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["fingerprint"] != digest:
        raise ValueError("fingerprint của checkpoint XSIM khác cấu hình khi đánh giá")
    nodes.load_state_dict(state["weights"], strict=True)
    del state
    step = int(nodes.step.item())
    if step != params.num_iters:
        raise ValueError(f"checkpoint XSIM ở bước {step}, run cần {params.num_iters}")
    observations = provider.observations
    test_provider = BundleScene(
        str(root / original.inputs["observations"]),
        train=False,
        lidar_policy=params.lidar_policy,
        lidar_step=math.radians(params.lidar_elevation_step_deg),
        lidar_cols=params.lidar_azimuth_bins,
    )
    split = observations.split
    names = {index: name for name, index in observations.camera_indices.items()}
    test = {tuple(pair) for pair in split["test"]}
    train = {tuple(pair) for pair in split["train"]}
    cameras = {}
    for name in split["cameras"]:
        sensor = observations.camera_indices[name]
        if sensor in subset.camera_ids:
            info = observations.cameras[sensor]
            cameras[sensor] = (name, info.width, info.height)
    timestamps = {
        frame: (observations.source_frame(frame), observations.timestamps_us[t])
        for t, frame in enumerate(observations.frames)
    }
    renders = out / RENDERS_DIR
    count = 0
    with torch.no_grad(), TemporaryDirectory(dir=out) as temporary:
        pairs = merge(
            render_pairs(provider, subset, nodes, "cuda"),
            render_pairs(test_provider, subset, nodes, "cuda"),
            key=lambda pair: pair[:2],
        )
        for frame, sensor, rgb, _, _ in write_videos(
            pairs, Path(temporary), step, cameras, train, test, timestamps
        ):
            if (frame, names[sensor]) in test:
                save_render(rgb, renders, frame, sensor)
                count += 1
        folder = out / "videos"
        folder.mkdir(exist_ok=True)
        for artifact in Path(temporary).iterdir():
            artifact.replace(folder / artifact.name)
    (out / RENDERS_INDEX).write_text(
        json.dumps({"step": step, "num_test_images": count, "split_id": split["split_id"]}),
        encoding="utf-8",
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
            raise ValueError("loại job XSIM chưa được hỗ trợ")
        for path in [job.output_dir, *job.inputs.values()]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("đường dẫn job phải tương đối với workdir")
        out = root / job.output_dir
        if out.resolve() != result_path.resolve().parent:
            raise ValueError("thư mục kết quả khác thư mục stage")
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
                error=ErrorInfo(code="stage_failed", message=f"worker XSIM gặp lỗi: {exc}"),
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
