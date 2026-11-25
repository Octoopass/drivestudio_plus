"""Stage ``train.nurec``: dựng cảnh bằng Instant NuRec (feed-forward, không train theo cảnh).

Chạy ``run_inference.py`` của Instant NuRec trong môi trường riêng (``drive2sim/external/nurec``)
trên clip NCore của ``prep.ncore``, như ``ReconstructionManager`` của Fisheye Fusion Lab: chạy
trong thư mục mã nguồn NuRec, ``INSTANT_NUREC_FULL_PT`` trỏ tới checkpoint, Hugging Face ở chế độ
offline, ghép các đoạn (``--merge``) thành một PLY. Kết quả là thư mục ``model`` có ``nurec/``
(đầu ra gốc: PLY, sky, ảnh xem trước nếu bật), ``scene_nurec.ply`` (PLY đã ghép, hệ world của
clip NCore) và ``train_stats.json`` (thời gian dựng, số Gaussian).

Ba model phát hành: ``pa-multiview`` (nhiều camera, cả camera mắt cá đã resample),
``pa-front`` (một camera trước), ``pq-front`` (đường; NuRec cố định nó vào camera
``camera_front_wide_120fov`` mà clip KITTI-360 không có, nên approach chưa cho chọn). Camera ngữ
cảnh truyền bằng ``--camera-id`` theo thứ tự chuẩn của NuRec (``context_cameras``).

File này chứa:
- ``TrainNurecParams``: tham số.
- ``build_command``, ``nurec_env``: câu lệnh và biến môi trường (tách riêng để test).
- ``TrainNurecStage``: stage đăng ký với engine.
"""

import json
import os
import time
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine import machine
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.prep.ncore import CLIP_FILE

MERGED_PLY = "scene_nurec.ply"
FRAMES_PER_CHUNK = 18  # n_frames_per_sample của các model phát hành (instant_nurec/pretrained.py)
MAX_CHUNKS = 8
STATS_FILE = "train_stats.json"
RAW_DIR = "nurec"


class TrainNurecParams(BaseModel):
    """Tham số của ``train.nurec`` (khớp ràng buộc của Fisheye Fusion Lab).

    Attributes:
        model: Model phát hành dùng để dựng.
        n_gaussians: Số Gaussian mỗi đoạn.
        max_chunks: Số đoạn thời gian ghép lại; ``0`` (mặc định) là tự tính để phủ hết clip:
            mỗi đoạn NuRec dựng ``FRAMES_PER_CHUNK`` khung, tối đa 8 đoạn.
        include_dynamic: Dựng cả vật chuyển động (chỉ model ``pa-*``).
        render_preview: Ghi thêm ảnh render xem trước (cần gsplat, nvdiffrast trong môi trường
            NuRec).
    """

    model_config = ConfigDict(extra="forbid")

    model: Literal["pa-multiview", "pa-front", "pq-front"] = "pa-multiview"
    n_gaussians: int = Field(default=500000, ge=100000, le=2000000)
    max_chunks: int = Field(default=0, ge=0, le=8)
    include_dynamic: bool = False
    render_preview: bool = False


# Vai trò camera ngữ cảnh theo thứ tự chuẩn của NuRec (trước, chéo trái, chéo phải) → camera
# KITTI-360; camera mắt cá đã resample về camera chéo 120° của NVIDIA (prep.ncore, rectify120).
CONTEXT_ROLES = ("image_00", "image_02", "image_03")


def context_cameras(sensors: dict[str, str], model: str) -> list[str]:
    """Tên sensor NCore của camera ngữ cảnh cho ``--camera-id``, theo thứ tự chuẩn của NuRec.

    ``pa-multiview`` nhận 1, 3 hoặc 5 camera (trước, chéo trái, chéo phải...), ``pa-front`` một
    camera trước. ``pq-front`` bị NuRec cố định vào camera ``camera_front_wide_120fov`` mà clip
    KITTI-360 không có.

    Args:
        sensors: ``camera_id`` → tên sensor NCore (``clip.json`` của ``prep.ncore``).
        model: Model NuRec.

    Raises:
        ValueError: Thiếu camera trước, hoặc model ``pq-front``.
    """
    if model == "pq-front":
        raise ValueError(
            "pq-front bị Instant NuRec cố định vào camera_front_wide_120fov, clip KITTI-360 "
            "không có camera đó; chọn pa-multiview hoặc pa-front"
        )
    if "image_00" not in sensors:
        raise ValueError("Instant NuRec cần camera trước image_00 trong split")
    if model == "pa-multiview" and all(role in sensors for role in CONTEXT_ROLES):
        return [sensors[role] for role in CONTEXT_ROLES]
    return [sensors["image_00"]]


def chunks_for(params: TrainNurecParams, num_frames: int) -> int:
    """Số đoạn thời gian: ``params.max_chunks``, hoặc khi là 0 thì đủ phủ ``num_frames`` khung."""
    if params.max_chunks:
        return params.max_chunks
    return max(1, min(MAX_CHUNKS, -(-num_frames // FRAMES_PER_CHUNK)))


def build_command(
    params: TrainNurecParams, clip: Path, output_dir: Path, cameras: list[str]
) -> list[str]:
    """Câu lệnh ``run_inference.py`` bằng Python của venv NuRec.

    Args:
        params: Tham số.
        clip: File JSON của clip NCore.
        output_dir: Thư mục đầu ra gốc của NuRec.
        cameras: Camera ngữ cảnh (``context_cameras``).

    Raises:
        ValueError: ``include_dynamic`` với ``pq-front``.
    """
    if params.include_dynamic and params.model == "pq-front":
        raise ValueError("include_dynamic chỉ dùng với model pa-front hoặc pa-multiview")
    command = [
        str(nurec.nurec_python()), "-u", str(nurec.nurec_dir() / "run_inference.py"),
        "--model", params.model, "--ncore-path", str(clip), "--output-dir", str(output_dir),
        "--merge", "--n-gaussians", str(params.n_gaussians),
        "--max-chunks", str(params.max_chunks or 1),
    ]  # fmt: skip
    if params.include_dynamic:
        command.append("--include-dynamic")
    if params.render_preview:
        command.append("--render-preview")
    for camera in cameras:
        command += ["--camera-id", camera]
    return command


def nurec_env(checkpoint: Path) -> dict[str, str]:
    """Biến môi trường cho ``run_inference.py`` (như Fisheye Fusion Lab).

    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` như stage train, eval khác: bước gộp
    Gaussian (voxelize lớp tĩnh) xin một khối lớn liền; trên GPU dùng chung, vài GB bị giữ mà
    phân mảnh là đủ hết bộ nhớ. Biến đã đặt ở môi trường ngoài thì giữ nguyên.

    Máy nhiều GPU mà chưa đặt ``CUDA_VISIBLE_DEVICES``: chạy trên GPU còn trống nhiều nhất
    (``machine.freest_gpu``), vì GPU 0 thường đã có job khác.
    """
    python_bin = str(nurec.nurec_python().parent)
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
        "INSTANT_NUREC_FULL_PT": str(checkpoint),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MAX_JOBS": "2",
        "VIRTUAL_ENV": str(nurec.nurec_python().parent.parent),
        "PATH": python_bin + os.pathsep + os.environ.get("PATH", ""),
    }
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu = machine.freest_gpu()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _merged_ply(raw_dir: Path) -> Path:
    """PLY Gaussian lớn nhất trong đầu ra (bỏ file ``.sky.*``)."""
    plies = [p for p in raw_dir.rglob("*.ply") if ".sky" not in p.name and p.stat().st_size]
    if not plies:
        raise RuntimeError("Instant NuRec xong nhưng không ghi PLY nào")
    return max(plies, key=lambda p: p.stat().st_size)


def _vertex_count(ply: Path) -> int:
    with ply.open("rb") as stream:
        for _ in range(200):
            line = stream.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    return 0


@register_stage("train.nurec")
class TrainNurecStage(Stage):
    """Dựng cảnh từ clip NCore bằng Instant NuRec."""

    version: ClassVar[str] = "0.2.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"clip": "ncore_clip"}
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = TrainNurecParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy Instant NuRec, chép PLY đã ghép thành ``scene_nurec.ply``.

        Returns:
            ``{"model": <thư mục stage>}``.

        Raises:
            RuntimeError: Thiếu checkpoint hoặc NuRec không ghi PLY.
        """
        assert isinstance(params, TrainNurecParams)
        checkpoint = nurec.checkpoint_path(params.model)
        if not checkpoint.is_file():
            raise RuntimeError(
                f"chưa có checkpoint {params.model} ({checkpoint}): bash envs/install_nurec.sh "
                f"--models {params.model} hoặc nút cài trên trang Chạy mới"
            )
        record = json.loads((inputs["clip"] / CLIP_FILE).read_text(encoding="utf-8"))
        clip = inputs["clip"] / record["clip"]
        cameras = context_cameras(record["sensors"], params.model)
        chunks = chunks_for(params, len(record["frame_ids"]))
        params = params.model_copy(update={"max_chunks": chunks})
        raw_dir = ctx.output_dir / RAW_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)
        ctx.progress(0.0, f"Instant NuRec {params.model}, {chunks} đoạn")
        began = time.monotonic()
        run_logged(
            build_command(params, clip, raw_dir, cameras),
            ctx.output_dir / LOG_FILE,
            cwd=nurec.nurec_dir(),
            should_stop=ctx.should_stop,
            env=nurec_env(checkpoint),
        )
        elapsed = time.monotonic() - began
        merged = _merged_ply(raw_dir)
        target = ctx.output_dir / MERGED_PLY
        target.write_bytes(merged.read_bytes())
        stats = {
            "train_time_s": round(elapsed, 1),
            "num_gaussians": _vertex_count(target),
            "num_iters": 0,
            "model": params.model,
            "context_cameras": cameras,
            "max_chunks": chunks,
            "source_ply": str(merged.relative_to(ctx.output_dir)),
        }
        (ctx.output_dir / STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n")
        ctx.progress(1.0, f"{stats['num_gaussians']:,} Gaussian")
        return {"model": ctx.output_dir}
