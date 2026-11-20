"""Stage ``eval.ncore``: chấm PLY của Instant NuRec trên các khung test của split.

Hai bước như ``PipelineManager.evaluate`` của Fisheye Fusion Lab:
1. ``ncore_eval_export.py`` (môi trường NuRec, đọc clip NCore): xuất pose, thông số camera, ảnh
   thật và mask thân xe của các khung test (``--frame-indices``).
2. ``ncore_eval_render.py`` (môi trường chính, gsplat 1.5.3): vẽ PLY bằng gsplat (3DGUT cho camera
   mắt cá) và tính PSNR, SSIM, LPIPS trên vùng hợp lệ (ngoài mask thân xe, alpha > 0.9).
Kết quả đổi sang ``summary.json`` (CONTRACTS mục 6.3) để Studio và ``d2s report`` đặt NuRec cạnh
các approach khác.

Khác protocol của ``eval.standard``: NuRec là mô hình feed-forward, dựng từ mọi khung của clip
(cả khung test), nên điểm test không phải ảnh "chưa thấy" như với approach train theo cảnh; LPIPS
đo trên ảnh mà pixel ngoài vùng hợp lệ lấy theo ảnh thật. Hai điều này ghi trong
``nurec_eval.json``.

File này chứa:
- ``clip_test_indices``: chỉ số khung test trong clip NCore.
- ``make_summary``: ``metrics.json`` của bước 2 → ``Summary``.
- ``EvalNcoreStage``: stage đăng ký với engine.
"""

import json
import sys
from importlib import metadata
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, EvalProtocol, LpipsSetting, Metrics, Summary
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.eval.standard import PROTOCOL_VERSION, SUMMARY_FILE
from drive2sim.stages.prep.ncore import CLIP_FILE
from drive2sim.stages.train.nurec import MERGED_PLY, STATS_FILE

VIEWS_FILE = "views.npz"
METRICS_FILE = "metrics.json"
NOTE_FILE = "nurec_eval.json"


def clip_test_indices(clip: dict, scene_frames: list[int]) -> list[int]:
    """Chỉ số trong clip NCore của các khung test (theo số khung gốc), bỏ khung clip không có.

    Args:
        clip: Nội dung ``clip.json`` của ``prep.ncore``: ``frame_ids`` (số khung gốc của clip),
            ``scene_frame_ids`` (số khung gốc của đoạn split), ``scene_frames`` (đoạn khung
            của split trong cảnh).
        scene_frames: Chỉ số khung test trong cảnh (``split.test``).
    """
    position = {frame: i for i, frame in enumerate(clip["frame_ids"])}
    scene_ids = clip["scene_frame_ids"]
    first = clip["scene_frames"][0]
    indices = []
    for frame in sorted(set(scene_frames)):
        offset = frame - first
        if 0 <= offset < len(scene_ids) and scene_ids[offset] in position:
            indices.append(position[scene_ids[offset]])
    return indices


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def make_summary(
    metrics: dict, split_id: str, sensors: dict[str, str], stats: Optional[dict] = None
) -> Summary:
    """Dựng ``Summary`` từ ``metrics.json`` của ``ncore_eval_render.py``.

    Args:
        metrics: ``{<sensor>: {"psnr_mean", "ssim_mean", "lpips_mean", "views"},
            "_summary": {...}}``.
        split_id: Split của run.
        sensors: ``camera_id`` → tên sensor NCore.
        stats: ``train_stats.json`` của ``train.nurec``.

    Raises:
        KeyError: Nếu thiếu ``_summary`` hoặc metric tổng.
    """
    overall = metrics["_summary"]
    camera_of = {sensor: camera for camera, sensor in sensors.items()}

    def as_metrics(item: dict) -> Metrics:
        return Metrics(psnr=item["psnr_mean"], ssim=item["ssim_mean"], lpips=item["lpips_mean"])

    per_camera = {
        camera_of.get(sensor, sensor): as_metrics(item)
        for sensor, item in metrics.items()
        if not sensor.startswith("_")
    }
    stats = stats or {}
    return Summary(
        schema_version=SCHEMA_VERSIONS["summary"],
        protocol=EvalProtocol(
            version=PROTOCOL_VERSION,
            image_dtype="uint8",
            resolution=str(overall.get("render_resolution", "unknown")),
            lpips=LpipsSetting(
                net="alex",
                package="torchmetrics",
                package_version=_package_version("torchmetrics"),
            ),
        ),
        split_id=split_id,
        overall=as_metrics(overall),
        per_camera=per_camera,
        train_time_s=stats.get("train_time_s"),
        peak_vram_mb=int(overall["peak_vram_gib"] * 1024) if "peak_vram_gib" in overall else None,
        num_gaussians=overall.get("gaussians") or stats.get("num_gaussians"),
        num_iters=0,
        render_fps=overall.get("render_fps"),
        render_resolution=overall.get("render_resolution"),
    )


@register_stage("eval.ncore")
class EvalNcoreStage(Stage):
    """Chấm PLY của Instant NuRec trên khung test của split, ghi ``summary.json``."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model", "clip": "ncore_clip", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"report": "eval_report"}
    Params: ClassVar[type[BaseModel]] = NoParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Xuất view test, vẽ và chấm, ghi ``summary.json``.

        Returns:
            ``{"report": <thư mục stage>}``.

        Raises:
            RuntimeError: Split không có khung test nào trong clip, hoặc một bước lỗi.
        """
        clip = json.loads((inputs["clip"] / CLIP_FILE).read_text(encoding="utf-8"))
        indices = clip_test_indices(clip, [frame for frame, _ in ctx.split.test])
        if not indices:
            raise RuntimeError("không có khung test nào của split nằm trong clip NCore")
        sensors = {c: s for c, s in clip["sensors"].items() if c in set(ctx.split.cameras)}
        log = ctx.output_dir / LOG_FILE
        views = ctx.output_dir / VIEWS_FILE
        ctx.progress(0.1, f"xuất {len(indices)} khung test")
        run_logged(
            [
                str(nurec.nurec_python()), "-u", str(nurec.TOOLS_DIR / "ncore_eval_export.py"),
                str(inputs["clip"] / clip["clip"]), "--out", str(views),
                "--frame-indices", ",".join(map(str, indices)),
                "--cameras", ",".join(sensors.values()),
            ],
            log,
            should_stop=ctx.should_stop,
        )  # fmt: skip
        ctx.progress(0.5, "vẽ và chấm bằng gsplat")
        run_logged(
            [
                sys.executable, "-u", str(nurec.TOOLS_DIR / "ncore_eval_render.py"),
                str(inputs["model"] / MERGED_PLY), str(views), "--out", str(ctx.output_dir),
            ],
            log,
            should_stop=ctx.should_stop,
        )  # fmt: skip
        views.unlink(missing_ok=True)  # ảnh test đã nằm trong cảnh; file này lớn
        metrics = json.loads((ctx.output_dir / METRICS_FILE).read_text(encoding="utf-8"))
        stats_path = inputs["model"] / STATS_FILE
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        files.dump(
            make_summary(metrics, ctx.split.split_id, sensors, stats),
            ctx.output_dir / SUMMARY_FILE,
        )
        (ctx.output_dir / NOTE_FILE).write_text(
            json.dumps(
                {
                    "test_frames_in_clip": indices,
                    "note": "Instant NuRec dựng từ mọi khung của clip NCore, kể cả khung test; "
                    "LPIPS đo trên ảnh mà pixel ngoài vùng hợp lệ lấy theo ảnh thật",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        return {"report": ctx.output_dir}
