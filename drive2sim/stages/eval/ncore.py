"""Stage ``eval.ncore``: chấm PLY của Instant NuRec trên các khung test của split.

Hai bước như ``PipelineManager.evaluate`` của Fisheye Fusion Lab:
1. ``ncore_eval_export.py`` (môi trường NuRec, đọc clip NCore): xuất pose, thông số camera, ảnh
   thật và mask thân xe của các khung test (``--frame-indices``).
2. ``ncore_eval_render.py`` (môi trường chính, gsplat 1.5.3): vẽ PLY bằng gsplat (3DGUT cho camera
   mắt cá) và tính PSNR, SSIM, LPIPS trên vùng hợp lệ (ngoài mask thân xe, alpha > 0.9).
Kết quả đổi sang ``summary.json`` (CONTRACTS mục 6.3) để Studio và ``d2s report`` đặt NuRec cạnh
các approach khác.

Chấm như ``eval.standard`` (từ 0.2.0): bước 2 nhận ``views.json`` liệt kê đúng các cặp (khung,
camera) test của split cùng tên file mask của cảnh, rồi đo bằng ``compute_image_metrics`` của
drive2sim (hàm ``eval.standard`` dùng qua ``train.py``) trên ảnh 8 bit, trong vùng hợp lệ: ngoài
thân xe, camera mắt cá chỉ lấy pixel trong góc dùng được (``max_angle`` của manifest, tối đa
``RENDERER_MAX_ANGLE`` như khi train 3DGUT). Có mask của ``prep.masks_segformer`` thì thêm PSNR
vùng không phải bầu trời, vật động, người, xe như DriveStudio. Bản 0.1.0 chỉ chấm pixel có
alpha > 0.9 (chỗ PLY đã phủ), cho điểm cao hơn thực tế ở chỗ PLY bỏ trống; con số đó vẫn ghi trong
``metrics.json`` (``psnr_covered``) cùng tỉ lệ phủ (``coverage``).

Khác còn lại so với ``eval.standard``: NuRec là mô hình feed-forward, dựng từ mọi khung của clip
(cả khung test), nên điểm test không phải ảnh "chưa thấy" như với approach train theo cảnh. Điều
này ghi trong ``nurec_eval.json``.

File này chứa:
- ``clip_test_indices``: chỉ số khung test trong clip NCore.
- ``mean_coverage``: tỉ lệ điểm ảnh được chấm, trung bình trên mọi view.
- ``clip_test_views``: cặp (khung, camera) test của split → khung trong clip và tên file mask.
- ``fisheye_max_angles``: góc lệch trục dùng được của từng camera mắt cá.
- ``make_summary``: ``metrics.json`` của bước 2 → ``Summary``.
- ``EvalNcoreStage``: stage đăng ký với engine.
"""

import json
import math
import sys
from importlib import metadata
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import (
    SCHEMA_VERSIONS,
    EvalProtocol,
    LpipsSetting,
    Manifest,
    Metrics,
    Summary,
)
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.eval.standard import PROTOCOL_VERSION, SUMMARY_FILE
from drive2sim.stages.prep.ncore import CLIP_FILE
from drive2sim.stages.train.drivestudio import RENDERER_MAX_ANGLE
from drive2sim.stages.train.nurec import MERGED_PLY, STATS_FILE

VIEWS_FILE = "views.npz"
PLAN_FILE = "views.json"
METRICS_FILE = "metrics.json"
NOTE_FILE = "nurec_eval.json"
# Phiên bản protocol riêng của stage này: SSIM và vùng hợp lệ (ngoài mask thân xe, alpha > 0,9)
# khác ``eval.standard``, nên số đo không được coi là cùng bộ đo với protocol của stage đó.
NCORE_PROTOCOL_VERSION = "ncore-0.3"
REGION_FIELDS = ("occupied_psnr", "dynamic_psnr", "human_psnr", "vehicle_psnr")


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


def clip_test_views(
    clip: dict, test: list[tuple[int, str]], sensors: dict[str, str], camera_index: dict[str, int]
) -> dict[str, dict[str, str]]:
    """Cặp (khung, camera) test của split → khung trong clip NCore và tên file mask của cảnh.

    Args:
        clip: Nội dung ``clip.json`` (xem ``clip_test_indices``).
        test: ``split.test``: các cặp (chỉ số khung trong cảnh, ``camera_id``).
        sensors: ``camera_id`` → tên sensor NCore, chỉ các camera của split.
        camera_index: ``camera_id`` → vị trí trong ``manifest.yaml``; mask của cảnh tên
            ``<khung 3 chữ số>_<vị trí camera>.png`` như ảnh.

    Returns:
        ``{sensor: {"<khung trong clip>": "<tên mask không đuôi>"}}``; cặp mà clip không có khung
        thì bỏ.
    """
    views: dict[str, dict[str, str]] = {}
    for frame, camera in sorted({(int(f), str(c)) for f, c in test}):
        if camera not in sensors:
            continue
        found = clip_test_indices(clip, [frame])
        if found:
            stem = f"{frame:03d}_{camera_index[camera]}"
            views.setdefault(sensors[camera], {})[str(found[0])] = stem
    return views


def fisheye_max_angles(manifest: Manifest, sensors: dict[str, str]) -> dict[str, float]:
    """Góc lệch trục lớn nhất (rad) được chấm của từng camera mắt cá, theo tên sensor NCore.

    Như loader KITTI-360 khi train bằng 3DGUT: góc nhỏ hơn giữa ``max_angle`` của camera trong
    manifest và ``RENDERER_MAX_ANGLE``; camera không ghi ``max_angle`` lấy ``RENDERER_MAX_ANGLE``.
    """
    angles = {}
    for camera in manifest.cameras:
        if camera.camera_id in sensors and camera.model != "opencv_pinhole":
            limit = camera.max_angle or RENDERER_MAX_ANGLE
            angles[sensors[camera.camera_id]] = min(limit, RENDERER_MAX_ANGLE)
    return angles


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def mean_coverage(per_view: list[dict]) -> Optional[float]:
    """Trung bình tỉ lệ điểm ảnh được chấm (``coverage``) trên mọi view.

    ``ncore_eval_render.py`` chỉ chấm điểm ảnh ngoài mask thân xe và có alpha > 0,9; tỉ lệ này cho
    người đọc biết PSNR đo trên bao nhiêu phần ảnh (SPEC/CONTRACTS mục 6.3).

    Args:
        per_view: Nội dung ``per_view.json``: mỗi phần tử có ``coverage`` trong [0, 1].

    Returns:
        Trung bình các ``coverage``; ``None`` khi không có view nào ghi ``coverage``.
    """
    values = [view["coverage"] for view in per_view if view.get("coverage") is not None]
    return sum(values) / len(values) if values else None


def _is_finite_metrics(item: dict) -> bool:
    """``True`` khi ba metric trung bình của camera đều hữu hạn (không ``null``, ``NaN``)."""
    values = (item.get("psnr_mean"), item.get("ssim_mean"), item.get("lpips_mean"))
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def make_summary(
    metrics: dict,
    split_id: str,
    sensors: dict[str, str],
    stats: Optional[dict] = None,
    coverage: Optional[float] = None,
) -> Summary:
    """Dựng ``Summary`` từ ``metrics.json`` của ``ncore_eval_render.py``.

    Camera mà mọi view đều không có điểm hợp lệ (metric ``null`` hoặc ``NaN``) bị bỏ khỏi
    ``per_camera``: không có số đo thì không ghi số.

    Args:
        metrics: ``{<sensor>: {"psnr_mean", "ssim_mean", "lpips_mean", "views"},
            "_summary": {...}}``.
        split_id: Split của run.
        sensors: ``camera_id`` → tên sensor NCore.
        stats: ``train_stats.json`` của ``train.nurec``.
        coverage: Tỉ lệ điểm ảnh được chấm, trung bình trên mọi view (``mean_coverage``).

    Raises:
        KeyError: Nếu thiếu ``_summary`` hoặc metric tổng.
        ValueError: Nếu metric tổng không phải số hữu hạn (không view nào có điểm hợp lệ).
    """
    overall = metrics["_summary"]
    if not _is_finite_metrics(overall):
        raise ValueError(
            "metric tổng không hữu hạn: không view nào có điểm ảnh hợp lệ, hoặc thiếu LPIPS"
        )
    camera_of = {sensor: camera for camera, sensor in sensors.items()}

    def as_metrics(item: dict) -> Metrics:
        return Metrics(psnr=item["psnr_mean"], ssim=item["ssim_mean"], lpips=item["lpips_mean"])

    per_camera = {
        camera_of.get(sensor, sensor): as_metrics(item)
        for sensor, item in metrics.items()
        if not sensor.startswith("_") and _is_finite_metrics(item)
    }
    stats = stats or {}
    return Summary(
        schema_version=SCHEMA_VERSIONS["summary"],
        protocol=EvalProtocol(
            version=overall.get("protocol_version", NCORE_PROTOCOL_VERSION),
            image_dtype="uint8",
            resolution=str(overall.get("render_resolution", "unknown")),
            lpips=LpipsSetting(
                net="alex",
                package="torchmetrics",
                package_version=_package_version("torchmetrics"),
            ),
        ),
        split_id=split_id,
        overall=Metrics(
            psnr=overall["psnr_mean"],
            ssim=overall["ssim_mean"],
            lpips=overall["lpips_mean"],
            **{name: overall[name] for name in REGION_FIELDS if name in overall},
        ),
        per_camera=per_camera,
        train_time_s=stats.get("train_time_s"),
        peak_vram_mb=int(overall["peak_vram_gib"] * 1024) if "peak_vram_gib" in overall else None,
        num_gaussians=overall.get("gaussians") or stats.get("num_gaussians"),
        num_iters=0,
        render_fps=overall.get("render_fps"),
        render_resolution=overall.get("render_resolution"),
        coverage=overall.get("coverage", coverage),
    )


@register_stage("eval.ncore")
class EvalNcoreStage(Stage):
    """Chấm PLY của Instant NuRec trên khung test của split, ghi ``summary.json``."""

    version: ClassVar[str] = "0.2.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {
        "model": "model",
        "clip": "ncore_clip",
        "split": "split",
        "masks": "masks",
    }
    optional_inputs: ClassVar[frozenset[str]] = frozenset({"masks"})
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
        sensors = {c: s for c, s in clip["sensors"].items() if c in set(ctx.split.cameras)}
        manifest = ctx.scene.manifest
        views_of = clip_test_views(
            clip,
            ctx.split.test,
            sensors,
            {camera.camera_id: i for i, camera in enumerate(manifest.cameras)},
        )
        indices = sorted({int(i) for frames in views_of.values() for i in frames})
        if not indices:
            raise RuntimeError("không có khung test nào của split nằm trong clip NCore")
        plan = {
            "protocol_version": PROTOCOL_VERSION,
            "views": views_of,
            "masks_dir": str(inputs["masks"]) if "masks" in inputs else None,
            "max_angle": fisheye_max_angles(manifest, sensors),
        }
        (ctx.output_dir / PLAN_FILE).write_text(json.dumps(plan, indent=2) + "\n")
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
                "--views-json", str(ctx.output_dir / PLAN_FILE),
            ],
            log,
            should_stop=ctx.should_stop,
        )  # fmt: skip
        views.unlink(missing_ok=True)  # ảnh test đã nằm trong cảnh; file này lớn
        metrics = json.loads((ctx.output_dir / METRICS_FILE).read_text(encoding="utf-8"))
        stats_path = inputs["model"] / STATS_FILE
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        per_view = json.loads((ctx.output_dir / "per_view.json").read_text(encoding="utf-8"))
        files.dump(
            make_summary(
                metrics, ctx.split.split_id, sensors, stats, coverage=mean_coverage(per_view)
            ),
            ctx.output_dir / SUMMARY_FILE,
        )
        (ctx.output_dir / NOTE_FILE).write_text(
            json.dumps(
                {
                    "test_frames_in_clip": indices,
                    "masks": "masks" in inputs,
                    "coverage": metrics["_summary"].get("coverage"),
                    "psnr_covered": metrics["_summary"].get("psnr_covered"),
                    "note": "Instant NuRec dựng từ mọi khung của clip NCore, kể cả khung test. "
                    "Chấm như eval.standard trên mọi pixel hợp lệ (ngoài thân xe, mắt cá trong "
                    "góc dùng được), kể cả chỗ PLY bỏ trống; psnr_covered chỉ tính chỗ "
                    "alpha > 0.9, coverage là tỉ lệ pixel hợp lệ có alpha > 0.9",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        return {"report": ctx.output_dir}
