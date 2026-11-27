"""Stage ``eval.mcgs``: chấm bản đồ Gaussian của MCGS-SLAM trên các khung test của split.

MCGS-SLAM tự đo PSNR trên keyframe đã dùng để train (``eval_rendering_kf``), nên con số đó không so
được với các approach khác. Stage này đo lại trên ảnh test theo cùng cách như ``eval.ncore``:
1. Căn quỹ đạo keyframe ``traj_mcgs.txt`` vào pose camera tham chiếu của cảnh bằng phép đồng dạng
   (Umeyama), ghi ATE; đổi pose thật của các khung test sang hệ bản đồ (co 0,2 lần).
2. Ghi ``views.npz`` (``external/mcgs/views.py``) rồi chạy ``ncore_eval_render.py`` ở môi trường
   chính: vẽ ``3dgs_final.ply`` bằng gsplat (3DGUT cho camera mắt cá, ``with_ut``) và tính PSNR,
   SSIM, LPIPS trên vùng hợp lệ (ngoài mask thân xe và góc quá ``max_angle``, alpha > 0,9).
Kết quả đổi sang ``summary.json`` và ``frames.jsonl`` (CONTRACTS mục 6.3).

Giới hạn cần đọc khi dùng số này (ghi trong ``mcgs_eval.json``):
- Bản đồ chỉ dựng từ ảnh pinhole (SLAM không hỗ trợ fisheye), nên camera fisheye nhìn nhiều nơi
  bản đồ chưa có Gaussian (alpha thấp); PSNR fisheye chỉ tính trên phần bản đồ phủ tới.
- Pose khung test là pose thật đã căn, không phải pose SLAM ước lượng; sai số căn quỹ đạo (ATE)
  làm ảnh vẽ lệch nhẹ.
- ``held_out`` cho biết khung test có bị loại khỏi đầu vào của SLAM không; ``False`` thì điểm là
  điểm trên ảnh đã thấy.
- SH bậc 0 (``ncore_eval_render.load_ply`` chỉ đọc ``f_dc``): màu phụ thuộc hướng nhìn bị bỏ.

Khi chạy sau ``train.mcgs`` (approach ``mcgs``), ``held_out`` và camera tham chiếu lấy từ
``mcgs_prep.json`` mà ``prep.mcgs`` ghi và ``train.mcgs`` chép sang thư mục model; thiếu file này
(thư mục output của ``demo.py`` chạy tay, trỏ bằng ``model_dir``) thì giữ mặc định cũ: ``False`` và
``image_00``. Split có camera chỉ test thì ``overall`` chỉ tính trên camera đã train và camera chỉ
test vào ``Summary.shifted`` như ``eval.standard`` (protocol 0.5, CONTRACTS mục 6.3).

Chưa chạy thật (cần một lần chạy MCGS-SLAM, ARCHITECTURE mục 7.1).

File này chứa:
- ``EvalMcgsParams``: tham số.
- ``prep_settings``, ``resolve_settings``: lựa chọn của ``prep.mcgs`` và giá trị cuối cùng dùng.
- ``frame_records``: bản ghi ``frames.jsonl``.
- ``summarize``: ``metrics.json`` và bản ghi từng ảnh → ``Summary`` (không cần GPU).
- ``EvalMcgsStage``: stage đăng ký với engine.
"""

import json
import math
import sys
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.core import files
from drive2sim.core.schema import ShiftedMetrics, Split, Summary
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external import mcgs, nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.eval.ncore import METRICS_FILE, VIEWS_FILE, make_summary, mean_coverage
from drive2sim.stages.eval.standard import SUMMARY_FILE, aggregate_frames, split_frames_by_role
from drive2sim.stages.prep.mcgs import PREP_FILE
from drive2sim.stages.train.drivestudio import RENDERER_MAX_ANGLE

FRAMES_FILE = "frames.jsonl"
PER_VIEW_FILE = "per_view.json"
NOTE_FILE = "mcgs_eval.json"
STATS_FILE = "train_stats.json"
DEFAULT_REFERENCE_CAMERA = "image_00"


class EvalMcgsParams(BaseModel):
    """Tham số của ``eval.mcgs``.

    Attributes:
        model_dir: Thư mục output của ``demo.py`` (có quỹ đạo và PLY). Bỏ trống thì dùng đầu vào
            ``model`` của stage trước.
        ply: Tên file Gaussian trong thư mục đó.
        trajectory: Tên file quỹ đạo TUM trong thư mục đó.
        reference_camera: ``camera_id`` của camera 0 của SLAM (thư mục ảnh đầu tiên khi chạy);
            ``None`` thì lấy từ ``mcgs_prep.json``, không có thì ``image_00``.
        frames_per_stamp: Nhân timestamp trong quỹ đạo để ra chỉ số khung của cảnh; 1 khi tên file
            ảnh đưa vào SLAM là chỉ số khung.
        with_scale: Căn cả tỉ lệ; ``False`` khi tin tỉ lệ mét của chế độ stereo.
        pose_scale: Hệ số co của bản đồ so với quỹ đạo (``mcgs.POSE_SCALE``).
        max_angle: Góc lệch trục tối đa (rad) của pixel fisheye khi chấm; mặc định như train.
        held_out: Khung test đã bị loại khỏi đầu vào của SLAM (ghi vào ghi chú kết quả); ``None``
            thì lấy từ ``mcgs_prep.json``, không có thì ``False``.
    """

    model_config = ConfigDict(extra="forbid")

    model_dir: Optional[str] = None
    ply: str = mcgs.OUTPUT_GAUSSIANS
    trajectory: str = mcgs.OUTPUT_TRAJECTORY
    reference_camera: Optional[str] = None
    frames_per_stamp: float = Field(default=1.0, gt=0)
    with_scale: bool = True
    pose_scale: float = Field(default=mcgs.POSE_SCALE, gt=0)
    max_angle: float = Field(default=RENDERER_MAX_ANGLE, gt=0, le=math.pi)
    held_out: Optional[bool] = None


def prep_settings(model_dir: Path) -> dict:
    """Nội dung ``mcgs_prep.json`` trong thư mục model, hoặc ``{}`` nếu không có.

    Args:
        model_dir: Thư mục model (output của ``train.mcgs`` hoặc của ``demo.py`` chạy tay).

    Returns:
        Từ điển ``layout``, ``reference_camera``, ``held_out``, ...; rỗng với thư mục không do
        ``train.mcgs`` tạo (run cũ), để giữ hành vi cũ.
    """
    path = model_dir / PREP_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def resolve_settings(params: EvalMcgsParams, prep: dict) -> tuple[bool, str]:
    """Giá trị ``held_out`` và camera tham chiếu cuối cùng: tham số, rồi ``prep``, rồi mặc định.

    Args:
        params: Tham số của stage; giá trị khác ``None`` thắng.
        prep: Kết quả ``prep_settings``.

    Returns:
        ``(held_out, reference_camera)``; mặc định ``(False, "image_00")`` như trước khi có
        ``prep.mcgs``.
    """
    held_out = (
        params.held_out if params.held_out is not None else bool(prep.get("held_out", False))
    )
    reference = params.reference_camera or prep.get("reference_camera") or DEFAULT_REFERENCE_CAMERA
    return held_out, reference


def frame_records(views_info: list[dict], per_view: list[dict]) -> list[dict]:
    """Bản ghi ``frames.jsonl``, mỗi ảnh test một dòng (CONTRACTS mục 6.3).

    Args:
        views_info: Kết quả ``views.build_views``: ``frame``, ``camera_index``, ``position``.
        per_view: ``per_view.json`` của ``ncore_eval_render.py``, cùng thứ tự; ``psnr``, ``ssim``,
            ``lpips`` là ``None`` khi view không có điểm ảnh hợp lệ.

    Returns:
        Danh sách bản ghi; ``depth`` và ``time`` luôn ``None``.
    """
    return [
        {
            "frame_index": view["frame"],
            "camera_index": view["camera_index"],
            "time": None,
            "position": view["position"],
            "psnr": result["psnr"],
            "ssim": result["ssim"],
            "lpips": result["lpips"],
            "coverage": result["coverage"],
            "depth": None,
        }
        for view, result in zip(views_info, per_view)
    ]


def summarize(
    metrics: dict,
    records: list[dict],
    split: Split,
    camera_ids: list[str],
    stats: Optional[dict],
    coverage: Optional[float],
) -> Summary:
    """Dựng ``Summary`` từ ``metrics.json`` và bản ghi từng ảnh.

    Không có camera chỉ test thì ``overall`` lấy từ ``metrics.json``. Có camera chỉ test thì
    ``overall`` tính lại từ ``records`` chỉ trên camera đã train (R1) và camera chỉ test vào
    ``shifted`` (R2), như ``eval.standard`` protocol 0.5; ảnh có ``psnr`` là ``None`` không vào
    trung bình. Không có ``psnr_cc``, ``lpips_cc`` vì stage này không chỉnh màu.

    Args:
        metrics: ``metrics.json`` của ``ncore_eval_render.py``; khoá camera là ``camera_id``.
        records: Kết quả ``frame_records``.
        split: Split của run.
        camera_ids: ``camera_id`` theo thứ tự trong manifest (để đổi ``camera_index``).
        stats: ``train_stats.json``.
        coverage: Tỉ lệ điểm ảnh được chấm (``mean_coverage``).

    Returns:
        Đối tượng ``Summary``; ``shifted`` để trống nếu không có camera chỉ test hoặc camera đó
        không có ảnh nào chấm được.

    Raises:
        ValueError: Có camera chỉ test mà không ảnh nào của camera đã train chấm được.
    """
    summary = make_summary(
        metrics, split.split_id, {c: c for c in split.cameras}, stats, coverage=coverage
    )
    if not split.test_only_cameras:
        return summary
    test_only_indices = {camera_ids.index(camera) for camera in split.test_only_cameras}
    trained, shifted_frames = split_frames_by_role(records, test_only_indices)
    trained_values = aggregate_frames(trained)
    if any(trained_values[name] is None for name in ("psnr", "ssim", "lpips")):
        raise ValueError("không ảnh test nào của camera đã train chấm được (không có điểm hợp lệ)")
    overall = summary.overall.model_copy(
        update={name: trained_values[name] for name in ("psnr", "ssim", "lpips")}
    )
    scored = [item for item in shifted_frames if item["psnr"] is not None]
    shifted = None
    if scored:
        values = aggregate_frames(scored)
        shifted = ShiftedMetrics(
            cameras=list(split.test_only_cameras),
            num_images=len(scored),
            psnr=values["psnr"],
            ssim=values["ssim"],
            lpips=values["lpips"],
        )
    return summary.model_copy(update={"overall": overall, "shifted": shifted})


@register_stage("eval.mcgs")
class EvalMcgsStage(Stage):
    """Chấm ``3dgs_final.ply`` của MCGS-SLAM trên khung test của split, ghi ``summary.json``."""

    version: ClassVar[str] = "0.3.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model", "split": "split"}
    optional_inputs: ClassVar[frozenset[str]] = frozenset({"model"})
    outputs: ClassVar[dict[str, str]] = {"report": "eval_report"}
    Params: ClassVar[type[BaseModel]] = EvalMcgsParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Căn quỹ đạo, xuất view test, vẽ và chấm, ghi ``summary.json`` và ``frames.jsonl``.

        Returns:
            ``{"report": <thư mục stage>}``.

        Raises:
            ValueError: Thiếu thư mục model, hoặc split không có khung test nào.
            FileNotFoundError: Thiếu quỹ đạo hoặc PLY trong thư mục model.
            RuntimeError: Nếu bước vẽ lỗi.
        """
        assert isinstance(params, EvalMcgsParams)
        model_dir = Path(params.model_dir) if params.model_dir else inputs.get("model")
        if model_dir is None:
            raise ValueError("eval.mcgs cần model_dir hoặc đầu vào model của stage train")
        trajectory, ply = model_dir / params.trajectory, model_dir / params.ply
        for needed in (trajectory, ply):
            if not needed.is_file():
                raise FileNotFoundError(f"{needed}: MCGS-SLAM chưa ghi file này (demo.py)")
        pairs = [(frame, camera) for frame, camera in ctx.split.test]
        if not pairs:
            raise ValueError("split không có ảnh test")

        from drive2sim.external.mcgs import views  # cần numpy và OpenCV: nạp khi stage chạy

        manifest = ctx.scene.manifest
        camera_ids = [camera.camera_id for camera in manifest.cameras]
        camera_models = {camera.camera_id: camera.model for camera in manifest.cameras}
        held_out, reference_camera = resolve_settings(params, prep_settings(model_dir))
        if reference_camera not in camera_ids:
            raise ValueError(
                f"reference_camera {reference_camera} không có trong cảnh "
                f"({', '.join(camera_ids)}); đặt camera 0 của SLAM bằng --set"
            )
        ctx.progress(0.05, "căn quỹ đạo SLAM vào pose thật")
        dataset = manifest.source.dataset
        poses = views.ScenePoses(ctx.scene.root, dataset, manifest.timeline.num_frames)
        alignment = views.fit_alignment(
            trajectory,
            poses,
            camera_ids.index(reference_camera),
            params.frames_per_stamp,
            params.with_scale,
        )
        ctx.progress(0.15, f"xuất {len(pairs)} ảnh test")
        views_path = ctx.output_dir / VIEWS_FILE
        info = views.build_views(
            ctx.scene.root,
            poses,
            camera_ids,
            camera_models,
            pairs,
            alignment,
            views_path,
            max_angle=params.max_angle,
            dataset=dataset,
            pose_scale=params.pose_scale,
        )
        ctx.progress(0.4, "vẽ và chấm bằng gsplat")
        run_logged(
            [
                sys.executable, "-u", str(nurec.TOOLS_DIR / "ncore_eval_render.py"),
                str(ply), str(views_path), "--out", str(ctx.output_dir),
            ],
            ctx.output_dir / LOG_FILE,
            should_stop=ctx.should_stop,
        )  # fmt: skip
        views_path.unlink(missing_ok=True)  # chứa ảnh thật của mọi khung test; file này lớn

        metrics = json.loads((ctx.output_dir / METRICS_FILE).read_text(encoding="utf-8"))
        per_view = json.loads((ctx.output_dir / PER_VIEW_FILE).read_text(encoding="utf-8"))
        records = frame_records(info, per_view)
        with (ctx.output_dir / FRAMES_FILE).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        stats_path = model_dir / STATS_FILE
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        coverage = mean_coverage(per_view)
        files.dump(
            summarize(metrics, records, ctx.split, camera_ids, stats, coverage),
            ctx.output_dir / SUMMARY_FILE,
        )
        scale = alignment["scale"]
        (ctx.output_dir / NOTE_FILE).write_text(
            json.dumps(
                {
                    "alignment_scale": scale,
                    "ate_rmse_m": alignment["ate_rmse_m"],
                    "keyframes": alignment["keyframes"],
                    "held_out": held_out,
                    "reference_camera": reference_camera,
                    "coverage": coverage,
                    "ply": params.ply,
                    "note": "Bản đồ chỉ dựng từ ảnh pinhole; pose khung test là pose thật đã "
                    "căn bằng quỹ đạo keyframe; màu SH bậc 0; PSNR chỉ đo trên phần ảnh "
                    "có alpha > 0,9 (coverage). "
                    + (
                        ""
                        if held_out
                        else "Khung test nằm trong đầu vào của SLAM nên điểm không phải ảnh "
                        "chưa thấy."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        ctx.progress(1.0, f"ATE {alignment['ate_rmse_m']:.2f} m, tỉ lệ {scale:.3f}")
        return {"report": ctx.output_dir}
