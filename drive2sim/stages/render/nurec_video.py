"""Stage ``render.nurec_video``: video đi dọc cả đoạn đường từ PLY của Instant NuRec.

Video của ``eval.ncore`` chỉ có khung test của một camera (split ``every8`` trên 120 khung là 15
khung, cỡ 2 giây). Stage này vẽ PLY ở **mọi khung** của clip NCore cho các camera của split, như
``render_flythrough.py`` của Fisheye Fusion Lab và video dựng cả đoạn của các approach DriveStudio:
1. ``ncore_eval_export.py --every 1`` (môi trường NuRec): pose, thông số camera, ảnh thật của mọi
   khung.
2. ``ncore_render_video.py`` (môi trường chính, gsplat; camera mắt cá bằng 3DGUT): mỗi khung là
   một hàng ảnh dựng của các camera, mặc định thêm hàng ảnh thật ở trên; ghi ``flythrough.mp4``
   (H.264) và ``flythrough.json`` (số khung, fps, thời lượng). Mặc định video chạy đúng tốc độ gốc:
   mỗi khung giữ trên màn hình bằng khoảng thời gian thật tới khung sau (KITTI-360 trong NuRec
   thường 0,5 s một khung, nên 120 khung là cỡ 1 phút); ``speed`` khác 1 thì nhanh, chậm hơn.
   ``interpolate`` (mặc định bật, 30 fps): khung video giữa hai khung nguồn được vẽ thật từ pose
   camera nội suy, nên chuyển động mượt dù nguồn chỉ 2 ảnh/giây; tắt thì lặp khung nguồn.
   ``dense_gt`` (cảnh KITTI-360 còn dữ liệu gốc): ``kitti360_dense_gt.py`` lấy ảnh thật 10 Hz nằm
   giữa các khung clip từ ``data_2d_raw``, nên hàng ảnh thật cũng mượt; thiếu dữ liệu gốc thì bỏ
   qua. ``gt_flow`` (mặc định bật): không có ảnh thật ở giữa thì ảnh giữa hai ảnh thật được nội suy
   bằng optical flow (ô ghi ``ground truth (interp)``), chỉ ở đoạn cuối ``gt_flow_window`` của mỗi
   quãng (mặc định 20%, phần còn lại giữ ảnh thật); tắt hoặc 0 thì hàng ảnh thật giữ ảnh gần nhất.
Tab Video của Studio liệt kê mọi MP4 trong thư mục stage nên video hiện ra ở đó.

Nằm trong danh sách stage của approach ``nurec`` (sau ``export.nurec_ply``); run tạo trước đó thì
thêm bằng ``d2s export video <run_id>`` hoặc nút "Xuất video đi dọc đường" trên Studio.

File này chứa ``NurecVideoParams``, ``build_render_command`` và ``RenderNurecVideoStage``.
"""

import json
import sys
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.prep.ncore import CLIP_FILE, missing_raw_files, raw_kitti360_dir
from drive2sim.stages.train.nurec import MERGED_PLY

VIDEO_FILE = "flythrough.mp4"
VIEWS_FILE = "views_all.npz"
DENSE_GT_FILE = "gt_dense.npz"


class NurecVideoParams(BaseModel):
    """Tham số của ``render.nurec_video``.

    Attributes:
        fps: Khung hình mỗi giây của file video.
        speed: Tốc độ so với thời gian thật (1 là tốc độ gốc, 2 nhanh gấp đôi); 0 là mỗi khung
            nguồn một khung video (bỏ qua mốc thời gian, nhanh như video của ``eval.ncore``).
        height: Chiều cao (pixel) của ảnh mỗi camera trong hàng.
        with_gt: Thêm hàng ảnh thật ở trên hàng ảnh dựng.
        interpolate: Vẽ khung giữa hai khung nguồn từ pose nội suy (mượt; thời gian vẽ tăng theo
            fps) thay vì lặp lại khung nguồn.
        dense_gt: Cùng ``interpolate``: hàng ảnh thật dùng ảnh 10 Hz gốc của KITTI-360 nằm giữa
            các khung clip (cần dữ liệu gốc trên máy).
        gt_flow: Cùng ``interpolate``: không có ảnh thật ở giữa thì nội suy hàng ảnh thật bằng
            optical flow (ô ghi ``ground truth (interp)``); tắt thì giữ ảnh gần nhất.
        gt_flow_window: Phần cuối mỗi quãng giữa hai ảnh thật được nội suy (0..1); phần trước
            giữ nguyên ảnh thật. 0 là không nội suy, 1 là nội suy cả quãng.
    """

    model_config = ConfigDict(extra="forbid")

    fps: float = Field(default=30, gt=0, le=60)
    speed: float = Field(default=1, ge=0, le=32)
    height: int = Field(default=360, ge=120, le=1080)
    with_gt: bool = True
    interpolate: bool = True
    dense_gt: bool = True
    gt_flow: bool = True
    gt_flow_window: float = Field(default=0.2, ge=0, le=1)


def build_render_command(
    params: NurecVideoParams,
    ply: Path,
    views: Path,
    out: Path,
    cameras: list[str],
    dense_gt: Optional[Path] = None,
) -> list[str]:
    """Câu lệnh ``ncore_render_video.py`` (môi trường chính).

    Args:
        params: Tham số đã kiểm tra.
        ply: ``scene_nurec.ply`` của ``train.nurec``.
        views: ``views_all.npz`` của bước xuất.
        out: File MP4 cần ghi.
        cameras: Tên sensor NCore theo thứ tự camera của split.
        dense_gt: ``gt_dense.npz`` (ảnh thật giữa các khung), nếu có.
    """
    command = [
        sys.executable, "-u", str(nurec.TOOLS_DIR / "ncore_render_video.py"),
        str(ply), str(views), "--out", str(out),
        "--cameras", ",".join(cameras), "--fps", str(params.fps), "--speed", str(params.speed),
        "--height", str(params.height),
    ]  # fmt: skip
    if not params.with_gt:
        command.append("--no-gt")
    if params.interpolate:
        command.append("--interp")
        if dense_gt is not None and params.with_gt:
            command += ["--gt-dense", str(dense_gt)]
        if params.gt_flow and params.with_gt and params.gt_flow_window > 0:
            command += ["--gt-flow", "--gt-flow-window", str(params.gt_flow_window)]
    return command


def dense_gt_command(
    raw_dir: Path, sequence: str, clip_json: Path, out: Path, height: int
) -> list[str]:
    """Câu lệnh ``kitti360_dense_gt.py`` bằng Python của môi trường NuRec."""
    return [
        str(nurec.nurec_python()), "-u", str(nurec.TOOLS_DIR / "kitti360_dense_gt.py"),
        "--data", str(raw_dir), "--sequence", sequence, "--clip", str(clip_json),
        "--out", str(out), "--height", str(height),
    ]  # fmt: skip


@register_stage("render.nurec_video")
class RenderNurecVideoStage(Stage):
    """Vẽ PLY của NuRec ở mọi khung của clip thành video ``flythrough.mp4``."""

    version: ClassVar[str] = "0.2.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model", "clip": "ncore_clip", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"videos": "videos"}
    Params: ClassVar[type[BaseModel]] = NurecVideoParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Xuất mọi khung của clip rồi vẽ video.

        Returns:
            ``{"videos": <thư mục stage>}``.

        Raises:
            RuntimeError: Clip không có camera nào của split, hoặc một bước lỗi.
        """
        assert isinstance(params, NurecVideoParams)
        clip = json.loads((inputs["clip"] / CLIP_FILE).read_text(encoding="utf-8"))
        sensors = [clip["sensors"][c] for c in ctx.split.cameras if c in clip["sensors"]]
        if not sensors:
            raise RuntimeError("clip NCore không có camera nào của split")
        log = ctx.output_dir / LOG_FILE
        views = ctx.output_dir / VIEWS_FILE
        ctx.progress(0.05, f"xuất mọi khung của clip ({len(sensors)} camera)")
        run_logged(
            [
                str(nurec.nurec_python()), "-u", str(nurec.TOOLS_DIR / "ncore_eval_export.py"),
                str(inputs["clip"] / clip["clip"]), "--out", str(views), "--every", "1",
                "--cameras", ",".join(sensors),
            ],
            log,
            should_stop=ctx.should_stop,
        )  # fmt: skip
        dense = self.dense_gt(ctx, inputs["clip"], clip, params, log)
        ctx.progress(0.4, "vẽ từng khung bằng gsplat")
        command = build_render_command(
            params,
            inputs["model"] / MERGED_PLY,
            views,
            ctx.output_dir / VIDEO_FILE,
            sensors,
            dense,
        )
        try:
            run_logged(command, log, should_stop=ctx.should_stop)
        finally:
            views.unlink(missing_ok=True)  # ảnh thật mọi khung: lớn, đã có trong cảnh
            if dense is not None:
                dense.unlink(missing_ok=True)
        if not (ctx.output_dir / VIDEO_FILE).is_file():
            raise RuntimeError(f"ncore_render_video.py xong nhưng không có {VIDEO_FILE}")
        return {"videos": ctx.output_dir}

    def dense_gt(
        self, ctx: StageContext, clip_dir: Path, clip: dict, params: NurecVideoParams, log: Path
    ) -> Optional[Path]:
        """Ghi ``gt_dense.npz`` (ảnh thật 10 Hz giữa các khung clip) khi làm được; không thì None.

        Chỉ với cảnh KITTI-360 còn dữ liệu gốc, clip thưa hơn nguồn, ``with_gt``, ``interpolate``
        và ``dense_gt``. Lỗi ở bước này không làm hỏng video: ghi lý do vào log rồi bỏ qua.
        """
        manifest = ctx.scene.manifest
        ids = [int(f) for f in clip.get("frame_ids", [])]
        if not (params.dense_gt and params.with_gt and params.interpolate):
            return None
        if manifest.source.dataset != "kitti360" or not any(
            b - a > 1 for a, b in zip(ids, ids[1:])
        ):
            return None
        sequence = manifest.source.sequence_id
        try:
            raw = raw_kitti360_dir(ctx.scene.root)
            missing = missing_raw_files(raw, sequence, list(clip["sensors"]), lidar=False)
            if missing:
                raise RuntimeError(f"thiếu {', '.join(missing)} trong {raw}")
            out = ctx.output_dir / DENSE_GT_FILE
            ctx.progress(0.3, "lấy ảnh thật 10 Hz giữa các khung từ KITTI-360 gốc")
            run_logged(
                dense_gt_command(raw, sequence, clip_dir / CLIP_FILE, out, params.height),
                log,
                should_stop=ctx.should_stop,
            )
            return out if out.is_file() else None
        except RuntimeError as err:
            if ctx.should_stop():
                raise
            with log.open("a", encoding="utf-8") as f:
                f.write(f"bỏ qua ảnh thật 10 Hz (hàng ảnh thật giữ ảnh gần nhất): {err}\n")
            return None
