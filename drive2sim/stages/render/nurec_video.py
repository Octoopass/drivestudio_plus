"""Stage ``render.nurec_video``: video đi dọc cả đoạn đường từ PLY của Instant NuRec.

Video của ``eval.ncore`` chỉ có khung test của một camera (split ``every8`` trên 120 khung là 15
khung, cỡ 2 giây). Stage này vẽ PLY ở **mọi khung** của clip NCore cho các camera của split, như
``render_flythrough.py`` của Fisheye Fusion Lab và video dựng cả đoạn của các approach DriveStudio:
1. ``ncore_eval_export.py --every 1`` (môi trường NuRec): pose, thông số camera, ảnh thật của mọi
   khung.
2. ``ncore_render_video.py`` (môi trường chính, gsplat; camera mắt cá bằng 3DGUT): mỗi khung là
   một hàng ảnh dựng của các camera, mặc định thêm hàng ảnh thật ở trên; ghi ``flythrough.mp4``
   (H.264) và ``flythrough.json`` (số khung, fps, thời lượng).
Tab Video của Studio liệt kê mọi MP4 trong thư mục stage nên video hiện ra ở đó.

Nằm trong danh sách stage của approach ``nurec`` (sau ``export.nurec_ply``); run tạo trước đó thì
thêm bằng ``d2s export video <run_id>`` hoặc nút "Xuất video đi dọc đường" trên Studio.

File này chứa ``NurecVideoParams``, ``build_render_command`` và ``RenderNurecVideoStage``.
"""

import json
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.prep.ncore import CLIP_FILE
from drive2sim.stages.train.nurec import MERGED_PLY

VIDEO_FILE = "flythrough.mp4"
VIEWS_FILE = "views_all.npz"


class NurecVideoParams(BaseModel):
    """Tham số của ``render.nurec_video``.

    Attributes:
        fps: Khung hình mỗi giây của video.
        height: Chiều cao (pixel) của ảnh mỗi camera trong hàng.
        with_gt: Thêm hàng ảnh thật ở trên hàng ảnh dựng.
    """

    model_config = ConfigDict(extra="forbid")

    fps: float = Field(default=10, gt=0, le=60)
    height: int = Field(default=360, ge=120, le=1080)
    with_gt: bool = True


def build_render_command(
    params: NurecVideoParams, ply: Path, views: Path, out: Path, cameras: list[str]
) -> list[str]:
    """Câu lệnh ``ncore_render_video.py`` (môi trường chính).

    Args:
        params: Tham số đã kiểm tra.
        ply: ``scene_nurec.ply`` của ``train.nurec``.
        views: ``views_all.npz`` của bước xuất.
        out: File MP4 cần ghi.
        cameras: Tên sensor NCore theo thứ tự camera của split.
    """
    command = [
        sys.executable, "-u", str(nurec.TOOLS_DIR / "ncore_render_video.py"),
        str(ply), str(views), "--out", str(out),
        "--cameras", ",".join(cameras), "--fps", str(params.fps), "--height", str(params.height),
    ]  # fmt: skip
    if not params.with_gt:
        command.append("--no-gt")
    return command


@register_stage("render.nurec_video")
class RenderNurecVideoStage(Stage):
    """Vẽ PLY của NuRec ở mọi khung của clip thành video ``flythrough.mp4``."""

    version: ClassVar[str] = "0.1.0"
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
        ctx.progress(0.4, "vẽ từng khung bằng gsplat")
        command = build_render_command(
            params, inputs["model"] / MERGED_PLY, views, ctx.output_dir / VIDEO_FILE, sensors
        )
        try:
            run_logged(command, log, should_stop=ctx.should_stop)
        finally:
            views.unlink(missing_ok=True)  # ảnh thật mọi khung: lớn, đã có trong cảnh
        if not (ctx.output_dir / VIDEO_FILE).is_file():
            raise RuntimeError(f"ncore_render_video.py xong nhưng không có {VIDEO_FILE}")
        return {"videos": ctx.output_dir}
