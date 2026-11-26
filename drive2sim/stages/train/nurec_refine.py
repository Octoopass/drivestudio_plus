"""Stage ``train.nurec_refine``: tối ưu tiếp PLY của Instant NuRec trên ảnh train của clip.

Instant NuRec đoán Gaussian trong một lượt từ vài khung ngữ cảnh, nên ảnh dựng nhoè và nhạt (độ
tương phản thấp hơn ảnh thật nhiều). Stage này lấy ``scene_nurec.ply`` của ``train.nurec`` làm
điểm xuất phát rồi tối ưu tiếp như 3DGS:
1. ``ncore_eval_export.py`` (môi trường NuRec): pose, thông số camera, ảnh, mask ego của mọi khung
   train của split trong clip (khung có trong phần test của split bị bỏ, để ``eval.ncore`` vẫn
   chấm trên khung chưa thấy).
2. ``ncore_refine.py`` (môi trường chính, gsplat; camera mắt cá bằng 3DGUT): L1 + D-SSIM, giữ
   nguyên số Gaussian (``strategy: mcmc`` thêm Gaussian, còn thử nghiệm).

Kết quả ``scene_nurec.ply`` cùng định dạng với PLY của ``train.nurec`` nên ``eval.ncore``,
``export.nurec_ply``, ``render.nurec_video``, ``render.carla`` đứng sau dùng luôn (đầu vào
``model`` nhận stage gần nhất phía trước). ``train_stats.json`` cộng thời gian tối ưu vào thời gian
dựng. ``iters`` = 0 thì chỉ chép PLY (tắt tối ưu).

File này chứa ``RefineParams``, ``train_indices``, ``refine_command`` và ``TrainNurecRefineStage``.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.eval.ncore import clip_test_indices
from drive2sim.stages.prep.ncore import CLIP_FILE
from drive2sim.stages.train.nurec import MERGED_PLY, STATS_FILE

VIEWS_FILE = "train_views.npz"
REFINE_FILE = "refine.json"


class RefineParams(BaseModel):
    """Tham số của ``train.nurec_refine``.

    Attributes:
        iters: Số bước tối ưu; 0 là tắt (chép PLY của NuRec).
        strategy: ``none`` (mặc định) giữ nguyên số Gaussian; ``mcmc`` dời và thêm Gaussian
            (thử nghiệm: với drive_0006 làm PSNR giảm).
        cap_factor: Với ``mcmc``: số Gaussian tối đa so với PLY của NuRec.
        max_side: Ảnh có cạnh dài hơn bị thu nhỏ về cỡ này khi tối ưu.
        means_lr: Tốc độ học vị trí Gaussian (mét mỗi bước), giảm dần còn 1% ở bước cuối.
        ssim_weight: Trọng số D-SSIM trong hàm lỗi (phần còn lại là L1).
    """

    model_config = ConfigDict(extra="forbid")

    iters: int = Field(default=30000, ge=0, le=200000)
    strategy: Literal["none", "mcmc"] = "none"
    cap_factor: float = Field(default=1.5, ge=1, le=4)
    max_side: int = Field(default=1408, ge=256, le=4096)
    means_lr: float = Field(default=1e-3, gt=0, le=0.1)
    ssim_weight: float = Field(default=0.2, ge=0, le=1)


def train_indices(clip: dict, test_frames: list[int]) -> list[int]:
    """Chỉ số khung của clip dùng để tối ưu: mọi khung trừ khung test của split.

    Args:
        clip: ``clip.json`` của ``prep.ncore``.
        test_frames: Chỉ số khung test trong cảnh (``split.test``).
    """
    test = set(clip_test_indices(clip, test_frames))
    return [i for i in range(len(clip["frame_ids"])) if i not in test]


def refine_command(params: RefineParams, ply: Path, views: Path, out: Path) -> list[str]:
    """Câu lệnh ``ncore_refine.py`` (môi trường chính, cần gsplat)."""
    return [
        sys.executable, "-u", str(nurec.TOOLS_DIR / "ncore_refine.py"), str(ply), str(views),
        "--out", str(out), "--iters", str(params.iters), "--strategy", params.strategy,
        "--cap-factor", str(params.cap_factor), "--max-side", str(params.max_side),
        "--means-lr", str(params.means_lr), "--ssim-weight", str(params.ssim_weight),
    ]  # fmt: skip


@register_stage("train.nurec_refine")
class TrainNurecRefineStage(Stage):
    """Tối ưu tiếp PLY của Instant NuRec trên các khung train của clip NCore."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model", "clip": "ncore_clip", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = RefineParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Xuất khung train, tối ưu, ghi ``scene_nurec.ply`` và ``train_stats.json``.

        Returns:
            ``{"model": <thư mục stage>}``.

        Raises:
            RuntimeError: Split không có khung train nào trong clip, hoặc một bước lỗi.
        """
        assert isinstance(params, RefineParams)
        out = ctx.output_dir
        source = inputs["model"] / MERGED_PLY
        stats_path = inputs["model"] / STATS_FILE
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        if params.iters == 0:
            shutil.copy2(source, out / MERGED_PLY)
            (out / STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n")
            return {"model": out}
        clip = json.loads((inputs["clip"] / CLIP_FILE).read_text(encoding="utf-8"))
        indices = train_indices(clip, [frame for frame, _ in ctx.split.test])
        if not indices:
            raise RuntimeError("không có khung train nào của split nằm trong clip NCore")
        sensors = [clip["sensors"][c] for c in ctx.split.cameras if c in clip["sensors"]]
        log = out / LOG_FILE
        views = out / VIEWS_FILE
        ctx.progress(0.02, f"xuất {len(indices)} khung train ({len(sensors)} camera)")
        try:
            run_logged(
                [
                    str(nurec.nurec_python()), "-u", str(nurec.TOOLS_DIR / "ncore_eval_export.py"),
                    str(inputs["clip"] / clip["clip"]), "--out", str(views),
                    "--frame-indices", ",".join(map(str, indices)),
                    "--cameras", ",".join(sensors),
                ],
                log,
                should_stop=ctx.should_stop,
            )  # fmt: skip
            ctx.progress(0.1, f"tối ưu {params.iters} bước bằng gsplat ({params.strategy})")
            run_logged(
                refine_command(params, source, views, out / MERGED_PLY),
                log,
                should_stop=ctx.should_stop,
            )
        finally:
            views.unlink(missing_ok=True)  # ảnh mọi khung train: hàng chục GB, đã có trong cảnh
        refine = json.loads((out / REFINE_FILE).read_text(encoding="utf-8"))
        ctx.report_peak_vram(int(refine.get("peak_vram_mb", 0)))
        stats.update(
            {
                "train_time_s": round(
                    float(stats.get("train_time_s") or 0) + refine["refine_time_s"], 1
                ),
                "num_gaussians": refine["gaussians_after"],
                "num_iters": params.iters,
                "refine": refine,
            }
        )
        (out / STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n")
        return {"model": out}
