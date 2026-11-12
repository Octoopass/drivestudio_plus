"""Stage ``export.preview``: xuất gói xem trước để xem cảnh trong trình duyệt bằng Spark.

Stage này chạy ``drive2sim/recon/tools/export_preview.py`` trên ``checkpoint_final.pth`` của
stage train và ghi gói theo CONTRACTS mục 7: ``background.ply``, ``actors/<track_id>.ply``,
``actor_poses.json``, ``preview.json``. Studio mở gói ở trang xem 3D, phát lại xe theo khung;
máy xem chỉ cần WebGL2, không cần GPU CUDA (ARCHITECTURE mục 2.13).

File này chứa:
- ``ExportPreviewParams``: tham số của stage.
- ``build_command``: dựng câu lệnh export; tách riêng để test được trên máy không có GPU.
- ``ExportPreviewStage``: stage đăng ký với engine.

Không nằm trong approach mặc định; thêm vào run bằng ``d2s export preview <run_id>`` hoặc nút
trên Studio. Chưa chạy thật: cần checkpoint thật và PyTorch.
"""

import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.core import files
from drive2sim.core.schema import Preview
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.train.drivestudio import FINAL_CHECKPOINT

PREVIEW_FILE = "preview.json"


class ExportPreviewParams(BaseModel):
    """Tham số của stage ``export.preview``.

    Attributes:
        min_opacity: Bỏ các Gaussian có độ mờ đục nhỏ hơn giá trị này, cho file nhỏ hơn.
    """

    model_config = ConfigDict(extra="forbid")

    min_opacity: float = Field(default=0.005, ge=0, le=1)


def build_command(
    params: ExportPreviewParams, model_dir: Path, output_dir: Path, source_run: str
) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.tools.export_preview``.

    Args:
        params: Tham số đã kiểm tra.
        model_dir: Thư mục kết quả của stage train, chứa ``checkpoint_final.pth``.
        output_dir: Thư mục kết quả của stage này.
        source_run: Thư mục run, tương đối với thư mục làm việc; ghi vào ``preview.json``.

    Returns:
        Câu lệnh dạng danh sách.
    """
    return [
        sys.executable,
        "-m",
        "drive2sim.recon.tools.export_preview",
        "--ckpt",
        str(model_dir / FINAL_CHECKPOINT),
        "--out",
        str(output_dir),
        "--source_run",
        source_run,
        "--min_opacity",
        str(params.min_opacity),
    ]


@register_stage("export.preview")
class ExportPreviewStage(Stage):
    """Xuất checkpoint cuối của run ra gói xem trước cho Spark."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model"}
    outputs: ClassVar[dict[str, str]] = {"preview": "preview"}
    Params: ClassVar[type[BaseModel]] = ExportPreviewParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy ``export_preview.py`` rồi kiểm tra ``preview.json`` đúng schema.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: ``{"model": <thư mục train>}``.
            params: ``ExportPreviewParams`` đã kiểm tra.

        Returns:
            ``{"preview": <thư mục stage>}``.

        Raises:
            RuntimeError: Nếu lệnh lỗi hoặc không ghi ra ``background.ply``, ``preview.json``.
        """
        assert isinstance(params, ExportPreviewParams)
        run_dir = ctx.output_dir.parents[2]  # <run>/d2s/stages/export.preview
        source_run = Path(*run_dir.parts[-4:]).as_posix()  # runs/<cảnh>/<approach>/<run_id>
        command = build_command(params, inputs["model"], ctx.output_dir, source_run)
        run_logged(command, ctx.output_dir / LOG_FILE, should_stop=ctx.should_stop)
        for name in ("background.ply", PREVIEW_FILE):
            if not (ctx.output_dir / name).is_file():
                raise RuntimeError(f"export_preview.py xong nhưng không có {name}")
        files.load(ctx.output_dir / PREVIEW_FILE, Preview)
        return {"preview": ctx.output_dir}
