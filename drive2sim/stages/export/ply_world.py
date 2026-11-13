"""Stage ``export.ply_world``: xuất model đã train ra file PLY 3DGS chuẩn trong hệ ``world``.

Stage này bọc công cụ có sẵn ``drive2sim/recon/tools/export_ply.py``: đọc ``checkpoint_final.pth``
của stage train, ghi ``scene.ply`` (định dạng PLY của 3DGS gốc, mở được bằng các viewer 3DGS
thông dụng) và ``scene.json`` mô tả file (ARCHITECTURE mục 4.3).

File này chứa:
- ``ExportPlyWorldParams``: tham số của stage.
- ``build_command``: dựng câu lệnh export; tách riêng để test được trên máy không có GPU.
- ``ExportPlyWorldStage``: stage đăng ký với engine.

Giới hạn của file PLY, theo ``export_ply.py``: chỉ có nền tĩnh và (tuỳ chọn) xe đặt cố định ở một
khung; không có bầu trời, người SMPL và vật biến dạng (ARCHITECTURE mục 7.2).

Chưa chạy thật: cần checkpoint thật và PyTorch.
"""

import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.commands import LOG_FILE, copy_cameras, run_logged
from drive2sim.stages.train.drivestudio import FINAL_CHECKPOINT

PLY_FILE = "scene.ply"


class ExportPlyWorldParams(BaseModel):
    """Tham số của stage ``export.ply_world``.

    Attributes:
        rigid_frame: Khung dùng để đặt các xe vào cảnh; ``-1`` thì không xuất xe, chỉ xuất nền.
        min_opacity: Bỏ các Gaussian có độ mờ đục nhỏ hơn giá trị này, cho file nhỏ hơn.
    """

    model_config = ConfigDict(extra="forbid")

    rigid_frame: int = Field(default=-1, ge=-1)
    min_opacity: float = Field(default=0.005, ge=0, le=1)


def build_command(params: ExportPlyWorldParams, model_dir: Path, output_dir: Path) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.tools.export_ply``.

    Args:
        params: Tham số đã kiểm tra.
        model_dir: Thư mục kết quả của stage train, chứa ``checkpoint_final.pth``.
        output_dir: Thư mục kết quả của stage này.

    Returns:
        Câu lệnh dạng danh sách.
    """
    return [
        sys.executable,
        "-m",
        "drive2sim.recon.tools.export_ply",
        "--ckpt",
        str(model_dir / FINAL_CHECKPOINT),
        "--out",
        str(output_dir),
        "--rigid_frame",
        str(params.rigid_frame),
        "--min_opacity",
        str(params.min_opacity),
    ]


@register_stage("export.ply_world")
class ExportPlyWorldStage(Stage):
    """Xuất checkpoint cuối của run ra ``scene.ply`` trong hệ ``world``."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model"}
    outputs: ClassVar[dict[str, str]] = {"ply": "export_ply"}
    Params: ClassVar[type[BaseModel]] = ExportPlyWorldParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy ``export_ply.py``, ghi log vào ``log.txt`` của stage.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: ``{"model": <thư mục train>}``.
            params: ``ExportPlyWorldParams`` đã kiểm tra.

        Returns:
            ``{"ply": <thư mục stage>/scene.ply}``.

        Raises:
            RuntimeError: Nếu lệnh export lỗi hoặc không ghi ra ``scene.ply``.
        """
        assert isinstance(params, ExportPlyWorldParams)
        run_logged(
            build_command(params, inputs["model"], ctx.output_dir),
            ctx.output_dir / LOG_FILE,
            should_stop=ctx.should_stop,
        )
        ply = ctx.output_dir / PLY_FILE
        if not ply.is_file():
            raise RuntimeError(f"export_ply.py xong nhưng không có {PLY_FILE}")
        copy_cameras(inputs["model"], ctx.output_dir)
        return {"ply": ply}
