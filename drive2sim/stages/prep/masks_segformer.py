"""Stage ``prep.masks_segformer``: sinh mask bầu trời (và mask vật động tinh) bằng SegFormer.

Stage này bọc script có sẵn ``drive2sim/recon/preprocess/masks/segformer_hf.py`` (bản dùng
SegFormer-B5 của Hugging Face, chạy trong môi trường chính). Script đọc ảnh của cảnh và ghi:
- ``sky_masks/<khung>_<camera>.png``: mask bầu trời, cần cho phần ``Sky`` của OmniRe và Street
  Gaussians;
- ``fine_dynamic_masks/{all,human,vehicle}/<khung>_<camera>.png`` khi ``dynamic`` bật: mask vật
  động tinh, giao giữa mask thô từ box (``dynamic_masks/``, do ingest tạo) và phân đoạn ngữ nghĩa.

Kết quả là artifact loại ``masks`` trong ``cache/`` (stage có cache): mọi approach trên cùng cảnh
dùng chung, chỉ sinh một lần (ARCHITECTURE mục 2.7, 4.3). Script chạy trên một thư mục cảnh ảo
dựng bằng symlink (``drive2sim/stages/scene_view.py``), nên dữ liệu chuẩn không bị sửa.

File này chứa ``PrepMasksSegformerParams``, ``build_command`` và ``PrepMasksSegformerStage``.

Cần GPU (mặc định ``cuda:0``) và gói ``transformers``. Chưa chạy thật.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.scene_view import VIEW_SCENE, build_view, collect_outputs

OUTPUT_NAMES = ("sky_masks", "fine_dynamic_masks")
VIEW_DIR = "view"
DEVICE_ENV = "D2S_DEVICE"  # đổi thiết bị mà không đổi khoá cache, ví dụ D2S_DEVICE=cpu


class PrepMasksSegformerParams(BaseModel):
    """Tham số của stage ``prep.masks_segformer``.

    Attributes:
        model: Model SegFormer trên Hugging Face.
        dynamic: Sinh thêm mask vật động tinh; cần cảnh có ``dynamic_masks/`` (từ box).
    """

    model_config = ConfigDict(extra="forbid")

    model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    dynamic: bool = True


def build_command(params: PrepMasksSegformerParams, view_root: Path) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.preprocess.masks.segformer_hf``.

    Thiết bị (GPU hay CPU) lấy từ biến môi trường ``D2S_DEVICE`` (mặc định ``cuda:0``), không phải
    tham số, để đổi thiết bị không làm đổi khoá cache: mask sinh trên GPU hay CPU là như nhau.

    Args:
        params: Tham số đã kiểm tra.
        view_root: Thư mục chứa cảnh ảo, truyền làm ``--data_root``.

    Returns:
        Câu lệnh dạng danh sách.
    """
    command = [
        sys.executable,
        "-m",
        "drive2sim.recon.preprocess.masks.segformer_hf",
        "--data_root",
        str(view_root),
        "--scene_ids",
        str(int(VIEW_SCENE)),
        "--model",
        params.model,
        "--device",
        os.environ.get(DEVICE_ENV, "cuda:0"),
    ]
    if params.dynamic:
        command.append("--process_dynamic_mask")
    return command


@register_stage("prep.masks_segformer")
class PrepMasksSegformerStage(Stage):
    """Sinh mask bầu trời và mask vật động tinh cho một cảnh, ghi vào cache."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = True
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"scene": "scene"}
    outputs: ClassVar[dict[str, str]] = {"masks": "masks"}
    Params: ClassVar[type[BaseModel]] = PrepMasksSegformerParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Dựng cảnh ảo, chạy SegFormer, chuyển mask ra thư mục của stage.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: Không có đầu vào artifact.
            params: ``PrepMasksSegformerParams`` đã kiểm tra.

        Returns:
            ``{"masks": <thư mục stage>}``, chứa ``sky_masks/`` và (nếu bật)
            ``fine_dynamic_masks/``.

        Raises:
            RuntimeError: Nếu script lỗi hoặc không sinh ra ``sky_masks/``.
        """
        assert isinstance(params, PrepMasksSegformerParams)
        view_root = ctx.output_dir / VIEW_DIR
        view = build_view(ctx.scene.root, view_root, exclude=OUTPUT_NAMES)
        ctx.progress(0.0, f"sinh mask bằng {params.model}")
        run_logged(
            build_command(params, view_root),
            ctx.output_dir / LOG_FILE,
            should_stop=ctx.should_stop,
        )
        moved = collect_outputs(view, ctx.output_dir, OUTPUT_NAMES)
        shutil.rmtree(view_root)
        if "sky_masks" not in moved:
            raise RuntimeError("segformer_hf.py xong nhưng không có sky_masks/")
        ctx.progress(1.0, "sinh mask xong")
        return {"masks": ctx.output_dir}
