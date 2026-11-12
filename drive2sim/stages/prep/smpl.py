"""Stage ``prep.smpl``: sinh tư thế người (SMPL) theo khung cho các người đi bộ trong cảnh.

Stage này bọc quy trình có sẵn ``drive2sim/recon/preprocess/smpl/run.py`` (hướng dẫn
``docs/datasets/human_pose.md``): chiếu box người lên ảnh, chạy 4D-Humans để ước lượng SMPL, ghép
kết quả với box, rồi ghi ``humanpose/smpl.pkl`` (cùng các file trung gian trong
``humanpose/temp/``, ``humanpose/postprocess/``). OmniRe cần file này cho ``SMPLNodes``; Street
Gaussians không cần.

Kết quả là artifact loại ``smpl`` trong ``cache/`` (stage có cache). Script chạy trên một thư mục
cảnh ảo dựng bằng symlink (``drive2sim/stages/scene_view.py``), nên dữ liệu chuẩn không bị sửa.

File này chứa ``build_command`` và ``PrepSmplStage``.

Cần GPU, gói ``hmr2`` (``third_party/Humans4D``) và cảnh có box 3D. Chưa chạy thật.
"""

import shutil
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.scene_view import (
    DRIVESTUDIO_DATASETS,
    VIEW_SCENE,
    build_view,
    collect_outputs,
)

OUTPUT_NAME = "humanpose"
VIEW_DIR = "view"


def build_command(dataset: str, view_root: Path) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.preprocess.smpl.run``.

    Args:
        dataset: Dataset gốc trong manifest, ví dụ ``nuscenes``.
        view_root: Thư mục chứa cảnh ảo, truyền làm ``--data_root``.

    Returns:
        Câu lệnh dạng danh sách.

    Raises:
        ValueError: Nếu dataset chưa có bước chiếu box người của DriveStudio.
    """
    if dataset not in DRIVESTUDIO_DATASETS:
        raise ValueError(f"prep.smpl chưa hỗ trợ dataset {dataset}")
    return [
        sys.executable,
        "-m",
        "drive2sim.recon.preprocess.smpl.run",
        "--data_root",
        str(view_root),
        "--dataset",
        DRIVESTUDIO_DATASETS[dataset],
        "--scene_ids",
        str(int(VIEW_SCENE)),
    ]


@register_stage("prep.smpl")
class PrepSmplStage(Stage):
    """Sinh SMPL cho người đi bộ trong một cảnh, ghi vào cache."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = True
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"scene": "scene"}
    outputs: ClassVar[dict[str, str]] = {"smpl": "smpl"}
    Params: ClassVar[type[BaseModel]] = NoParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Dựng cảnh ảo, chạy quy trình SMPL, chuyển ``humanpose/`` ra thư mục của stage.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: Không có đầu vào artifact.
            params: Không có tham số.

        Returns:
            ``{"smpl": <thư mục stage>}``, chứa ``humanpose/smpl.pkl``.

        Raises:
            RuntimeError: Nếu quy trình lỗi hoặc không ghi ra ``humanpose/smpl.pkl``.
        """
        view_root = ctx.output_dir / VIEW_DIR
        view = build_view(ctx.scene.root, view_root, exclude=[OUTPUT_NAME])
        ctx.progress(0.0, "sinh SMPL bằng 4D-Humans")
        command = build_command(ctx.scene.manifest.source.dataset, view_root)
        run_logged(command, ctx.output_dir / LOG_FILE, should_stop=ctx.should_stop)
        collect_outputs(view, ctx.output_dir, [OUTPUT_NAME])
        shutil.rmtree(view_root)
        if not (ctx.output_dir / OUTPUT_NAME / "smpl.pkl").is_file():
            raise RuntimeError("smpl/run.py xong nhưng không có humanpose/smpl.pkl")
        ctx.progress(1.0, "sinh SMPL xong")
        return {"smpl": ctx.output_dir}
