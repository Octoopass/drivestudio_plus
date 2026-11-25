"""Stage ``eval.ply``: đo export gap của ``.ply`` nền tĩnh (yêu cầu R3 của benchmark).

Export gap là PSNR của render gốc trừ PSNR của render từ file ``.ply`` xuất ra, trên **cùng** mask
nền tĩnh (điểm ảnh hợp lệ, không phải trời, ngoài hình chiếu box vật động). Nó đo phần chất lượng
mất đi khi đưa model ra định dạng 3DGS chuẩn, để chọn method mà ``.ply`` dùng được ở simulator.

Stage chạy hai lệnh trong tiến trình con:
1. ``drive2sim.recon.tools.export_ply --rigid_frame -1``: xuất ``.ply`` chỉ có nền tĩnh vào
   ``<stage>/background/scene.ply`` (riêng cho stage, nên không phụ thuộc cách ``export.ply_world``
   được đặt tham số);
2. ``drive2sim.recon.tools.eval_ply``: nạp lại dataset của run, vẽ ``.ply`` bằng renderer 3DGS
   chung ở các ảnh test của camera đã train, và ghi ``ply_gap.json``.

PSNR của render gốc lấy từ ``psnr_static`` trong ``frames.jsonl`` của ``eval.standard`` (protocol
0.5), nên stage cần đầu vào ``report``; ``frames.jsonl`` do protocol cũ ghi không có khoá này và bị
từ chối trước khi dùng GPU. ``d2s benchmark`` đọc ``export_gap_db`` trong ``ply_gap.json``
(CONTRACTS mục 6.5).

File này chứa:
- ``check_frames``: kiểm ``frames.jsonl`` đã có ``psnr_static``.
- ``build_eval_command``: dựng câu lệnh ``eval_ply``; tách riêng để test được trên máy không GPU.
- ``EvalPlyStage``: stage đăng ký với engine.

Chưa chạy thật: cần checkpoint, GPU và gsplat.
"""

import json
import os
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.paths import SMPL_MODEL_ENV
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.eval.standard import FRAMES_FILE
from drive2sim.stages.export.ply_world import PLY_FILE, ExportPlyWorldParams
from drive2sim.stages.export.ply_world import build_command as build_export_command
from drive2sim.stages.scene_view import build_view
from drive2sim.stages.train.drivestudio import (
    MASK_ENTRIES,
    SMPL_ENTRY,
    VIEW_DIR,
    smpl_model_for,
    view_extras,
)

PLY_GAP_FILE = "ply_gap.json"
BACKGROUND_DIR = "background"


def check_frames(frames: Path) -> None:
    """Kiểm ``frames.jsonl`` của ``eval.standard`` đã có metric nền tĩnh.

    Chỉ đọc dòng đầu: mọi dòng của một file do cùng một phiên bản code đo ghi.

    Args:
        frames: Đường dẫn ``frames.jsonl``.

    Raises:
        ValueError: Nếu không có file, file rỗng, hoặc không có khoá ``psnr_static`` (file do
            protocol dưới 0.5 ghi; cần chạy lại ``eval.standard`` để đo lại từ checkpoint).
    """
    if not frames.is_file():
        raise ValueError(
            f"không có {FRAMES_FILE} của eval.standard ({frames}); chạy eval.standard"
        )
    with frames.open(encoding="utf-8") as handle:
        first = handle.readline().strip()
    if not first or "psnr_static" not in json.loads(first):
        raise ValueError(
            f"{FRAMES_FILE} chưa có psnr_static (đo bằng protocol dưới 0.5): chạy lại "
            "eval.standard để đo lại từ checkpoint bằng protocol 0.5, rồi chạy eval.ply"
        )


def build_eval_command(
    model_dir: Path, ply: Path, frames: Path, output_dir: Path, view_root: Path
) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.tools.eval_ply``.

    Args:
        model_dir: Thư mục kết quả của stage train (``checkpoint_final.pth``, ``config.yaml``).
        ply: ``.ply`` nền tĩnh cần đo.
        frames: ``frames.jsonl`` của ``eval.standard``, nguồn PSNR nền tĩnh của render gốc.
        output_dir: Thư mục ghi ``ply_gap.json``.
        view_root: Thư mục cảnh ảo (cảnh cộng mask và SMPL), truyền làm ``data.data_root``.

    Returns:
        Câu lệnh dạng danh sách.
    """
    return [
        sys.executable,
        "-m",
        "drive2sim.recon.tools.eval_ply",
        "--model_dir",
        str(model_dir),
        "--ply",
        str(ply),
        "--frames",
        str(frames),
        "--out",
        str(output_dir),
        f"data.data_root={view_root}",
    ]


@register_stage("eval.ply")
class EvalPlyStage(Stage):
    """Đo export gap của ``.ply`` nền tĩnh so với render gốc của model."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {
        "model": "model",
        "report": "eval_report",
        "split": "split",
        "masks": "masks",
        "smpl": "smpl",
    }
    optional_inputs: ClassVar[frozenset[str]] = frozenset({"masks", "smpl"})
    # ``ply_gap.json`` nằm trong thư mục stage; dùng cùng loại artifact với eval.standard vì chưa
    # có stage nào đọc nó như đầu vào (d2s benchmark đọc thẳng file).
    outputs: ClassVar[dict[str, str]] = {"gap": "eval_report"}
    Params: ClassVar[type[BaseModel]] = NoParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Xuất ``.ply`` chỉ nền, rồi đo export gap bằng ``eval_ply.py``.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: ``{"model": <thư mục train>, "report": <thư mục eval.standard>}`` và tuỳ chọn
                ``masks``, ``smpl``.
            params: Không có tham số.

        Returns:
            ``{"gap": <thư mục stage>}``, chứa ``ply_gap.json`` và ``background/scene.ply``.

        Raises:
            ValueError: Nếu ``frames.jsonl`` chưa có ``psnr_static``.
            RuntimeError: Nếu một lệnh lỗi hoặc không ra file kết quả.
        """
        frames = inputs["report"] / FRAMES_FILE
        check_frames(frames)  # trước mọi việc tốn GPU
        log_path = ctx.output_dir / LOG_FILE
        background = ctx.output_dir / BACKGROUND_DIR
        ctx.progress(0.0, "xuất .ply chỉ có nền tĩnh")
        run_logged(
            build_export_command(
                ExportPlyWorldParams(rigid_frame=-1), inputs["model"], background
            ),
            log_path,
            should_stop=ctx.should_stop,
        )
        ply = background / PLY_FILE
        if not ply.is_file():
            raise RuntimeError(f"export_ply.py xong nhưng không có {PLY_FILE}")
        # Cảnh ảo dựng lại như lúc đo ở eval.standard: cảnh, mask trời và vật động, SMPL từ cache.
        extra = view_extras(inputs)
        view_root = ctx.output_dir / VIEW_DIR
        build_view(ctx.scene.root, view_root, exclude=(*MASK_ENTRIES, SMPL_ENTRY), extra=extra)
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
        }
        if SMPL_ENTRY in extra:
            env[SMPL_MODEL_ENV] = str(smpl_model_for(ctx.output_dir))
        ctx.progress(0.3, "vẽ .ply ở các ảnh test và tính export gap")
        run_logged(
            build_eval_command(inputs["model"], ply, frames, ctx.output_dir, view_root),
            log_path,
            should_stop=ctx.should_stop,
            env=env,
        )
        if not (ctx.output_dir / PLY_GAP_FILE).is_file():
            raise RuntimeError(f"eval_ply.py xong nhưng không có {PLY_GAP_FILE}")
        return {"gap": ctx.output_dir}
