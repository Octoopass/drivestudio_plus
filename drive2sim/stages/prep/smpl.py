"""Stage ``prep.smpl``: sinh tư thế người (SMPL) theo khung cho các người đi bộ trong cảnh.

Stage này bọc quy trình có sẵn ``drive2sim/recon/preprocess/smpl/run.py`` (hướng dẫn
``docs/datasets/human_pose.md``): chiếu box người lên ảnh, chạy 4D-Humans để ước lượng SMPL, ghép
kết quả với box, rồi ghi ``humanpose/smpl.pkl`` (cùng các file trung gian trong
``humanpose/temp/``, ``humanpose/postprocess/``). OmniRe cần file này cho ``SMPLNodes``; Street
Gaussians không cần.

Kết quả là artifact loại ``smpl`` trong ``cache/`` (stage có cache). Script chạy trên một thư mục
cảnh ảo dựng bằng symlink (``drive2sim/stages/scene_view.py``), nên dữ liệu chuẩn không bị sửa.

File này chứa ``build_command``, ``link_smpl_model`` và ``PrepSmplStage``.

Cần GPU, gói ``hmr2`` (``third_party/Humans4D``) và ``phalp`` (``envs/install_main.sh`` bước 8),
file ``SMPL_NEUTRAL.pkl`` (``drive2sim.paths.find_smpl_model``) và cảnh có box 3D.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.paths import find_smpl_model
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.scene_view import (
    DRIVESTUDIO_DATASETS,
    VIEW_SCENE,
    build_view,
    collect_outputs,
)

OUTPUT_NAME = "humanpose"
VIEW_DIR = "view"
# Chỗ PHALP và 4D-Humans tìm SMPL, tính từ ~/.cache. Thiếu file thì PHALP tự wget một link đã chết
# (github.com/classner/up, 404) rồi bỏ qua cảnh, script vẫn thoát 0.
SMPL_CACHE_PATHS = (
    Path("phalp/3D/models/smpl/SMPL_NEUTRAL.pkl"),
    Path("4DHumans/data/smpl/SMPL_NEUTRAL.pkl"),
)


def link_smpl_model(model: Path, cache_dir: Path) -> list[Path]:
    """Trỏ các chỗ PHALP và 4D-Humans tìm SMPL tới file của người dùng (symlink).

    File đã có ở đó (người dùng tự đặt) thì giữ nguyên.

    Args:
        model: File ``SMPL_NEUTRAL.pkl`` tìm được.
        cache_dir: Thư mục cache của hai thư viện (``~/.cache``).

    Returns:
        Các đường dẫn vừa tạo.
    """
    created = []
    for relative in SMPL_CACHE_PATHS:
        target = cache_dir / relative
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()  # symlink hỏng
        target.symlink_to(model)
        created.append(target)
    return created


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
        # Cảnh nằm ở <thư mục làm việc>/data/<cảnh>/ (trên Kaggle: dưới /kaggle/input, khi đó
        # find_smpl_model tìm trong các dataset đã gắn).
        model = find_smpl_model(ctx.scene.root.parent.parent)
        if model is None:
            raise RuntimeError(
                "prep.smpl cần SMPL_NEUTRAL.pkl: đặt ở <thư mục làm việc>/smpl_models/ hoặc trỏ "
                "biến D2S_SMPL_MODEL (Kaggle: biến D2S_KAGGLE_SMPL_DATASET khi đẩy)"
            )
        link_smpl_model(model, Path(os.environ.get("HOME", "~")).expanduser() / ".cache")
        view_root = ctx.output_dir / VIEW_DIR
        view = build_view(ctx.scene.root, view_root, exclude=[OUTPUT_NAME])
        ctx.progress(0.0, "sinh SMPL bằng 4D-Humans")
        command = build_command(ctx.scene.manifest.source.dataset, view_root)
        run_logged(command, ctx.output_dir / LOG_FILE, should_stop=ctx.should_stop)
        collect_outputs(view, ctx.output_dir, [OUTPUT_NAME])
        shutil.rmtree(view_root)
        if not (ctx.output_dir / OUTPUT_NAME / "smpl.pkl").is_file():
            # smpl/run.py bắt lỗi từng cảnh rồi vẫn thoát 0; lỗi thật nằm trong log.
            lines = (ctx.output_dir / LOG_FILE).read_text("utf-8", errors="replace").splitlines()
            errors = [line for line in lines if "ERROR" in line or "Error" in line][-5:]
            raise RuntimeError(
                "smpl/run.py xong nhưng không có humanpose/smpl.pkl; lỗi trong log.txt:\n"
                + "\n".join(errors or lines[-20:])
            )
        ctx.progress(1.0, "sinh SMPL xong")
        return {"smpl": ctx.output_dir}
