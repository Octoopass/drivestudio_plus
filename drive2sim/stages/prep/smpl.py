"""Stage ``prep.smpl``: sinh tư thế người (SMPL) theo khung cho các người đi bộ trong cảnh.

Stage này bọc quy trình có sẵn ``drive2sim/recon/preprocess/smpl/run.py`` (hướng dẫn
``docs/datasets/human_pose.md``): chiếu box người lên ảnh, chạy 4D-Humans để ước lượng SMPL, ghép
kết quả với box, rồi ghi ``humanpose/smpl.pkl`` (cùng các file trung gian trong
``humanpose/temp/``, ``humanpose/postprocess/``). OmniRe cần file này cho ``SMPLNodes``; Street
Gaussians không cần.

Kết quả là artifact loại ``smpl`` trong ``cache/`` (stage có cache). Script chạy trên một thư mục
cảnh ảo dựng bằng symlink (``drive2sim/stages/scene_view.py``), nên dữ liệu chuẩn không bị sửa.

File này chứa ``pinhole_cameras``, ``build_command``, ``link_smpl_model`` và ``PrepSmplStage``.

Cần GPU, gói ``hmr2`` (``third_party/Humans4D``) và ``phalp`` (``envs/install_main.sh`` bước 8),
file ``SMPL_NEUTRAL.pkl`` (``drive2sim.paths.find_smpl_model``) và cảnh có box 3D.

Đoạn khung của split không có box người (``available.human_frames`` của manifest,
``drive2sim/core/instances.py``) thì ``unavailable_reason`` khoá stage: kết quả sẽ không được
dùng vì trainer chỉ nạp các khung của split. Mục cấu hình ``smpl: auto`` của OmniRe khi đó chọn
``none`` và train không có node người SMPL.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel

from drive2sim.core.instances import no_humans_reason
from drive2sim.core.schema import Manifest, Split
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
# Dataset có bước chiếu box người (nhánh --dataset của recon/preprocess/smpl/run.py): mọi dataset
# của DriveStudio, cộng KITTI-360 (dùng lại nhánh KITTI, recon/preprocess/kitti360/human.py).
SMPL_DATASETS = set(DRIVESTUDIO_DATASETS)
# Dataset theo khung KITTI: lớp người trong instances_info.json là HUMAN_CLASSES, nên đếm được
# người để bỏ qua 4D-Humans khi cảnh không có ai.
KITTI_LAYOUT_DATASETS = frozenset({"kitti", "kitti360", "phenikaa"})
# Dataset mà số và thứ tự camera khác nhau giữa các cảnh: smpl/run.py nhận chỉ số camera pinhole
# của cảnh qua --cameras thay vì CAMERA_LIST cố định.
CAMERAS_FROM_MANIFEST = frozenset({"phenikaa"})
VIEW_DIR = "view"
# Chỗ PHALP và 4D-Humans tìm SMPL, tính từ ~/.cache. Thiếu file thì PHALP tự wget một link đã chết
# (github.com/classner/up, 404) rồi bỏ qua cảnh, script vẫn thoát 0.
SMPL_CACHE_PATHS = (
    Path("phalp/3D/models/smpl/SMPL_NEUTRAL.pkl"),
    Path("4DHumans/data/smpl/SMPL_NEUTRAL.pkl"),
)


# Lớp người trong instances_info.json của khung KITTI (loader KITTI, SMPLNODE_CLASSES).
HUMAN_CLASSES = frozenset({"Pedestrian", "Person_sitting"})


def scene_humans(scene_root: Path) -> int:
    """Số người (lớp ``HUMAN_CLASSES``) trong ``instances/instances_info.json`` của cảnh."""
    path = scene_root / "instances" / "instances_info.json"
    if not path.is_file():
        return 0
    info = json.loads(path.read_text("utf-8"))
    return sum(value.get("class_name") in HUMAN_CLASSES for value in info.values())


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


def pinhole_cameras(manifest: Manifest) -> list[int]:
    """Chỉ số các camera pinhole, theo thứ tự trong manifest (tên file ``<khung>_<chỉ số>``)."""
    return [i for i, camera in enumerate(manifest.cameras) if camera.model == "opencv_pinhole"]


def build_command(dataset: str, view_root: Path, cameras: Optional[list[int]] = None) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.preprocess.smpl.run``.

    Args:
        dataset: Dataset gốc trong manifest, ví dụ ``nuscenes``.
        view_root: Thư mục chứa cảnh ảo, truyền làm ``--data_root``.
        cameras: Chỉ số camera cần dựng SMPL (``--cameras``); bắt buộc với dataset trong
            ``CAMERAS_FROM_MANIFEST``, ``None`` thì script dùng ``CAMERA_LIST`` của dataset.

    Returns:
        Câu lệnh dạng danh sách.

    Raises:
        ValueError: Nếu dataset chưa có bước chiếu box người của DriveStudio, hoặc cần
            ``cameras`` mà không có.
    """
    if dataset not in SMPL_DATASETS:
        raise ValueError(f"prep.smpl chưa hỗ trợ dataset {dataset}")
    if dataset in CAMERAS_FROM_MANIFEST and not cameras:
        raise ValueError(f"prep.smpl với dataset {dataset} cần chỉ số camera pinhole")
    extra = ["--cameras", *(str(i) for i in cameras)] if cameras else []
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
        *extra,
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

    @classmethod
    def unavailable_reason(
        cls, manifest: Manifest, split: Optional[Split] = None
    ) -> Optional[str]:
        """Khoá khi biết chắc đoạn khung của split không có người (``no_humans_reason``)."""
        return no_humans_reason(manifest, split)

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
        dataset = ctx.scene.manifest.source.dataset
        if dataset in KITTI_LAYOUT_DATASETS and not scene_humans(ctx.scene.root):
            # Cảnh không có người: không có gì để dựng, khỏi chạy 4D-Humans (tốn GPU). Trả thư mục
            # rỗng; train.drivestudio thấy thiếu smpl.pkl thì tắt load_smpl.
            (ctx.output_dir / LOG_FILE).write_text(
                f"cảnh {dataset} không có người: bỏ qua bước dựng SMPL\n", "utf-8"
            )
            ctx.progress(1.0, "cảnh không có người, bỏ qua SMPL")
            return {"smpl": ctx.output_dir}
        cameras = pinhole_cameras(ctx.scene.manifest) if dataset in CAMERAS_FROM_MANIFEST else None
        if cameras == []:
            # 4D-Humans và phép chiếu box chỉ dùng camera pinhole: cảnh chỉ có mắt cá thì người
            # được dựng bằng node biến dạng, như người chỉ hiện ở camera mắt cá của KITTI-360.
            (ctx.output_dir / LOG_FILE).write_text(
                f"cảnh {dataset} không có camera pinhole: bỏ qua bước dựng SMPL\n", "utf-8"
            )
            ctx.progress(1.0, "cảnh không có camera pinhole, bỏ qua SMPL")
            return {"smpl": ctx.output_dir}
        # Trên Kaggle cảnh nằm dưới /kaggle/input; find_smpl_model tìm cả trong các dataset đã gắn.
        model = find_smpl_model(ctx.workdir)
        if model is None:
            raise RuntimeError(
                "prep.smpl cần SMPL_NEUTRAL.pkl: đặt ở <thư mục làm việc>/smpl_models/ hoặc trỏ "
                "biến D2S_SMPL_MODEL (Kaggle: biến D2S_KAGGLE_SMPL_DATASET khi đẩy)"
            )
        link_smpl_model(model, Path(os.environ.get("HOME", "~")).expanduser() / ".cache")
        view_root = ctx.output_dir / VIEW_DIR
        view = build_view(ctx.scene.root, view_root, exclude=[OUTPUT_NAME])
        ctx.progress(0.0, "sinh SMPL bằng 4D-Humans")
        command = build_command(dataset, view_root, cameras)
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
