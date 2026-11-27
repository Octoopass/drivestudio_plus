"""Xuất gói quan sát từ đường dữ liệu DriveStudio cho approach chạy ở môi trường khác.

``PrepObservationsStage`` chạy ``drive2sim.recon.tools.export_observations`` trong môi trường
chính với đúng câu lệnh ``train.drivestudio`` dựng cho cùng cảnh, split, method và renderer
(``build_command``), nên gói chứa đúng ảnh, mask, pose, LiDAR, box và SMPL mà OmniRe nhận và được
chấm điểm trên đó. Approach XSIM đọc gói này thay cho thư mục cảnh gốc (CONTRACTS mục 5.4, loại
``observations``).
"""

import json
import shutil
import sys
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.core.schema import CAMERA_TRACK, Manifest, Split, parse_strategy
from drive2sim.engine.cache import input_key_for_split
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external.xsim import OBSERVATIONS_INDEX
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.scene_view import build_view
from drive2sim.stages.train.drivestudio import (
    MASK_ENTRIES,
    SMPL_ENTRY,
    VIEW_DIR,
    Method,
    Renderer,
    TrainDrivestudioParams,
    build_command,
    has_dynamic_masks,
    view_extras,
    write_method_config,
)

# Tham số của train.py không thuộc phần dữ liệu: bỏ khi chuyển câu lệnh sang công cụ xuất.
_TRAIN_ONLY_FLAGS = ("--output_root", "--project", "--run_name", "--resume_from")


class PrepObservationsParams(BaseModel):
    """Chọn data config DriveStudio mà gói quan sát phải giống.

    Attributes:
        method: Method DriveStudio có phần ``data`` cần giống (mặc định OmniRe).
        renderer: Renderer của approach so sánh; với camera mắt cá, 3DGUT và 3DGEER che pixel
            từ 88 độ (``RENDERER_MAX_ANGLE``), nên gói phải che như vậy mới chấm cùng pixel.
            Bỏ trống: ``3dgut`` khi split có camera mắt cá, ``3dgs`` khi chỉ có pinhole.
        downscale_when_loading: Hệ số thu ảnh khi nạp cho mọi camera; ``dataset`` thì theo config
            dataset (ví dụ KITTI-360 là 2), như OmniRe.
        dataset: Config dataset DriveStudio, như ``train.drivestudio``.
        fisheye_max_angle: Như ``train.drivestudio``.
        overrides: Khoá ``data.*`` khác cần đè.
    """

    model_config = ConfigDict(extra="forbid")
    method: Method = "omnire"
    renderer: Optional[Renderer] = None
    downscale_when_loading: Union[Literal["dataset"], Literal[1, 2, 4]] = "dataset"
    dataset: Optional[str] = None
    fisheye_max_angle: Optional[float] = Field(default=None, gt=0)
    overrides: dict[str, object] = {}


def train_params(
    params: PrepObservationsParams, split: Split, manifest: Manifest
) -> TrainDrivestudioParams:
    """Tham số ``train.drivestudio`` có cùng phần dữ liệu với gói quan sát."""
    renderer = params.renderer or (
        "3dgut"
        if any(CAMERA_TRACK[manifest.camera(c).model] == "fisheye" for c in split.cameras)
        else "3dgs"
    )
    overrides = dict(params.overrides)
    if params.downscale_when_loading != "dataset":
        overrides["data.pixel_source.downscale_when_loading"] = [
            params.downscale_when_loading
        ] * len(split.cameras)
    return TrainDrivestudioParams(
        method=params.method,
        renderer=renderer,
        dataset=params.dataset,
        fisheye_max_angle=params.fisheye_max_angle,
        overrides=overrides,
        eval_at=[],
    )


def export_command(
    params: PrepObservationsParams,
    split: Split,
    manifest: Manifest,
    view_root: Path,
    output_dir: Path,
    has_sky_masks: bool,
    has_smpl: bool,
    config_file: Path,
    index_file: Path,
    scene_dir: Path,
) -> list[str]:
    """Câu lệnh xuất gói: câu lệnh ``train.drivestudio`` đổi module, bỏ cờ chỉ dùng khi train."""
    train = train_params(params, split, manifest)
    command = build_command(
        train, split, manifest, view_root, output_dir, has_sky_masks, has_smpl, config_file
    )
    module = command.index("drive2sim.recon.tools.train")
    command[module] = "drive2sim.recon.tools.export_observations"
    kept: list[str] = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        if item in _TRAIN_ONLY_FLAGS:
            skip = True
            continue
        if item.startswith(("trainer.", "logging.")):
            continue
        kept.append(item)
    position = kept.index("--config_file") + 2
    extra = [
        "--out", str(output_dir), "--scene_dir", str(scene_dir), "--index_json", str(index_file),
    ]  # fmt: skip
    return kept[:position] + extra + kept[position:]


@register_stage("prep.observations")
class PrepObservationsStage(Stage):
    """Dựng ``DrivingDataset`` như trainer DriveStudio rồi ghi quan sát của nó ra file."""

    version: ClassVar[str] = "0.2.0"
    cacheable: ClassVar[bool] = True
    uses_lidar: ClassVar[bool] = True
    split_content_key: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {
        "scene": "scene",
        "split": "split",
        "masks": "masks",
        "smpl": "smpl",
    }
    optional_inputs: ClassVar[frozenset[str]] = frozenset({"masks", "smpl"})
    outputs: ClassVar[dict[str, str]] = {"observations": "observations"}
    Params: ClassVar[type[BaseModel]] = PrepObservationsParams

    @classmethod
    def unavailable_reason(
        cls, manifest: Manifest, split: Optional[Split] = None
    ) -> Optional[str]:
        """Gói đi qua loader DriveStudio: cần split theo bước khung như ``train.drivestudio``."""
        if split is not None:
            if parse_strategy(split.strategy)[0] != "every_nth_frame":
                return "gói quan sát dùng split every_nth_frame như train.drivestudio"
            if not split.train or not split.test:
                return "cần cả ảnh train và ảnh test"
        return None

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy công cụ xuất trong môi trường chính và đối chiếu split của loader với run.

        Raises:
            ValueError: Nếu khung test loader chọn khác các cặp test của split.
        """
        assert isinstance(params, PrepObservationsParams)
        scene, split = ctx.scene, ctx.split
        extra = view_extras(inputs)
        overrides = dict(params.overrides)
        if not has_dynamic_masks(scene.root, extra):
            overrides.setdefault("data.pixel_source.load_dynamic_mask", False)
        if not (scene.root / "instances" / "instances_info.json").is_file():
            overrides.setdefault("data.pixel_source.load_objects", False)
        params = params.model_copy(update={"overrides": overrides})
        work = ctx.output_dir / ".work"
        view_root = work / VIEW_DIR
        build_view(scene.root, view_root, exclude=(*MASK_ENTRIES, SMPL_ENTRY), extra=extra)
        index = {
            "scene_id": scene.scene_id,
            "converter_version": scene.manifest.source.converter_version,
            "split_key": input_key_for_split(split.split_id, split),
            "split": split.model_dump(mode="json"),
            "camera_ids": [camera.camera_id for camera in scene.manifest.cameras],
            "camera_indices": {
                c.camera_id: i
                for i, c in enumerate(scene.manifest.cameras)
                if c.camera_id in split.cameras
            },
            "method": params.method,
        }
        index_file = work / "index.json"
        index_file.write_text(json.dumps(index), encoding="utf-8")
        # Method config của đúng tổ hợp method, renderer: phần data phải như run DriveStudio đó.
        method = train_params(params, split, scene.manifest)
        command = export_command(
            params,
            split,
            scene.manifest,
            view_root,
            ctx.output_dir,
            has_sky_masks="sky_masks" in extra,
            has_smpl=SMPL_ENTRY in extra,
            config_file=write_method_config(method, work),
            index_file=index_file,
            scene_dir=scene.root,
        )
        command[0] = sys.executable
        ctx.progress(0.0, "xuất quan sát từ loader DriveStudio")
        run_logged(command, ctx.output_dir / LOG_FILE, should_stop=ctx.should_stop)
        written = json.loads((ctx.output_dir / OBSERVATIONS_INDEX).read_text("utf-8"))
        expected = sorted(
            {frame for frame, camera in split.test if camera not in split.test_only_cameras}
        )
        if written["loader_test_frames"] != expected:
            raise ValueError(
                f"loader chọn khung test {written['loader_test_frames']}, split có {expected}"
            )
        # Cảnh ảo là symlink tới thư mục cảnh: không để lại trong cache (CONTRACTS mục 5.2).
        shutil.rmtree(work, ignore_errors=True)
        ctx.progress(1.0, "đã xuất quan sát")
        return {"observations": ctx.output_dir}
