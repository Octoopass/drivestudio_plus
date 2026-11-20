"""Stage ``prep.ncore``: chuyển đoạn khung của split sang một clip NCore V4 cho Instant NuRec.

Instant NuRec đọc clip NCore (NVIDIA), không đọc thư mục cảnh của DriveStudio. Converter đã kiểm
ở Fisheye Fusion Lab (``drive2sim/external/nurec/tools/kitti360_to_ncore.py``) đọc thẳng dữ liệu
KITTI-360 gốc (bộ có ``ImageSets-rosa-6class``, ``data_poses.zip``, ``data_2d_raw/*.tar``,
``data_3d_raw/*.tar``) và chạy bằng Python của môi trường NuRec (có ``nvidia-ncore``). Nó làm
phần hình học khó: camera mắt cá Mei → F-theta 120° của NVIDIA (``rectify120``, cái mô hình
multiview đã học) hoặc F-theta đầy đủ, mask thân xe, LiDAR. Stage lấy chuỗi từ manifest, đoạn
khung và camera từ split; thư mục gốc từ ``raw_source.json`` mà ingest KITTI-360 ghi, hoặc biến
``D2S_KITTI360_ROOT``.

Hiện chỉ có đường KITTI-360 (như bản Fisheye Fusion Lab); dataset khác bị khoá kèm lý do.

File này chứa:
- ``PrepNcoreParams``: tham số.
- ``raw_kitti360_dir``: thư mục KITTI-360 gốc của một cảnh.
- ``build_command``: câu lệnh converter (tách riêng để test không cần NuRec).
- ``PrepNcoreStage``: stage đăng ký với engine; kết quả là ``clip.json`` trỏ tới file NCore.
"""

import json
import os
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict

from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged

RAW_SOURCE_FILE = "raw_source.json"
RAW_ROOT_ENV = "D2S_KITTI360_ROOT"
CLIP_FILE = "clip.json"
ROSA_META = "ImageSets-rosa-6class"


class PrepNcoreParams(BaseModel):
    """Tham số của ``prep.ncore``.

    Attributes:
        fisheye_model: ``rectify120``: camera mắt cá resample về camera F-theta 120° của NVIDIA
            (mô hình ``pa-multiview`` học trên loại này); ``ftheta``: giữ cả vòng mắt cá.
        lidar: Ghi LiDAR vào clip (cảnh không có LiDAR thì tự bỏ).
    """

    model_config = ConfigDict(extra="forbid")

    fisheye_model: Literal["rectify120", "ftheta"] = "rectify120"
    lidar: bool = True


def raw_kitti360_dir(scene_root: Path) -> Path:
    """Thư mục KITTI-360 gốc của cảnh: ``raw_source.json`` của cảnh, rồi ``D2S_KITTI360_ROOT``.

    Raises:
        RuntimeError: Nếu không biết thư mục gốc, hoặc thư mục thiếu metadata ROSA.
    """
    candidates = []
    record = scene_root / RAW_SOURCE_FILE
    if record.is_file():
        candidates.append(Path(json.loads(record.read_text(encoding="utf-8"))["raw_dir"]))
    if os.environ.get(RAW_ROOT_ENV):
        candidates.append(Path(os.environ[RAW_ROOT_ENV]).expanduser())
    for path in candidates:
        if (path / ROSA_META).is_dir() and (path / "data_poses.zip").is_file():
            return path
    tried = ", ".join(str(p) for p in candidates) or "chưa có"
    raise RuntimeError(
        "không tìm thấy dữ liệu KITTI-360 gốc (có ImageSets-rosa-6class/, data_poses.zip) mà "
        f"converter NCore cần; đã thử: {tried}. Ingest lại cảnh (ghi {RAW_SOURCE_FILE}) hoặc "
        f"đặt {RAW_ROOT_ENV}"
    )


def build_command(
    raw_dir: Path,
    out_dir: Path,
    sequence: str,
    start: int,
    frames: int,
    cameras: list[str],
    params: PrepNcoreParams,
    sequence_id: str,
) -> list[str]:
    """Câu lệnh ``kitti360_to_ncore.py`` bằng Python của môi trường NuRec."""
    command = [
        str(nurec.nurec_python()), "-u", str(nurec.TOOLS_DIR / "kitti360_to_ncore.py"),
        "--data", str(raw_dir), "--out", str(out_dir), "--sequence", sequence,
        "--start", str(start), "--frames", str(frames), "--cameras", ",".join(cameras),
        "--sequence-id", sequence_id, "--fisheye-model", params.fisheye_model,
    ]  # fmt: skip
    if not params.lidar:
        command.append("--no-lidar")
    return command


@register_stage("prep.ncore")
class PrepNcoreStage(Stage):
    """Chuyển đoạn khung của split sang clip NCore V4 (Instant NuRec đọc)."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = True
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"scene": "scene", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"clip": "ncore_clip"}
    Params: ClassVar[type[BaseModel]] = PrepNcoreParams

    @classmethod
    def unavailable_reason(
        cls, manifest: Manifest, split: Optional[Split] = None
    ) -> Optional[str]:
        """Chỉ cảnh KITTI-360 có đường chuyển sang NCore."""
        if manifest.source.dataset != "kitti360":
            return "chuyển sang NCore cho Instant NuRec hiện chỉ có với cảnh KITTI-360"
        if split is not None and not set(split.cameras) & set(nurec.SENSORS):
            return "split không có camera KITTI-360 nào (image_00..image_03)"
        return None

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy converter; ghi ``clip.json`` (file NCore, khung, camera) vào thư mục stage.

        Returns:
            ``{"clip": <thư mục stage>}``.

        Raises:
            RuntimeError: Thiếu dữ liệu gốc, môi trường NuRec, hoặc converter không ghi clip.
        """
        assert isinstance(params, PrepNcoreParams)
        python = nurec.nurec_python()
        if not python.is_file():
            raise RuntimeError(
                f"chưa cài Instant NuRec ({python} không có): bash envs/install_nurec.sh hoặc "
                "nút cài trên trang Chạy mới"
            )
        raw = raw_kitti360_dir(ctx.scene.root)
        frame_ids = [
            int(line)
            for line in (ctx.scene.root / "frame_ids.txt").read_text().split()
            if line.strip()
        ]
        first, last = ctx.split.frame_range
        wanted = frame_ids[first : last + 1]
        cameras = [c for c in ctx.split.cameras if c in nurec.SENSORS]
        manifest = ctx.scene.manifest
        lidar = params.lidar and bool(manifest.lidars)
        options = params.model_copy(update={"lidar": lidar})
        sequence_id = f"{ctx.scene.scene_id}_{first}_{last}".replace("-", "_")
        ctx.progress(0.0, f"chuyển {len(wanted)} khung, {len(cameras)} camera sang NCore")
        run_logged(
            build_command(
                raw, ctx.output_dir, manifest.source.sequence_id, wanted[0], len(wanted),
                cameras, options, sequence_id,
            ),
            ctx.output_dir / LOG_FILE,
            should_stop=ctx.should_stop,
        )  # fmt: skip
        clip = ctx.output_dir / f"{sequence_id}.json"
        if not clip.is_file():
            raise RuntimeError(f"converter xong nhưng không có {clip.name}")
        meta = json.loads(clip.read_text(encoding="utf-8"))
        converted = _frame_ids(meta)
        if converted and converted[0] != wanted[0]:
            raise RuntimeError(
                f"clip NCore bắt đầu ở khung {converted[0]}, split bắt đầu ở {wanted[0]}: hệ toạ "
                "độ sẽ lệch; chọn split có đoạn khung liền nhau"
            )
        (ctx.output_dir / CLIP_FILE).write_text(
            json.dumps(
                {
                    "clip": clip.name,
                    "frame_ids": converted or wanted,
                    "scene_frame_ids": wanted,
                    "scene_frames": [first, last],
                    "sensors": {c: nurec.SENSORS[c] for c in cameras},
                    "fisheye_model": params.fisheye_model,
                    "lidar": lidar,
                },
                indent=2,
            )
            + "\n"
        )
        return {"clip": ctx.output_dir}


def _frame_ids(meta: dict) -> list[int]:
    """Số khung gốc converter ghi trong metadata của clip (``generic_meta_data.frame_ids``)."""
    found: list[int] = []

    def walk(value: object) -> None:
        if found:
            return
        if isinstance(value, dict):
            ids = value.get("frame_ids")
            if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
                found.extend(ids)
                return
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(meta)
    return found
