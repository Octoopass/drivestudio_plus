"""Chuẩn bị chỉ mục cảnh KITTI-360 cho worker XSIM theo CONTRACTS mục 5.4.

PrepXsimStage nối chỉ số ảnh của split với camera trong dữ liệu ingest và với thời điểm
của khung. Worker dùng chỉ mục trong cache để mở đúng ảnh, calibration và sweep.
"""

import json
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel

from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.cache import input_key_for_split
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext


@register_stage("prep.xsim")
class PrepXsimStage(Stage):
    """Ghi chỉ mục camera, khung, thời gian và tập ảnh của run cho worker XSIM."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = True
    uses_lidar: ClassVar[bool] = True
    split_content_key: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {"scene": "scene", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"clip": "xsim_scene"}

    @classmethod
    def unavailable_reason(
        cls, manifest: Manifest, split: Optional[Split] = None
    ) -> Optional[str]:
        """Kiểm tra cảnh và các camera thực sự được chọn cho baseline KITTI-360."""
        if manifest.source.dataset != "kitti360":
            return "provider XSIM hiện cần cảnh KITTI-360"
        if split is not None:
            if not split.train or not split.test:
                return "XSIM L1 cần cả ảnh train và ảnh test"
            supported = {
                c.camera_id
                for c in manifest.cameras
                if c.model in {"opencv_pinhole", "opencv_fisheye"} and c.shutter == "GLOBAL"
            }
            if set(split.cameras) - supported:
                return "camera của split chưa được provider XSIM hỗ trợ"
        return None

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Ghi scene.json để tra cứu các file theo quy ước ingest KITTI-360.

        Raises:
            ValueError: Nếu split khác cảnh, timeline thiếu dữ liệu hoặc sai thứ tự.
        """
        scene, split = ctx.scene, ctx.split
        reason = self.unavailable_reason(scene.manifest, split)
        if reason:
            raise ValueError(reason)
        first, last = split.frame_range
        count = scene.manifest.timeline.num_frames
        if scene.scene_id != split.scene_id or last >= count:
            raise ValueError("split không khớp cảnh hoặc vượt quá timeline")
        frame_ids, stamps = (
            [int(value) for value in (scene.root / name).read_text().split()]
            for name in ("frame_ids.txt", "timestamps.txt")
        )
        for sequence in (frame_ids, stamps):
            if len(sequence) != count or any(a >= b for a, b in zip(sequence, sequence[1:])):
                raise ValueError("số khung hoặc thứ tự frame_ids/timestamps không hợp lệ")
        cameras = {c.camera_id: i for i, c in enumerate(scene.manifest.cameras)}
        index = {
            "format_version": 1,
            "scene_id": scene.scene_id,
            "converter_version": scene.manifest.source.converter_version,
            "split_key": input_key_for_split(split.split_id, split),
            "split": split.model_dump(mode="json"),
            "camera_indices": {name: cameras[name] for name in split.cameras},
            "source_frames": frame_ids[first : last + 1],
            "time_origin_us": stamps[first],
            "relative_time_us": [stamp - stamps[first] for stamp in stamps[first : last + 1]],
            "lidar_policy": "train_frame_only",
            "lidar_timing": "sweep_timestamp",
        }
        (ctx.output_dir / "scene.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        return {"clip": ctx.output_dir}
