"""Stage ``export.nurec_ply``: PLY của Instant NuRec đổi sang hệ ``world`` của DriveStudio.

PLY của NuRec nằm trong hệ world của clip NCore (rig FLU ở khung đầu, gốc ở ``image_00``); model
DriveStudio của cùng cảnh nằm trong hệ ``world`` của loader KITTI (Velodyne ở khung đầu của
split). Stage đổi vị trí và hướng Gaussian (``drive2sim/external/nurec/ply.py``; PLY của NuRec
không có SH bậc cao nên màu không phải xoay), ghi ``scene.ply`` với dòng ``comment`` trục lên như
``export.ply_world`` và ``scene.json`` mô tả file, để trang Xem 3D, So sánh và ``d2s export`` dùng
như PLY của approach khác.

File này chứa ``ExportNurecPlyStage``.
"""

import json
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.stages.export.ply_world import PLY_FILE
from drive2sim.stages.train.nurec import MERGED_PLY

UP_COMMENT = "comment drive2sim up=+z frame=drivestudio_world units=m"


@register_stage("export.nurec_ply")
class ExportNurecPlyStage(Stage):
    """Đổi PLY của Instant NuRec sang hệ ``world`` của cảnh."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model", "scene": "scene"}
    outputs: ClassVar[dict[str, str]] = {"ply": "export_ply"}
    Params: ClassVar[type[BaseModel]] = NoParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Đọc ``scene_nurec.ply``, đổi hệ theo ``extrinsics/0.txt``, ghi ``scene.ply``.

        Returns:
            ``{"ply": <thư mục stage>/scene.ply}``.
        """
        import numpy as np

        from drive2sim.external.nurec import ply

        cam0 = np.loadtxt(ctx.scene.root / "extrinsics" / "0.txt").reshape(4, 4)
        transform = ply.ncore_to_world(cam0)
        header, data = ply.read_ply(inputs["model"] / MERGED_PLY)
        moved = ply.transform_ply(data, transform)
        header = [line for line in header if not line.startswith("comment drive2sim")]
        header.insert(header.index("end_header"), UP_COMMENT)
        target = ctx.output_dir / PLY_FILE
        ply.write_ply(target, header, moved)
        (ctx.output_dir / "scene.json").write_text(
            json.dumps(
                {
                    "source": str(inputs["model"] / MERGED_PLY),
                    "frame": "drivestudio world frame (KITTI-360: Velodyne at the split's first "
                    "frame), from the NCore clip world by extrinsics/0.txt x rig->camera",
                    "up_axis": "+z",
                    "units": "metres",
                    "layout": "Instant NuRec PLY: xyz, f_dc (SH0 only), opacity, scale, rot(wxyz)",
                    "ncore_to_world": transform.tolist(),
                    "num_gaussians_written": int(len(moved)),
                    "file_size_mb": round(target.stat().st_size / 2**20, 1),
                },
                indent=2,
            )
            + "\n"
        )
        return {"ply": target}
