"""Stage train XSIM bằng interpreter riêng và hợp đồng job/result ở CONTRACTS mục 9.

Engine gọi TrainXsimStage với clip đã chuẩn bị và mask của cảnh. Stage trao đổi
đường dẫn tương đối với worker để cùng job có thể chạy ở workdir khác.
"""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.xsim_worker import run_xsim_worker


class TrainXsimParams(BaseModel):
    """Tham số cho baseline camera và LiDAR với cấu hình native waymo_default."""

    model_config = ConfigDict(extra="forbid")
    num_iters: int = Field(default=40000, ge=1)
    save_every: int = Field(default=1000, ge=1)
    downscale: float = Field(default=1.0, ge=1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


@register_stage("train.xsim")
class TrainXsimStage(Stage):
    """Khởi chạy worker XSIM và xác nhận checkpoint của lần train trước khi báo xong."""

    version: ClassVar[str] = "0.1.0"
    uses_lidar: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {"scene": "scene", "clip": "xsim_scene", "masks": "masks"}
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = TrainXsimParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Ghi job, chuyển yêu cầu huỷ tới subprocess và đối chiếu kết quả."""
        return run_xsim_worker(ctx, inputs, params, self.stage_id, "model", "checkpoint.pt")
