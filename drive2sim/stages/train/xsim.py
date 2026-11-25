"""Stage train XSIM bằng interpreter riêng và hợp đồng job/result ở CONTRACTS mục 9.

Engine gọi TrainXsimStage với gói quan sát của ``prep.observations``. Stage trao đổi đường dẫn
tương đối với worker để cùng job có thể chạy ở workdir khác.
"""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.xsim_worker import run_xsim_worker


class TrainXsimParams(BaseModel):
    """Tham số train XSIM với cấu hình native ``waymo_default`` (camera, LiDAR, CNN decoder).

    Attributes:
        num_iters: Số bước tối ưu (mặc định như ``trainer.max_steps`` của XSIM).
        save_every: Bao nhiêu bước ghi checkpoint một lần (để chạy tiếp khi bị ngắt).
        seed: Seed ngẫu nhiên.
        preload_images: Nạp sẵn ảnh train vào RAM như cấu hình gốc của XSIM; tắt thì đọc PNG mỗi
            bước (ít RAM, chậm hơn).
        num_workers: Số tiến trình đọc dữ liệu của XSIM lúc nạp.
        lidar_policy: ``all`` dùng LiDAR của mọi khung loader giữ (như DriveStudio khởi tạo từ
            LiDAR cả đoạn); ``train`` chỉ LiDAR ở khung train.
        lidar_elevation_step_deg: Bước góc nâng của ảnh cầu LiDAR (độ).
        lidar_azimuth_bins: Số cột của ảnh cầu LiDAR.
    """

    model_config = ConfigDict(extra="forbid")
    num_iters: int = Field(default=40000, ge=1)
    save_every: int = Field(default=1000, ge=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    preload_images: bool = False
    num_workers: int = Field(default=0, ge=0)
    lidar_policy: Literal["all", "train"] = "all"
    lidar_elevation_step_deg: float = Field(default=0.2, gt=0, le=5)
    lidar_azimuth_bins: int = Field(default=2048, ge=256, le=8192)


@register_stage("train.xsim")
class TrainXsimStage(Stage):
    """Khởi chạy worker XSIM và xác nhận checkpoint của lần train trước khi báo xong."""

    version: ClassVar[str] = "0.2.0"
    uses_lidar: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {"observations": "observations"}
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = TrainXsimParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Ghi job, chuyển yêu cầu huỷ tới subprocess và đối chiếu kết quả."""
        assert isinstance(params, TrainXsimParams)
        return run_xsim_worker(
            ctx, inputs, params, self.stage_id, "model", "checkpoint.pt", params.num_iters
        )
