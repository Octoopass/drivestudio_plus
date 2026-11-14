"""Stage ``train.gsplat_3dgut``: train một cảnh tĩnh bằng Gaussian 3DGUT của gsplat.

Stage này chạy trainer ``drive2sim/recon/tools/train_gsplat_3dgut.py`` (port trainer KITTI-360
của teammate, repo thaianh20021/fisheye-fusion-lab) trong một tiến trình con của môi trường chính,
giống ``train.drivestudio``: engine và CLI không phải nạp PyTorch, lỗi CUDA chỉ làm hỏng tiến
trình con. Trainer đọc thẳng thư mục cảnh chuẩn và split của run, vẽ camera pinhole và mắt cá cùng
lúc bằng 3DGUT (``docs/methods/gsplat_3dgut.md``).

File này chứa:
- ``TrainGsplat3dgutParams``: tham số của stage (số iteration, độ phân giải, vòm trời, mask nắp
  capô, SSIM, optimizer...).
- ``build_command``: dựng câu lệnh train; tách riêng để test được trên máy không có GPU.
- ``TrainGsplat3dgutStage``: stage đăng ký với engine.

Kết quả trong thư mục của stage có cùng tên file với ``train.drivestudio`` ở những chỗ stage sau
đọc: ``checkpoint_final.pth`` (node ``Background``, cho ``export.ply_world`` và
``export.preview``), ``metrics/images_test_<thời điểm>.json`` và ``train_stats.json`` (cho
``eval.standard``), ``cameras.json`` (cho trang Xem 3D). Chạy tiếp từ ``checkpoint_resume.pth``
khi run bị ngắt (CONTRACTS mục 5.2).
"""

import math
import sys
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.core import files
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.commands import LOG_FILE, run_logged

TRAINER_MODULE = "drive2sim.recon.tools.train_gsplat_3dgut"
FINAL_CHECKPOINT = "checkpoint_final.pth"
RESUME_CHECKPOINT = "checkpoint_resume.pth"
# Split của run chép vào thư mục stage để trainer đọc, không phụ thuộc đường dẫn của workdir.
SPLIT_FILE = "split.json"


class TrainGsplat3dgutParams(BaseModel):
    """Tham số của stage ``train.gsplat_3dgut``.

    Mặc định là cấu hình tốt nhất của teammate (vòm trời, mask nắp capô, fused-ssim,
    SelectiveAdam; ``docs/kitti360-3dgut.md`` của repo teammate).

    Attributes:
        num_iters: Số bước train (mỗi bước một ảnh).
        downscale: Thu nhỏ ảnh bao nhiêu lần khi nạp (teammate: 2).
        max_init_gaussians: Số Gaussian khởi tạo tối đa từ LiDAR.
        cap_max: Số Gaussian tối đa khi MCMC thêm Gaussian.
        sky: Thêm Gaussian lớn trên vòm trời 200 m (LiDAR không chạm tới trời).
        hood_mask: Bỏ nắp capô và vùng đen ngoài vòng tròn mắt cá khỏi loss. Đánh giá luôn bỏ hai
            vùng này.
        ssim: ``fused`` (gói fused-ssim), ``torch`` (conv2d) hoặc ``none`` (chỉ L1).
        optimizer: ``selective_adam`` (gsplat, chỉ cập nhật Gaussian thấy được; không hiệu chỉnh
            bias của Adam) hoặc ``adam`` (``torch.optim.Adam``).
        fisheye_max_angle: Góc tia lớn nhất (rad) của camera mắt cá được train và đánh giá; phải
            nhỏ hơn pi/2 (giới hạn của renderer UT).
        checkpoint_every: Lưu ``checkpoint_resume.pth`` sau mỗi số bước này; ``0`` là không lưu.
        seed: Seed ngẫu nhiên.
    """

    model_config = ConfigDict(extra="forbid")

    num_iters: int = Field(default=10000, ge=1)
    downscale: int = Field(default=2, ge=1)
    max_init_gaussians: int = Field(default=120000, ge=100)
    cap_max: int = Field(default=1000000, ge=1000)
    sky: bool = True
    hood_mask: bool = True
    ssim: Literal["fused", "torch", "none"] = "fused"
    optimizer: Literal["selective_adam", "adam"] = "selective_adam"
    fisheye_max_angle: float = Field(default=1.45, gt=0, lt=math.pi / 2)
    checkpoint_every: int = Field(default=2500, ge=0)
    seed: int = 42


def build_command(
    params: TrainGsplat3dgutParams, scene_dir: Path, split_file: Path, output_dir: Path
) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.tools.train_gsplat_3dgut``.

    Nếu thư mục kết quả đã có ``checkpoint_resume.pth`` (run bị ngắt), thêm ``--resume_from`` để
    train tiếp.

    Args:
        params: Tham số đã kiểm tra.
        scene_dir: Thư mục cảnh ``data/<scene_id>/``.
        split_file: File split của run (bản chép trong thư mục stage).
        output_dir: Thư mục kết quả của stage.

    Returns:
        Câu lệnh dạng danh sách.
    """
    command = [
        sys.executable,
        "-m",
        TRAINER_MODULE,
        "--scene_dir",
        str(scene_dir),
        "--split_file",
        str(split_file),
        "--output_dir",
        str(output_dir),
        "--num_iters",
        str(params.num_iters),
        "--downscale",
        str(params.downscale),
        "--max_init_gaussians",
        str(params.max_init_gaussians),
        "--cap_max",
        str(params.cap_max),
        "--sky",
        "on" if params.sky else "off",
        "--hood_mask",
        "on" if params.hood_mask else "off",
        "--ssim",
        params.ssim,
        "--optimizer",
        params.optimizer,
        "--fisheye_max_angle",
        str(params.fisheye_max_angle),
        "--checkpoint_every",
        str(params.checkpoint_every),
        "--seed",
        str(params.seed),
    ]
    resume = output_dir / RESUME_CHECKPOINT
    if resume.is_file():
        command += ["--resume_from", str(resume)]
    return command


@register_stage("train.gsplat_3dgut")
class TrainGsplat3dgutStage(Stage):
    """Train cảnh tĩnh bằng 3DGUT của gsplat, qua một tiến trình con.

    Đầu vào là cảnh và split; đầu ra là thư mục train chứa ``checkpoint_final.pth``. Khởi tạo
    Gaussian từ LiDAR nên cần LiDAR.
    """

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {"scene": "scene", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = TrainGsplat3dgutParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy trainer, ghi toàn bộ stdout và stderr vào ``log.txt`` của stage.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: Không có artifact đầu vào; cảnh và split lấy từ ``ctx``.
            params: ``TrainGsplat3dgutParams`` đã kiểm tra.

        Returns:
            ``{"model": <thư mục stage>}``.

        Raises:
            RuntimeError: Nếu trainer thoát với mã khác 0 hoặc không ghi ra checkpoint cuối.
        """
        assert isinstance(params, TrainGsplat3dgutParams)
        split_file = ctx.output_dir / SPLIT_FILE
        files.dump(ctx.split, split_file)
        command = build_command(params, ctx.scene.root, split_file, ctx.output_dir)
        ctx.progress(0.0, f"train 3DGUT {params.num_iters} bước")
        run_logged(command, ctx.output_dir / LOG_FILE, should_stop=ctx.should_stop)
        if not (ctx.output_dir / FINAL_CHECKPOINT).is_file():
            raise RuntimeError(f"trainer xong nhưng không có {FINAL_CHECKPOINT}")
        ctx.progress(1.0, "train xong")
        return {"model": ctx.output_dir}
