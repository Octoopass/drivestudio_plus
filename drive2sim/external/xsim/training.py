"""Dựng pipeline train XSIM từ gói quan sát trong interpreter riêng của XSIM.

Worker dùng build_training_pipeline để khởi tạo bằng tập train đã giới hạn
ở provider. train_step lấy đúng một batch đồng bộ để checkpoint giữa hai bước
không phải tái tạo hàng đợi prefetch. Xem kế hoạch tích hợp mục A05 và A08.
"""

from pathlib import Path

import torch
from jetcon import read
from xsim.data import SceneReconstructionDataset, SensorType, Subset
from xsim.data.loaders.sensor_loader import MultiLoaderDataset
from xsim.data.reconstruction.init import prepare_initialization
from xsim.modeling.nodes import NodeCollection
from xsim.modeling.scene import Scene
from xsim.utils.data_move import torch_move

from drive2sim.external.xsim.checkpoint import restore_checkpoint, save_checkpoint
from drive2sim.external.xsim.provider import BundleScene


def build_training_pipeline(
    provider: BundleScene,
    native_config: Path,
    device: str,
    preload_images: bool = False,
    num_workers: int = 0,
) -> tuple[Subset, NodeCollection]:
    """Dựng node và sampler từ đúng các quan sát mà provider train cho phép.

    Ảnh trong gói đã ở độ phân giải loader DriveStudio dùng, nên XSIM không thu nhỏ hay khử méo
    thêm (``image_downscale_factor`` 1, ``undistort_images`` tắt: xsimgs vẽ thẳng camera pinhole
    có méo và mắt cá).

    Args:
        provider: Provider được tạo cho phần train của split.
        native_config: Đường dẫn YAML native gồm cấu hình initialization và nodes.
        device: Thiết bị chứa model và batch khi tính gradient.
        preload_images: Nạp sẵn mọi ảnh train vào RAM như cấu hình gốc của XSIM.
        num_workers: Số tiến trình đọc dữ liệu của XSIM.

    Returns:
        Subset train và cây node trước khi tạo optimizer để có thể nạp checkpoint.
    """
    cfg = read(str(native_config))
    dataset_config = dict(cfg.dataset)
    dataset_config["scene"] = {
        **dataset_config.get("scene", {}),
        "undistort_images": False,
        "image_downscale_factor": 1.0,
        "image_resample": "bilinear",
        "load_segmentation": True,
    }
    dataset_config["load"] = {
        "device": "cpu",
        "preload_images": preload_images,
        "num_workers": num_workers,
    }
    dataset_config["batch"] = {**dataset_config.get("batch", {}), "max_lidars_per_batch": 1}
    dataset = SceneReconstructionDataset(provider, dataset_config)
    camera_ids = {
        sensor: list(range(len(frames))) for sensor, frames in provider.index.cameras.items()
    }
    lidar_ids = {0: list(range(len(provider.index.lidar_frames)))}
    subset = Subset(dataset, camera_ids, lidar_ids, split="train")
    init = prepare_initialization(dataset, cfg.init.build())
    init["train_split"] = subset
    nodes = cfg.nodes.build()(dataset, init).to(device)
    return subset, nodes


def train_step(subset: Subset, nodes: NodeCollection, device: str) -> float:
    """Lấy một mẫu theo lỗi ảnh và cập nhật toàn bộ node trainable của XSIM.

    Args:
        subset: Tập train chứa các chỉ số đã đồng bộ ảnh, sky mask và LiDAR.
        nodes: Cây node đã khởi tạo optimizer hoặc khôi phục từ checkpoint.
        device: Thiết bị thực hiện render và backward.

    Returns:
        Tổng loss của batch vừa hoàn thành.

    Raises:
        FloatingPointError: Nếu tổng loss không hữu hạn.
    """
    sample = subset.img_sampler.propose_sample()
    loaders = MultiLoaderDataset(subset.loaders, subset.multi_indices)
    batch = subset.dataset.make_sensor_batch(torch_move(loaders[sample], device))
    nodes.zero_grad()
    scene = Scene.from_sensor_batch(subset.dataset, batch)
    outputs = nodes(scene)
    loss = sum(nodes.compute_losses(scene, outputs).values())
    if not torch.isfinite(loss).item():
        raise FloatingPointError("XSIM training loss is not finite")
    with torch.no_grad():
        for render, target in zip(scene.renders, scene.batch):
            if target.sensor_type == SensorType.CAMERA:
                subset.img_sampler.update_error(
                    target.sensor_id,
                    target.sample_idx,
                    render.result.color.clamp(0, 1),
                    target.image,
                    target.gt_mask,
                )
    nodes.pre_backward(scene, outputs)
    loss.backward()
    nodes.post_backward(scene, outputs)
    nodes.optimizer_step(scene)
    return loss.detach().item()


def train(
    subset: Subset,
    nodes: NodeCollection,
    device: str,
    *,
    checkpoint: Path,
    fingerprint: str,
    max_steps: int,
    save_every: int,
) -> None:
    """Tối ưu cho tới bước cuối, dùng lại checkpoint hợp lệ sau lần gián đoạn.

    Args:
        subset: Tập train đã giới hạn quan sát theo split của drive2sim.
        nodes: Cây node trên thiết bị thực thi, chưa tạo optimizer.
        device: Thiết bị dùng cho batch và render.
        checkpoint: File trạng thái có thể nạp lại trong thư mục kết quả của stage.
        fingerprint: Định danh cấu hình, dữ liệu và revision của lần chạy.
        max_steps: Số bước tối ưu mục tiêu của cả run.
        save_every: Khoảng cách giữa hai lần ghi checkpoint theo số bước.

    Raises:
        ValueError: Nếu giới hạn bước sai hoặc checkpoint vượt số bước yêu cầu.
        RuntimeError: Nếu bộ đếm node không tăng đúng sau một bước tối ưu.
    """
    if min(max_steps, save_every) < 1:
        raise ValueError("XSIM step counts must be positive")
    if checkpoint.is_file():
        completed = restore_checkpoint(checkpoint, fingerprint, nodes, subset)
    else:
        nodes.init_optimization()
        completed = int(nodes.step.item())
    if not 0 <= completed <= max_steps:
        raise ValueError("XSIM checkpoint step outside requested training range")
    for step in range(completed + 1, max_steps + 1):
        loss = train_step(subset, nodes, device)
        if int(nodes.step.item()) != step:
            raise RuntimeError("XSIM node step differs from worker step")
        if step == 1 or step % 100 == 0 or step == max_steps:
            print(f"XSIM step={step}/{max_steps} loss={loss:.6f}", flush=True)
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(checkpoint, fingerprint, nodes, subset)
