"""Vẽ các camera của split bằng model XSIM đã học.

Worker dùng đúng embedding camera và hệ thời gian của run train. Provider render chỉ cung cấp pose
camera tại khung cần vẽ; ảnh thật lấy từ gói quan sát. Ảnh render được lưu PNG 8-bit để stage
``eval.xsim`` chấm điểm trong môi trường chính bằng cùng hàm và cùng mask như ``eval.standard``.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from xsim.data import SensorType, Subset
from xsim.modeling import RenderInfo
from xsim.modeling.nodes import NodeCollection
from xsim.modeling.scene import Scene

from drive2sim.external.xsim.provider import BundleScene

Pair = tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]


def to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """Ảnh HWC trong [0, 1] → uint8, cùng cách làm tròn với ``eval.standard``."""
    return torch.round(rgb.clamp(0, 1) * 255.0).byte().cpu().numpy()


def save_render(rgb: torch.Tensor, folder: Path, frame: int, sensor: int) -> None:
    """Ghi ảnh render của một cặp thành ``<frame>_<camera>.png``."""
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(rgb)).save(folder / f"{frame:03d}_{sensor}.png", compress_level=1)


def render_pairs(
    provider: BundleScene, trained: Subset, nodes: NodeCollection, device: str
) -> Iterator[Pair]:
    """Vẽ lần lượt các cặp khung và camera theo thứ tự thời gian.

    Args:
        provider: Scene của phần split cần vẽ.
        trained: Subset dùng khi dựng model, cung cấp hệ thời gian và các camera đã học.
        nodes: Cây node đã nạp checkpoint.
        device: Thiết bị của model.

    Yields:
        Khung, chỉ số camera, ảnh render và ảnh thật HWC trong [0, 1], mask hiển thị HW (mask
        đánh giá của gói, cả ảnh khi không có).

    Raises:
        ValueError: Nếu camera chưa có embedding train hoặc ảnh trả về sai kích thước.
    """
    cameras = provider.cameras
    unseen = set(cameras.indices) - set(trained.camera_ids)
    if unseen:
        raise ValueError(f"camera thiếu embedding đã train: {sorted(unseen)}")
    observations = provider.observations
    masks = {}
    for sensor in cameras.indices:
        mask = observations.eval_mask(sensor)
        info = observations.cameras[sensor]
        if mask is None:
            mask = np.ones((info.height, info.width), dtype=bool)
        masks[sensor] = torch.from_numpy(mask.copy()).to(device)
    nodes.eval()
    pairs = sorted(
        (frame, sensor, sample)
        for sensor, frames in provider.index.cameras.items()
        for sample, frame in enumerate(frames)
    )
    for frame, sensor, sample in pairs:
        with torch.no_grad():
            gt = cameras(sensor, sample).permute(1, 2, 0).to(device).float() / 255
            height, width = gt.shape[:2]
            info = RenderInfo(
                sensor_type=SensorType.CAMERA,
                sensor_id=sensor,
                sample_idx=sample,
                camera=cameras.get_camera(sensor, sample).to(device),
                width=width,
                height=height,
            )
            scene = Scene(time_bounds=trained.dataset.time_range, renders=[info])
            nodes(scene)
            rgb = info.result.color
            if rgb is None or rgb.shape != gt.shape or masks[sensor].shape != gt.shape[:2]:
                raise ValueError(
                    f"sai kích thước render hoặc mask: khung {frame}, camera {sensor}"
                )
            rgb = rgb.clamp(0, 1)
        yield frame, sensor, rgb, gt, masks[sensor]
