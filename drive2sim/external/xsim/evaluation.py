"""Vẽ camera của split bằng model XSIM đã học và chấm ảnh theo SPEC mục 9.

Worker dùng đúng embedding camera và hệ thời gian của run train. Provider đánh
giá chỉ cung cấp pose camera, ảnh tham chiếu và vùng hợp lệ tại frame cần vẽ.
"""

from collections.abc import Iterator

import torch
from xsim.data import SensorType, Subset
from xsim.data.loaders.image_loaders import ImageLoadingDataset, resize_mask
from xsim.modeling import RenderInfo
from xsim.modeling.nodes import NodeCollection
from xsim.modeling.scene import Scene

from drive2sim.external.xsim.provider import KittiScene


def render_pairs(
    provider: KittiScene, trained: Subset, nodes: NodeCollection, downscale: float, device: str
) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Vẽ lần lượt các cặp frame và camera theo thứ tự thời gian.

    Args:
        provider: Scene của phần split cần vẽ.
        trained: Subset dùng khi dựng model, cung cấp hệ thời gian và các camera đã học.
        nodes: Cây node đã nạp checkpoint.
        downscale: Hệ số thu ảnh dùng khi train.
        device: Thiết bị của model.

    Yields:
        Frame ingest, chỉ số camera, ảnh render và ảnh thật HWC trong [0,1], cùng mask HW.

    Raises:
        ValueError: Nếu camera test chưa có embedding train hoặc ảnh trả về sai kích thước.
    """
    cameras = provider.cameras
    unseen = set(cameras.indices) - set(trained.camera_ids)
    if unseen:
        raise ValueError(f"Camera test thiếu embedding đã train: {sorted(unseen)}")
    loader = ImageLoadingDataset(cameras.get_loader(), None, downscale, "nearest")
    masks = {
        sensor: resize_mask(cameras.get_mask(sensor), downscale).bool().to(device)
        for sensor in cameras.indices
    }
    nodes.eval()
    pairs = sorted(
        (frame, sensor, sample)
        for sensor, frames in provider.index.cameras.items()
        for sample, frame in enumerate(frames)
    )
    for frame, sensor, sample in pairs:
        with torch.no_grad():
            _, gt = loader.load_image(sensor, sample)
            gt = gt.to(device).float() / 255
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
                    f"Kích thước render hoặc mask sai tại frame {frame} camera {sensor}"
                )
            rgb = rgb.clamp(0, 1)
        yield frame, sensor, rgb, gt, masks[sensor]
