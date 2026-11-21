"""Cung cấp camera và sweep KITTI-360 trong môi trường native của XSIM.

Worker dùng KittiSensors để dựng rig theo index của prep.xsim. Mỗi sensor chỉ mở
những frame trong phần split đã chọn. Xem CONTRACTS mục 5.4.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from toast import SE3, LinearMotion
from torchvision.io import ImageReadMode, read_image
from xsim.data.loaders.sensor_data import SensorData
from xsim.data.loaders.sensor_loader import SensorLoader
from xsim.structures import Sweep
from xsimgs.cameras import (
    DistortedPerspectiveCamera,
    FisheyeCamera,
    RollingShutter,
    RSCamera,
    SphericalCamera,
)

from drive2sim.core.camera import MEI_FIT_MAX_ANGLE
from drive2sim.core.schema import ManifestCamera
from drive2sim.external.xsim.sweep import read_sweep_grid
from drive2sim.recon.utils.geometry import camera_model_rays


class KittiSensors(SensorData, SensorLoader):
    """Rig camera hoặc Velodyne với loader dùng được trong DataLoader của XSIM."""

    def __init__(
        self,
        root: Path,
        frames: dict[int, list[int]],
        names: dict[int, str],
        times: dict[int, float],
        poses: dict[int, torch.Tensor],
        cameras: Optional[dict[int, ManifestCamera]] = None,
        grid_shape: tuple[int, int] = (128, 2048),
    ) -> None:
        self.root = root
        self.frames = frames
        self.names = names
        self.poses = poses
        self.is_lidar = cameras is None
        self.grid_shape = grid_shape
        self.rig: dict[int, SE3] = {}
        self.native: dict[int, RSCamera] = {}
        self.masks: dict[int, torch.Tensor] = {}
        for sensor, selected in frames.items():
            transform = (
                torch.eye(4)
                if self.is_lidar
                else torch.tensor(
                    np.loadtxt(root / "extrinsics" / f"{sensor}.txt"), dtype=torch.float32
                )
            )
            self.rig[sensor] = SE3.from_matrix(transform)
            time = torch.tensor([times[f] for f in selected]).view(-1, 1)
            pose = SE3.from_matrix(torch.stack([poses[f] @ transform for f in selected]))
            motion = LinearMotion(
                pose=pose,
                pose_time=time,
                linear_velocity=torch.zeros_like(pose.t),
                angular_velocity=torch.zeros_like(pose.t),
            )
            shutter = RollingShutter(time, torch.zeros(len(time), 2))
            if cameras is None:
                limits = torch.tensor([0.0, 2 * math.pi, math.pi / 2, -math.pi])
                self.native[sensor] = SphericalCamera(
                    motion, shutter, limits.expand(len(time), -1).contiguous(), camera_name=sensor
                )
                self.masks[sensor] = torch.ones(grid_shape, dtype=torch.bool)
                continue
            camera = cameras[sensor]
            w, h = camera.resolution
            intr = np.loadtxt(root / "intrinsics" / f"{sensor}.txt")
            fisheye = camera.model == "opencv_fisheye"
            if intr.shape != (8 if fisheye else 9,) or not np.isfinite(intr).all():
                raise ValueError(f"intrinsics camera {sensor} không hợp lệ")
            fx, fy, cx, cy = intr[:4]
            cls = FisheyeCamera if fisheye else DistortedPerspectiveCamera
            normalized = cls.intrinsics_from_sized(fx, fy, cx, cy, w, h)
            dist = torch.tensor(intr[4:] if fisheye else np.pad(intr[4:], (0, 3))).float()
            self.native[sensor] = cls(
                world_se3_camera=motion,
                shutter=shutter,
                normalized_intrinsics=normalized.expand(len(time), -1).contiguous(),
                distortion=dist.expand(len(time), -1).contiguous(),
                camera_name=sensor,
            )
            if fisheye:
                matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
                _, mask = camera_model_rays(
                    h,
                    w,
                    matrix,
                    "fisheye",
                    dist,
                    fisheye_max_angle=camera.max_angle or MEI_FIT_MAX_ANGLE,
                )
                self.masks[sensor] = mask
            else:
                self.masks[sensor] = torch.ones((h, w), dtype=torch.bool)

    @property
    def indices(self) -> list[int]:
        """Trả chỉ số sensor có mẫu trong split."""
        return list(self.frames)

    @property
    def captions(self) -> list[str]:
        """Trả tên sensor theo thứ tự index."""
        return [self.names[i] for i in self.indices]

    def num_images(self, sensor_id: int) -> int:
        """Đếm mẫu đã chọn của sensor."""
        return len(self.frames[sensor_id])

    def get_camera(self, sensor_id: int, sample_idx: int) -> RSCamera:
        """Trả pose và calibration của một mẫu."""
        return self.native[sensor_id][sample_idx]

    def get_cameras(self, sensor_id: int) -> RSCamera:
        """Trả mọi pose của sensor dưới dạng batch native."""
        return self.native[sensor_id]

    def get_mask(self, sensor_id: int) -> torch.Tensor:
        """Trả vùng pixel hợp lệ theo hiệu chuẩn ống kính."""
        return self.masks[sensor_id]

    def get_ego_se3_sensor(self, sensor_id: int) -> SE3:
        """Trả phép biến đổi từ sensor sang Velodyne."""
        return self.rig[sensor_id]

    def get_loader(self) -> SensorLoader:
        """Trả loader mà dataset native gọi theo sensor và sample."""
        return self

    def __call__(self, sensor_id: int, sample_idx: int) -> object:
        """Đọc ảnh RGB hoặc sweep trong hệ world tại đúng thời điểm khung."""
        frame = self.frames[sensor_id][sample_idx]
        if not self.is_lidar:
            return read_image(
                str(self.root / "images" / f"{frame:03d}_{sensor_id}.png"), ImageReadMode.RGB
            )
        grid, valid = read_sweep_grid(
            self.root / "lidar" / f"{frame:03d}.bin", *self.grid_shape, math.pi / 2, -math.pi / 2
        )
        pose = self.poses[frame]
        xyz = torch.from_numpy(grid[:, :, :3]) @ pose[:3, :3].T + pose[:3, 3]
        return Sweep(
            xyz=xyz,
            origin=pose[:3, 3],
            intensity=torch.from_numpy(grid[:, :, 3:]),
            mask=torch.from_numpy(valid),
        )
