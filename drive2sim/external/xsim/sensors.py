"""Camera và LiDAR của gói quan sát dưới dạng sensor của XSIM.

Worker dựng ``BundleCameras`` và ``BundleLidar`` theo chỉ mục của split. Mỗi sensor chỉ mở những
khung chỉ mục cho phép. Pose, intrinsics, hệ số méo và mask lấy nguyên từ gói (CONTRACTS mục 5.4),
tức đúng những gì loader DriveStudio đưa cho OmniRe.
"""

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

from drive2sim.external.xsim.bundle import Observations
from drive2sim.external.xsim.sweep import GridSpec, sweep_grid


def _static_motion(poses: np.ndarray, times: list[float]) -> tuple[LinearMotion, RollingShutter]:
    """Pose cố định trong từng ảnh (global shutter), vận tốc bằng 0."""
    time = torch.tensor(times, dtype=torch.float32).view(-1, 1)
    pose = SE3.from_matrix(torch.from_numpy(np.asarray(poses)).float())
    motion = LinearMotion(
        pose=pose,
        pose_time=time,
        linear_velocity=torch.zeros_like(pose.t),
        angular_velocity=torch.zeros_like(pose.t),
    )
    return motion, RollingShutter(time, torch.zeros(len(times), 2))


class BundleCameras(SensorData, SensorLoader):
    """Các camera của split, mỗi camera chỉ gồm các khung được phép đọc."""

    def __init__(self, observations: Observations, frames: dict[int, list[int]]) -> None:
        self.observations = observations
        self.frames = frames
        self.native: dict[int, RSCamera] = {}
        self.masks: dict[int, torch.Tensor] = {}
        self.rig: dict[int, SE3] = {}
        for sensor, selected in frames.items():
            info = observations.cameras[sensor]
            poses = np.stack([observations.cam_to_world(sensor, f) for f in selected])
            motion, shutter = _static_motion(poses, [observations.time_s(f) for f in selected])
            fisheye = info.model == "fisheye"
            cls = FisheyeCamera if fisheye else DistortedPerspectiveCamera
            normalized = cls.intrinsics_from_sized(*info.intrinsics, info.width, info.height)
            coeffs = np.asarray(info.distortion, dtype=np.float32)
            if fisheye and coeffs.shape != (4,):
                raise ValueError(f"camera {info.camera_id}: mắt cá cần 4 hệ số méo")
            if not fisheye:
                # OpenCV k1, k2, p1, p2, k3 → 8 hệ số của DistortedPerspectiveCamera.
                coeffs = np.pad(coeffs, (0, 8 - len(coeffs)))
            dist = torch.from_numpy(coeffs)
            self.native[sensor] = cls(
                world_se3_camera=motion,
                shutter=shutter,
                normalized_intrinsics=normalized.expand(len(selected), -1).contiguous(),
                distortion=dist.expand(len(selected), -1).contiguous(),
                camera_name=sensor,
            )
            self.masks[sensor] = torch.from_numpy(observations.train_mask(sensor).copy())
            # XSIM chỉ dùng rig để vẽ, không để học: lấy pose camera so với LiDAR ở khung đầu.
            lidar = observations.lidar_frames
            ref = selected[0]
            ego = observations.lidar(ref)["sensor_to_world"] if ref in lidar else np.eye(4)
            self.rig[sensor] = SE3.from_matrix(
                torch.from_numpy(np.linalg.inv(ego) @ poses[0]).float()
            )

    @property
    def indices(self) -> list[int]:
        """Chỉ số sensor có mẫu."""
        return list(self.frames)

    @property
    def captions(self) -> list[str]:
        """Tên camera theo thứ tự index."""
        return [self.observations.cameras[i].camera_id for i in self.indices]

    def num_images(self, sensor_id: int) -> int:
        """Số mẫu của camera."""
        return len(self.frames[sensor_id])

    def get_camera(self, sensor_id: int, sample_idx: int) -> RSCamera:
        """Pose và calibration của một mẫu."""
        return self.native[sensor_id][sample_idx]

    def get_cameras(self, sensor_id: int) -> RSCamera:
        """Mọi pose của camera dưới dạng batch native."""
        return self.native[sensor_id]

    def get_mask(self, sensor_id: int) -> torch.Tensor:
        """Pixel loader cho phép giám sát (vành mắt cá, viền tối, thân xe bị loại)."""
        return self.masks[sensor_id]

    def get_ego_se3_sensor(self, sensor_id: int) -> SE3:
        """Camera so với LiDAR."""
        return self.rig[sensor_id]

    def get_loader(self) -> SensorLoader:
        """Loader mà dataset native gọi theo sensor và sample."""
        return self

    def __call__(self, sensor_id: int, sample_idx: int) -> torch.Tensor:
        """Ảnh RGB uint8 CHW của mẫu."""
        frame = self.frames[sensor_id][sample_idx]
        return read_image(str(self.observations.image_path(frame, sensor_id)), ImageReadMode.RGB)


class BundleLidar(SensorData, SensorLoader):
    """LiDAR của gói, mỗi khung một ảnh cầu theo ``GridSpec`` của cảnh."""

    def __init__(self, observations: Observations, frames: list[int], spec: GridSpec) -> None:
        self.observations = observations
        self.frames = {0: frames}
        self.spec = spec
        poses = np.stack([observations.lidar(f)["sensor_to_world"] for f in frames])
        motion, shutter = _static_motion(poses, [observations.time_s(f) for f in frames])
        limits = torch.tensor(spec.limits, dtype=torch.float32)
        self.native = SphericalCamera(
            motion, shutter, limits.expand(len(frames), -1).contiguous(), camera_name=0
        )

    @property
    def indices(self) -> list[int]:
        """Một LiDAR, chỉ số 0."""
        return [0]

    @property
    def captions(self) -> list[str]:
        """Tên LiDAR."""
        return ["lidar"]

    def num_images(self, sensor_id: int) -> int:
        """Số sweep."""
        return len(self.frames[sensor_id])

    def get_camera(self, sensor_id: int, sample_idx: int) -> RSCamera:
        """Ảnh cầu của một sweep."""
        return self.native[sample_idx]

    def get_cameras(self, sensor_id: int) -> RSCamera:
        """Mọi sweep dưới dạng batch native."""
        return self.native

    def get_mask(self, sensor_id: int) -> torch.Tensor:
        """Mọi ô của ảnh cầu đều hợp lệ; ô không đo do ``Sweep.mask`` loại."""
        return torch.ones((self.spec.rows, self.spec.cols), dtype=torch.bool)

    def get_ego_se3_sensor(self, sensor_id: int) -> SE3:
        """LiDAR là gốc của rig."""
        return SE3.from_matrix(torch.eye(4))

    def get_loader(self) -> SensorLoader:
        """Loader mà dataset native gọi theo sensor và sample."""
        return self

    def __call__(self, sensor_id: int, sample_idx: int) -> Sweep:
        """Sweep world của khung đặt vào lưới cầu."""
        frame = self.frames[sensor_id][sample_idx]
        data = self.observations.lidar(frame)
        xyz, origin, valid, _ = sweep_grid(
            data["points"], data["origins"], data["sensor_to_world"], self.spec
        )
        return Sweep(
            xyz=torch.from_numpy(xyz),
            origin=torch.from_numpy(origin),
            mask=torch.from_numpy(valid),
        )
