"""Provider XSIM đọc gói quan sát của ``prep.observations``.

``BundleScene`` đưa XSIM đúng các quan sát loader DriveStudio đưa OmniRe: ảnh và mask của từng
camera, LiDAR, box theo khung và SMPL của người. Box dùng loại node DriveStudio đã gán (xe cứng,
người SMPL, vật biến dạng như người đi xe) để chọn nhóm của XSIM; SMPL đổi sang dạng XSIM bằng
cùng các bước ``xsim.data.utils.smpl`` làm với Waymo. Worker tạo đối tượng này bên trong môi
trường XSIM. CONTRACTS mục 5.4 mô tả gói.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import toast
import torch
from toast import SE3, LinearMotion
from torchvision.io import ImageReadMode, read_image
from xsim.data.loaders.sensor_data import SensorType
from xsim.data.loaders.sensor_loader import SensorLoader
from xsim.data.provider.basic_scene import DatasetScene
from xsim.data.utils.smpl import SMPLData
from xsimgs.structures import BoxSensorLink, OrientedBoxes

from drive2sim.core.schema import Split
from drive2sim.external.xsim.bundle import Observations
from drive2sim.external.xsim.observations import ObservationIndex
from drive2sim.external.xsim.sensors import BundleCameras, BundleLidar
from drive2sim.external.xsim.sweep import (
    DEFAULT_AZIMUTH_BINS,
    DEFAULT_ELEVATION_STEP,
    GridSpec,
    elevations,
    grid_for,
    local_points,
    sweep_grid,
)

# ModelType của DriveStudio (recon/data/base/scene_dataset.py) → BoxType của XSIM (Waymo):
# xe cứng → VEHICLE (1), người SMPL → PEDESTRIAN (2), vật biến dạng (người đi xe) → CYCLIST (4).
CATEGORY = {0: 1, 1: 2, 2: 4}
RIGID, SMPL_NODE, DEFORMABLE = 0, 1, 2


class SkyLoader(SensorLoader):
    """Mask bầu trời của gói ở kênh thứ ba, như loader segmentation của XSIM (argmax = 2)."""

    def __init__(self, observations: Observations, frames: dict[int, list[int]]) -> None:
        self.observations = observations
        self.frames = frames

    def __call__(self, sensor_id: int, sample_idx: int) -> torch.Tensor:
        """Ba kênh uint8 cùng kích thước ảnh; không có mask thì không có pixel trời."""
        frame = self.frames[sensor_id][sample_idx]
        path = self.observations.sky_path(frame, sensor_id)
        if path is None:
            info = self.observations.cameras[sensor_id]
            sky = torch.zeros((1, info.height, info.width), dtype=torch.uint8)
        else:
            sky = read_image(str(path), ImageReadMode.GRAY).gt(127).byte() * 255
        return torch.cat([torch.zeros_like(sky), torch.zeros_like(sky), sky])


def lidar_grid(
    observations: Observations,
    frames: list[int],
    step: float = DEFAULT_ELEVATION_STEP,
    cols: int = DEFAULT_AZIMUTH_BINS,
) -> tuple[GridSpec, float]:
    """Lưới cầu phủ dải góc nâng của các sweep, và tỉ lệ điểm lưới đó giữ được."""
    ranges = []
    for frame in frames:
        data = observations.lidar(frame)
        elevation = elevations(local_points(data["points"], data["sensor_to_world"]))
        if len(elevation):
            ranges.append((float(elevation.min()), float(elevation.max())))
    spec = grid_for(ranges, step, cols)
    kept = total = 0
    for frame in frames:
        data = observations.lidar(frame)
        kept += sweep_grid(data["points"], data["origins"], data["sensor_to_world"], spec)[3]
        total += len(data["points"])
    return spec, kept / total if total else 0.0


def smpl_for_keyframes(
    rotations: np.ndarray, betas: np.ndarray, valid: np.ndarray, keyframes: torch.Tensor
) -> Optional[SMPLData]:
    """SMPL của một người tại các keyframe box, theo ``xsim.data.utils.smpl``.

    Args:
        rotations: Ma trận quay (F, 24, 3, 3): khớp gốc trong hệ world, 23 khớp còn lại cục bộ,
            như ``smpl_quats`` của loader DriveStudio.
        betas: Hình dạng theo khung (F, 10).
        valid: Khung có SMPL (F,).
        keyframes: Vị trí khung của từng keyframe box (K,), tăng dần.

    Returns:
        ``SMPLData`` hoặc ``None`` nếu không keyframe nào của người này có SMPL (khi đó XSIM dựng
        người như vật biến dạng, không dùng góc khớp giữ chỗ).
    """
    index = keyframes.long()
    usable = np.flatnonzero(valid[index.numpy()])
    if not len(usable):
        return None
    quats = toast.Quat.from_matrix(torch.from_numpy(rotations).float()).std().data[index]
    times = keyframes.float()
    mask = torch.from_numpy(valid)[index]
    # Mỗi khớp một lần gọi với thời gian một chiều (K,), thay vì cả 24 khớp với (24, K) như
    # xsim.data.utils.smpl: với query hai chiều, sample_trajectory (torch backend, pytoast 6dadc76)
    # gán out_quats[invalid_mask, 0] = 1 bằng mask bool hai chiều kèm chỉ số nguyên, PyTorch báo
    # IndexError (gặp trên Kaggle, PyTorch 2.8). Mọi khớp chung mask nên kết quả như nhau (đã so
    # trên cảnh PHENIKAA thật: lệch 0).
    joints = []
    with toast.use_backend("torch"):
        for joint in range(quats.shape[1]):
            joints.append(
                toast.sample_trajectory(
                    query_times=times,
                    seq_times=times,
                    seq_quats=quats[:, joint],
                    seq_mask=mask,
                    extrapolate=False,
                    use_only_valid_keyframes=True,
                    return_mask=True,
                )[0]
            )
    first = int(index[int(usable[0])])
    return SMPLData(
        betas=torch.from_numpy(betas[first]).float(),
        joint_q=torch.stack(joints, dim=1).contiguous(),
    )


class BundleScene(DatasetScene):
    """Scene của một phần split (train hoặc test) dựng từ gói quan sát."""

    def __init__(
        self,
        observations_dir: str,
        train: bool = True,
        lidar_policy: str = "all",
        lidar_step: float = DEFAULT_ELEVATION_STEP,
        lidar_cols: int = DEFAULT_AZIMUTH_BINS,
    ) -> None:
        """Mở gói và dựng sensor, box, SMPL của phần split đã chọn.

        Args:
            observations_dir: Thư mục gói.
            train: Phần train (khởi tạo, tối ưu) hay test (render).
            lidar_policy: ``all`` dùng mọi sweep loader giữ (như DriveStudio khởi tạo từ LiDAR
                của cả đoạn); ``train`` chỉ sweep ở khung train.
            lidar_step: Bước góc nâng của lưới LiDAR (rad).
            lidar_cols: Số cột của lưới LiDAR.

        Raises:
            ValueError: Nếu gói thiếu LiDAR hoặc chính sách LiDAR lạ.
        """
        super().__init__()
        self.observations = observations = Observations(Path(observations_dir))
        split = Split.model_validate(observations.split)
        self.scene_idx = split.scene_id
        if lidar_policy not in ("all", "train"):
            raise ValueError(f"chính sách LiDAR lạ: {lidar_policy}")
        available = set(observations.lidar_frames)
        if not available:
            raise ValueError("gói quan sát không có LiDAR; XSIM khởi tạo và giám sát bằng LiDAR")
        if lidar_policy == "train":
            available &= {frame for frame, _ in split.train}
        self.index = ObservationIndex.from_split(
            split, observations.camera_indices, train=train, lidar_frames=sorted(available)
        )
        self.sky_loader = SkyLoader(observations, self.index.cameras)
        self.sensors[SensorType.CAMERA] = BundleCameras(observations, self.index.cameras)
        spec, self.lidar_retention = lidar_grid(
            observations, self.index.lidar_frames, lidar_step, lidar_cols
        )
        self.lidar_spec = spec
        self.sensors[SensorType.LIDAR] = BundleLidar(observations, self.index.lidar_frames, spec)
        self.boxes: list[OrientedBoxes] = []
        self.model_types: list[int] = []
        self.smpl_data: dict[int, SMPLData] = {}
        self._read_instances()

    def _read_instances(self) -> None:
        """Box của loader tại các khung có sweep: pose world, kích thước, vận tốc sai phân."""
        observations = self.observations
        instances = observations.instances()
        if instances is None:
            return
        smpl = observations.smpl()
        smpl_rows = {int(c): i for i, c in enumerate(smpl["columns"])} if smpl else {}
        lookup = {frame: i for i, frame in enumerate(self.index.lidar_frames)}
        positions = [observations.position(f) for f in self.index.lidar_frames]
        for column in range(instances["sizes"].shape[0]):
            rows = [
                (sweep, t)
                for sweep, t in zip(range(len(positions)), positions)
                if instances["visible"][t, column]
            ]
            if not rows:
                continue
            frames = [self.index.lidar_frames[s] for s, _ in rows]
            pose = SE3.from_matrix(
                torch.from_numpy(instances["poses"][[t for _, t in rows], column]).float()
            )
            stamp = torch.tensor([observations.time_s(f) for f in frames]).view(-1, 1)
            velocity = torch.zeros_like(pose.t)
            if len(rows) > 1:
                gaps = stamp[1:] - stamp[:-1]
                velocity[:-1] = (pose.t[1:] - pose.t[:-1]) / gaps
                velocity[-1] = velocity[-2]
            kind = int(instances["model_types"][column])
            object_id = len(self.boxes)
            count = len(rows)
            size = torch.from_numpy(instances["sizes"][column]).float()
            self.boxes.append(
                OrientedBoxes(
                    motion=LinearMotion(
                        pose=pose,
                        pose_time=stamp,
                        linear_velocity=velocity,
                        angular_velocity=torch.zeros_like(velocity),
                    ),
                    sizes=size.expand(count, -1).contiguous(),
                    category=torch.full((count,), CATEGORY.get(kind, 1), dtype=torch.uint8),
                    object_ids=torch.full((count,), object_id, dtype=torch.int32),
                    sensor_link=BoxSensorLink(
                        sensor_type=torch.full((count, 1), SensorType.LIDAR.value),
                        sensor_id=torch.zeros((count, 1), dtype=torch.long),
                        sample_idx=torch.tensor([lookup[f] for f in frames]).view(-1, 1),
                    ),
                )
            )
            self.model_types.append(kind)
            if kind == SMPL_NODE and column in smpl_rows:
                row = smpl_rows[column]
                data = smpl_for_keyframes(
                    smpl["rotations"][row],
                    smpl["betas"][row],
                    smpl["valid"][row],
                    torch.tensor([t for _, t in rows]),
                )
                if data is not None:
                    self.smpl_data[object_id] = data

    @property
    def num_objects(self) -> int:
        """Số box có trong các khung đã chọn."""
        return len(self.boxes)

    @property
    def vehicle_ids(self) -> list[int]:
        """Box DriveStudio dựng bằng node cứng."""
        return [i for i, kind in enumerate(self.model_types) if kind == RIGID]

    @property
    def human_ids(self) -> list[int]:
        """Người đi bộ (SMPL) và người đi xe (biến dạng), như XSIM gộp Pedestrian và Cyclist."""
        return [i for i, kind in enumerate(self.model_types) if kind in (SMPL_NODE, DEFORMABLE)]

    def get_boxes(self, object_id: int) -> OrientedBoxes:
        """Box theo thời gian của một vật trong hệ world."""
        return self.boxes[object_id]

    def get_smpl_data(self, object_id: int) -> Optional[SMPLData]:
        """SMPL của người, ``None`` khi gói không có SMPL cho người này."""
        return self.smpl_data.get(object_id)

    def get_segmentation_loader(self) -> SensorLoader:
        """Loader mask bầu trời của các mẫu được phép đọc."""
        return self.sky_loader

    def camera_lidar_synchronization(self) -> tuple[dict, dict, list]:
        """Ghép mẫu camera và sweep theo khung; mẫu không có sweep cùng khung thì không ghép."""
        return (
            {0: self.index.cameras_for_lidar},
            {
                sensor: [[] if i is None else [(0, i)] for i in items]
                for sensor, items in self.index.lidar_for_camera.items()
            },
            [(0,)],
        )
