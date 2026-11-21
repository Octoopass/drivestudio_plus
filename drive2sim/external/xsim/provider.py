"""Provider XSIM đọc scene KITTI-360 đã ingest và chỉ mục của prep.xsim.

KittiScene ghép camera với Velodyne và box theo các frame được phép đọc trong split.
Worker tạo đối tượng này bên trong interpreter XSIM. CONTRACTS mục 5.4 mô tả artifact.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from toast import SE3, LinearMotion
from torchvision.io import ImageReadMode, read_image
from xsim.data.loaders.sensor_data import SensorType
from xsim.data.loaders.sensor_loader import SensorLoader
from xsim.data.provider.basic_scene import DatasetScene
from xsim.data.utils.smpl import SMPLData
from xsimgs.structures import BoxSensorLink, OrientedBoxes

from drive2sim.core import files
from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.cache import input_key_for_split
from drive2sim.external.xsim.observations import ObservationIndex
from drive2sim.external.xsim.sensors import KittiSensors


class SkyLoader(SensorLoader):
    """Mở sky mask của drive2sim và đưa sky vào kênh thứ ba theo loader XSIM."""

    def __init__(self, masks: Path, index: ObservationIndex) -> None:
        self.masks = masks
        self.index = index

    def __call__(self, sensor_id: int, sample_idx: int) -> torch.Tensor:
        """Trả ba kênh uint8 cùng kích thước ảnh gốc."""
        frame = self.index.cameras[sensor_id][sample_idx]
        path = self.masks / "sky_masks" / f"{frame:03d}_{sensor_id}.png"
        sky = read_image(str(path), ImageReadMode.GRAY).gt(0).byte() * 255
        return torch.cat([torch.zeros_like(sky), torch.zeros_like(sky), sky])


class KittiScene(DatasetScene):
    """Scene KITTI-360 cho cấu hình XSIM dùng LiDAR theo thời điểm khung."""

    def __init__(self, scene_dir: str, clip_dir: str, masks_dir: str, train: bool = True) -> None:
        super().__init__()
        root = Path(scene_dir)
        manifest = files.load(root / "manifest.yaml", Manifest)
        clip = json.loads((Path(clip_dir) / "scene.json").read_text())
        split = Split.model_validate(clip["split"])
        if (
            clip["format_version"] != 1
            or manifest.source.dataset != "kitti360"
            or clip["converter_version"] != manifest.source.converter_version
            or manifest.scene_id != split.scene_id
            or clip["scene_id"] != split.scene_id
            or clip["split_key"] != input_key_for_split(split.split_id, split)
        ):
            raise ValueError("artifact prep.xsim không khớp scene hoặc split")
        ids = clip["camera_indices"]
        expected = {
            c.camera_id: i for i, c in enumerate(manifest.cameras) if c.camera_id in split.cameras
        }
        if expected != ids:
            raise ValueError("camera trong artifact XSIM khác manifest")
        self.scene_idx = manifest.scene_id
        self.index = ObservationIndex.from_split(split, ids, train=train)
        self.sky_loader = SkyLoader(Path(masks_dir), self.index)
        first = split.frame_range[0]
        frames = self.index.lidar_frames
        times = {f: clip["relative_time_us"][f - first] / 1e6 for f in frames}
        poses = {
            f: torch.from_numpy(np.loadtxt(root / "ego_pose" / f"{f:03d}.txt")).float()
            for f in frames
        }
        names = {i: name for name, i in ids.items()}
        camera_meta = {i: manifest.camera(name) for i, name in names.items()}
        self.sensors[SensorType.CAMERA] = KittiSensors(
            root, self.index.cameras, names, times, poses, cameras=camera_meta
        )
        self.sensors[SensorType.LIDAR] = KittiSensors(
            root, {0: frames}, {0: "velodyne"}, times, poses
        )
        self.boxes: list[OrientedBoxes] = []
        self.track_names: list[str] = []
        self.classes: list[str] = []
        track_path = root / "instances" / "instances_info.json"
        if track_path.exists():
            self._read_tracks(track_path, times)
        elif manifest.available.cuboids == "gt":
            raise ValueError("manifest có cuboid GT nhưng thiếu instances_info.json")

    def _read_tracks(self, path: Path, times: dict[int, float]) -> None:
        """Chuyển pose world và kích thước dài/rộng/cao của track sang OrientedBoxes."""
        lookup = {f: i for i, f in enumerate(self.index.lidar_frames)}
        for key, track in sorted(json.loads(path.read_text()).items()):
            ann = track["frame_annotations"]
            rows = [(i, f) for i, f in enumerate(ann["frame_idx"]) if f in lookup]
            if not rows:
                continue
            frame_indices = torch.tensor([lookup[f] for _, f in rows])
            pose = SE3.from_matrix(torch.tensor([ann["obj_to_world"][i] for i, _ in rows]).float())
            stamp = torch.tensor([times[f] for _, f in rows]).view(-1, 1)
            velocity = torch.zeros_like(pose.t)
            if len(rows) > 1:
                gaps = stamp[1:] - stamp[:-1]
                if not (gaps > 0).all():
                    raise ValueError(f"timestamp track không tăng: {key}")
                velocity[:-1] = (pose.t[1:] - pose.t[:-1]) / gaps
                velocity[-1] = velocity[-2]
            name = track["class_name"]
            category = 2 if name == "Pedestrian" else 4 if name == "Cyclist" else 1
            self.boxes.append(
                OrientedBoxes(
                    motion=LinearMotion(
                        pose=pose,
                        pose_time=stamp,
                        linear_velocity=velocity,
                        angular_velocity=torch.zeros_like(velocity),
                    ),
                    sizes=torch.tensor([ann["box_size"][i] for i, _ in rows]).float(),
                    category=torch.full((len(rows),), category, dtype=torch.uint8),
                    object_ids=torch.full((len(rows),), len(self.boxes), dtype=torch.int32),
                    sensor_link=BoxSensorLink(
                        sensor_type=torch.full((len(rows), 1), SensorType.LIDAR.value),
                        sensor_id=torch.zeros((len(rows), 1), dtype=torch.long),
                        sample_idx=frame_indices.view(-1, 1),
                    ),
                )
            )
            self.classes.append(name)
            self.track_names.append(key)

    @property
    def num_objects(self) -> int:
        """Đếm track quan sát được trong phần split đã chọn."""
        return len(self.boxes)

    @property
    def vehicle_ids(self) -> list[int]:
        """Trả ID liên tiếp của xe có box."""
        return [i for i, c in enumerate(self.classes) if c in {"Car", "Truck", "Tram", "Van"}]

    @property
    def human_ids(self) -> list[int]:
        """Trả ID các track người đi bộ."""
        return [i for i, c in enumerate(self.classes) if c == "Pedestrian"]

    def get_boxes(self, object_id: int) -> OrientedBoxes:
        """Trả pose đã chuyển cho track trong hệ world."""
        return self.boxes[object_id]

    def get_smpl_data(self, object_id: int) -> Optional[SMPLData]:
        """Baseline dùng box cho vật động và không có khớp SMPL."""
        return None

    def get_segmentation_loader(self) -> SensorLoader:
        """Trả loader chỉ đọc mask thuộc split."""
        return self.sky_loader

    def camera_lidar_synchronization(self) -> tuple[dict, dict, list]:
        """Ghép chỉ số camera và sweep bằng frame ingest thật của cả hai sensor."""
        return (
            {0: self.index.cameras_for_lidar},
            {
                sensor: [[(0, i)] for i in items]
                for sensor, items in self.index.lidar_for_camera.items()
            },
            [(0,)],
        )
