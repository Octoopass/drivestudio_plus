"""Ánh xạ các cặp (khung, camera) của split và các lượt quét LiDAR sang chỉ số của XSIM.

ObservationIndex giữ bảng ghép camera với sweep cùng khung. Worker tạo chỉ mục train trước
initialization để loader chỉ mở được ảnh thuộc tập train; LiDAR theo chính sách của run: mọi khung
có LiDAR (như DriveStudio khởi tạo từ LiDAR của cả đoạn) hoặc chỉ khung train. Chỉ mục test dùng ở
bước render. Xem CONTRACTS mục 5.4.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from drive2sim.core.schema import Split


@dataclass(frozen=True)
class ObservationIndex:
    """Bảng tra khung của các mẫu worker được phép đọc.

    Attributes:
        cameras: Chỉ số camera trong manifest → các khung của nó theo thời gian.
        lidar_frames: Các khung có sweep được dùng, tăng dần.
        cameras_for_lidar: Mỗi sweep đi với các cặp (sensor_id, sample_idx) cùng khung.
        lidar_for_camera: Chỉ số sweep cùng khung của từng mẫu camera, ``None`` khi khung đó
            không có sweep.
    """

    cameras: dict[int, list[int]]
    lidar_frames: list[int]
    cameras_for_lidar: list[list[tuple[int, int]]]
    lidar_for_camera: dict[int, list[Optional[int]]]

    @classmethod
    def from_split(
        cls,
        split: Split,
        camera_indices: dict[str, int],
        *,
        train: bool,
        lidar_frames: Optional[Sequence[int]] = None,
    ) -> "ObservationIndex":
        """Dựng bảng hai chiều từ danh sách cặp tường minh của split.

        Args:
            split: Split đã được kiểm tra schema.
            camera_indices: Ánh xạ tên camera sang chỉ số manifest.
            train: Chọn ảnh train cho khởi tạo và tối ưu, hoặc ảnh test cho đánh giá.
            lidar_frames: Các khung có sweep được phép dùng; ``None`` là đúng các khung có ảnh
                trong phần split đã chọn.

        Returns:
            Chỉ mục gồm những camera có ít nhất một mẫu trong phần split đã chọn.

        Raises:
            ValueError: Nếu thiếu ảnh, thiếu camera hoặc chỉ số bị trùng.
        """
        pairs = split.train if train else split.test
        if not pairs:
            raise ValueError("phần split XSIM đã chọn không có ảnh")
        if set(camera_indices) != set(split.cameras) or len(set(camera_indices.values())) != len(
            camera_indices
        ):
            raise ValueError("ánh xạ camera của gói không khớp split")
        cameras: dict[int, list[int]] = {camera_indices[name]: [] for name in split.cameras}
        sweeps = sorted(
            set(lidar_frames) if lidar_frames is not None else {frame for frame, _ in pairs}
        )
        frame_to_sweep = {frame: i for i, frame in enumerate(sweeps)}
        links: list[list[tuple[int, int]]] = [[] for _ in sweeps]
        for frame, name in sorted(pairs):
            sensor = camera_indices[name]
            if frame in frame_to_sweep:
                links[frame_to_sweep[frame]].append((sensor, len(cameras[sensor])))
            cameras[sensor].append(frame)
        cameras = {sensor: seq for sensor, seq in cameras.items() if seq}
        return cls(
            cameras,
            sweeps,
            links,
            {sensor: [frame_to_sweep.get(f) for f in seq] for sensor, seq in cameras.items()},
        )
