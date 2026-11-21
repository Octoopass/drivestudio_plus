"""Ánh xạ các mẫu camera và Velodyne của KITTI-360 sang chỉ số của worker XSIM.

ObservationIndex giữ bảng ghép camera với sweep cùng khung theo các cặp ảnh được cấp.
Worker tạo chỉ mục train trước initialization để loader chỉ có thể mở RGB và LiDAR
thuộc tập train. Chỉ mục test dùng ở bước render. Xem CONTRACTS mục 5.4.
"""

from dataclasses import dataclass

from drive2sim.core.schema import Split


@dataclass(frozen=True)
class ObservationIndex:
    """Bảng tra khung ingest của các mẫu worker được phép đọc.

    Attributes:
        cameras: Chỉ số camera trong manifest sang các khung ingest của nó theo thời gian.
        lidar_frames: Các khung có ít nhất một ảnh trong phần split đang dùng.
        cameras_for_lidar: Mỗi sweep đi với các cặp (sensor_id, sample_idx) cùng khung.
        lidar_for_camera: Chỉ số sweep cùng khung của từng mẫu trong camera.
    """

    cameras: dict[int, list[int]]
    lidar_frames: list[int]
    cameras_for_lidar: list[list[tuple[int, int]]]
    lidar_for_camera: dict[int, list[int]]

    @classmethod
    def from_split(
        cls, split: Split, camera_indices: dict[str, int], *, train: bool
    ) -> "ObservationIndex":
        """Dựng bảng hai chiều từ danh sách cặp tường minh của split.

        Args:
            split: Split đã được kiểm tra schema.
            camera_indices: Ánh xạ tên camera sang chỉ số file ingest.
            train: Chọn ảnh train cho khởi tạo và tối ưu, hoặc ảnh test cho đánh giá.

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
            raise ValueError("ánh xạ camera của artifact không khớp split")
        cameras: dict[int, list[int]] = {camera_indices[name]: [] for name in split.cameras}
        frames = sorted({frame for frame, _ in pairs})
        frame_to_sweep = {frame: i for i, frame in enumerate(frames)}
        links: list[list[tuple[int, int]]] = [[] for _ in frames]
        for frame, name in sorted(pairs):
            sensor = camera_indices[name]
            links[frame_to_sweep[frame]].append((sensor, len(cameras[sensor])))
            cameras[sensor].append(frame)
        cameras = {sensor: seq for sensor, seq in cameras.items() if seq}
        return cls(
            cameras,
            frames,
            links,
            {sensor: [frame_to_sweep[f] for f in seq] for sensor, seq in cameras.items()},
        )
