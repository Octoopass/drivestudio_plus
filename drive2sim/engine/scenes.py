"""Liệt kê cảnh, liệt kê và tạo split trong thư mục làm việc.

CLI (``d2s scene list``, ``d2s split create/list``) và API của Studio (CONTRACTS mục 11) cùng gọi
các hàm ở đây, để hai nơi luôn kiểm tra giống nhau và báo cùng một lỗi.

File này chứa:
- ``list_scenes``: đọc manifest của mọi cảnh trong ``data/``.
- ``list_splits``: đọc mọi split của một cảnh.
- ``default_split_id``: đặt tên split theo CONTRACTS mục 10.1.
- ``create_split``: kiểm tra yêu cầu, tạo split ``every_nth_frame:N`` và ghi ra file.
- ``SplitError``, ``SplitExistsError``: lỗi khi yêu cầu tạo split sai hoặc split đã có.
"""

from collections.abc import Sequence
from typing import Optional

from drive2sim.core import files
from drive2sim.core.schema import Manifest, Split, parse_strategy
from drive2sim.core.split import make_split
from drive2sim.engine.workdir import Workdir


class SplitError(ValueError):
    """Yêu cầu tạo split sai: camera không có, đoạn khung sai, chiến lược sai.

    Ứng với mã lỗi ``invalid_request`` (CONTRACTS mục 11.3).
    """


class SplitExistsError(SplitError):
    """Split cùng tên đã có; ứng với mã lỗi ``conflict``."""


def list_scenes(workdir: Workdir) -> list[Manifest]:
    """Đọc manifest của mọi cảnh trong ``data/``.

    Args:
        workdir: Thư mục làm việc.

    Returns:
        Manifest của các cảnh, sắp theo ``scene_id``.

    Raises:
        ValueError: Nếu một manifest sai schema.
    """
    return [
        files.load(path, Manifest) for path in sorted(workdir.data_dir.glob("*/manifest.yaml"))
    ]


def list_splits(workdir: Workdir, scene_id: str) -> list[Split]:
    """Đọc mọi split của một cảnh.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Định danh cảnh.

    Returns:
        Các split, sắp theo ``split_id``.

    Raises:
        WorkdirError: Nếu cảnh chưa ingest.
    """
    workdir.load_scene(scene_id)
    return [
        files.load(path, Split)
        for path in sorted((workdir.scene_dir(scene_id) / "splits").glob("*.json"))
    ]


def default_split_id(strategy: str, cameras: Sequence[str], manifest: Manifest) -> str:
    """Đặt tên split theo CONTRACTS mục 10.1.

    Args:
        strategy: Chuỗi chiến lược, ví dụ ``every_nth_frame:8``.
        cameras: Camera dùng trong split.
        manifest: Manifest của cảnh.

    Returns:
        ``every<N>_allcams`` khi dùng mọi camera của cảnh, ngược lại ``every<N>_<k>cams``, ví dụ
        ``every8_3cams``.
    """
    _, n = parse_strategy(strategy)
    all_cameras = [camera.camera_id for camera in manifest.cameras]
    group = "allcams" if sorted(cameras) == sorted(all_cameras) else f"{len(cameras)}cams"
    return f"every{n}_{group}"


def create_split(
    workdir: Workdir,
    scene_id: str,
    strategy: str,
    cameras: Optional[Sequence[str]] = None,
    frame_range: Optional[tuple[int, int]] = None,
    split_id: Optional[str] = None,
) -> Split:
    """Tạo split ``every_nth_frame:N`` cho một cảnh và ghi vào ``data/<scene_id>/splits/``.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Định danh cảnh.
        strategy: Chuỗi chiến lược, ví dụ ``every_nth_frame:8``.
        cameras: Camera dùng trong split; mặc định mọi camera của cảnh.
        frame_range: Khung đầu và khung cuối, tính cả hai đầu; mặc định cả cảnh.
        split_id: Tên split; mặc định theo ``default_split_id``.

    Returns:
        Split đã ghi.

    Raises:
        WorkdirError: Nếu cảnh chưa ingest.
        SplitError: Nếu camera không có trong cảnh, đoạn khung nằm ngoài cảnh, hoặc chiến lược
            sai; thông báo tiếng Việt nói rõ giá trị hợp lệ.
        SplitExistsError: Nếu split cùng tên đã có.
    """
    scene = workdir.load_scene(scene_id)
    num_frames = scene.manifest.timeline.num_frames
    all_cameras = [camera.camera_id for camera in scene.manifest.cameras]
    cameras = list(cameras) if cameras else all_cameras
    unknown = [camera for camera in cameras if camera not in all_cameras]
    if unknown:
        raise SplitError(
            f"camera không có trong cảnh: {', '.join(unknown)}; cảnh có: {', '.join(all_cameras)}"
        )
    frame_range = frame_range or (0, num_frames - 1)
    if not 0 <= frame_range[0] <= frame_range[1] < num_frames:
        raise SplitError(
            f"đoạn khung {frame_range[0]},{frame_range[1]} nằm ngoài cảnh (0 tới {num_frames - 1})"
        )
    try:
        split_id = split_id or default_split_id(strategy, cameras, scene.manifest)
        split = make_split(scene_id, cameras, frame_range, strategy, split_id)
    except ValueError as error:
        raise SplitError(f"không tạo được split: {error}") from None
    path = workdir.split_path(scene_id, split.split_id)
    if path.exists():
        raise SplitExistsError(f"split {split.split_id} đã có; đặt tên khác")
    files.dump(split, path)
    return split
