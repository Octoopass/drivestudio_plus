"""Tạo split train/test (CONTRACTS mục 3.1)."""

from collections.abc import Sequence
from typing import Optional

from drive2sim.core.schema import SCHEMA_VERSIONS, Split, parse_strategy

Pair = tuple[int, str]  # (chỉ số khung, camera_id)


def every_nth_frame_pairs(
    frame_range: tuple[int, int], cameras: Sequence[str], n: int
) -> tuple[list[Pair], list[Pair]]:
    """Tính các cặp train và test theo chiến lược ``every_nth_frame``.

    Các khung ``first + N``, ``first + 2N``, ... là test với mọi camera; các khung còn lại, kể cả
    khung đầu, là train. Quy tắc này trùng với ``split_train_test`` dùng ``test_image_stride``
    trong ``drive2sim/recon/data/driving_dataset.py``, để split của tool và của DriveStudio
    khớp nhau.

    Args:
        frame_range: Khung đầu và khung cuối của đoạn, ``(first, last)``, tính cả hai đầu.
        cameras: Danh sách ``camera_id`` dùng trong split.
        n: Khoảng cách giữa hai khung test, ``n >= 1``.

    Returns:
        Bộ ``(train, test)``, mỗi phần là danh sách cặp ``(chỉ số khung, camera_id)``.

    Raises:
        ValueError: Nếu ``n < 1``.
    """
    first, last = frame_range
    if n < 1:
        raise ValueError("n must be >= 1")
    test_frames = set(range(first + n, last + 1, n))
    train = [
        (frame, camera)
        for frame in range(first, last + 1)
        if frame not in test_frames
        for camera in cameras
    ]
    test = [(frame, camera) for frame in sorted(test_frames) for camera in cameras]
    return train, test


def make_split(
    scene_id: str,
    cameras: Sequence[str],
    frame_range: tuple[int, int],
    strategy: str,
    split_id: Optional[str] = None,
) -> Split:
    """Tạo một split theo chiến lược ``every_nth_frame:N``.

    Split kiểu ``custom`` được viết tay, không tạo bằng hàm này.

    Args:
        scene_id: Định danh cảnh.
        cameras: Danh sách ``camera_id`` dùng trong split.
        frame_range: Khung đầu và khung cuối của đoạn, tính cả hai đầu.
        strategy: Chuỗi chiến lược, ví dụ ``"every_nth_frame:8"``.
        split_id: Tên split; mặc định là ``every<N>_cams<số camera>``.

    Returns:
        Split đã kiểm tra theo schema.

    Raises:
        ValueError: Nếu chiến lược không phải ``every_nth_frame`` hoặc dữ liệu sai schema.
    """
    strategy_name, n = parse_strategy(strategy)
    if strategy_name != "every_nth_frame" or n is None:
        raise ValueError(
            "make_split builds every_nth_frame splits only; write custom splits by hand"
        )
    train, test = every_nth_frame_pairs(frame_range, cameras, n)
    return Split(
        schema_version=SCHEMA_VERSIONS["split"],
        split_id=split_id or f"every{n}_cams{len(cameras)}",
        scene_id=scene_id,
        strategy=strategy,
        frame_range=frame_range,
        cameras=list(cameras),
        train=train,
        test=test,
    )


def matches_strategy(split: Split) -> bool:
    """Kiểm tra một split ``every_nth_frame`` có liệt kê đúng các cặp mà chiến lược sinh ra.

    Dùng trước khi giao split cho DriveStudio: DriveStudio tự chia lại theo
    ``test_image_stride`` nên split ghi trong file phải trùng với cách chia đó.

    Args:
        split: Split cần kiểm tra.

    Returns:
        ``True`` nếu khớp, hoặc nếu split là kiểu ``custom``.
    """
    strategy_name, n = parse_strategy(split.strategy)
    if strategy_name != "every_nth_frame" or n is None:
        return True
    train, test = every_nth_frame_pairs(split.frame_range, split.cameras, n)
    return set(train) == set(split.train) and set(test) == set(split.test)
