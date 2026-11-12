"""Nhóm lệnh ``d2s split``: tạo và xem split train/test của một cảnh.

File này chứa:
- ``register``: thêm các lệnh ``split create``, ``split list``, ``split show`` vào parser.
- ``create_split``: tạo split ``every_nth_frame:N``, ghi vào
  ``data/<scene_id>/splits/<split_id>.json``.
- ``list_splits``, ``show_split``: liệt kê và xem split.
- ``default_split_id``: đặt tên split theo CONTRACTS mục 10.1.
"""

import argparse
from typing import Optional

from drive2sim.cli.output import UsageError, emit, format_table, info
from drive2sim.core import files
from drive2sim.core.schema import Manifest, Split, parse_strategy
from drive2sim.core.split import make_split
from drive2sim.engine.workdir import Workdir


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``split`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("split", help="tạo và xem split train/test")
    actions = group.add_subparsers(dest="action", required=True)

    create = actions.add_parser("create", parents=[common], help="tạo split every_nth_frame")
    create.add_argument("scene_id")
    create.add_argument("--strategy", required=True, help="ví dụ every_nth_frame:8")
    create.add_argument("--cameras", help="camera_id cách nhau bằng dấu phẩy; mặc định mọi camera")
    create.add_argument("--frame-range", help="khung đầu,khung cuối; mặc định cả cảnh")
    create.add_argument("--split", dest="split_id", help="tên split; mặc định theo CONTRACTS 10.1")
    create.set_defaults(handler=create_split)

    list_parser = actions.add_parser("list", parents=[common], help="liệt kê split của một cảnh")
    list_parser.add_argument("scene_id")
    list_parser.set_defaults(handler=list_splits)

    show = actions.add_parser("show", parents=[common], help="xem một split")
    show.add_argument("scene_id")
    show.add_argument("split_id")
    show.set_defaults(handler=show_split)


def default_split_id(strategy: str, cameras: list[str], manifest: Manifest) -> str:
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


def _parse_frame_range(raw: Optional[str], num_frames: int) -> tuple[int, int]:
    """Đọc ``--frame-range a,b``; mặc định là cả cảnh ``0,num_frames-1``.

    Raises:
        UsageError: Nếu không đúng dạng hai số nguyên.
    """
    if raw is None:
        return (0, num_frames - 1)
    parts = raw.split(",")
    if len(parts) != 2 or not all(part.strip().lstrip("-").isdigit() for part in parts):
        raise UsageError(f"--frame-range {raw!r}: cần dạng <khung đầu>,<khung cuối>")
    return (int(parts[0]), int(parts[1]))


def create_split(args: argparse.Namespace) -> int:
    """Tạo split ``every_nth_frame:N`` cho một cảnh và ghi ra file.

    Args:
        args: Tham số dòng lệnh: ``scene_id``, ``strategy``, ``cameras``, ``frame_range``,
            ``split_id``.

    Returns:
        Mã thoát ``0``.

    Raises:
        UsageError: Nếu camera không có trong cảnh, đoạn khung sai, hoặc split đã tồn tại.
        WorkdirError: Nếu cảnh chưa ingest.
    """
    workdir = Workdir.from_env(args.workdir)
    scene = workdir.load_scene(args.scene_id)
    all_cameras = [camera.camera_id for camera in scene.manifest.cameras]
    cameras = args.cameras.split(",") if args.cameras else all_cameras
    unknown = [camera for camera in cameras if camera not in all_cameras]
    if unknown:
        raise UsageError(
            f"camera không có trong cảnh: {', '.join(unknown)}; cảnh có: {', '.join(all_cameras)}"
        )
    frame_range = _parse_frame_range(args.frame_range, scene.manifest.timeline.num_frames)
    if not 0 <= frame_range[0] <= frame_range[1] < scene.manifest.timeline.num_frames:
        raise UsageError(
            f"--frame-range {frame_range[0]},{frame_range[1]} nằm ngoài cảnh "
            f"(0 tới {scene.manifest.timeline.num_frames - 1})"
        )
    try:
        split_id = args.split_id or default_split_id(args.strategy, cameras, scene.manifest)
        split = make_split(args.scene_id, cameras, frame_range, args.strategy, split_id)
    except ValueError as error:
        raise UsageError(f"không tạo được split: {error}") from None
    path = workdir.split_path(args.scene_id, split.split_id)
    if path.exists():
        raise UsageError(f"split {split.split_id} đã có; đặt tên khác bằng --split")
    files.dump(split, path)
    info(args, f"đã ghi {workdir.relative(path)}")
    emit(
        args,
        split.model_dump(mode="json"),
        f"{split.split_id}: {len(split.train)} ảnh train, {len(split.test)} ảnh test",
    )
    return 0


def list_splits(args: argparse.Namespace) -> int:
    """Liệt kê các split của một cảnh.

    Args:
        args: Tham số dòng lệnh, có ``scene_id``.

    Returns:
        Mã thoát ``0``.
    """
    workdir = Workdir.from_env(args.workdir)
    workdir.load_scene(args.scene_id)
    splits = [
        files.load(path, Split)
        for path in sorted((workdir.scene_dir(args.scene_id) / "splits").glob("*.json"))
    ]
    rows = [[s.split_id, s.strategy, len(s.cameras), len(s.train), len(s.test)] for s in splits]
    emit(
        args,
        [{"split_id": s.split_id, "strategy": s.strategy, "cameras": s.cameras} for s in splits],
        format_table(["split_id", "chiến lược", "camera", "train", "test"], rows),
    )
    return 0


def show_split(args: argparse.Namespace) -> int:
    """In một split.

    Args:
        args: Tham số dòng lệnh, có ``scene_id`` và ``split_id``.

    Returns:
        Mã thoát ``0``.
    """
    split = Workdir.from_env(args.workdir).load_split(args.scene_id, args.split_id)
    test_frames = sorted({frame for frame, _ in split.test})
    text = "\n".join(
        [
            f"split      {split.split_id} ({split.strategy})",
            f"khung      {split.frame_range[0]} tới {split.frame_range[1]}",
            f"camera     {', '.join(split.cameras)}",
            f"train      {len(split.train)} ảnh",
            f"test       {len(split.test)} ảnh, khung {test_frames}",
        ]
    )
    emit(args, split.model_dump(mode="json"), text)
    return 0
