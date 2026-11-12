"""Nhóm lệnh ``d2s split``: tạo và xem split train/test của một cảnh.

File này chứa:
- ``register``: thêm các lệnh ``split create``, ``split list``, ``split show`` vào parser.
- ``create_split``: tạo split ``every_nth_frame:N``, ghi vào
  ``data/<scene_id>/splits/<split_id>.json``.
- ``list_splits``, ``show_split``: liệt kê và xem split.
- ``delete_split``: xoá một split không còn run nào dùng.

Phần kiểm tra và ghi file nằm ở ``drive2sim/engine/scenes.py``, dùng chung với API của Studio;
file này chỉ đọc tham số dòng lệnh và in kết quả.
"""

import argparse
from typing import Optional

from drive2sim.cli.output import UsageError, confirm, emit, format_table, info
from drive2sim.engine import housekeeping, scenes
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

    delete = actions.add_parser("delete", parents=[common], help="xoá một split")
    delete.add_argument("scene_id")
    delete.add_argument("split_id")
    delete.set_defaults(handler=delete_split)


def _parse_frame_range(raw: Optional[str]) -> Optional[tuple[int, int]]:
    """Đọc ``--frame-range a,b``; không có thì trả ``None`` (cả cảnh).

    Raises:
        UsageError: Nếu không đúng dạng hai số nguyên.
    """
    if raw is None:
        return None
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
    try:
        split = scenes.create_split(
            workdir,
            args.scene_id,
            args.strategy,
            cameras=args.cameras.split(",") if args.cameras else None,
            frame_range=_parse_frame_range(args.frame_range),
            split_id=args.split_id,
        )
    except scenes.SplitError as error:
        raise UsageError(str(error)) from None
    info(args, f"đã ghi {workdir.relative(workdir.split_path(args.scene_id, split.split_id))}")
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
    splits = scenes.list_splits(Workdir.from_env(args.workdir), args.scene_id)
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


def delete_split(args: argparse.Namespace) -> int:
    """Xoá một split sau khi hỏi xác nhận.

    Args:
        args: Tham số dòng lệnh, có ``scene_id``, ``split_id``, ``yes``.

    Returns:
        Mã thoát ``0``.

    Raises:
        InUseError: Nếu còn run dùng split này.
    """
    workdir = Workdir.from_env(args.workdir)
    path = workdir.split_path(args.scene_id, args.split_id)
    confirm(args, f"Xoá split {args.split_id}?", [workdir.relative(path)])
    removed = housekeeping.delete_split(workdir, args.scene_id, args.split_id)
    emit(args, {"removed": removed}, f"đã xoá {', '.join(removed)}")
    return 0
