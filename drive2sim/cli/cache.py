"""Nhóm lệnh ``d2s cache``: xem và dọn ``cache/`` (ARCHITECTURE mục 2.7).

File này chứa:
- ``register``: thêm các lệnh ``cache list`` và ``cache gc``.
- ``list_cache``: liệt kê kết quả trong cache kèm dung lượng và run nào còn dùng.
- ``gc``: xoá kết quả không run nào dùng và thư mục ``.partial`` bỏ dở, sau khi hỏi xác nhận.
"""

import argparse
from dataclasses import asdict

from drive2sim.cli.output import confirm, emit, format_table
from drive2sim.engine import housekeeping
from drive2sim.engine.workdir import Workdir


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``cache`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("cache", help="xem và dọn cache")
    actions = group.add_subparsers(dest="action", required=True)
    for name, handler, help_text in (
        ("list", list_cache, "liệt kê kết quả trong cache"),
        ("gc", gc, "xoá kết quả không run nào dùng"),
    ):
        parser = actions.add_parser(name, parents=[common], help=help_text)
        parser.add_argument("--scene", dest="scene_id")
        parser.set_defaults(handler=handler)


def _mb(size: int) -> str:
    """Viết dung lượng bằng MB, một chữ số thập phân."""
    return f"{size / 2**20:.1f}"


def list_cache(args: argparse.Namespace) -> int:
    """Liệt kê kết quả trong cache.

    Args:
        args: Tham số dòng lệnh; ``scene_id`` để lọc theo cảnh.

    Returns:
        Mã thoát ``0``.
    """
    entries = housekeeping.cache_entries(Workdir.from_env(args.workdir), args.scene_id)
    rows = [
        [
            e.cache_key + (" (.partial)" if e.partial else ""),
            _mb(e.size_bytes),
            len(e.used_by) if not e.partial else ("bỏ dở" if e.stale else "đang ghi"),
        ]
        for e in entries
    ]
    emit(
        args,
        [asdict(e) for e in entries],
        format_table(["cache_key", "MB", "số run dùng"], rows),
    )
    return 0


def gc(args: argparse.Namespace) -> int:
    """Xoá kết quả cache không run nào dùng và thư mục ``.partial`` bỏ dở.

    Args:
        args: Tham số dòng lệnh; ``scene_id`` để chỉ dọn một cảnh; ``yes`` để không hỏi.

    Returns:
        Mã thoát ``0``.
    """
    workdir = Workdir.from_env(args.workdir)
    doomed = housekeeping.gc_cache(workdir, args.scene_id, dry_run=True)
    if not doomed:
        emit(args, {"removed": []}, "không có gì để dọn")
        return 0
    total = sum(e.size_bytes for e in doomed)
    confirm(
        args,
        f"Xoá {len(doomed)} kết quả cache ({_mb(total)} MB)?",
        [e.cache_key for e in doomed],
    )
    removed = housekeeping.gc_cache(workdir, args.scene_id)
    emit(
        args,
        {"removed": [e.cache_key for e in removed]},
        f"đã xoá {len(removed)} kết quả, {_mb(total)} MB",
    )
    return 0
