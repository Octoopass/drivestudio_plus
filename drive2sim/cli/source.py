"""Nhóm lệnh ``d2s source``: xem trước thư mục dữ liệu gốc, trước khi ingest.

File này chứa:
- ``register``: thêm lệnh ``source inspect`` vào parser.
- ``inspect``: in kết quả của ``drive2sim.ingest.detect.inspect_source`` và gợi ý lệnh ingest.

Lệnh chỉ đọc, không ghi gì (CONTRACTS mục 10.4).
"""

import argparse
from pathlib import Path

from drive2sim.cli.output import emit, format_table
from drive2sim.ingest.detect import SourceReport, inspect_source


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``source`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("source", help="xem trước thư mục dữ liệu gốc")
    actions = group.add_subparsers(dest="action", required=True)
    inspect_parser = actions.add_parser(
        "inspect", parents=[common], help="nhận ra dataset, báo có gì, gợi ý lệnh ingest"
    )
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.set_defaults(handler=inspect)


def _yes_no(value: object) -> str:
    """Viết ``True``/``False``/``None`` thành ``có``/``không``/``không rõ``."""
    return {True: "có", False: "không"}.get(value, "không rõ")  # type: ignore[call-overload]


def _suggestion(report: SourceReport) -> str:
    """Dựng lệnh ingest gợi ý cho một nguồn đã nhận ra."""
    command = f"d2s scene ingest {report.path} --source-type {report.dataset}"
    for version, sequences in report.versions.items():
        if sequences:
            if version:
                command += f" --version {version}"
            return command + f" --sequence {sequences[0]}"
    return command


def inspect(args: argparse.Namespace) -> int:
    """In những gì có trong một thư mục dữ liệu gốc.

    Args:
        args: Tham số dòng lệnh, có ``path``.

    Returns:
        Mã thoát ``0``.

    Raises:
        FileNotFoundError: Nếu thư mục không tồn tại; ``main`` đổi thành lỗi ``not_found``.
    """
    report = inspect_source(args.path)
    lines = [f"nhận ra:  {report.dataset or 'không nhận ra'}"]
    for version, sequences in report.versions.items():
        shown = ", ".join(sequences[:5]) + (", ..." if len(sequences) > 5 else "")
        label = f"{version}: " if version else ""
        lines.append(f"chuỗi:    {label}{len(sequences)} ({shown})")
    if report.cameras:
        lines.append(f"camera:   {len(report.cameras)} ({', '.join(report.cameras)})")
    if report.dataset:
        lines.append(f"LiDAR:    {_yes_no(report.has_lidar)}")
        lines.append(f"box 3D:   {_yes_no(report.has_cuboids)}")
    if report.inventory:
        images = [item for item in report.inventory if item["kind"] == "images"]
        others = [item for item in report.inventory if item["kind"] == "files"]
        if images:
            rows = [
                [i["path"], i["count"], "x".join(map(str, i["resolution"] or ["?"]))]
                for i in images
            ]
            lines.append(format_table(["thư mục ảnh", "số ảnh", "độ phân giải"], rows))
        if others:
            lines.append(
                "file khác: " + ", ".join(f"{i['suffix']} ({i['count']})" for i in others)
            )
    lines += report.notes
    if report.ingest_supported:
        lines.append(f"lệnh tiếp theo:\n  {_suggestion(report)}")
    emit(args, report.to_dict(), "\n".join(lines))
    return 0
