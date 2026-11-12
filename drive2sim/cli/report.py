"""Lệnh ``d2s report``: bảng so sánh nhiều run (SPEC mục 8 nhóm D, mục 9).

In bảng các run cùng cảnh và split, xếp theo PSNR; ``--csv`` để xuất cho báo cáo. Dùng chung
``drive2sim.engine.report`` với trang So sánh của Studio.
"""

import argparse
from dataclasses import asdict

from drive2sim.cli.output import emit, format_table
from drive2sim.engine.report import report_rows, rows_to_csv
from drive2sim.engine.workdir import Workdir


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm lệnh ``report`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    parser = subparsers.add_parser("report", parents=[common], help="bảng so sánh các run")
    parser.add_argument("run_ids", nargs="*", metavar="run_id", help="chỉ lấy các run này")
    parser.add_argument("--scene", dest="scene_id")
    parser.add_argument("--split", dest="split_id")
    parser.add_argument("--csv", action="store_true", help="in CSV ra stdout")
    parser.set_defaults(handler=report)


def _value(value: object, digits: int = 2) -> str:
    """Viết số với ``digits`` chữ số thập phân; ``None`` thành ``–``."""
    return "–" if value is None else f"{value:.{digits}f}"


def report(args: argparse.Namespace) -> int:
    """In bảng so sánh các run.

    Args:
        args: Tham số dòng lệnh: ``run_ids``, ``scene_id``, ``split_id``, ``csv``.

    Returns:
        Mã thoát ``0``.
    """
    rows = report_rows(
        Workdir.from_env(args.workdir), args.scene_id, args.split_id, args.run_ids or None
    )
    if args.csv:
        print(rows_to_csv(rows), end="")
        return 0
    table = [
        [
            r.run_id,
            r.split_id,
            r.approach_id,
            ", ".join(f"{k}={v}" for k, v in r.options.items()),
            _value(r.psnr),
            _value(r.ssim, 3),
            _value(r.lpips, 3),
            _value(r.train_time_s / 60 if r.train_time_s is not None else None, 1),
            r.peak_vram_mb or "–",
        ]
        for r in rows
    ]
    headers = ["run_id", "split", "approach", "ô chọn", "PSNR", "SSIM", "LPIPS", "phút", "VRAM"]
    emit(args, [asdict(r) for r in rows], format_table(headers, table))
    return 0
