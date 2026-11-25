"""Lệnh ``d2s benchmark``: bảng benchmark và method thắng trong từng nhóm run.

Đọc file định nghĩa ``benchmarks/<tên>.yaml`` (nhóm run theo track, ô, cảnh, split), gom kết quả
của các run bằng ``drive2sim.engine.benchmark`` và in bảng xếp hạng. ``--out`` ghi thêm
``d2s-benchresult``. Chỉ đọc file kết quả đã có, không chạy GPU. Xem CONTRACTS mục 6.5.
"""

import argparse
from pathlib import Path

from drive2sim.cli.output import UsageError, emit, format_table, info
from drive2sim.core import files
from drive2sim.engine.benchmark import run_benchmark
from drive2sim.engine.workdir import Workdir


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm lệnh ``benchmark`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    parser = subparsers.add_parser(
        "benchmark", parents=[common], help="bảng benchmark theo file định nghĩa"
    )
    parser.add_argument("definition", type=Path, help="file benchmarks/<tên>.yaml")
    parser.add_argument("--out", type=Path, help="ghi kết quả (d2s-benchresult) ra file JSON")
    parser.set_defaults(handler=benchmark)


def _value(value: object, digits: int = 2) -> str:
    """Viết số với ``digits`` chữ số thập phân; ``None`` thành ``–`` (chưa đo)."""
    return "–" if value is None else f"{value:.{digits}f}"


def benchmark(args: argparse.Namespace) -> int:
    """In bảng benchmark của từng nhóm run.

    Args:
        args: Tham số dòng lệnh: ``definition``, ``out``.

    Returns:
        Mã thoát ``0``.

    Raises:
        UsageError: Nếu file định nghĩa sai hoặc một run không hợp lệ.
    """
    try:
        result = run_benchmark(Workdir.from_env(args.workdir), args.definition)
    except (OSError, ValueError) as error:
        raise UsageError(str(error)) from error
    if args.out:
        files.dump(result, args.out)
        info(args, f"đã ghi {args.out}")
    blocks = []
    for group in result.groups:
        title = f"track {group.bench_track}" + (f", ô {group.cell}" if group.cell else "")
        headers = [
            "#", "run_id", "approach", "PSNR", "Avg", "pinhole", "fisheye", "R2 cc", "AbsRel",
            "gap dB", "đoạn xấu", "phút", "VRAM",
        ]  # fmt: skip
        table = [
            [
                f"{r.rank}{'*' if r.tied_with_best and r.rank > 1 else ''}",
                r.run_id,
                r.approach_id,
                _value(r.psnr),
                _value(r.avg_error, 4),
                _value(r.psnr_pinhole),
                _value(r.psnr_fisheye),
                _value(r.shift_psnr_cc),
                _value(r.depth.abs_rel if r.depth else None, 3),
                _value(r.export_gap_db) if r.export_applicable else "n/a",
                sum(len(v) for v in r.bad_segments.values()) if r.frames_measured else "–",
                _value(r.train_time_s / 60 if r.train_time_s is not None else None, 1),
                r.peak_vram_mb or "–",
            ]
            for r in group.rows
        ]
        lines = [f"{title}: {group.scene_id}, {group.split_id}", format_table(headers, table)]
        decided = f", quyết định bởi {group.decided_by}" if group.decided_by else ""
        lines.append(f"thắng: {group.winner_run_id or '–'}{decided}  (* = ngang run tốt nhất)")
        lines.extend(f"cảnh báo: {w}" for w in group.warnings)
        blocks.append("\n".join(lines))
    emit(args, result.model_dump(mode="json"), "\n\n".join(blocks))
    return 0
