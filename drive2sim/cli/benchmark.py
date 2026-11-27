"""Lệnh ``d2s benchmark``: bảng benchmark và method thắng trong từng nhóm run.

Đọc file định nghĩa ``benchmarks/<tên>.yaml`` (nhóm run theo track, ô, cảnh, split), gom kết quả
của các run bằng ``drive2sim.engine.benchmark`` và in bảng xếp hạng. ``--out`` ghi thêm
``d2s-benchresult``. Chỉ đọc file kết quả đã có, không chạy GPU. Xem CONTRACTS mục 6.5.
"""

import argparse
from pathlib import Path

from drive2sim.cli.output import UsageError, emit, format_table, info
from drive2sim.core import files
from drive2sim.core.schema import BenchmarkChunkedGroup
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


def _chunked_block(item: BenchmarkChunkedGroup) -> str:
    """Khối chữ của một lần dựng chia chunk: số gộp, từng chunk và từng chỗ nối."""
    lines = [
        f"track {item.bench_track}, chia chunk: {item.label} ({item.scene_id}, "
        f"{len(item.chunks)} chunk, {', '.join(item.approach_ids)})",
        f"gộp: PSNR {_value(item.psnr)}, SSIM {_value(item.ssim, 3)}, LPIPS "
        f"{_value(item.lpips, 3)}, Avg {_value(item.avg_error, 4)}, R2 cc "
        f"{_value(item.shift_psnr_cc)}, quãng đường {_value(item.distance_m, 0)} m, train "
        f"{_value(item.train_time_s / 60 if item.train_time_s is not None else None, 1)} phút "
        f"({_value(item.train_s_per_100m, 0)} s/100 m), .ply {_value(item.ply_mb_per_100m, 1)} "
        "MB/100 m",
        format_table(
            ["chunk", "run_id", "khung", "PSNR", "phút", "VRAM"],
            [
                [
                    c.split_id,
                    c.run_id,
                    f"{c.frame_range[0]}–{c.frame_range[1]}",
                    _value(c.psnr),
                    _value(c.train_time_s / 60 if c.train_time_s is not None else None, 1),
                    c.peak_vram_mb or "–",
                ]
                for c in item.chunks
            ],
        ),
        format_table(
            ["nối", "đoạn chồng", "m", "PSNR quanh", "kém trung vị", "chồng: trước / sau"],
            [
                [
                    f"{s.index}|{s.index + 1}",
                    f"{s.overlap[0]}–{s.overlap[1]}" if s.overlap else "hở",
                    _value(s.at_m, 0),
                    f"{_value(s.psnr_near)} ({s.num_frames} khung)",
                    _value(s.drop_db),
                    f"{_value(s.overlap_psnr_before)} / {_value(s.overlap_psnr_after)} "
                    f"({s.overlap_frames})",
                ]
                for s in item.seams
            ],
        ),
    ]
    segments = sum(len(v) for v in item.bad_segments.values())
    lines.append(f"đoạn xấu: {segments if item.frames_measured else '–'}")
    lines.extend(f"cảnh báo: {w}" for w in item.warnings)
    return "\n".join(lines)


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
    blocks = [f"ngưỡng hoà: {result.tie_threshold_db:.2f} dB ({result.tie_threshold_source})"]
    blocks[0] += "".join(f"\ncảnh báo: {w}" for w in result.warnings)
    for group in result.groups:
        title = f"track {group.bench_track}" + (f", ô {group.cell}" if group.cell else "")
        headers = [
            "#", "run_id", "approach", "PSNR", "Avg", "pinhole", "fisheye", "R2 cc", "AbsRel",
            "R3 .ply", "gap dB", "đoạn xấu", "phút", "VRAM",
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
                (_value(r.ply_psnr) + ("" if r.export_static else "†"))
                if r.export_applicable
                else "n/a",
                _value(r.export_gap_db) if r.export_applicable else "n/a",
                sum(len(v) for v in r.bad_segments.values()) if r.frames_measured else "–",
                _value(r.train_time_s / 60 if r.train_time_s is not None else None, 1),
                r.peak_vram_mb or "–",
            ]
            for r in group.rows
        ]
        lines = [f"{title}: {group.scene_id}, {group.split_id}", format_table(headers, table)]
        decided = f", quyết định bởi {group.decided_by}" if group.decided_by else ""
        lines.append(
            f"thắng: {group.winner_run_id or '–'}{decided}  (* = ngang run tốt nhất; "
            "† = .ply chưa phải nền tĩnh, R3 không phân xử)"
        )
        lines.extend(f"cảnh báo: {w}" for w in group.warnings)
        blocks.append("\n".join(lines))
    blocks.extend(_chunked_block(item) for item in result.chunked)
    emit(args, result.model_dump(mode="json"), "\n\n".join(blocks))
    return 0
