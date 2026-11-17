"""Lệnh ``d2s pack``: kiểm kê hoặc gói các run đầy đủ để bàn giao (CONTRACTS mục 10.4)."""

import argparse
from pathlib import Path

from drive2sim.cli.output import CommandError, emit
from drive2sim.engine.pack import plan_pack, write_pack
from drive2sim.engine.workdir import Workdir


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Đăng ký lệnh đơn pack, nhận run và file tar đích.

    Args:
        subparsers: Parser của các lệnh cấp đầu, do CLI main tạo.
        common: Tuỳ chọn chung như workdir, JSON và mức chi tiết.
    """
    parser = subparsers.add_parser(
        "pack", parents=[common], help="gói run, cảnh và cache để bàn giao"
    )
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="file .tar đích (không ghi đè)")
    parser.add_argument("--dry-run", action="store_true", help="chỉ kiểm kê, không ghi gói")
    parser.set_defaults(handler=pack)


def pack(args: argparse.Namespace) -> int:
    """Kiểm kê, ghi gói nếu được yêu cầu, rồi báo dung lượng cho người dùng.

    Args:
        args: Có run_ids, output, dry_run và các tuỳ chọn chung của CLI.

    Returns:
        Mã thoát 0 nếu kiểm kê hoặc đóng gói thành công.

    Raises:
        CommandError: Nếu file đích đã có, với mã lỗi conflict.
        ValueError: Nếu run đang chạy, symlink không hợp lệ hoặc không đủ chỗ.
        FileNotFoundError: Nếu dữ liệu cảnh hoặc cache chưa có đầy đủ trên máy.
    """
    plan = plan_pack(Workdir.from_env(args.workdir), args.run_ids)
    try:
        result = plan.to_dict() if args.dry_run else write_pack(plan, args.output)
    except FileExistsError as error:
        raise CommandError(str(error), code="conflict") from error
    result["dry_run"] = args.dry_run
    emit(
        args,
        result,
        f"{len(plan.run_ids)} run, {result['file_count']} file, "
        f"{result['size_bytes'] / 2**30:.2f} GiB"
        + (" (chỉ kiểm kê)" if args.dry_run else f" → {args.output}"),
    )
    return 0
