"""Nhóm lệnh ``d2s libraries``: xem và cài thư viện, môi trường trên máy này (CONTRACTS mục 10.4).

Cùng danh mục và cách cài với trang Thư viện của Studio (``drive2sim/engine/libraries.py``,
``InstallFlow``), dùng được trên máy chủ không màn hình:

    d2s libraries list
    d2s libraries install sdk-nuscenes anthropic
    d2s libraries install --all     # như nút "Cài tất cả": mọi mục còn thiếu, bỏ Waymo, nuPlan

Log in ra màn hình và ghi vào ``<thư mục làm việc>/logs/install-<mục>.log``.
"""

import argparse
import sys

from drive2sim.cli.output import UsageError, emit
from drive2sim.engine import libraries
from drive2sim.engine.env_install import InstallFlow
from drive2sim.engine.workdir import Workdir

MARK = {"ok": "đủ", "partial": "thiếu một phần", "missing": "chưa có"}


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``libraries`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("libraries", help="xem và cài thư viện, môi trường")
    actions = group.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list", parents=[common], help="danh mục và trạng thái")
    listing.set_defaults(handler=list_items)
    install = actions.add_parser("install", parents=[common], help="cài một số mục hoặc tất cả")
    install.add_argument("items", nargs="*", help="id mục (xem d2s libraries list)")
    install.add_argument("--all", action="store_true", help="mọi mục còn thiếu, như Studio")
    install.add_argument("--skip", default="", help="id mục bỏ qua, cách nhau bằng dấu phẩy")
    install.set_defaults(handler=install_items)


def list_items(args: argparse.Namespace) -> int:
    """In danh mục kèm trạng thái."""
    data = libraries.catalog()
    lines = [
        f"{item['id']:<16} {MARK[item['status']]:<15} {item['name']}"
        + ("" if item["bulk"] or not item["installable"] else "  (không nằm trong --all)")
        for item in data["items"]
    ]
    emit(args, data, "\n".join(lines))
    return 0


def install_items(args: argparse.Namespace) -> int:
    """Cài các mục đã chọn (hoặc ``--all``) lần lượt, in log, tóm tắt kết quả.

    Returns:
        ``0`` nếu không mục nào lỗi, ``1`` nếu có.

    Raises:
        UsageError: Không chọn mục nào, hoặc id lạ.
    """
    skip = {i for i in args.skip.split(",") if i}
    ids = [i for i in (libraries.bulk_ids() if args.all else list(args.items)) if i not in skip]
    if not ids:
        raise UsageError("chọn mục cần cài hoặc --all (xem d2s libraries list)")
    unknown = [i for i in ids if i not in libraries.BY_ID]
    if unknown:
        raise UsageError(f"không có mục {', '.join(unknown)} (xem d2s libraries list)")
    workdir = Workdir.from_env(args.workdir)
    flow = InstallFlow(workdir.root / "logs")
    if not args.quiet:
        flow.echo = lambda line: (sys.stderr.write(line), sys.stderr.flush())
    flow.start_all(ids)
    view = flow.wait()
    summary = view["all"]
    lines = [
        f"{i['id']:<16} {i['result']}" + (f": {i['reason']}" if i["reason"] else "")
        for i in summary["items"]
    ]
    emit(args, summary, "\n".join(lines))
    return 1 if summary["state"] == "failed" else 0
