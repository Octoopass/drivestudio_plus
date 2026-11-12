"""Điểm vào của lệnh ``d2s``: dựng parser, gọi lệnh, đổi lỗi thành mã thoát.

Lệnh có dạng ``d2s <nhóm> <hành động> [tuỳ chọn]`` (CONTRACTS mục 10.4). File này chứa:
- ``build_parser``: dựng parser với mọi nhóm lệnh (``source``, ``scene``, ``split``,
  ``approach``, ``run``, ``report``, ``export``, ``cache``, ``kaggle``, ``studio``,
  ``schema``) và tuỳ chọn chung ``--workdir``, ``--json``, ``--verbose``, ``--quiet``,
  ``--yes``.
- ``main``: phân tích dòng lệnh, nạp mọi stage, gọi lệnh, và đổi lỗi thành mã thoát:
  ``0`` thành công; ``1`` lỗi khi chạy; ``2`` sai cú pháp hoặc sai giá trị; ``3`` bị khoá vì
  không thoả điều kiện hoặc ràng buộc; ``130`` bị ngắt bằng Ctrl-C.

Thông báo lỗi in tiếng Việt ra stderr; khi có ``--json`` thì in ra stdout dạng
``{"error": {"code", "message", "details"}}`` (CONTRACTS mục 11.3). Lỗi kỹ thuật tiếng Anh từ
``drive2sim.core`` được bọc thêm câu tiếng Việt (CONTRACTS mục 10.6).

``pyproject.toml`` khai báo lệnh ``d2s`` trỏ tới ``main``.
"""

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from drive2sim.cli import (
    approach,
    cache,
    export,
    kaggle,
    report,
    run,
    scene,
    schema,
    source,
    split,
    studio,
)
from drive2sim.cli.output import CommandError, UsageError
from drive2sim.engine.cache import CacheBusyError
from drive2sim.engine.housekeeping import InUseError
from drive2sim.engine.registry import load_builtin_stages
from drive2sim.engine.resolve import BlockedError, ResolveError
from drive2sim.engine.workdir import WorkdirError
from drive2sim.kaggle.client import KaggleError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3
EXIT_INTERRUPTED = 130


def _version() -> str:
    """Trả về phiên bản của package ``drive2sim`` đã cài, hoặc ``"unknown"``."""
    try:
        from importlib import metadata

        return metadata.version("drive2sim")
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Dựng parser của ``d2s``.

    Tuỳ chọn chung nằm trong một parser cha gắn vào từng lệnh con, nên viết sau tên lệnh:
    ``d2s run list --json``. Không gắn vào parser gốc, vì ``argparse`` để giá trị mặc định của
    lệnh con đè lên giá trị đặt ở parser gốc.

    Returns:
        Parser hoàn chỉnh.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workdir", type=Path, help="thư mục chứa data/, cache/, runs/ (mặc định D2S_WORKDIR)"
    )
    common.add_argument("--json", action="store_true", help="in kết quả dạng JSON ra stdout")
    common.add_argument("--verbose", action="store_true", help="in thêm metric và traceback")
    common.add_argument("--quiet", action="store_true", help="không in tiến độ ra stderr")
    common.add_argument("--yes", action="store_true", help="không hỏi xác nhận khi xoá")

    parser = argparse.ArgumentParser(
        prog="d2s",
        description="drive2sim: dựng cảnh lái xe thành cảnh 3D Gaussian Splatting",
    )
    parser.add_argument("--version", action="version", version=f"d2s {_version()}")
    groups = parser.add_subparsers(dest="group", required=True)
    for module in (
        source,
        scene,
        split,
        approach,
        run,
        report,
        export,
        cache,
        kaggle,
        studio,
        schema,
    ):
        module.register(groups, common)
    return parser


def _print_error(
    args: Optional[argparse.Namespace], code: str, message: str, details: Optional[dict] = None
) -> None:
    """In một lỗi: JSON ra stdout khi có ``--json``, chữ ra stderr khi không."""
    if args is not None and getattr(args, "json", False):
        payload = {"error": {"code": code, "message": message, "details": details or {}}}
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"lỗi: {message}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Chạy một lệnh ``d2s`` và trả mã thoát.

    Args:
        argv: Các đối số dòng lệnh, không có tên chương trình; mặc định lấy từ ``sys.argv``.

    Returns:
        Mã thoát theo CONTRACTS mục 10.4.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_info:  # argparse thoát với mã 2 khi sai cú pháp, 0 khi --help
        return int(exit_info.code or 0)

    try:
        load_builtin_stages()
        return int(args.handler(args))
    except BlockedError as error:
        details = {
            "reasons": [
                {"message": r.message, "option": r.option, "choice": r.choice}
                for r in error.reasons
            ]
        }
        lines = "\n".join(f"  - {reason.message}" for reason in error.reasons)
        _print_error(args, "blocked", f"approach không chạy được:\n{lines}", details)
        return EXIT_BLOCKED
    except (ResolveError, UsageError) as error:
        _print_error(args, "invalid_request", str(error))
        return EXIT_USAGE
    except (WorkdirError, FileNotFoundError) as error:
        _print_error(args, "not_found", str(error))
        return EXIT_ERROR
    except InUseError as error:
        _print_error(args, "conflict", str(error))
        return EXIT_ERROR
    except CacheBusyError as error:
        _print_error(args, "conflict", str(error))
        return EXIT_ERROR
    except KaggleError as error:
        _print_error(args, "kaggle_failed", str(error))
        return EXIT_ERROR
    except CommandError as error:
        _print_error(args, error.code, str(error))
        return EXIT_ERROR
    except ValueError as error:
        # Lỗi kỹ thuật từ drive2sim.core (file sai schema...), viết tiếng Anh; bọc tiếng Việt.
        _print_error(args, "invalid_request", f"dữ liệu không hợp lệ: {error}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        _print_error(args, "cancelled", "đã dừng theo yêu cầu (Ctrl-C)")
        return EXIT_INTERRUPTED
    except Exception as error:
        _print_error(args, "internal", f"lỗi không lường trước: {type(error).__name__}: {error}")
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
