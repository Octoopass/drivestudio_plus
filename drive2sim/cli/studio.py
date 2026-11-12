"""Lệnh ``d2s studio``: bật Studio (API và giao diện) trên máy đang dùng.

Studio chỉ nghe trên ``127.0.0.1`` (ARCHITECTURE mục 2.10): máy khác không truy cập được, nên
không cần tài khoản hay xác thực. Trên máy GPU thuê, mở Studio qua port forward của SSH, ví dụ
``ssh -L 8000:127.0.0.1:8000 <máy>``.

File này chứa ``register`` (thêm lệnh vào parser) và ``run_studio`` (chạy server bằng uvicorn).
Studio cần các gói của nhóm ``studio`` (``pip install -e ".[studio]"``).
"""

import argparse

from drive2sim.cli.output import CommandError, info
from drive2sim.engine.workdir import Workdir

HOST = "127.0.0.1"


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm lệnh ``studio`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    parser = subparsers.add_parser("studio", parents=[common], help="bật Studio trên 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="cổng; mặc định 8000")
    parser.set_defaults(handler=run_studio)


def run_studio(args: argparse.Namespace) -> int:
    """Chạy Studio cho tới khi người dùng bấm Ctrl-C.

    Args:
        args: Tham số dòng lệnh, có ``port`` và ``workdir``.

    Returns:
        Mã thoát ``0`` khi server dừng.

    Raises:
        CommandError: Nếu chưa cài ``fastapi`` hoặc ``uvicorn``.
    """
    try:
        import uvicorn

        from drive2sim.studio.api.app import create_app
    except ImportError as error:
        raise CommandError(
            f'Studio cần gói {error.name}; cài bằng: pip install -e ".[studio]"'
        ) from None
    workdir = Workdir.from_env(args.workdir)
    info(args, f"Studio ở http://{HOST}:{args.port} (thư mục làm việc {workdir.root})")
    uvicorn.run(create_app(workdir), host=HOST, port=args.port, log_level="warning")
    return 0
