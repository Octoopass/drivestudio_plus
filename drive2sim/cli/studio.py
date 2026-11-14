"""Lệnh ``d2s studio``: bật Studio (API và giao diện) trên máy đang dùng.

Studio chỉ nghe trên ``127.0.0.1`` (ARCHITECTURE mục 2.10): máy khác không truy cập được, nên
không cần tài khoản hay xác thực. Trên máy GPU thuê, mở Studio qua port forward của SSH, ví dụ
``ssh -L 8000:127.0.0.1:8000 <máy>``.

File này chứa ``register`` (thêm lệnh vào parser) và ``run_studio`` (chạy server bằng uvicorn).
Studio cần các gói của nhóm ``studio`` (``pip install -e ".[studio]"``).
"""

import argparse
import socket

from drive2sim.cli.output import CommandError, info
from drive2sim.engine.workdir import Workdir

HOST = "127.0.0.1"
GRACEFUL_SHUTDOWN_S = 3


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
    parser.add_argument(
        "--kaggle-watch-minutes",
        type=float,
        default=5,
        help="cứ chừng ấy phút hỏi trạng thái run trên Kaggle, xong thì tự tải về; 0 để tắt",
    )
    parser.add_argument(
        "--colab-watch-minutes",
        type=float,
        default=5,
        help="cứ chừng ấy phút hỏi trạng thái job trên Colab, xong thì tự tải về; 0 để tắt",
    )
    parser.set_defaults(handler=run_studio)


def run_studio(args: argparse.Namespace) -> int:
    """Chạy Studio cho tới khi người dùng bấm Ctrl-C.

    Args:
        args: Tham số dòng lệnh, có ``port``, ``workdir``, ``kaggle_watch_minutes`` và
            ``colab_watch_minutes``.

    Returns:
        Mã thoát ``0`` khi server dừng.

    Raises:
        CommandError: Nếu chưa cài ``fastapi`` hoặc ``uvicorn``, hoặc cổng đang bận.
    """
    try:
        import uvicorn

        from drive2sim.studio.api.app import create_app
    except ImportError as error:
        raise CommandError(
            f'Studio cần gói {error.name}; cài bằng: pip install -e ".[studio]"'
        ) from None
    workdir = Workdir.from_env(args.workdir)
    # Thử giữ cổng trước khi báo địa chỉ, để không in "Studio ở ..." rồi mới báo cổng bận.
    # SO_REUSEADDR giống uvicorn: kết nối cũ còn ở TIME_WAIT sau khi tắt Studio không bị tính là
    # cổng bận; chỉ một chương trình đang nghe mới làm bind lỗi.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, args.port))
        except OSError:
            raise CommandError(
                f"cổng {args.port} đang có chương trình khác dùng; chọn cổng khác bằng --port",
                "conflict",
            ) from None
    info(args, f"Studio ở http://{HOST}:{args.port} (thư mục làm việc {workdir.root})")
    watch = args.kaggle_watch_minutes * 60 if args.kaggle_watch_minutes > 0 else None
    colab_watch = args.colab_watch_minutes * 60 if args.colab_watch_minutes > 0 else None
    app = create_app(workdir, kaggle_watch_s=watch, colab_watch_s=colab_watch)
    # Luồng sự kiện (SSE) của giao diện đang mở không tự đóng; không có hạn chót thì Ctrl+C chờ
    # mãi ở "Waiting for connections to close" và không tới bước dọn dẹp (tắt viewer GPU, hàng
    # đợi). Sau GRACEFUL_SHUTDOWN_S giây uvicorn cắt các kết nối còn lại.
    uvicorn.run(
        app,
        host=HOST,
        port=args.port,
        log_level="warning",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
    return 0
