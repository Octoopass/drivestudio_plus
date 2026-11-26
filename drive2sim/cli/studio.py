"""Lệnh ``d2s studio``: bật Studio (API và giao diện) trên máy đang dùng.

Mặc định Studio chỉ nghe trên ``127.0.0.1`` (ARCHITECTURE mục 2.10): máy khác không truy cập
được, nên không cần tài khoản hay xác thực. Trên máy GPU thuê, mở Studio qua port forward của SSH,
ví dụ ``ssh -L 8000:127.0.0.1:8000 <máy>``.

``--host <địa chỉ>`` cho Studio nghe trên một địa chỉ khác, dành cho mạng riêng chỉ nhóm vào được
(ví dụ IP Tailscale của máy): mọi người mở thẳng ``http://<địa chỉ>:<cổng>``. Studio vẫn không có
đăng nhập, nên ai vào được mạng đó cũng tạo, xoá run và dùng GPU được; không dùng IP công khai.
Kiểm tra ``Host``/``Origin`` (chống DNS rebinding, POST chéo trang) nhận thêm đúng địa chỉ đó.

``--allow-host <tên>`` chỉ nhận thêm tên máy trong ``Host``/``Origin`` mà vẫn nghe trên
địa chỉ của ``--host``. Dùng khi Tailscale chạy ``--tun=userspace-networking`` (máy không có
sudo): IP Tailscale không gắn vào card mạng nào nên không nghe trực tiếp được, còn tailscaled
chuyển kết nối tới IP đó về ``127.0.0.1`` cùng cổng, với ``Host`` là IP Tailscale.

File này chứa ``register`` (thêm lệnh vào parser) và ``run_studio`` (chạy server bằng uvicorn).
Studio cần các gói của nhóm ``studio`` (``pip install -e ".[studio]"``).
"""

import argparse
import errno
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
        "--host",
        default=HOST,
        help="địa chỉ nghe; mặc định 127.0.0.1. Chỉ dùng IP của mạng riêng (ví dụ Tailscale): "
        "Studio không có đăng nhập",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="TÊN",
        help="nhận thêm tên máy này trong Host/Origin (lặp được), ví dụ IP Tailscale khi "
        "tailscaled chạy userspace-networking và chuyển kết nối về 127.0.0.1",
    )
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
        args: Tham số dòng lệnh, có ``host``, ``allow_host``, ``port``, ``workdir``,
            ``kaggle_watch_minutes`` và ``colab_watch_minutes``.

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
            probe.bind((args.host, args.port))
        except OSError as error:
            if error.errno == errno.EADDRNOTAVAIL:
                raise CommandError(
                    f"máy không có địa chỉ {args.host} trên card mạng nào; với Tailscale "
                    "userspace-networking dùng --allow-host thay cho --host",
                    "invalid_request",
                ) from None
            raise CommandError(
                f"cổng {args.port} đang có chương trình khác dùng; chọn cổng khác bằng --port",
                "conflict",
            ) from None
    info(args, f"Studio ở http://{args.host}:{args.port} (thư mục làm việc {workdir.root})")
    extra_hosts = [*args.allow_host, *(() if args.host == HOST else (args.host,))]
    if extra_hosts:
        info(
            args,
            f"nhận thêm {', '.join(extra_hosts)}; Studio không có đăng nhập: ai vào được địa chỉ "
            "đó cũng tạo, xoá run được",
        )
    watch = args.kaggle_watch_minutes * 60 if args.kaggle_watch_minutes > 0 else None
    colab_watch = args.colab_watch_minutes * 60 if args.colab_watch_minutes > 0 else None
    app = create_app(
        workdir,
        kaggle_watch_s=watch,
        colab_watch_s=colab_watch,
        extra_hosts=extra_hosts,
    )
    # Luồng sự kiện (SSE) của giao diện đang mở không tự đóng; không có hạn chót thì Ctrl+C chờ
    # mãi ở "Waiting for connections to close" và không tới bước dọn dẹp (tắt viewer GPU, hàng
    # đợi). Sau GRACEFUL_SHUTDOWN_S giây uvicorn cắt các kết nối còn lại.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
    return 0
