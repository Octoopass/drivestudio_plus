"""Nhóm lệnh ``d2s colab``: chạy run trên VM Google Colab (CONTRACTS mục 9.3).

File này chứa:
- ``register``: thêm các lệnh ``colab push``, ``status``, ``logs``, ``pull``, ``stop``.
- ``push``: đẩy một run lên VM Colab và cho chạy tách khỏi kết nối.
- ``status``: trạng thái job và các dòng log cuối.
- ``logs``: in log của job; ``--follow`` đọc tiếp tới khi job xong.
- ``pull``: tải kết quả về, ghép vào thư mục làm việc; run xong thì tắt phiên.
- ``stop``: tắt phiên Colab của run.

``d2s run start --executor colab`` tạo run rồi gọi ``push``. Trên VM, job chạy chính lệnh
``d2s kaggle run`` với cùng gói job như notebook Kaggle.
"""

import argparse
import sys

from drive2sim.cli.output import emit
from drive2sim.colab import executor
from drive2sim.colab.client import GPUS
from drive2sim.engine.workdir import Workdir


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``colab`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("colab", help="chạy run trên VM Google Colab")
    actions = group.add_subparsers(dest="action", required=True)
    for name, handler, help_text in (
        ("push", push, "đẩy run lên VM Colab và cho chạy"),
        ("status", status, "xem trạng thái job và log cuối"),
        ("logs", logs, "in log của job"),
        ("pull", pull, "tải kết quả về thư mục làm việc"),
        ("stop", stop, "tắt phiên Colab của run (trả VM)"),
    ):
        parser = actions.add_parser(name, parents=[common], help=help_text)
        parser.add_argument("run_id")
        parser.set_defaults(handler=handler)
        if name == "push":
            add_push_arguments(parser)
        if name == "logs":
            parser.add_argument(
                "--follow", action="store_true", help="đọc tiếp log tới khi job xong"
            )
        if name == "pull":
            parser.add_argument(
                "--keep-session",
                action="store_true",
                help="không tắt phiên Colab kể cả khi run đã xong",
            )


def add_push_arguments(parser: argparse.ArgumentParser) -> None:
    """Thêm tuỳ chọn của ``colab push`` (``run start --executor colab`` dùng lại).

    Args:
        parser: Parser của lệnh.
    """
    parser.add_argument("--branch", default=executor.DEFAULT_BRANCH)
    parser.add_argument(
        "--gpu",
        default=executor.DEFAULT_GPU,
        choices=GPUS,
        help="GPU khi phải tạo phiên mới (mặc định T4)",
    )
    parser.add_argument(
        "--session",
        default=executor.DEFAULT_SESSION,
        help="tên phiên Colab; các run dùng chung phiên thì chỉ cài môi trường một lần",
    )


def push(args: argparse.Namespace) -> int:
    """Đẩy một run lên VM Colab.

    Args:
        args: Tham số dòng lệnh, có ``run_id``, ``branch``, ``gpu``, ``session``.

    Returns:
        Mã thoát ``0``.
    """
    workdir = Workdir.from_env(args.workdir)
    state = executor.push_run(
        workdir, args.run_id, branch=args.branch, gpu=args.gpu, session=args.session
    )
    emit(
        args,
        state.to_dict(),
        f"đã chạy run {args.run_id} trên phiên Colab {state.session} ({state.gpu}), job "
        f"{state.job}; theo dõi: d2s colab logs {args.run_id} --follow; tải về: "
        f"d2s colab pull {args.run_id}",
    )
    return 0


def status(args: argparse.Namespace) -> int:
    """In trạng thái job Colab của một run và các dòng log cuối.

    Returns:
        Mã thoát ``0``.
    """
    data = executor.job_state(Workdir.from_env(args.workdir), args.run_id)
    lines = [f"{data['job']} trên phiên {data['session']}: {data['status'] or 'không rõ'}"]
    if data["status_error"]:
        lines.append(data["status_error"])
    lines.extend(data["log_tail"])
    emit(args, data, "\n".join(lines))
    return 0


def logs(args: argparse.Namespace) -> int:
    """In log của job; với ``--follow`` đọc tiếp tới khi job xong (Ctrl-C chỉ dừng việc đọc).

    Returns:
        ``0``; với ``--follow``: ``0`` nếu job xong thành công, ``1`` nếu không.
    """
    workdir = Workdir.from_env(args.workdir)
    if not args.follow:
        data = executor.job_state(workdir, args.run_id)
        emit(args, data["log_tail"], "\n".join(data["log_tail"]))
        return 0

    def show(line: str) -> None:
        print(line, file=sys.stderr if args.json else sys.stdout, flush=True)

    final = executor.follow_log(workdir, args.run_id, show)
    emit(
        args,
        {"run_id": args.run_id, "status": final},
        f"job xong: {final}; tải về: d2s colab pull {args.run_id}",
    )
    return 0 if final == "succeeded" else 1


def pull(args: argparse.Namespace) -> int:
    """Tải kết quả của run trên Colab về.

    Returns:
        ``0`` nếu run đã xong; ``1`` nếu chưa (đẩy lại để chạy tiếp, hoặc xem lỗi).
    """
    record = executor.pull_run(
        Workdir.from_env(args.workdir), args.run_id, keep_session=args.keep_session
    )
    hint = "" if record.status == "succeeded" else f"; chạy tiếp: d2s colab push {args.run_id}"
    emit(args, record.model_dump(mode="json"), f"run {record.run_id}: {record.status}{hint}")
    return 0 if record.status == "succeeded" else 1


def stop(args: argparse.Namespace) -> int:
    """Tắt phiên Colab của một run.

    Returns:
        Mã thoát ``0``.
    """
    data = executor.stop_session(Workdir.from_env(args.workdir), args.run_id)
    emit(args, data, f"đã tắt phiên Colab {data['session']}")
    return 0
