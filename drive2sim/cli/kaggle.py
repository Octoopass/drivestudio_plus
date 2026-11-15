"""Nhóm lệnh ``d2s kaggle``: chạy run trên Kaggle (ARCHITECTURE mục 2.8, CONTRACTS mục 9.2).

File này chứa:
- ``register``: thêm các lệnh ``kaggle push``, ``status``, ``pull``, ``ingest``, ``run``.
- ``push``: đẩy một run lên Kaggle (lần sau chạy tiếp từ output lần trước).
- ``status``: trạng thái notebook của run.
- ``pull``: tải kết quả về, ghép vào thư mục làm việc.
- ``ingest``: ingest một cảnh trên Kaggle từ dữ liệu gốc đã có trên Kaggle (CONTRACTS mục 9.4).
- ``run_bundle``: chạy trên Kaggle (notebook gọi): dựng thư mục làm việc từ gói job rồi chạy run.

``d2s run start --executor kaggle`` tạo run rồi gọi ``push``.
"""

import argparse
import json
from pathlib import Path

from drive2sim.cli.output import emit, event_printer, info
from drive2sim.cli.scene import add_ingest_options, ingest_arguments
from drive2sim.engine.run import execute_run
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle import executor
from drive2sim.kaggle.bundle import materialize, read_bundle
from drive2sim.kaggle.ingest import IngestRequest, ingest_on_kaggle


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``kaggle`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("kaggle", help="chạy run trên Kaggle")
    actions = group.add_subparsers(dest="action", required=True)
    for name, handler, help_text in (
        ("push", push, "đẩy run lên Kaggle và cho chạy"),
        ("status", status, "xem trạng thái notebook của run"),
        ("pull", pull, "tải kết quả về thư mục làm việc"),
    ):
        parser = actions.add_parser(name, parents=[common], help=help_text)
        parser.add_argument("run_id")
        if name == "push":
            parser.add_argument("--branch", default=executor.DEFAULT_BRANCH)
            parser.add_argument(
                "--accelerator",
                default=executor.DEFAULT_ACCELERATOR,
                help="GPU của Kaggle, ví dụ NvidiaTeslaT4 (mặc định), NvidiaL4",
            )
        parser.set_defaults(handler=handler)
    ingest_parser = actions.add_parser(
        "ingest", parents=[common], help="ingest một cảnh trên Kaggle từ dữ liệu gốc trên Kaggle"
    )
    ingest_parser.add_argument(
        "source", help="dataset hoặc notebook chứa dữ liệu gốc, <người dùng>/<tên>"
    )
    ingest_parser.add_argument(
        "--path", default="", help="thư mục con trong nguồn chứa dữ liệu gốc, ví dụ kitti360"
    )
    ingest_parser.add_argument("--branch", default=executor.DEFAULT_BRANCH)
    add_ingest_options(ingest_parser)
    ingest_parser.set_defaults(handler=ingest)
    run_parser = actions.add_parser(
        "run", parents=[common], help="(trong notebook Kaggle) chạy run từ gói job"
    )
    run_parser.add_argument("bundle", type=Path, help="thư mục gói job")
    run_parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    run_parser.set_defaults(handler=run_bundle)


def push(args: argparse.Namespace) -> int:
    """Đẩy một run lên Kaggle.

    Args:
        args: Tham số dòng lệnh, có ``run_id`` và ``branch``.

    Returns:
        Mã thoát ``0``.
    """
    workdir = Workdir.from_env(args.workdir)
    state = executor.push_run(
        workdir, args.run_id, branch=args.branch, accelerator=args.accelerator
    )
    emit(
        args,
        state.to_dict(),
        f"đã đẩy run {args.run_id} lên notebook {state.kernel} (lần {state.attempt}); "
        f"xem bằng d2s kaggle status {args.run_id}",
    )
    return 0


def status(args: argparse.Namespace) -> int:
    """In trạng thái notebook của một run.

    Returns:
        Mã thoát ``0``.
    """
    data = executor.kernel_state(Workdir.from_env(args.workdir), args.run_id)
    emit(args, data, f"{data['kernel']} (lần {data['attempt']}): {data['status']}")
    return 0


def pull(args: argparse.Namespace) -> int:
    """Tải kết quả của run trên Kaggle về.

    Returns:
        ``0`` nếu run đã xong; ``1`` nếu chưa (đẩy lại để chạy tiếp, hoặc xem lỗi).
    """
    record = executor.pull_run(Workdir.from_env(args.workdir), args.run_id)
    hint = "" if record.status == "succeeded" else f"; chạy tiếp: d2s kaggle push {args.run_id}"
    emit(args, record.model_dump(mode="json"), f"run {record.run_id}: {record.status}{hint}")
    return 0 if record.status == "succeeded" else 1


def ingest(args: argparse.Namespace) -> int:
    """Ingest một cảnh trên Kaggle; chờ notebook xong rồi ghi file nhỏ của cảnh về máy.

    Args:
        args: Tham số dòng lệnh: ``source``, ``path``, ``branch`` và tuỳ chọn của ``scene ingest``.

    Returns:
        ``0`` nếu báo cáo kiểm tra của cảnh không có lỗi, ``1`` nếu có.
    """
    workdir = Workdir.from_env(args.workdir)
    request = IngestRequest(args.source, args.path, tuple(ingest_arguments(args)))
    scene_dir = ingest_on_kaggle(
        workdir, request, branch=args.branch, notify=lambda message: info(args, message)
    )
    report = json.loads((scene_dir / "reports" / "validation.json").read_text("utf-8"))
    summary = report["summary"]
    emit(
        args,
        {"scene_id": scene_dir.name, "validation": report},
        f"cảnh {scene_dir.name} nằm trên Kaggle, đã ghi file nhỏ vào "
        f"{workdir.relative(scene_dir)}; kiểm tra {summary['errors']} lỗi, "
        f"{summary['warnings']} cảnh báo",
    )
    return 0 if summary["errors"] == 0 else 1


def run_bundle(args: argparse.Namespace) -> int:
    """Chạy run từ gói job; notebook Kaggle gọi lệnh này.

    Args:
        args: Tham số dòng lệnh: ``bundle``, ``input_root``, ``workdir``.

    Returns:
        ``0`` nếu run xong, ``1`` nếu không.
    """
    workdir = Workdir.from_env(args.workdir)
    run_dir = materialize(read_bundle(args.bundle), workdir, args.input_root)
    info(args, f"chạy {workdir.relative(run_dir)}")
    record = execute_run(run_dir, workdir, sink=event_printer(args))
    emit(args, record.model_dump(mode="json"), f"run {record.run_id}: {record.status}")
    return 0 if record.status == "succeeded" else 1
