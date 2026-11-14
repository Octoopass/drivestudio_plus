"""Nhóm lệnh ``d2s run``: chạy, chạy tiếp và theo dõi run.

File này chứa:
- ``register``: thêm các lệnh ``run start``, ``run resume``, ``run status``, ``run list``.
- ``start_run``: kiểm tra như ``approach check``, tạo run rồi chạy hết các stage.
- ``resume_run``: chạy tiếp một run dở hoặc lỗi; stage đã xong không chạy lại.
- ``status_run``, ``list_runs``: xem trạng thái một run và liệt kê các run.

Tiến độ in ra stderr trong lúc chạy; kết quả cuối in ra stdout. Run lỗi trả mã thoát ``1``.
- ``cancel_run``: yêu cầu huỷ một run đang chạy hoặc đang chờ.
- ``delete_run``: xoá thư mục của một run sau khi hỏi xác nhận.
"""

import argparse

from drive2sim.cli.approach import add_plan_arguments, resolve_from_args
from drive2sim.cli.output import (
    confirm,
    emit,
    event_printer,
    format_table,
    format_value,
    info,
)
from drive2sim.colab import executor as colab_executor
from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine import housekeeping
from drive2sim.engine.run import D2S_DIR, RUN_FILE, create_run, execute_run, request_cancel
from drive2sim.engine.run import list_runs as read_runs
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle.executor import push_run


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``run`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("run", help="chạy và theo dõi run")
    actions = group.add_subparsers(dest="action", required=True)

    start = actions.add_parser("start", parents=[common], help="tạo run mới và chạy")
    add_plan_arguments(start)
    start.add_argument("--executor", choices=["local", "kaggle", "colab"], default="local")
    start.set_defaults(handler=start_run)

    resume = actions.add_parser("resume", parents=[common], help="chạy tiếp một run")
    resume.add_argument("run_id")
    resume.set_defaults(handler=resume_run)

    status = actions.add_parser("status", parents=[common], help="xem trạng thái một run")
    status.add_argument("run_id")
    status.set_defaults(handler=status_run)

    cancel = actions.add_parser("cancel", parents=[common], help="huỷ một run")
    cancel.add_argument("run_id")
    cancel.set_defaults(handler=cancel_run)

    delete = actions.add_parser("delete", parents=[common], help="xoá thư mục của một run")
    delete.add_argument("run_id")
    delete.set_defaults(handler=delete_run)

    list_parser = actions.add_parser("list", parents=[common], help="liệt kê run")
    list_parser.add_argument("--scene", dest="scene_id")
    list_parser.set_defaults(handler=list_runs)


def _report(args: argparse.Namespace, record: RunRecord) -> int:
    """In kết quả cuối của một run và trả mã thoát tương ứng.

    Returns:
        ``0`` nếu run ``succeeded``, ``1`` nếu không.
    """
    failed = [stage for stage in record.stages if stage.error is not None]
    text = f"run {record.run_id}: {record.status}"
    if failed:
        text += f"\n{failed[0].error.message}"  # type: ignore[union-attr]
    emit(args, record.model_dump(mode="json"), text)
    return 0 if record.status == "succeeded" else 1


def start_run(args: argparse.Namespace) -> int:
    """Tạo một run mới rồi chạy hết các stage, trên máy, trên Kaggle hoặc trên Colab.

    Với ``--executor kaggle``, run được tạo trên máy rồi đẩy lên Kaggle (``d2s kaggle push``);
    lệnh trả về ngay, kết quả tải về bằng ``d2s kaggle pull``. ``--executor colab`` tương tự với
    ``d2s colab push`` (phiên ``d2s``, GPU T4) và ``d2s colab pull``.

    Args:
        args: Tham số dòng lệnh như ``approach check``, thêm ``executor``.

    Returns:
        ``0`` nếu run xong (hoặc đã đẩy lên Kaggle, Colab), ``1`` nếu một stage lỗi.
    """
    workdir, scene, split, plan = resolve_from_args(args)
    run_dir = create_run(plan, workdir, scene, split, executor=args.executor)
    info(args, f"tạo run {run_dir.name} ở {workdir.relative(run_dir)}")
    if args.executor == "kaggle":
        state = push_run(workdir, run_dir.name)
        emit(
            args,
            {"run_id": run_dir.name, "kaggle": state.to_dict()},
            f"đã đẩy run {run_dir.name} lên notebook {state.kernel}; "
            f"xem: d2s kaggle status {run_dir.name}; tải về: d2s kaggle pull {run_dir.name}",
        )
        return 0
    if args.executor == "colab":
        colab_state = colab_executor.push_run(workdir, run_dir.name)
        emit(
            args,
            {"run_id": run_dir.name, "colab": colab_state.to_dict()},
            f"đã chạy run {run_dir.name} trên phiên Colab {colab_state.session}; "
            f"xem: d2s colab logs {run_dir.name} --follow; tải về: d2s colab pull {run_dir.name}",
        )
        return 0
    return _report(args, execute_run(run_dir, workdir, sink=event_printer(args)))


def resume_run(args: argparse.Namespace) -> int:
    """Chạy tiếp một run; stage đã xong không chạy lại.

    Args:
        args: Tham số dòng lệnh, có ``run_id``.

    Returns:
        ``0`` nếu run xong, ``1`` nếu một stage lỗi.
    """
    workdir = Workdir.from_env(args.workdir)
    run_dir = workdir.find_run(args.run_id)
    return _report(args, execute_run(run_dir, workdir, sink=event_printer(args)))


def status_run(args: argparse.Namespace) -> int:
    """In trạng thái của một run và từng stage.

    Args:
        args: Tham số dòng lệnh, có ``run_id``.

    Returns:
        Mã thoát ``0``.
    """
    workdir = Workdir.from_env(args.workdir)
    record = files.load(workdir.find_run(args.run_id) / D2S_DIR / RUN_FILE, RunRecord)
    rows = [
        [
            stage.stage_id,
            stage.status,
            "có" if stage.cache_hit else "",
            stage.duration_s,
            stage.error.message if stage.error else "",
        ]
        for stage in record.stages
    ]
    options = ", ".join(f"{k}={format_value(v)}" for k, v in record.approach.options.items())
    text = "\n".join(
        [
            f"run {record.run_id}: {record.status}",
            f"cảnh {record.scene_id}, split {record.split_id}, "
            f"approach {record.approach.approach_id}" + (f" ({options})" if options else ""),
            format_table(["stage", "trạng thái", "cache", "giây", "lỗi"], rows),
        ]
    )
    emit(args, record.model_dump(mode="json"), text)
    return 0


def list_runs(args: argparse.Namespace) -> int:
    """Liệt kê các run trong thư mục làm việc, mới nhất trước.

    Args:
        args: Tham số dòng lệnh; ``scene_id`` để lọc theo cảnh.

    Returns:
        Mã thoát ``0``.
    """
    records = read_runs(Workdir.from_env(args.workdir), args.scene_id)
    rows = [[r.run_id, r.scene_id, r.split_id, r.approach.approach_id, r.status] for r in records]
    emit(
        args,
        [r.model_dump(mode="json") for r in records],
        format_table(["run_id", "cảnh", "split", "approach", "trạng thái"], rows),
    )
    return 0


def cancel_run(args: argparse.Namespace) -> int:
    """Yêu cầu huỷ một run.

    Run đang chạy dừng trong khoảng một giây (lệnh ngoài của stage bị dừng); run đang chờ bị huỷ
    khi tới lượt; run ghi ``running`` nhưng tiến trình đã chết được ghi ``cancelled`` ngay.

    Args:
        args: Tham số dòng lệnh, có ``run_id``.

    Returns:
        Mã thoát ``0``.
    """
    workdir = Workdir.from_env(args.workdir)
    record = request_cancel(workdir.find_run(args.run_id))
    emit(args, record.model_dump(mode="json"), f"run {record.run_id}: đã gửi yêu cầu huỷ")
    return 0


def delete_run(args: argparse.Namespace) -> int:
    """Xoá thư mục của một run sau khi hỏi xác nhận.

    Args:
        args: Tham số dòng lệnh, có ``run_id`` và ``yes``.

    Returns:
        Mã thoát ``0``.

    Raises:
        InUseError: Nếu run đang chạy.
    """
    workdir = Workdir.from_env(args.workdir)
    run_dir = workdir.find_run(args.run_id)
    confirm(args, f"Xoá run {args.run_id}?", [workdir.relative(run_dir)])
    removed = housekeeping.delete_run(workdir, args.run_id)
    emit(args, {"removed": removed}, f"đã xoá {', '.join(removed)}")
    return 0
