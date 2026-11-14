"""Nhóm lệnh ``d2s scene``: xem các cảnh đã ingest trong thư mục làm việc.

File này chứa:
- ``register``: thêm các lệnh ``scene list`` và ``scene show`` vào parser.
- ``list_scenes``: liệt kê các cảnh có ``manifest.yaml`` trong ``data/``.
- ``show_scene``: in manifest và tóm tắt báo cáo kiểm tra của một cảnh.
- ``ingest_scene``: chuyển một chuỗi của dữ liệu gốc thành ``data/<scene_id>/`` (mọi dataset có
  script tiền xử lý của DriveStudio, và KITTI-360), ghi manifest và báo cáo kiểm tra.

- ``validate_scene``: kiểm tra lại một cảnh, ghi ``reports/validation.json``.
- ``delete_scene``: xoá cảnh cùng split, cache và run của nó, sau khi hỏi xác nhận.
"""

import argparse
import json
from pathlib import Path

from drive2sim.cli.output import CommandError, UsageError, confirm, emit, format_table, info
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine import housekeeping, scenes
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest import kitti360
from drive2sim.ingest.detect import inspect_source
from drive2sim.ingest.dispatch import IngestRequestError, plan_ingest, run_ingest
from drive2sim.ingest.drivestudio import IngestError
from drive2sim.ingest.validate import validate_scene


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``scene`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung (``--workdir``, ``--json``...).
    """
    group = subparsers.add_parser("scene", help="xem các cảnh đã ingest")
    actions = group.add_subparsers(dest="action", required=True)
    list_parser = actions.add_parser("list", parents=[common], help="liệt kê cảnh")
    list_parser.set_defaults(handler=list_scenes)
    show_parser = actions.add_parser("show", parents=[common], help="xem manifest của một cảnh")
    show_parser.add_argument("scene_id")
    show_parser.set_defaults(handler=show_scene)
    ingest = actions.add_parser(
        "ingest", parents=[common], help="chuyển một cảnh của dữ liệu gốc sang dữ liệu chuẩn"
    )
    ingest.add_argument("path", type=Path, help="thư mục dữ liệu gốc")
    ingest.add_argument("--source-type", help="dataset gốc; bỏ trống thì tự nhận dạng")
    ingest.add_argument("--version", help="phiên bản hoặc split, ví dụ v1.0-mini, training")
    ingest.add_argument("--sequence", help="chuỗi (cảnh gốc) cần ingest, ví dụ scene-0061")
    ingest.add_argument("--scene", dest="scene_id", help="scene_id trong tool; mặc định tự đặt")
    ingest.add_argument(
        "--interpolate-n",
        type=int,
        default=4,
        help="nuScenes: số khung nội suy giữa hai keyframe; 4 cho 10 Hz (mặc định), 0 cho 2 Hz",
    )
    ingest.add_argument("--workers", type=int, default=4, help="số tiến trình của script")
    ingest.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="KITTI-360: số hiệu khung gốc nhỏ nhất được xét (mặc định 0)",
    )
    ingest.add_argument(
        "--num-frames",
        type=int,
        help="KITTI-360: số khung tối đa; bỏ trống thì lấy cả đoạn liền nhau dài nhất",
    )
    ingest.add_argument(
        "--cameras",
        help="KITTI-360: camera cần ghi, cách nhau bằng dấu phẩy; mặc định "
        "image_00,image_01,image_02,image_03",
    )
    ingest.add_argument(
        "--max-gap",
        type=int,
        default=kitti360.DEFAULT_MAX_GAP,
        help="KITTI-360: hai khung liền nhau cách nhau quá số này (theo số hiệu gốc) thì cắt "
        f"đoạn (mặc định {kitti360.DEFAULT_MAX_GAP})",
    )
    ingest.set_defaults(handler=ingest_scene)
    validate = actions.add_parser("validate", parents=[common], help="kiểm tra lại một cảnh")
    validate.add_argument("scene_id")
    validate.set_defaults(handler=validate_scene_command)
    delete = actions.add_parser("delete", parents=[common], help="xoá cảnh, split, cache và run")
    delete.add_argument("scene_id")
    delete.set_defaults(handler=delete_scene)


def list_scenes(args: argparse.Namespace) -> int:
    """Liệt kê mọi cảnh có ``manifest.yaml`` trong ``data/``.

    Args:
        args: Tham số dòng lệnh.

    Returns:
        Mã thoát ``0``.

    Raises:
        ValueError: Nếu một manifest sai schema.
    """
    manifests = scenes.list_scenes(Workdir.from_env(args.workdir))
    rows = [
        [m.scene_id, m.track, len(m.cameras), m.timeline.num_frames, m.available.lidar]
        for m in manifests
    ]
    emit(
        args,
        [m.model_dump(mode="json") for m in manifests],
        format_table(["scene_id", "track", "camera", "khung", "lidar"], rows),
    )
    return 0


def show_scene(args: argparse.Namespace) -> int:
    """In manifest của một cảnh và tóm tắt báo cáo kiểm tra nếu có.

    Args:
        args: Tham số dòng lệnh, có ``scene_id``.

    Returns:
        Mã thoát ``0``.

    Raises:
        WorkdirError: Nếu cảnh chưa ingest.
    """
    workdir = Workdir.from_env(args.workdir)
    scene = workdir.load_scene(args.scene_id)
    data = scene.manifest.model_dump(mode="json")
    report_path = scene.root / "reports" / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    data_out = {"manifest": data, "validation_summary": report["summary"] if report else None}
    text = [
        f"cảnh     {scene.scene_id} ({scene.manifest.source.dataset}, "
        f"track {scene.manifest.track})",
        f"khung    {scene.manifest.timeline.num_frames}",
        f"lidar    {', '.join(scene.manifest.lidars) or 'không có'}",
        f"box 3D   {scene.manifest.available.cuboids}",
        "camera:",
        format_table(
            ["camera_id", "mô hình", "độ phân giải", "shutter"],
            [
                [c.camera_id, c.model, f"{c.resolution[0]}x{c.resolution[1]}", c.shutter]
                for c in scene.manifest.cameras
            ],
        ),
    ]
    if report:
        summary = report["summary"]
        text.append(f"kiểm tra {summary['errors']} lỗi, {summary['warnings']} cảnh báo")
    emit(args, data_out, "\n".join(text))
    return 0


def ingest_scene(args: argparse.Namespace) -> int:
    """Chuyển một chuỗi của dữ liệu gốc sang ``data/<scene_id>/``.

    Bỏ ``--source-type`` thì nhận dạng như ``source inspect``. Phiên bản và chuỗi được chọn tự
    động khi chỉ có một khả năng (``drive2sim.ingest.dispatch.plan_ingest``).

    Args:
        args: Tham số dòng lệnh: ``path``, ``source_type``, ``version``, ``sequence``,
            ``scene_id``, ``interpolate_n``, ``workers``; với KITTI-360 thêm ``start_frame``,
            ``num_frames``, ``cameras``, ``max_gap``.

    Returns:
        ``0`` nếu ingest xong và báo cáo kiểm tra không có lỗi; ``1`` nếu báo cáo có lỗi.

    Raises:
        UsageError: Nếu thiếu hoặc sai dataset, phiên bản, chuỗi.
        CommandError: Nếu ingest lỗi.
    """
    workdir = Workdir.from_env(args.workdir)
    try:
        target = plan_ingest(
            inspect_source(args.path), args.source_type, args.version, args.sequence
        )
    except IngestRequestError as error:
        raise UsageError(str(error)) from None
    options = kitti360.Kitti360Options(
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        cameras=tuple(args.cameras.split(",")) if args.cameras else kitti360.CAMERAS,
        max_gap=args.max_gap,
    )
    try:
        scene_dir = run_ingest(
            args.path,
            workdir,
            target,
            scene_id=args.scene_id,
            interpolate_n=args.interpolate_n,
            workers=args.workers,
            kitti360_options=options,
        )
    except IngestError as error:
        raise CommandError(str(error)) from None
    info(args, f"đã ghi {workdir.relative(scene_dir)}")
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    report_data = json.loads((scene_dir / "reports" / "validation.json").read_text("utf-8"))
    summary = report_data["summary"]
    emit(
        args,
        {"manifest": manifest.model_dump(mode="json"), "validation": report_data},
        f"cảnh {manifest.scene_id}: {manifest.timeline.num_frames} khung, "
        f"{len(manifest.cameras)} camera; kiểm tra {summary['errors']} lỗi, "
        f"{summary['warnings']} cảnh báo",
    )
    return 0 if summary["errors"] == 0 else 1


def validate_scene_command(args: argparse.Namespace) -> int:
    """Kiểm tra lại một cảnh và ghi ``reports/validation.json``.

    Dùng sau khi sửa tay thư mục cảnh, hoặc khi bộ kiểm tra có phép kiểm tra mới.

    Args:
        args: Tham số dòng lệnh, có ``scene_id``.

    Returns:
        ``0`` nếu không có lỗi; ``1`` nếu báo cáo có lỗi.
    """
    workdir = Workdir.from_env(args.workdir)
    scene = workdir.load_scene(args.scene_id)
    report = validate_scene(scene.root, scene.manifest)
    files.dump(report, scene.root / "reports" / "validation.json")
    lines = [f"{report.summary.errors} lỗi, {report.summary.warnings} cảnh báo"]
    lines += [f"  {c.severity}: {c.check_id} {c.message}" for c in report.checks]
    emit(args, report.model_dump(mode="json"), "\n".join(lines))
    return 0 if report.summary.errors == 0 else 1


def delete_scene(args: argparse.Namespace) -> int:
    """Xoá một cảnh cùng split, cache và run của nó, sau khi hỏi xác nhận.

    Args:
        args: Tham số dòng lệnh, có ``scene_id`` và ``yes``.

    Returns:
        Mã thoát ``0``.

    Raises:
        InUseError: Nếu một run của cảnh đang chạy.
    """
    workdir = Workdir.from_env(args.workdir)
    workdir.load_scene(args.scene_id)
    targets = [
        workdir.relative(path)
        for path in (
            workdir.scene_dir(args.scene_id),
            workdir.cache_dir / args.scene_id,
            workdir.runs_dir / args.scene_id,
        )
        if path.exists()
    ]
    confirm(args, f"Xoá cảnh {args.scene_id} cùng mọi split, cache và run của nó?", targets)
    removed = housekeeping.delete_scene(workdir, args.scene_id)
    emit(args, {"removed": removed}, f"đã xoá {', '.join(removed)}")
    return 0
