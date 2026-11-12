"""Nhóm lệnh ``d2s scene``: xem các cảnh đã ingest trong thư mục làm việc.

File này chứa:
- ``register``: thêm các lệnh ``scene list`` và ``scene show`` vào parser.
- ``list_scenes``: liệt kê các cảnh có ``manifest.yaml`` trong ``data/``.
- ``show_scene``: in manifest và tóm tắt báo cáo kiểm tra của một cảnh.
- ``ingest_scene``: chuyển một cảnh của dữ liệu gốc thành ``data/<scene_id>/`` (hiện hỗ trợ
  nuScenes), ghi manifest và báo cáo kiểm tra.

Lệnh ``scene validate`` và ``scene delete`` (CONTRACTS mục 10.4) chưa có.
"""

import argparse
import json
from pathlib import Path

from drive2sim.cli.output import CommandError, UsageError, emit, format_table, info
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine import scenes
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest.detect import inspect_source
from drive2sim.ingest.nuscenes import IngestError, ingest_nuscenes


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
    ingest.add_argument("--version", help="phiên bản dataset, ví dụ v1.0-mini (nuScenes)")
    ingest.add_argument("--sequence", help="chuỗi (cảnh gốc) cần ingest, ví dụ scene-0061")
    ingest.add_argument("--scene", dest="scene_id", help="scene_id trong tool; mặc định tự đặt")
    ingest.add_argument(
        "--interpolate-n",
        type=int,
        default=4,
        help="nuScenes: số khung nội suy giữa hai keyframe; 4 cho 10 Hz (mặc định), 0 cho 2 Hz",
    )
    ingest.add_argument("--workers", type=int, default=4, help="số tiến trình của script")
    ingest.set_defaults(handler=ingest_scene)


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
    """Chuyển một cảnh của dữ liệu gốc sang ``data/<scene_id>/``.

    Bỏ ``--source-type`` thì nhận dạng như ``source inspect``. Với nuScenes: ``--version`` được
    chọn tự động khi thư mục chỉ có một phiên bản; ``--sequence`` bắt buộc khi có nhiều cảnh.

    Args:
        args: Tham số dòng lệnh: ``path``, ``source_type``, ``version``, ``sequence``,
            ``scene_id``, ``interpolate_n``, ``workers``.

    Returns:
        ``0`` nếu ingest xong và báo cáo kiểm tra không có lỗi; ``1`` nếu báo cáo có lỗi.

    Raises:
        UsageError: Nếu thiếu hoặc sai ``--version``, ``--sequence``.
        CommandError: Nếu nguồn chưa được hỗ trợ, hoặc ingest lỗi.
    """
    workdir = Workdir.from_env(args.workdir)
    report = inspect_source(args.path)
    dataset = args.source_type or report.dataset
    if dataset is None:
        raise CommandError(
            "không nhận ra dataset; chạy d2s source inspect để xem thư mục có gì", "not_found"
        )
    if dataset != "nuscenes":
        raise CommandError(f"scene ingest chưa hỗ trợ {dataset}; hiện chỉ có nuscenes")
    versions = report.versions
    version = args.version or (next(iter(versions)) if len(versions) == 1 else None)
    if version is None or version not in versions:
        raise UsageError(f"chọn --version trong: {', '.join(versions) or '(không có)'}")
    sequences = versions[version]
    sequence = args.sequence or (sequences[0] if len(sequences) == 1 else None)
    if sequence is None:
        shown = ", ".join(sequences[:10]) + (", ..." if len(sequences) > 10 else "")
        raise UsageError(f"chọn một cảnh bằng --sequence; {version} có: {shown}")
    try:
        scene_dir = ingest_nuscenes(
            args.path,
            workdir,
            scene_name=sequence,
            version=version,
            scene_id=args.scene_id,
            interpolate_n=args.interpolate_n,
            workers=args.workers,
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
