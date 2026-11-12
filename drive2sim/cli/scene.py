"""Nhóm lệnh ``d2s scene``: xem các cảnh đã ingest trong thư mục làm việc.

File này chứa:
- ``register``: thêm các lệnh ``scene list`` và ``scene show`` vào parser.
- ``list_scenes``: liệt kê các cảnh có ``manifest.yaml`` trong ``data/``.
- ``show_scene``: in manifest và tóm tắt báo cáo kiểm tra của một cảnh.

Lệnh ``scene ingest`` và ``scene validate`` (CONTRACTS mục 10.4) có khi làm tầng dữ liệu.
"""

import argparse
import json

from drive2sim.cli.output import emit, format_table
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.workdir import Workdir


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


def list_scenes(args: argparse.Namespace) -> int:
    """Liệt kê mọi cảnh có ``manifest.yaml`` trong ``data/``.

    Args:
        args: Tham số dòng lệnh.

    Returns:
        Mã thoát ``0``.

    Raises:
        ValueError: Nếu một manifest sai schema.
    """
    workdir = Workdir.from_env(args.workdir)
    manifests = [
        files.load(path, Manifest) for path in sorted(workdir.data_dir.glob("*/manifest.yaml"))
    ]
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
