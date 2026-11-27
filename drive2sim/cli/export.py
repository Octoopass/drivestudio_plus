"""Nhóm lệnh ``d2s export``: chạy lại stage xuất kết quả cho một run đã train.

``d2s export ply <run_id>`` chạy lại ``export.ply_world``; ``d2s export preview <run_id>`` chạy
``export.preview`` (gói xem trước cho Spark); ``d2s export video <run_id>`` chạy
``render.nurec_video`` (video đi dọc đoạn đường của run Instant NuRec; ``--fps`` đổi số khung mỗi
giây, ``--no-interp`` lặp khung nguồn thay vì vẽ khung nội suy); ``d2s export carla <run_id>``
chạy ``render.carla`` (cảnh NuRec trong CARLA 1.0, ghi video; ``--ego``, ``--max-seconds``). Stage
được thêm vào cuối run nếu approach không có, và các stage khác đã xong không chạy lại
(CONTRACTS mục 10.4).
"""

import argparse

from drive2sim.cli.output import emit, event_printer
from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine.run import D2S_DIR, RUN_FILE, execute_run, ply_export_stage, reset_stage
from drive2sim.engine.workdir import Workdir

STAGES = {
    "ply": "export.ply_world",
    "preview": "export.preview",
    "video": "render.nurec_video",
    "carla": "render.carla",
}


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``export`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("export", help="xuất lại kết quả của một run")
    actions = group.add_subparsers(dest="action", required=True)
    for name, stage_id in STAGES.items():
        parser = actions.add_parser(name, parents=[common], help=f"chạy lại {stage_id}")
        parser.add_argument("run_id")
        if name == "video":
            parser.add_argument("--fps", type=float, help="khung mỗi giây của video (mặc định 30)")
            parser.add_argument(
                "--no-interp", action="store_true", help="lặp khung nguồn, không vẽ khung nội suy"
            )
        if name == "carla":
            parser.add_argument(
                "--ego", choices=["replay", "trajectory", "autopilot"], help="cách lái xe ego"
            )
            parser.add_argument(
                "--max-seconds", type=float, help="giây mô phỏng (0: hết quỹ đạo; mặc định 30)"
            )
        parser.set_defaults(handler=export, stage_id=stage_id)


def export(args: argparse.Namespace) -> int:
    """Chạy lại một stage export của run.

    Args:
        args: Tham số dòng lệnh, có ``run_id`` và ``stage_id``.

    Returns:
        ``0`` nếu run xong, ``1`` nếu stage lỗi.
    """
    workdir = Workdir.from_env(args.workdir)
    run_dir = workdir.find_run(args.run_id)
    params: dict = {}
    if getattr(args, "fps", None):
        params["fps"] = args.fps
    if getattr(args, "no_interp", False):
        params["interpolate"] = False
    if getattr(args, "ego", None):
        params["ego"] = args.ego
    if getattr(args, "max_seconds", None) is not None:
        params["max_seconds"] = args.max_seconds
    stage_id = args.stage_id
    if stage_id == "export.ply_world":  # run Instant NuRec có stage PLY riêng
        record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
        stage_id = ply_export_stage([s.stage_id for s in record.stages])
    reset_stage(run_dir, stage_id, params=params or None)
    record = execute_run(run_dir, workdir, sink=event_printer(args))
    emit(args, record.model_dump(mode="json"), f"run {record.run_id}: {record.status}")
    return 0 if record.status == "succeeded" else 1
