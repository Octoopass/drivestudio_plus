"""Nhóm lệnh ``d2s approach``: xem các approach và kiểm tra trước khi chạy.

File này chứa:
- ``register``: thêm các lệnh ``approach list``, ``approach show``, ``approach check``.
- ``list_approaches``, ``show_approach``: liệt kê và xem mục cấu hình, preset, ràng buộc, stage.
- ``check_approach``: chạy đúng bước kiểm tra mà ``d2s run start`` sẽ chạy, rồi in danh sách stage
  sẽ chạy và stage nào dùng lại cache, mà không chạy gì (CONTRACTS mục 10.4).
- ``resolve_from_args``: đọc cảnh, split, approach và ``--set`` rồi gọi ``resolve``; ``d2s run
  start`` dùng lại hàm này để hai lệnh luôn cho cùng kết quả.
"""

import argparse

from drive2sim.cli.output import emit, format_table, format_value, parse_set
from drive2sim.core.schema import Split
from drive2sim.engine.approaches import list_approach_ids, load_approach
from drive2sim.engine.environment import environment_check
from drive2sim.engine.resolve import QUICK_NUM_ITERS, Plan, resolve
from drive2sim.engine.run import preview_reuse
from drive2sim.engine.stage import Scene
from drive2sim.engine.workdir import Workdir


def add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    """Thêm các tuỳ chọn chung của ``approach check`` và ``run start``.

    Args:
        parser: Parser của lệnh.
    """
    parser.add_argument("approach_id")
    parser.add_argument("--scene", dest="scene_id", required=True)
    parser.add_argument("--split", dest="split_id", required=True)
    parser.add_argument("--preset")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KHOÁ=GIÁ_TRỊ",
        help=(
            "đặt cấu hình approach (renderer=3dgeer) "
            "hoặc tham số stage (train.drivestudio.num_iters=1000)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"chạy thử nhanh: train {QUICK_NUM_ITERS} iteration để kiểm tra approach chạy được",
    )


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
    common: argparse.ArgumentParser,
) -> None:
    """Thêm nhóm lệnh ``approach`` vào parser của ``d2s``.

    Args:
        subparsers: Nơi thêm nhóm lệnh.
        common: Parser chứa tuỳ chọn chung.
    """
    group = subparsers.add_parser("approach", help="xem approach, kiểm tra trước khi chạy")
    actions = group.add_subparsers(dest="action", required=True)
    actions.add_parser("list", parents=[common], help="liệt kê approach").set_defaults(
        handler=list_approaches
    )
    show = actions.add_parser(
        "show", parents=[common], help="xem cấu hình approach, preset, stage"
    )
    show.add_argument("approach_id")
    show.set_defaults(handler=show_approach)
    check = actions.add_parser(
        "check", parents=[common], help="kiểm tra điều kiện, in danh sách stage sẽ chạy"
    )
    add_plan_arguments(check)
    check.set_defaults(handler=check_approach)


def list_approaches(args: argparse.Namespace) -> int:
    """Liệt kê các approach có sẵn.

    Args:
        args: Tham số dòng lệnh.

    Returns:
        Mã thoát ``0``.
    """
    approaches = [load_approach(approach_id) for approach_id in list_approach_ids()]
    rows = [[a.approach_id, a.name, a.track, a.level, a.codebase] for a in approaches]
    emit(
        args,
        [a.model_dump(mode="json") for a in approaches],
        format_table(["approach_id", "tên", "track", "mức", "codebase"], rows),
    )
    return 0


def show_approach(args: argparse.Namespace) -> int:
    """In mục cấu hình, preset, ràng buộc và các stage của một approach.

    Args:
        args: Tham số dòng lệnh, có ``approach_id``.

    Returns:
        Mã thoát ``0``.
    """
    approach = load_approach(args.approach_id)
    lines = [f"{approach.approach_id}: {approach.name} (track {approach.track}, {approach.level})"]
    lines.append("cấu hình approach:")
    for name, spec in approach.options.items():
        choices = ", ".join(format_value(choice) for choice in spec.choices)
        lines.append(f"  {name}: {choices} (mặc định {format_value(spec.default)})")
    lines.append("preset:")
    for name, values in approach.presets.items():
        lines.append(
            f"  {name}: " + ", ".join(f"{k}={format_value(v)}" for k, v in values.items())
        )
    lines.append("ràng buộc:")
    for constraint in approach.constraints:
        pairs = {**constraint.when, **constraint.forbid}
        combination = ", ".join(f"{k}={format_value(v)}" for k, v in pairs.items())
        lines.append(f"  không cho {combination}: {constraint.reason}")
    lines.append("stage:")
    lines += [f"  {index + 1}. {entry.stage}" for index, entry in enumerate(approach.stages)]
    emit(args, approach.model_dump(mode="json"), "\n".join(lines))
    return 0


def resolve_from_args(args: argparse.Namespace) -> tuple[Workdir, Scene, Split, Plan]:
    """Đọc cảnh, split, approach và ``--set`` từ dòng lệnh rồi quyết định danh sách stage.

    Args:
        args: Tham số dòng lệnh có ``approach_id``, ``scene_id``, ``split_id``, ``preset``,
            ``sets``, ``quick``, ``workdir``.

    Returns:
        Bộ ``(thư mục làm việc, cảnh, split, plan)``.

    Raises:
        WorkdirError: Nếu cảnh, split hoặc approach không có.
        UsageError: Nếu một ``--set`` sai dạng.
        ResolveError: Nếu preset, mục cấu hình, giá trị hoặc tham số không tồn tại.
        BlockedError: Nếu dữ liệu hoặc tổ hợp lựa chọn không cho chạy.
    """
    workdir = Workdir.from_env(args.workdir)
    scene = workdir.load_scene(args.scene_id)
    split = workdir.load_split(args.scene_id, args.split_id)
    approach = load_approach(args.approach_id)
    options, stage_params = parse_set(args.sets)
    plan = resolve(
        approach,
        scene.manifest,
        split,
        preset=args.preset,
        option_values=options,  # type: ignore[arg-type]
        stage_params=stage_params,
        quick=args.quick,
    )
    return workdir, scene, split, plan


def check_approach(args: argparse.Namespace) -> int:
    """Kiểm tra một approach chạy được trên cảnh và split không, in danh sách stage sẽ chạy.

    Args:
        args: Tham số dòng lệnh như ``run start``.

    Returns:
        Mã thoát ``0`` nếu chạy được; khi bị khoá thì ``main`` trả ``3``.
    """
    workdir, scene, split, plan = resolve_from_args(args)
    reused = preview_reuse(plan, scene, split, workdir)
    rows = [
        [index + 1, stage.stage_id, "dùng lại cache" if hit else "chạy", stage.params]
        for index, (stage, hit) in enumerate(zip(plan.stages, reused))
    ]
    options = ", ".join(f"{name}={format_value(value)}" for name, value in plan.options.items())
    # Môi trường Python của máy này (chỉ có ý nghĩa khi chạy trên máy; Kaggle cài trong notebook).
    env = environment_check(load_approach(plan.approach_id).codebase)
    env_lines = []
    if env["env"] and env["level"] != "ok":
        problems = [
            f"{c['id']} {c['found'] or 'chưa có'}"
            + (f" (cần {c['expected']})" if c["expected"] else "")
            for c in env["checks"]
            if c["level"] != "ok"
        ]
        env_lines.append(
            f"môi trường {env['env']} trên máy này: {', '.join(problems)}; "
            f"cài bằng {env['install']} (chạy trên Kaggle thì notebook tự cài)"
        )
    text = "\n".join(
        [
            f"approach {plan.approach_id} chạy được trên {scene.scene_id}, split {split.split_id}",
            f"cấu hình: {options or '(không có)'}",
            f"LiDAR dùng ở: {', '.join(plan.lidar_usage) or 'không dùng'}",
            *(["chạy thử nhanh: metric không so được với run đủ iteration"] if plan.quick else []),
            *env_lines,
            format_table(["#", "stage", "", "tham số"], rows),
        ]
    )
    data = {
        "approach_id": plan.approach_id,
        "preset": plan.preset,
        "quick": plan.quick,
        "options": plan.options,
        "lidar_usage": plan.lidar_usage,
        "stages": [
            {"stage_id": stage.stage_id, "params": stage.params, "cache_hit": hit}
            for stage, hit in zip(plan.stages, reused)
        ],
        "environment": env,
    }
    emit(args, data, text)
    return 0
