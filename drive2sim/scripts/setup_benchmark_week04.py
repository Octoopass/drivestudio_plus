"""Chuẩn bị benchmark tuần 04 trên KITTI-360 chuỗi 0003: split, lệnh chạy, file định nghĩa.

Ma trận chạy và lý do chọn scene: ``docs/experiments/benchmark_week04.md``. Script chỉ đọc, ghi
file trong thư mục làm việc (``D2S_WORKDIR``, mặc định gốc repo), không chạy GPU:

    python -m drive2sim.scripts.setup_benchmark_week04 splits [--dry-run]
    python -m drive2sim.scripts.setup_benchmark_week04 commands
    python -m drive2sim.scripts.setup_benchmark_week04 definition [--out benchmarks/week04.yaml]

- ``splits``: tạo các split của A1, A2, A3, B2 (và 6 chunk của B2) bằng ``engine.scenes``. A3 cho
  theo số hiệu khung gốc của KITTI-360 nên được đổi sang chỉ số khung của cảnh qua
  ``frame_ids.txt``; in thêm độ đổi hướng xe trong A3 để kiểm có rẽ. Cảnh chưa ingest thì bỏ qua
  và in lệnh ingest.
- ``commands``: in lệnh ``d2s run start`` cho từng run của ma trận, theo số thứ tự ở Sheet.
- ``definition``: đọc ``run.json`` của các run đã xong, xếp vào nhóm của ma trận và ghi
  ``benchmarks/week04.yaml`` (CONTRACTS mục 6.5) cho ``d2s benchmark``. Run chạy thử (``--quick``)
  bị bỏ; một ô có nhiều run khớp thì lấy run mới nhất và báo.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from drive2sim.core import files
from drive2sim.core.schema import (
    SCHEMA_VERSIONS,
    BenchmarkCalibration,
    BenchmarkEntry,
    BenchmarkFile,
    RunRecord,
)
from drive2sim.engine import scenes
from drive2sim.engine.run import list_runs
from drive2sim.engine.workdir import Workdir, WorkdirError

SHORT_SCENE = "kitti360_2013_05_28_drive_0003_sync"
"""Cảnh của A1, A2 và track C: 200 khung đầu của chuỗi 0003 (khung gốc 1–220)."""

LONG_SCENE = "kitti360_0003_long"
"""Cảnh của A3 và B2: đoạn khung gốc 167–1030 của chuỗi 0003 (864 khung, khoảng 1,3 km)."""

LONG_SCENE_INGEST = (
    "d2s kaggle ingest <người dùng>/d2s-raw-kitti360-0003 "
    "--sequence 2013_05_28_drive_0003_sync --start-frame 167 --scene " + LONG_SCENE
)

STRATEGY = "every_nth_frame:8"
A3_SOURCE_RANGE = (560, 779)
"""A3 theo khung gốc: chứa người đi bộ duy nhất của chuỗi (khung 629–662) và xe chạy."""

CHUNK_COUNT = 6
CHUNK_OVERLAP = 10
TURN_DEG = 30.0
"""Độ đổi hướng tối thiểu để coi đoạn A3 là có rẽ."""

STEREO = ("image_00", "image_01")
FISHEYE = ("image_02", "image_03")


@dataclass(frozen=True)
class SplitPlan:
    """Một split của ma trận.

    Attributes:
        scene_id: Cảnh.
        split_id: Tên split.
        cameras: Camera của split.
        test_only: Camera chỉ test (R2).
        source_range: Đoạn khung theo số hiệu gốc của KITTI-360 (tính cả hai đầu); đổi sang
            chỉ số khung của cảnh qua ``frame_ids.txt``.
        index_range: Đoạn khung theo chỉ số của cảnh; ``None`` cả hai là cả cảnh.
        purpose: Dùng cho scene nào của ma trận.
    """

    scene_id: str
    split_id: str
    cameras: tuple[str, ...]
    test_only: tuple[str, ...] = ()
    source_range: Optional[tuple[int, int]] = None
    index_range: Optional[tuple[int, int]] = None
    purpose: str = ""


FIXED_SPLITS = (
    SplitPlan(SHORT_SCENE, "every8_stereo", STEREO, ("image_01",), purpose="A1"),
    SplitPlan(
        SHORT_SCENE, "every8_stereo_fisheye2", STEREO + FISHEYE, ("image_01",), purpose="A2"
    ),
    SplitPlan(
        SHORT_SCENE,
        "every8_mixed3",
        ("image_00",) + FISHEYE,
        purpose="C (Instant NuRec; prep.ncore chưa bỏ camera chỉ test nên không có image_01)",
    ),
    SplitPlan(
        LONG_SCENE,
        "every8_stereo_a3",
        STEREO,
        ("image_01",),
        source_range=A3_SOURCE_RANGE,
        purpose="A3",
    ),
    SplitPlan(
        LONG_SCENE,
        "every8_pinhole2",
        STEREO,
        purpose="B2, MCGS-SLAM (cần cả hai pinhole để train)",
    ),
)


def chunk_ranges(
    num_frames: int, count: int = CHUNK_COUNT, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Chia ``num_frames`` khung thành ``count`` đoạn, hai đoạn kề chồng ``overlap`` khung.

    Args:
        num_frames: Số khung của cảnh.
        count: Số đoạn.
        overlap: Số khung chung giữa hai đoạn kề nhau.

    Returns:
        Danh sách ``(khung đầu, khung cuối)``, tính cả hai đầu; đoạn cuối kết thúc ở khung cuối.

    Raises:
        ValueError: Nếu cảnh quá ngắn để chia.
    """
    if count < 1:
        raise ValueError(f"số đoạn phải dương, nhận {count}")
    length = math.ceil((num_frames + (count - 1) * overlap) / count)
    if length <= overlap:
        raise ValueError(f"cảnh {num_frames} khung quá ngắn để chia {count} đoạn chồng {overlap}")
    step = length - overlap
    ranges = []
    for index in range(count):
        start = index * step
        end = min(start + length - 1, num_frames - 1)
        ranges.append((start, end))
    ranges[-1] = (ranges[-1][0], num_frames - 1)
    return ranges


def chunk_splits(num_frames: int) -> list[SplitPlan]:
    """Split của 6 chunk trên B2 (run drivestudio chia chunk rồi gộp).

    Args:
        num_frames: Số khung của cảnh B2.

    Returns:
        Các split ``every8_stereo_c1`` … ``every8_stereo_c6``.
    """
    return [
        SplitPlan(
            LONG_SCENE,
            f"every8_stereo_c{index}",
            STEREO,
            ("image_01",),
            index_range=frames,
            purpose=f"B2 chunk {index}",
        )
        for index, frames in enumerate(chunk_ranges(num_frames), start=1)
    ]


def read_frame_ids(scene_dir: Path) -> list[int]:
    """Đọc số hiệu khung gốc của từng khung trong cảnh (``frame_ids.txt``, mỗi dòng một số)."""
    text = (scene_dir / "frame_ids.txt").read_text(encoding="utf-8")
    return [int(line) for line in text.split()]


def source_to_index_range(
    frame_ids: Sequence[int], source_range: tuple[int, int]
) -> tuple[int, int]:
    """Đổi đoạn khung gốc sang đoạn chỉ số khung của cảnh.

    Args:
        frame_ids: Số hiệu gốc của từng khung, theo thứ tự trong cảnh.
        source_range: Khung gốc đầu và cuối, tính cả hai đầu.

    Returns:
        ``(chỉ số đầu, chỉ số cuối)`` của các khung có số hiệu gốc nằm trong đoạn.

    Raises:
        ValueError: Nếu không khung nào nằm trong đoạn.
    """
    low, high = source_range
    indices = [index for index, frame in enumerate(frame_ids) if low <= frame <= high]
    if not indices:
        raise ValueError(
            f"không khung nào có số hiệu gốc trong {low}–{high}; cảnh có "
            f"{frame_ids[0] if frame_ids else '?'}–{frame_ids[-1] if frame_ids else '?'}"
        )
    return (indices[0], indices[-1])


def heading_change_deg(scene_dir: Path, index_range: tuple[int, int]) -> float:
    """Độ đổi hướng của xe (độ) trong một đoạn khung, từ ``ego_pose/``.

    Hướng xe là góc của trục x của xe chiếu lên mặt phẳng ngang của hệ thế giới; trả về khoảng
    chênh giữa góc lớn nhất và nhỏ nhất sau khi nối liền góc qua ±180 độ.

    Args:
        scene_dir: Thư mục cảnh.
        index_range: Đoạn chỉ số khung, tính cả hai đầu.

    Returns:
        Độ đổi hướng, độ.
    """
    poses = sorted((scene_dir / "ego_pose").glob("*.txt"), key=lambda path: int(path.stem))
    headings: list[float] = []
    for path in poses[index_range[0] : index_range[1] + 1]:
        rows = [[float(value) for value in line.split()] for line in path.read_text().splitlines()]
        angle = math.atan2(rows[1][0], rows[0][0])
        if headings:
            while angle - headings[-1] > math.pi:
                angle -= 2 * math.pi
            while angle - headings[-1] < -math.pi:
                angle += 2 * math.pi
        headings.append(angle)
    if not headings:
        return 0.0
    return math.degrees(max(headings) - min(headings))


def create_splits(workdir: Workdir, dry_run: bool = False) -> int:
    """Tạo mọi split của ma trận; cảnh chưa ingest thì bỏ qua và in lệnh ingest.

    Args:
        workdir: Thư mục làm việc.
        dry_run: Chỉ in split sẽ tạo, không ghi file.

    Returns:
        Mã thoát: 0 nếu mọi split đã có hoặc vừa tạo, 1 nếu thiếu cảnh hay có lỗi.
    """
    status = 0
    plans = list(FIXED_SPLITS)
    try:
        long_scene = workdir.load_scene(LONG_SCENE)
        plans += chunk_splits(long_scene.manifest.timeline.num_frames)
    except WorkdirError:
        print(f"chưa có cảnh {LONG_SCENE} (A3, B2); ingest bằng:\n  {LONG_SCENE_INGEST}")
    for plan in plans:
        scene_dir = workdir.scene_dir(plan.scene_id)
        if not (scene_dir / "manifest.yaml").is_file():
            print(f"bỏ qua {plan.split_id} ({plan.purpose}): chưa có cảnh {plan.scene_id}")
            status = 1
            continue
        frame_range = plan.index_range
        if plan.source_range is not None:
            try:
                frame_range = source_to_index_range(read_frame_ids(scene_dir), plan.source_range)
            except (OSError, ValueError) as error:
                print(f"lỗi {plan.split_id}: {error}")
                status = 1
                continue
            turn = heading_change_deg(scene_dir, frame_range)
            verdict = "có rẽ" if turn >= TURN_DEG else f"KHÔNG có rẽ (dưới {TURN_DEG:.0f} độ)"
            print(f"{plan.split_id}: xe đổi hướng {turn:.1f} độ trong đoạn, {verdict}")
        where = f"khung {frame_range[0]}–{frame_range[1]}" if frame_range else "cả cảnh"
        label = f"{plan.scene_id}/{plan.split_id} ({plan.purpose}; {where})"
        if workdir.split_path(plan.scene_id, plan.split_id).is_file():
            print(f"đã có  {label}")
            continue
        if dry_run:
            print(f"sẽ tạo {label}")
            continue
        try:
            scenes.create_split(
                workdir,
                plan.scene_id,
                STRATEGY,
                cameras=list(plan.cameras),
                frame_range=frame_range,
                split_id=plan.split_id,
                test_only_cameras=list(plan.test_only),
            )
        except (scenes.SplitError, WorkdirError) as error:
            print(f"lỗi {label}: {error}")
            status = 1
            continue
        print(f"đã tạo {label}")
    return status


@dataclass(frozen=True)
class RunCommand:
    """Lệnh chạy một run của ma trận (số thứ tự khớp tab Ma trận của Sheet)."""

    number: str
    approach: str
    scene_id: str
    split_id: str
    sets: tuple[str, ...] = ()
    quick: bool = False
    note: str = ""

    def render(self) -> str:
        """Chuỗi lệnh ``d2s run start``."""
        parts = [
            "d2s run start",
            self.approach,
            "--scene",
            self.scene_id,
            "--split",
            self.split_id,
        ]
        for value in self.sets:
            parts += ["--set", value]
        if self.quick:
            parts.append("--quick")
        return " ".join(parts)


OMNIRE_FASTGS = ("fastgs=true",)
A_L_WINNER = "<thắng A-L: pvg|deformablegs>"
A_LB_WINNER = "<thắng A-LB: omnire + các --set của run thắng>"

RUN_COMMANDS = (
    RunCommand("1", "omnire", SHORT_SCENE, "every8_stereo", OMNIRE_FASTGS, True, "smoke R2"),
    RunCommand("2", "pvg", SHORT_SCENE, "every8_stereo", (), True, "smoke R2"),
    RunCommand("3", "deformablegs", SHORT_SCENE, "every8_stereo", (), True, "smoke R2"),
    RunCommand("4", "omnire", SHORT_SCENE, "every8_stereo", OMNIRE_FASTGS, note="A-LB, seed 0"),
    RunCommand("5", "pvg", SHORT_SCENE, "every8_stereo", note="A-L"),
    RunCommand("6", "deformablegs", SHORT_SCENE, "every8_stereo", note="A-L"),
    RunCommand(
        "7",
        "omnire",
        SHORT_SCENE,
        "every8_stereo",
        OMNIRE_FASTGS + ("train.drivestudio.seed=1",),
        note="hiệu chỉnh seed",
    ),
    RunCommand(
        "8",
        "omnire",
        SHORT_SCENE,
        "every8_stereo",
        OMNIRE_FASTGS + ("train.drivestudio.seed=2",),
        note="hiệu chỉnh seed",
    ),
    RunCommand(
        "9",
        A_L_WINNER,
        SHORT_SCENE,
        "every8_stereo",
        ("train.drivestudio.lidar=false",),
        note="A-0",
    ),
    RunCommand(
        "10",
        "omnire",
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        ("renderer=3dgut",) + OMNIRE_FASTGS,
        note="A-LB fisheye",
    ),
    RunCommand(
        "11",
        "omnire",
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        ("renderer=3dgeer",) + OMNIRE_FASTGS,
        note="A-LB fisheye",
    ),
    RunCommand(
        "12",
        A_L_WINNER,
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        ("renderer=3dgut", "campose=false"),
        note="A-L fisheye",
    ),
    RunCommand("13", "mcgs", LONG_SCENE, "every8_pinhole2", note="track B"),
    RunCommand("15", "nurec", SHORT_SCENE, "every8_mixed3", note="track C"),
    RunCommand("16", A_LB_WINNER, LONG_SCENE, "every8_stereo_a3", note="A3"),
)
SKIPPED_COMMANDS = (
    "#14 (smoke Instant NuRec): chạy trên clip mẫu NCore, không qua cảnh KITTI-360",
)


def print_commands(workdir: Workdir) -> int:
    """In lệnh chạy từng run của ma trận, kể cả 6 chunk của B2 (#17)."""
    for command in RUN_COMMANDS:
        print(f"# {command.number} {command.note}\n{command.render()}")
    num_frames = None
    try:
        num_frames = workdir.load_scene(LONG_SCENE).manifest.timeline.num_frames
    except WorkdirError:
        pass
    count = len(chunk_splits(num_frames)) if num_frames else CHUNK_COUNT
    print("# 17 track B, drivestudio chia chunk: mỗi chunk một run, cùng cấu hình thắng A-LB")
    for index in range(1, count + 1):
        print(RunCommand("17", A_LB_WINNER, LONG_SCENE, f"every8_stereo_c{index}").render())
    for line in SKIPPED_COMMANDS:
        print(f"# không có lệnh: {line}")
    return 0


@dataclass(frozen=True)
class Slot:
    """Một loại run trong nhóm của file định nghĩa.

    Attributes:
        approach_id: Approach.
        options: Lựa chọn phải khớp (tập con của ``approach.options`` của run).
        seed: Seed của ``train.drivestudio`` (0 khi run không có stage này).
        lidar: Tham số ``lidar`` của ``train.drivestudio`` (``True`` khi không có stage này).
    """

    approach_id: str
    options: tuple[tuple[str, Any], ...] = ()
    seed: int = 0
    lidar: bool = True


@dataclass(frozen=True)
class EntryPlan:
    """Một nhóm run được xếp hạng với nhau (một mục ``entries`` của file định nghĩa)."""

    bench_track: str
    cell: Optional[str]
    scene_id: str
    split_id: str
    slots: tuple[Slot, ...]
    label: str = ""


OMNIRE_A1 = Slot("omnire", (("renderer", "3dgs"), ("fastgs", True)))
PINHOLE = (("renderer", "3dgs"),)
FISHEYE_UT = (("renderer", "3dgut"),)
ENTRY_PLANS = (
    EntryPlan("A", "A-LB", SHORT_SCENE, "every8_stereo", (OMNIRE_A1, Slot("beziergs")), "A1 A-LB"),
    EntryPlan(
        "A",
        "A-L",
        SHORT_SCENE,
        "every8_stereo",
        (Slot("pvg", PINHOLE), Slot("deformablegs", PINHOLE)),
        "A1 A-L",
    ),
    EntryPlan(
        "A",
        "A-0",
        SHORT_SCENE,
        "every8_stereo",
        (Slot("pvg", PINHOLE, lidar=False), Slot("deformablegs", PINHOLE, lidar=False)),
        "A1 A-0",
    ),
    EntryPlan(
        "A",
        "A-LB",
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        (
            Slot("omnire", (("renderer", "3dgut"), ("fastgs", True))),
            Slot("omnire", (("renderer", "3dgeer"), ("fastgs", True))),
        ),
        "A2 A-LB",
    ),
    EntryPlan(
        "A",
        "A-L",
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        (Slot("pvg", FISHEYE_UT), Slot("deformablegs", FISHEYE_UT)),
        "A2 A-L",
    ),
    EntryPlan("A", "A-LB", LONG_SCENE, "every8_stereo_a3", (Slot("omnire"),), "A3 A-LB"),
    EntryPlan(
        "A",
        "A-L",
        LONG_SCENE,
        "every8_stereo_a3",
        (Slot("pvg", PINHOLE), Slot("deformablegs", PINHOLE)),
        "A3 A-L",
    ),
    EntryPlan("B", None, LONG_SCENE, "every8_pinhole2", (Slot("mcgs"),), "B2 MCGS-SLAM"),
    EntryPlan("C", None, SHORT_SCENE, "every8_mixed3", (Slot("nurec"),), "C Instant NuRec"),
)
CALIBRATION_SLOTS = tuple(
    Slot(OMNIRE_A1.approach_id, OMNIRE_A1.options, seed) for seed in (0, 1, 2)
)


def train_params(run: RunRecord) -> dict[str, Any]:
    """Tham số của stage ``train.drivestudio`` trong run; rỗng nếu run không có stage này."""
    for stage in run.stages:
        if stage.stage_id == "train.drivestudio":
            return stage.params
    return {}


def matches(run: RunRecord, scene_id: str, split_id: str, slot: Slot) -> bool:
    """Run đã xong, không phải chạy thử, đúng cảnh, split và loại run của ``slot``."""
    if run.status != "succeeded" or run.approach.quick:
        return False
    if (run.scene_id, run.split_id, run.approach.approach_id) != (
        scene_id,
        split_id,
        slot.approach_id,
    ):
        return False
    if any(run.approach.options.get(key) != value for key, value in slot.options):
        return False
    params = train_params(run)
    return (
        int(params.get("seed", 0)) == slot.seed and bool(params.get("lidar", True)) == slot.lidar
    )


@dataclass
class Definition:
    """Kết quả xếp run vào nhóm: file định nghĩa (nếu có nhóm nào) và ghi chú cho người dùng."""

    benchmark: Optional[BenchmarkFile]
    notes: list[str] = field(default_factory=list)


def pick(
    runs: Sequence[RunRecord], scene_id: str, split_id: str, slot: Slot, notes: list[str]
) -> Optional[RunRecord]:
    """Run mới nhất khớp ``slot``; ghi chú khi có nhiều run khớp. ``runs`` sắp mới nhất trước."""
    found = [run for run in runs if matches(run, scene_id, split_id, slot)]
    if len(found) > 1:
        others = ", ".join(run.run_id for run in found[1:])
        notes.append(f"{split_id} {slot.approach_id}: lấy {found[0].run_id}, bỏ {others}")
    return found[0] if found else None


def build_definition(runs: Sequence[RunRecord]) -> Definition:
    """Xếp các run vào nhóm của ma trận.

    Args:
        runs: Mọi run của thư mục làm việc, mới nhất trước (như ``list_runs``).

    Returns:
        File định nghĩa ``d2s-benchmark`` (``None`` nếu chưa nhóm nào có run) và ghi chú.
    """
    notes: list[str] = []
    entries = []
    for plan in ENTRY_PLANS:
        chosen = [pick(runs, plan.scene_id, plan.split_id, slot, notes) for slot in plan.slots]
        run_ids = list(dict.fromkeys(run.run_id for run in chosen if run is not None))
        if not run_ids:
            notes.append(f"{plan.label}: chưa có run nào xong, bỏ nhóm")
            continue
        entries.append(
            BenchmarkEntry(
                bench_track=plan.bench_track,
                cell=plan.cell,
                scene_id=plan.scene_id,
                split_id=plan.split_id,
                runs=run_ids,
            )
        )
    calibration_runs = [
        run.run_id
        for run in (
            pick(runs, SHORT_SCENE, "every8_stereo", slot, notes) for slot in CALIBRATION_SLOTS
        )
        if run is not None
    ]
    calibration = None
    if len(calibration_runs) >= 2:
        calibration = BenchmarkCalibration(
            scene_id=SHORT_SCENE, split_id="every8_stereo", runs=calibration_runs
        )
    else:
        notes.append(
            f"hiệu chỉnh seed: mới có {len(calibration_runs)} run (cần ít nhất 2), "
            "d2s benchmark dùng ngưỡng hoà mặc định 0,3 dB"
        )
    notes.append(
        "#17 (6 chunk của B2) chưa vào file định nghĩa: d2s benchmark chưa có nhóm chunked "
        "(ARCHITECTURE mục 2.17)"
    )
    if not entries:
        return Definition(None, notes)
    benchmark = BenchmarkFile(
        schema_version=SCHEMA_VERSIONS["benchmark"], calibration=calibration, entries=entries
    )
    return Definition(benchmark, notes)


def write_definition(workdir: Workdir, out: Optional[Path]) -> int:
    """Ghi ``benchmarks/week04.yaml`` từ các run đã xong; in ghi chú."""
    result = build_definition(list_runs(workdir))
    for note in result.notes:
        print(note)
    if result.benchmark is None:
        print("chưa ghi file: chưa nhóm nào có run đã xong")
        return 1
    path = files.dump(result.benchmark, out or workdir.benchmarks_dir / "week04.yaml")
    print(f"đã ghi {path} ({len(result.benchmark.entries)} nhóm); xem bằng: d2s benchmark {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, help="thư mục làm việc; mặc định D2S_WORKDIR")
    actions = parser.add_subparsers(dest="action", required=True)
    splits = actions.add_parser("splits", help="tạo split của ma trận")
    splits.add_argument("--dry-run", action="store_true", help="chỉ in, không ghi file")
    actions.add_parser("commands", help="in lệnh d2s run start của từng run")
    definition = actions.add_parser("definition", help="ghi file định nghĩa benchmark")
    definition.add_argument("--out", type=Path, help="mặc định benchmarks/week04.yaml")
    args = parser.parse_args(argv)
    workdir = Workdir.from_env(args.workdir)
    if args.action == "splits":
        return create_splits(workdir, args.dry_run)
    if args.action == "commands":
        return print_commands(workdir)
    return write_definition(workdir, args.out)


if __name__ == "__main__":
    sys.exit(main())
