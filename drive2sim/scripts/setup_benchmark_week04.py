"""Chuẩn bị benchmark tuần 04 (Waymo W1, KITTI-360 chuỗi 0003): split, lệnh chạy, file định nghĩa.

Ma trận chạy và lý do chọn scene: ``docs/experiments/benchmark_week04.md``. Script chỉ đọc, ghi
file trong thư mục làm việc (``D2S_WORKDIR``, mặc định gốc repo), không chạy GPU:

    python -m drive2sim.scripts.setup_benchmark_week04 splits [--dry-run]
    python -m drive2sim.scripts.setup_benchmark_week04 commands
    python -m drive2sim.scripts.setup_benchmark_week04 definition [--out benchmarks/week04.yaml]
    python -m drive2sim.scripts.setup_benchmark_week04 queue --wave 1 [--out <file.tsv>]
    python -m drive2sim.scripts.setup_benchmark_week04 waymo-stats

- ``splits``: tạo các split của W1, A2, A3, B2 (và 6 chunk của B2) bằng ``engine.scenes``.
  A3 cho theo số hiệu khung gốc của KITTI-360 nên được đổi sang chỉ số khung của cảnh qua
  ``frame_ids.txt``; in thêm độ đổi hướng xe trong A3 để kiểm có rẽ. Cảnh chưa ingest thì bỏ qua
  và in lệnh ingest.
- ``commands``: in lệnh ``d2s run start`` cho từng run của ma trận, theo số thứ tự ở Sheet
  (#1–#14 sau khi nhóm đánh số lại 26/11).
- ``definition``: đọc ``run.json`` của các run đã xong, xếp vào nhóm của ma trận và ghi
  ``benchmarks/week04.yaml`` (CONTRACTS mục 6.5) cho ``d2s benchmark``. Run chạy thử (``--quick``)
  bị bỏ; một ô có nhiều run khớp thì lấy run mới nhất và báo.
- ``queue``: ghi hàng đợi run của một đợt cho ``envs/bench_queue.sh`` (mỗi GPU một worker),
  mỗi dòng ``<số run><tab><tham số của d2s run start>``, run dài trước. Đợt 1 gồm các run không
  cần biết run thắng; đợt 2 cần ``--al-winner``, đợt 3 cần ``--alb-winner`` (chọn sau khi xem
  ``d2s benchmark``). Kế hoạch chạy: ``docs/experiments/benchmark_week04.md``.
- ``waymo-stats``: quãng đường, tốc độ, độ đổi hướng và số vật thể (xe, người, vật chạy) của các
  cảnh Waymo ứng viên cho W1 đã ingest, xếp quãng đường dài trước, để chọn ``D2S_W1_INDEX``.
"""

import argparse
import json
import math
import os
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from drive2sim.core import files
from drive2sim.core.schema import (
    SCHEMA_VERSIONS,
    BenchmarkCalibration,
    BenchmarkChunk,
    BenchmarkChunkedEntry,
    BenchmarkEntry,
    BenchmarkFile,
    RunRecord,
)
from drive2sim.engine import scenes
from drive2sim.engine.run import list_runs
from drive2sim.engine.workdir import Workdir, WorkdirError

SHORT_SCENE = "kitti360_2013_05_28_drive_0003_sync"
"""Cảnh của A2: 200 khung đầu của chuỗi 0003 (khung gốc 1–220)."""

LONG_SCENE = "kitti360_0003_long"
"""Cảnh của A3, B2 và track C: khung gốc 167–1030 của chuỗi 0003 (864 khung, khoảng 1,3 km)."""

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

WAYMO_TRAIN_LIST = Path(__file__).resolve().parents[1] / "assets/scene_lists/waymo_train_list.txt"
"""Danh sách segment Waymo của DriveStudio (EmerNeRF): chỉ số = số dòng trừ 1."""
WAYMO_W1_CANDIDATES = (2, 41, 46, 62, 71, 73, 82, 83)
"""Ứng viên W1: nhóm High-speed của Diverse-56 trong NOTR (EmerNeRF, ``docs/NOTR.md``), tức các
segment xe chạy nhanh nên đi xa nhất trong 20 giây của một segment Waymo. Chọn bằng
``waymo-stats`` sau khi ingest (quãng đường, số vật thể)."""
WAYMO_W1_PREVIOUS = 23
"""W1 của đợt 1, 2 (``segment-10455472356147194054``, khung 0–149): quãng đường ngắn, thay ngày
27/11/2026; run cũ trên cảnh đó giữ nguyên, không vào file định nghĩa mới."""
WAYMO_INDEX = int(os.environ.get("D2S_W1_INDEX", str(WAYMO_W1_CANDIDATES[0])))
"""Chỉ số W1 trong ``waymo_train_list.txt``; đổi bằng biến môi trường ``D2S_W1_INDEX`` (đặt cả ở
máy thuê và máy mình, vì ``definition`` nhận run theo ``scene_id``). Mặc định là ứng viên đầu, tạm
thời cho tới khi chọn bằng ``waymo-stats``."""


def waymo_segment(index: int) -> str:
    """Tên segment Waymo ở dòng ``index + 1`` của ``waymo_train_list.txt``.

    Raises:
        ValueError: Nếu chỉ số nằm ngoài danh sách.
    """
    names = WAYMO_TRAIN_LIST.read_text(encoding="utf-8").split()
    if not 0 <= index < len(names):
        raise ValueError(f"chỉ số Waymo {index} ngoài danh sách 0–{len(names) - 1}")
    return names[index]


WAYMO_SEGMENT = waymo_segment(WAYMO_INDEX)
WAYMO_SCENE = "waymo_" + WAYMO_SEGMENT
"""W1: scene pinhole nhiều camera (Waymo, 5 camera, cả segment). Thay A1 cho các ô xếp hạng
pinhole, vì A1 chỉ train một camera trước; từ 27/11/2026 cũng là cảnh của #8 (MCGS-SLAM)."""
WAYMO_INGEST = (
    "d2s scene ingest <thư mục có "
    + WAYMO_SEGMENT
    + ".tfrecord> --sequence "
    + WAYMO_SEGMENT
    + f"  (tải tfrecord bằng drive2sim.recon.preprocess.waymo.download --scene_ids {WAYMO_INDEX})"
)
WAYMO_CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT")
WAYMO_FRAMES: Optional[tuple[int, int]] = None
"""Cả segment (khoảng 198 khung): W1 mới chọn vì đi xa, nên không cắt còn 150 khung như W1 cũ."""
MCGS_W1_CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
"""Camera của #8 trên W1: ba camera trước cùng độ phân giải 1920×1280 (hai camera bên 1920×886),
như lần chạy MCGS-SLAM trên Waymo 017 (ARCHITECTURE mục 2.19)."""
WAYMO_SPLIT = "every8_5cams"

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
    SplitPlan(
        WAYMO_SCENE,
        WAYMO_SPLIT,
        WAYMO_CAMERAS,
        index_range=WAYMO_FRAMES,
        purpose="W1, pinhole xếp hạng (R2 không áp dụng)",
    ),
    SplitPlan(
        SHORT_SCENE, "every8_stereo_fisheye2", STEREO + FISHEYE, ("image_01",), purpose="A2"
    ),
    SplitPlan(
        LONG_SCENE,
        "every8_mixed3",
        ("image_00",) + FISHEYE,
        purpose="C, Instant NuRec trên cảnh dài, camera như A2 (prep.ncore chưa bỏ camera chỉ "
        "test nên không có image_01)",
    ),
    SplitPlan(
        LONG_SCENE,
        "every8_stereo_a3",
        STEREO,
        ("image_01",),
        source_range=A3_SOURCE_RANGE,
        purpose="A3",
    ),
)
"""Split ``every8_pinhole2`` (B2, MCGS-SLAM trên cảnh dài) bỏ từ 27/11/2026: #8 chuyển sang W1."""


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
    if not (workdir.scene_dir(WAYMO_SCENE) / "manifest.yaml").is_file():
        print(f"chưa có cảnh {WAYMO_SCENE} (W1); ingest bằng:\n  {WAYMO_INGEST}")
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

    def argv(self) -> list[str]:
        """Tham số của ``d2s run start`` (cũng là tham số của ``d2s approach check``)."""
        parts = [self.approach, "--scene", self.scene_id, "--split", self.split_id]
        for value in self.sets:
            parts += ["--set", value]
        if self.quick:
            parts.append("--quick")
        return parts

    def render(self) -> str:
        """Chuỗi lệnh ``d2s run start``."""
        return " ".join(["d2s run start", *self.argv()])


OMNIRE_FASTGS = ("fastgs=true",)
A_L_WINNER = "<thắng A-L: pvg|deformablegs>"
A_LB_WINNER = "<thắng A-LB: omnire + các --set của run thắng>"
A_L_CHOSEN = "pvg"
"""Thắng A-L do nhóm chốt 27/11/2026 (PVG, sau #2 và #3); mặc định của ``--al-winner``."""
A_LB_CHOSEN = "omnire renderer=3dgeer fastgs=true"
"""Thắng A-LB do nhóm chốt 27/11/2026 (OmniRe 3DGEER + FastGS, sau #5 và #6); mặc định của
``--alb-winner``."""

RUN_COMMANDS = (
    RunCommand("1", "omnire", WAYMO_SCENE, WAYMO_SPLIT, OMNIRE_FASTGS, note="A-LB"),
    RunCommand("2", "pvg", WAYMO_SCENE, WAYMO_SPLIT, note="A-L"),
    RunCommand("3", "deformablegs", WAYMO_SCENE, WAYMO_SPLIT, note="A-L"),
    RunCommand(
        "4",
        A_L_WINNER,
        WAYMO_SCENE,
        WAYMO_SPLIT,
        ("train.drivestudio.lidar=false",),
        note="A-0",
    ),
    RunCommand(
        "5",
        "omnire",
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        ("renderer=3dgut",) + OMNIRE_FASTGS,
        note="A-LB fisheye",
    ),
    RunCommand(
        "6",
        "omnire",
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        ("renderer=3dgeer",) + OMNIRE_FASTGS,
        note="A-LB fisheye",
    ),
    RunCommand(
        "7",
        A_L_WINNER,
        SHORT_SCENE,
        "every8_stereo_fisheye2",
        ("renderer=3dgeer", "campose=false"),  # renderer fisheye chốt 27/11: 3DGEER
        note="A-L fisheye",
    ),
    RunCommand(
        "8",
        "mcgs",
        WAYMO_SCENE,
        WAYMO_SPLIT,
        ("prep.mcgs.cameras=" + ",".join(MCGS_W1_CAMERAS),),  # --set nhận danh sách a,b,c
        note="track B, MCGS-SLAM trên W1 (ba camera trước)",
    ),
    RunCommand(
        "9",
        "nurec",
        LONG_SCENE,
        "every8_mixed3",
        note="track C; cảnh dài cần ~48 đoạn NuRec, tool giới hạn 8: chọn cách chia trước",
    ),
    RunCommand("10", A_LB_WINNER, LONG_SCENE, "every8_stereo_a3", note="A3"),
    RunCommand(
        "12",
        A_L_WINNER,
        LONG_SCENE,
        "every8_stereo_a3",
        note="A3, A-L (ưu tiên 2)",
    ),
    RunCommand("13", "beziergs", WAYMO_SCENE, WAYMO_SPLIT, note="W1, A-LB (ưu tiên 2)"),
)
CHUNK_RUN_NUMBER = 11  # #11: drivestudio chia chunk trên B2, một run cho mỗi chunk
SKIPPED_COMMANDS = (
    "#14 (SplatAD, ưu tiên 2, làm nếu thừa thời gian): chưa có approach, cần port vào trainer",
)


def print_commands(workdir: Workdir) -> int:
    """In lệnh chạy từng run của ma trận, kể cả 6 chunk của B2 (#11)."""

    def show(commands: Sequence[RunCommand]) -> None:
        for command in commands:
            command = fill_winners(command)
            print(f"# {command.number} {command.note}\n{command.render()}")

    show([c for c in RUN_COMMANDS if int(c.number) < CHUNK_RUN_NUMBER])
    num_frames = None
    try:
        num_frames = workdir.load_scene(LONG_SCENE).manifest.timeline.num_frames
    except WorkdirError:
        pass
    count = len(chunk_splits(num_frames)) if num_frames else CHUNK_COUNT
    print(
        f"# {CHUNK_RUN_NUMBER} track B, drivestudio chia chunk: mỗi chunk một run, "
        "cùng cấu hình thắng A-LB"
    )
    for index in range(1, count + 1):
        print(
            fill_winners(
                RunCommand(
                    str(CHUNK_RUN_NUMBER), A_LB_WINNER, LONG_SCENE, f"every8_stereo_c{index}"
                )
            ).render()
        )
    show([c for c in RUN_COMMANDS if int(c.number) > CHUNK_RUN_NUMBER])
    for line in SKIPPED_COMMANDS:
        print(f"# không có lệnh: {line}")
    return 0


EST_GPU_HOURS = {
    "1": 3.9,  # W1 mới cả segment (~198 khung, W1 cũ 150): ước lượng cũ nhân 1,3
    "2": 3.8,
    "3": 3.8,
    "4": 1.5,  # #4 đợt 2 (W1 cũ): 1172 s train; nhân 1,3 và cộng eval
    "5": 2.15,
    "6": 0.7,  # đo ở đợt 1 (393f): train 0,58 giờ + eval
    "7": 2.0,
    "8": 0.6,  # đợt 2 trên cảnh dài: 1539 s; W1 ngắn hơn nhưng ba camera 1920×1280
    "9": 2.5,
    "10": 1.2,
    "11": 5.0 / CHUNK_COUNT,
    "12": 1.7,
    "13": 3.9,
}
"""GPU-giờ ước lượng trên RTX 4090 (cột GPU-giờ của Sheet; #11 tính cho một chunk, #13 đoán)."""

WAVE_RUNS = {
    "1": ("1", "2", "3", "5", "6", "8"),
    "2": ("4", "6", "7", "8"),
    "3": ("10",),
    "4": ("1", "2", "3", "4", "8"),
}
"""Run ưu tiên 1 của từng đợt. Đợt 1 không cần run thắng nào; đợt 2 cần run thắng A-L (#2/#3);
đợt 3 cần cấu hình thắng A-LB (sau #1–#7) và thêm 6 chunk của #11.

Đợt 2 chạy lại #6 (3DGEER cắt viền ảnh pinhole có điểm chính lệch tâm, sửa ở
``recon/utils/geer_fov.py``) và #8 (MCGS-SLAM hỏng với rig 2 camera, sửa ở ``rig_layout``);
``definition`` lấy run mới nhất của mỗi ô nên run #6 cũ (``393f``) tự bị thay.

Đợt 4 (thêm 27/11/2026) chạy lại mọi run W1 trên W1 mới (``D2S_W1_INDEX``): #1–#4 và #8
(MCGS-SLAM chuyển từ cảnh dài sang W1); không phụ thuộc đợt 3 nên chạy cùng lúc được."""
A_L_PLACEHOLDERS = (A_L_WINNER,)  # chỉ các run này nhận --al-winner
WAVE_P2_RUNS = {"1": ("13",), "2": ("12",), "3": (), "4": ("13",)}
WAVE_NUREC = "1"
A_L_APPROACHES = ("pvg", "deformablegs")


class QueueError(ValueError):
    """Thiếu hoặc sai run thắng khi dựng hàng đợi."""


def parse_alb_winner(text: str) -> tuple[str, tuple[str, ...]]:
    """Tách ``"omnire renderer=3dgut fastgs=true"`` thành approach và các ``--set``.

    Raises:
        QueueError: Nếu không phải ``omnire`` theo sau là các cặp ``khoá=giá trị``.
    """
    tokens = shlex.split(text)
    if not tokens or tokens[0] != "omnire":
        raise QueueError("--alb-winner phải bắt đầu bằng omnire, ví dụ 'omnire fastgs=true'")
    sets = tuple(tokens[1:])
    bad = [token for token in sets if "=" not in token or token.startswith("=")]
    if bad:
        raise QueueError(f"--alb-winner: {', '.join(bad)} không có dạng khoá=giá trị")
    return tokens[0], sets


def fill_winners(
    command: RunCommand, al_winner: str = A_L_CHOSEN, alb_winner: str = A_LB_CHOSEN
) -> RunCommand:
    """Thay chỗ giữ chỗ run thắng của một lệnh bằng approach (và ``--set``) đã chốt.

    Với run thắng A-LB, ``--set`` của run thắng đứng trước ``--set`` riêng của lệnh.

    Raises:
        QueueError: Nếu ``alb_winner`` sai dạng.
    """
    if command.approach == A_L_WINNER:
        return RunCommand(
            command.number,
            al_winner,
            command.scene_id,
            command.split_id,
            command.sets,
            note=command.note,
        )
    if command.approach == A_LB_WINNER:
        approach, sets = parse_alb_winner(alb_winner)
        return RunCommand(
            command.number,
            approach,
            command.scene_id,
            command.split_id,
            sets + command.sets,
            note=command.note,
        )
    return command


def build_queue(
    wave: str,
    al_winner: Optional[str] = A_L_CHOSEN,
    alb_winner: Optional[str] = A_LB_CHOSEN,
    p2: bool = False,
    nurec: bool = False,
    num_chunks: int = CHUNK_COUNT,
) -> list[RunCommand]:
    """Các run của một đợt, đã thay run thắng, xếp run dài trước (worker nhận run theo thứ tự).

    Args:
        wave: ``"1"``, ``"2"``, ``"3"`` hoặc ``"4"``.
        al_winner: Approach thắng A-L (``pvg`` hoặc ``deformablegs``); cần cho đợt 2, 4; mặc
            định ``A_L_CHOSEN``.
        alb_winner: Cấu hình thắng A-LB, ví dụ ``"omnire fastgs=true"``; cần cho đợt 3; mặc
            định ``A_LB_CHOSEN``.
        p2: Thêm run ưu tiên 2 của đợt (#13 ở đợt 1, 4; #12 ở đợt 2).
        nurec: Thêm #9 (Instant NuRec) vào đợt 1, khi đã chọn cách chia đoạn.
        num_chunks: Số chunk của #11.

    Raises:
        QueueError: Nếu đợt không có hoặc thiếu, sai run thắng.
    """
    if wave not in WAVE_RUNS:
        raise QueueError(f"không có đợt {wave}; chọn {', '.join(sorted(WAVE_RUNS))}")
    numbers = list(WAVE_RUNS[wave])
    if p2:
        numbers += WAVE_P2_RUNS[wave]
    if nurec and wave == WAVE_NUREC:
        numbers.append("9")
    by_number = {command.number: command for command in RUN_COMMANDS}
    commands = [by_number[number] for number in numbers]
    if wave in ("2", "4"):
        if al_winner not in A_L_APPROACHES:
            raise QueueError(
                f"đợt {wave} cần --al-winner pvg hoặc deformablegs (thắng giữa #2 và #3)"
            )
        commands = [
            RunCommand(c.number, al_winner, c.scene_id, c.split_id, c.sets, note=c.note)
            if c.approach in A_L_PLACEHOLDERS
            else c
            for c in commands
        ]
    if wave == "3":
        if not alb_winner:
            raise QueueError("đợt 3 cần --alb-winner, ví dụ 'omnire fastgs=true'")
        approach, sets = parse_alb_winner(alb_winner)
        commands = [
            RunCommand(c.number, approach, c.scene_id, c.split_id, sets, note=c.note)
            for c in commands
        ]
        commands += [
            RunCommand(
                str(CHUNK_RUN_NUMBER),
                approach,
                LONG_SCENE,
                f"every8_stereo_c{index}",
                sets,
                note=f"track B, chunk {index}",
            )
            for index in range(1, num_chunks + 1)
        ]
    return sorted(commands, key=lambda c: -EST_GPU_HOURS[c.number])


def write_queue(
    workdir: Workdir,
    wave: str,
    out: Optional[Path],
    al_winner: Optional[str],
    alb_winner: Optional[str],
    p2: bool,
    nurec: bool,
) -> int:
    """In (và ghi nếu có ``out``) hàng đợi của một đợt; trả mã thoát."""
    num_chunks = CHUNK_COUNT
    try:
        num_chunks = len(chunk_splits(workdir.load_scene(LONG_SCENE).manifest.timeline.num_frames))
    except WorkdirError:
        pass
    try:
        commands = build_queue(wave, al_winner, alb_winner, p2, nurec, num_chunks)
    except QueueError as error:
        print(f"lỗi: {error}", file=sys.stderr)
        return 2
    lines = [f"{c.number}\t{shlex.join(c.argv())}" for c in commands]
    hours = sum(EST_GPU_HOURS[c.number] for c in commands)
    for line in lines:
        print(line)
    print(f"# đợt {wave}: {len(lines)} run, khoảng {hours:.1f} GPU-giờ RTX 4090", file=sys.stderr)
    if wave == WAVE_NUREC and not nurec:
        print(
            "# #9 (NuRec) chưa vào hàng đợi: thêm --nurec khi đã chọn cách chia đoạn",
            file=sys.stderr,
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"đã ghi {out}", file=sys.stderr)
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


OMNIRE_W1 = Slot("omnire", (("renderer", "3dgs"), ("fastgs", True)))
PINHOLE = (("renderer", "3dgs"),)
FISHEYE_GEER = (("renderer", "3dgeer"),)  # renderer fisheye của A-L chốt 27/11
ENTRY_PLANS = (
    EntryPlan("A", "A-LB", WAYMO_SCENE, WAYMO_SPLIT, (OMNIRE_W1, Slot("beziergs")), "W1 A-LB"),
    EntryPlan(
        "A",
        "A-L",
        WAYMO_SCENE,
        WAYMO_SPLIT,
        (Slot("pvg", PINHOLE), Slot("deformablegs", PINHOLE)),
        "W1 A-L",
    ),
    EntryPlan(
        "A",
        "A-0",
        WAYMO_SCENE,
        WAYMO_SPLIT,
        (Slot("pvg", PINHOLE, lidar=False), Slot("deformablegs", PINHOLE, lidar=False)),
        "W1 A-0",
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
        (Slot("pvg", FISHEYE_GEER), Slot("deformablegs", FISHEYE_GEER)),
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
    EntryPlan("B", None, WAYMO_SCENE, WAYMO_SPLIT, (Slot("mcgs"),), "W1 MCGS-SLAM"),
    EntryPlan("C", None, LONG_SCENE, "every8_mixed3", (Slot("nurec"),), "C Instant NuRec"),
)
CHUNK_SLOT = Slot("omnire", (("renderer", "3dgeer"), ("fastgs", True)))
"""Cấu hình của #11: thắng A-LB chốt 27/11/2026 (``A_LB_CHOSEN``)."""
CALIBRATION_SLOTS = tuple(
    Slot(OMNIRE_W1.approach_id, OMNIRE_W1.options, seed) for seed in (0, 1, 2)
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


def build_chunked(runs: Sequence[RunRecord], notes: list[str]) -> list[BenchmarkChunkedEntry]:
    """Mục ``chunked`` của #11: run mới nhất của mỗi chunk B2 với cấu hình ``CHUNK_SLOT``.

    Có dưới hai chunk đã xong thì bỏ mục; thiếu một vài chunk thì vẫn ghi các chunk đã có và báo
    (số gộp khi đó chỉ phủ một phần cảnh, chỗ hở có cảnh báo trong ``d2s benchmark``).

    Args:
        runs: Mọi run, mới nhất trước.
        notes: Ghi chú cho người dùng, được thêm vào.

    Returns:
        Danh sách rỗng hoặc một mục.
    """
    chunks = []
    missing = []
    for index in range(1, CHUNK_COUNT + 1):
        split_id = f"every8_stereo_c{index}"
        run = pick(runs, LONG_SCENE, split_id, CHUNK_SLOT, notes)
        if run is None:
            missing.append(split_id)
        else:
            chunks.append(BenchmarkChunk(split_id=split_id, run_id=run.run_id))
    if len(chunks) < 2:
        notes.append(f"#11: có {len(chunks)} chunk đã xong (cần ít nhất 2), bỏ mục chunked")
        return []
    if missing:
        notes.append(f"#11: thiếu chunk {', '.join(missing)}, số gộp chỉ phủ một phần cảnh")
    return [
        BenchmarkChunkedEntry(
            bench_track="B",
            label=f"#11 OmniRe 3DGEER + FastGS, {len(chunks)} chunk",
            scene_id=LONG_SCENE,
            chunks=chunks,
        )
    ]


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
            pick(runs, WAYMO_SCENE, WAYMO_SPLIT, slot, notes) for slot in CALIBRATION_SLOTS
        )
        if run is not None
    ]
    calibration = None
    if len(calibration_runs) >= 2:
        calibration = BenchmarkCalibration(
            scene_id=WAYMO_SCENE, split_id=WAYMO_SPLIT, runs=calibration_runs
        )
    else:
        notes.append(
            f"hiệu chỉnh seed: có {len(calibration_runs)} run (cần ít nhất 2; ma trận không còn "
            "run hiệu chỉnh seed), d2s benchmark dùng ngưỡng hoà mặc định 0,3 dB"
        )
    chunked = build_chunked(runs, notes)
    if not entries and not chunked:
        return Definition(None, notes)
    benchmark = BenchmarkFile(
        schema_version=SCHEMA_VERSIONS["benchmark"],
        calibration=calibration,
        entries=entries,
        chunked=chunked,
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
    print(
        f"đã ghi {path} ({len(result.benchmark.entries)} nhóm, "
        f"{len(result.benchmark.chunked)} mục chunked); xem bằng: d2s benchmark {path}"
    )
    return 0


@dataclass(frozen=True)
class SceneStats:
    """Số liệu để chọn W1 trong các cảnh Waymo ứng viên.

    Attributes:
        scene_id: Cảnh.
        num_frames: Số khung (số file ``ego_pose/``).
        distance_m: Quãng đường của xe, mét: tổng khoảng cách tâm xe giữa hai khung liền nhau.
        speed_mps: Tốc độ trung bình, m/s, với 10 khung mỗi giây (tần số khung của Waymo).
        turn_deg: Độ đổi hướng của xe trên cả cảnh, độ.
        vehicles: Số xe có box.
        pedestrians: Số người có box.
        moving: Số vật thể có tâm box dời quá ``MOVING_M`` mét trong cảnh.
    """

    scene_id: str
    num_frames: int
    distance_m: float
    speed_mps: float
    turn_deg: float
    vehicles: int
    pedestrians: int
    moving: int


WAYMO_FPS = 10.0
MOVING_M = 2.0
"""Vật thể dời quá ngần này mét (tâm box, hệ thế giới) thì coi là vật chạy."""


def scene_stats(scene_dir: Path) -> SceneStats:
    """Quãng đường, tốc độ, độ đổi hướng và số vật thể của một cảnh đã ingest.

    Đọc ``ego_pose/<khung>.txt`` (ma trận 4×4 xe → thế giới) và
    ``instances/instances_info.json`` (định dạng DriveStudio: ``class_name``,
    ``frame_annotations.obj_to_world``); thiếu file box thì số vật thể là 0.

    Args:
        scene_dir: Thư mục cảnh.

    Returns:
        ``SceneStats`` của cảnh.
    """
    poses = sorted((scene_dir / "ego_pose").glob("*.txt"), key=lambda path: int(path.stem))
    centres = []
    for path in poses:
        rows = [[float(v) for v in line.split()] for line in path.read_text().splitlines()]
        centres.append((rows[0][3], rows[1][3], rows[2][3]))
    distance = sum(math.dist(a, b) for a, b in zip(centres, centres[1:]))
    duration = max(len(centres) - 1, 1) / WAYMO_FPS
    turn = heading_change_deg(scene_dir, (0, len(poses) - 1)) if poses else 0.0
    vehicles = pedestrians = moving = 0
    info_path = scene_dir / "instances" / "instances_info.json"
    if info_path.is_file():
        instances = json.loads(info_path.read_text(encoding="utf-8"))
        for instance in instances.values():
            name = str(instance.get("class_name", "")).lower()
            vehicles += name == "vehicle"
            pedestrians += name == "pedestrian"
            boxes = instance.get("frame_annotations", {}).get("obj_to_world", [])
            points = [(box[0][3], box[1][3], box[2][3]) for box in boxes]
            if len(points) > 1 and max(math.dist(points[0], q) for q in points) > MOVING_M:
                moving += 1
    return SceneStats(
        scene_id=scene_dir.name,
        num_frames=len(centres),
        distance_m=distance,
        speed_mps=distance / duration,
        turn_deg=turn,
        vehicles=vehicles,
        pedestrians=pedestrians,
        moving=moving,
    )


def print_waymo_stats(workdir: Workdir) -> int:
    """In số liệu các cảnh ứng viên W1 đã ingest, quãng đường dài trước; trả mã thoát.

    Ứng viên chưa ingest được liệt kê cùng lệnh tải; trả 1 nếu không ứng viên nào đã ingest.
    """
    found = []
    for index in WAYMO_W1_CANDIDATES:
        scene_id = "waymo_" + waymo_segment(index)
        scene_dir = workdir.scene_dir(scene_id)
        if (scene_dir / "ego_pose").is_dir():
            found.append((index, scene_stats(scene_dir)))
        else:
            print(f"chưa ingest ứng viên {index}: {scene_id}")
    if not found:
        print(
            "chưa có ứng viên nào; tải: python -m drive2sim.recon.preprocess.waymo.download "
            "--scene_ids " + " ".join(str(i) for i in WAYMO_W1_CANDIDATES)
        )
        return 1
    print("chỉ số  khung  quãng đường  tốc độ   đổi hướng  xe  người  vật chạy  cảnh")
    for index, stats in sorted(found, key=lambda item: -item[1].distance_m):
        mark = "*" if index == WAYMO_INDEX else " "
        print(
            f"{index:>5}{mark} {stats.num_frames:>5}  {stats.distance_m:>9.0f} m  "
            f"{stats.speed_mps:>4.1f} m/s  {stats.turn_deg:>6.0f} độ  {stats.vehicles:>3}  "
            f"{stats.pedestrians:>5}  {stats.moving:>8}  {stats.scene_id}"
        )
    print(f"* = W1 hiện tại (D2S_W1_INDEX={WAYMO_INDEX}); đổi bằng: export D2S_W1_INDEX=<chỉ số>")
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
    queue = actions.add_parser("queue", help="hàng đợi run của một đợt cho envs/bench_queue.sh")
    queue.add_argument("--wave", required=True, choices=sorted(WAVE_RUNS), help="đợt chạy")
    queue.add_argument(
        "--al-winner",
        choices=A_L_APPROACHES,
        default=A_L_CHOSEN,
        help=f"thắng A-L (đợt 2, 4); mặc định {A_L_CHOSEN} (chốt 27/11)",
    )
    queue.add_argument(
        "--alb-winner",
        default=A_LB_CHOSEN,
        help=f"cấu hình thắng A-LB (đợt 3); mặc định '{A_LB_CHOSEN}' (chốt 27/11)",
    )
    queue.add_argument("--p2", action="store_true", help="thêm run ưu tiên 2 của đợt")
    queue.add_argument("--nurec", action="store_true", help="thêm #9 (NuRec) vào đợt 1")
    queue.add_argument("--out", type=Path, help="ghi hàng đợi ra file (vẫn in ra màn hình)")
    actions.add_parser("waymo-stats", help="số liệu các cảnh Waymo ứng viên cho W1")
    args = parser.parse_args(argv)
    workdir = Workdir.from_env(args.workdir)
    if args.action == "splits":
        return create_splits(workdir, args.dry_run)
    if args.action == "commands":
        return print_commands(workdir)
    if args.action == "waymo-stats":
        return print_waymo_stats(workdir)
    if args.action == "queue":
        return write_queue(
            workdir, args.wave, args.out, args.al_winner, args.alb_winner, args.p2, args.nurec
        )
    return write_definition(workdir, args.out)


if __name__ == "__main__":
    sys.exit(main())
