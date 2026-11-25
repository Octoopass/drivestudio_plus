"""Gom kết quả các run thành bảng benchmark và chọn method thắng trong từng nhóm.

File này chứa:
- ``bad_segments``, ``bad_segments_by_group``: tìm các đoạn đường hỏng (yêu cầu R7) từ metric từng
  ảnh test, riêng cho từng nhóm camera và chỉ trên camera đã train.
- ``order_group``: xếp hạng các run trong một nhóm theo Avg, rồi phân xử các run ngang nhau bằng
  R2, R3 và thời gian train (cùng GPU).
- ``build_row``: dựng một dòng kết quả (``BenchmarkRunRow``) từ ``summary.json``, ``frames.jsonl``
  và các file kết quả khác của run.
- ``run_benchmark``: đọc file định nghĩa benchmark và trả ``BenchmarkResult`` cho mọi nhóm.

``drive2sim/cli/benchmark.py`` (``d2s benchmark``) gọi ``run_benchmark``. Chỉ đọc file kết quả đã
có của các run, không chạy GPU; run chưa có metric nào thì ghi "chưa đo" bằng trường trống. Xem
CONTRACTS mục 6.5 và báo cáo tuần 03 mục 4.2 (Thiết kế benchmark).
"""

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from drive2sim.core import files
from drive2sim.core.schema import (
    SCHEMA_VERSIONS,
    BadSegment,
    BenchmarkEntry,
    BenchmarkFile,
    BenchmarkGroup,
    BenchmarkResult,
    BenchmarkRunRow,
    RunRecord,
    Summary,
)
from drive2sim.engine.report import ReportRow, report_rows
from drive2sim.engine.run import D2S_DIR, RUN_FILE, load_summary, summary_path
from drive2sim.engine.workdir import Workdir

# Hai method chênh PSNR dưới ngưỡng này được coi là ngang nhau (báo cáo tuần 03 mục 4.2): các run
# cũ cho thấy hai cấu hình gần giống nhau đã lệch 0,05 tới 0,2 dB.
TIE_THRESHOLD_DB = 0.3
# Hai run ngang nhau có export gap (R3) chênh dưới ngưỡng này coi như bằng nhau, xét tiếp chi phí.
EQUAL_GAP_DB = 0.3
# Các protocol của eval.standard đo PSNR, SSIM, LPIPS trên camera đã train theo cùng một cách
# (CONTRACTS 6.3), nên xếp hạng chung được; protocol khác (ví dụ ncore-0.3) thì không.
COMPARABLE_PROTOCOLS = frozenset({"0.3", "0.4", "0.5"})
# Đoạn đường hỏng: ít nhất ngần này khung test liền nhau, mỗi khung thấp hơn trung vị quá ngần
# này dB (R7).
BAD_SEGMENT_MIN_FRAMES = 5
BAD_SEGMENT_DROP_DB = 2.0
FRAMES_FILE = "frames.jsonl"
PLY_GAP_STAGE = "eval.ply"
PLY_GAP_FILE = "ply_gap.json"
MCGS_NOTE_FILE = "mcgs_eval.json"
# Stage xuất PLY nền tĩnh; approach không có stage nào trong số này thì R3 không áp dụng.
EXPORT_PLY_STAGES = frozenset({"export.ply_world", "export.nurec_ply"})
FISHEYE_MODELS = frozenset({"opencv_fisheye", "ftheta"})


def read_frames(path: Path) -> list[dict]:
    """Đọc ``frames.jsonl`` (mỗi dòng là metric của một ảnh test).

    Args:
        path: File ``frames.jsonl`` của stage ``eval.standard``.

    Returns:
        Danh sách bản ghi; rỗng nếu file không có.
    """
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _distance_by_frame(frames: Sequence[dict]) -> dict[int, tuple[float, float]]:
    """Với mỗi khung test, trả ``(quãng đường tích luỹ, PSNR trung bình các camera)``.

    Vị trí của khung là trung bình vị trí camera của các ảnh test cùng khung. Quãng đường là độ
    dài đường gấp khúc qua các khung test theo thứ tự chỉ số khung, nên thiếu phần đường giữa hai
    khung test liền nhau (giữa chúng là khung train).
    """
    by_frame: dict[int, list[dict]] = {}
    for record in frames:
        by_frame.setdefault(int(record["frame_index"]), []).append(record)
    result: dict[int, tuple[float, float]] = {}
    previous: Optional[list[float]] = None
    travelled = 0.0
    for index in sorted(by_frame):
        records = by_frame[index]
        position = [statistics.fmean(r["position"][axis] for r in records) for axis in range(3)]
        if previous is not None:
            travelled += math.dist(previous, position)
        previous = position
        result[index] = (travelled, statistics.fmean(r["psnr"] for r in records))
    return result


def bad_segments(
    frames: Sequence[dict],
    min_frames: int = BAD_SEGMENT_MIN_FRAMES,
    drop_db: float = BAD_SEGMENT_DROP_DB,
) -> list[BadSegment]:
    """Tìm đoạn đường hỏng: ``min_frames`` khung test liền nhau thấp hơn trung vị quá ``drop_db``.

    PSNR của một khung là trung bình các camera trong khung đó; trung vị lấy trên mọi khung test
    của run. "Liền nhau" là liền nhau trong danh sách khung test, không phải trong cảnh.

    Args:
        frames: Bản ghi từ ``read_frames``.
        min_frames: Số khung tối thiểu của một đoạn.
        drop_db: Mức thấp hơn trung vị, dB; phải thấp hơn quá mức này (nghiêm ngặt).

    Returns:
        Các đoạn hỏng theo thứ tự quãng đường; rỗng nếu không có hoặc không có khung nào.
    """
    per_frame = _distance_by_frame(frames)
    if not per_frame:
        return []
    median = statistics.median(psnr for _, psnr in per_frame.values())
    segments: list[BadSegment] = []
    run: list[tuple[float, float]] = []

    def close() -> None:
        if len(run) >= min_frames:
            segments.append(
                BadSegment(
                    start_m=round(run[0][0], 2),
                    end_m=round(run[-1][0], 2),
                    num_frames=len(run),
                    min_psnr=round(min(psnr for _, psnr in run), 3),
                )
            )
        run.clear()

    for index in sorted(per_frame):
        distance, psnr = per_frame[index]
        if psnr < median - drop_db:
            run.append((distance, psnr))
        else:
            close()
    close()
    return segments


def bad_segments_by_group(
    frames: Sequence[dict], fisheye_indices: set[int], exclude_indices: set[int]
) -> dict[str, list[BadSegment]]:
    """Tìm đoạn đường hỏng riêng cho nhóm camera pinhole và nhóm fisheye (R7).

    Ảnh của camera chỉ test (``exclude_indices``) bị bỏ trước khi tính: camera chưa train có PSNR
    thấp hơn hẳn và vị trí lệch, sẽ làm lệch trung vị và quãng đường. Hai nhóm tách riêng vì PSNR
    pinhole và fisheye khác nhau về bản chất; trộn lại thì đoạn hỏng của nhóm này bị che bởi nhóm
    kia.

    Args:
        frames: Bản ghi từ ``read_frames``.
        fisheye_indices: Vị trí trong manifest của các camera fisheye (``camera_index``).
        exclude_indices: Vị trí trong manifest của các camera chỉ test.

    Returns:
        ``{"pinhole": [...], "fisheye": [...]}`` chỉ gồm nhóm có ảnh; danh sách rỗng nếu nhóm đó
        không có đoạn hỏng.
    """
    kept = [r for r in frames if r["camera_index"] not in exclude_indices]
    groups = {
        "pinhole": [r for r in kept if r["camera_index"] not in fisheye_indices],
        "fisheye": [r for r in kept if r["camera_index"] in fisheye_indices],
    }
    return {name: bad_segments(records) for name, records in groups.items() if records}


def _protocol_family(version: str) -> str:
    """Nhóm protocol so được với nhau: ``0.3``, ``0.4``, ``0.5`` cùng một nhóm, còn lại riêng."""
    return "drivestudio" if version in COMPARABLE_PROTOCOLS else version


@dataclass
class TieInputs:
    """Số liệu của một run dùng để phân xử các run ngang nhau (R2, R3), kèm protocol.

    Trường để trống nghĩa là chưa đo; ``export_applicable`` ``False`` nghĩa là method không xuất
    được ``.ply`` (khác chưa đo).

    Attributes:
        shift_psnr_cc: PSNR sau chỉnh màu trên camera chỉ test (R2).
        export_gap_db: Chênh PSNR render gốc và render ``.ply`` (R3).
        export_applicable: Method có xuất được ``.ply`` nền tĩnh không.
        protocol: ``protocol.version`` của ``summary.json``.
    """

    shift_psnr_cc: Optional[float] = None
    export_gap_db: Optional[float] = None
    export_applicable: bool = True
    protocol: Optional[str] = None


@dataclass
class RankOrder:
    """Kết quả xếp hạng một nhóm.

    Attributes:
        run_ids: ``run_id`` theo thứ hạng từ 1: người thắng (nếu có), các run ngang còn lại theo
            Avg, các run đã đo còn lại theo Avg, rồi run chưa đo.
        tied: Các run ngang run có Avg tốt nhất (gồm chính nó), tính một lần.
        winner: Run thắng, ``None`` khi không phân xử được.
        decided_by: Bước phân xử đã chọn người thắng: ``avg``, ``r2``, ``r3`` hoặc
            ``train_time``; ``None`` khi không có người thắng.
        warnings: Lý do không phân xử được.
    """

    run_ids: list[str]
    tied: set[str] = field(default_factory=set)
    winner: Optional[str] = None
    decided_by: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def order_group(
    rows: Sequence[ReportRow],
    inputs: dict[str, TieInputs],
    threshold_db: float = TIE_THRESHOLD_DB,
) -> RankOrder:
    """Xếp các run của một nhóm theo Avg, rồi phân xử các run ngang nhau.

    Quy tắc (yêu cầu 25/11, mục 3 của spec): run tốt nhất là run có Avg thấp nhất; run có PSNR kém
    nó dưới ``threshold_db`` là ngang, tập này xác định một lần. Trong tập ngang phân xử lần lượt:
    R2 (PSNR sau chỉnh màu ở camera chỉ test; chênh dưới ngưỡng coi như bằng), R3 (export gap
    nhỏ hơn; chênh dưới ``EQUAL_GAP_DB`` coi như bằng), rồi thời gian train, chỉ khi mọi run cùng
    GPU. Một bước chưa đo hoặc không áp dụng với một run nào của tập thì bị bỏ qua cho cả tập.
    Các run đo bằng protocol không so được với nhau thì không xếp hạng tự động.

    Args:
        rows: Dòng báo cáo của các run trong nhóm.
        inputs: ``run_id`` → ``TieInputs``; thiếu run nào thì coi như chưa đo gì.
        threshold_db: Ngưỡng ngang nhau của PSNR, dB.

    Returns:
        ``RankOrder`` của nhóm.
    """
    measured = sorted(
        (r for r in rows if r.avg_error is not None and r.psnr is not None),
        key=lambda r: r.avg_error,
    )
    measured_ids = {r.run_id for r in measured}
    unmeasured_ids = [r.run_id for r in rows if r.run_id not in measured_ids]
    if not measured:
        return RankOrder(run_ids=unmeasured_ids)
    by_avg = [r.run_id for r in measured]

    def inputs_of(run_id: str) -> TieInputs:
        return inputs.get(run_id, TieInputs())

    protocol_names = {inputs_of(r.run_id).protocol for r in measured} - {None}
    if len({_protocol_family(name) for name in protocol_names}) > 1:
        return RankOrder(
            run_ids=by_avg + unmeasured_ids,
            warnings=[
                "các run đo bằng protocol khác nhau, không xếp hạng tự động: "
                + ", ".join(sorted(protocol_names))
            ],
        )

    best = measured[0]
    tied_rows = [r for r in measured if best.psnr - r.psnr < threshold_db]
    tied = {r.run_id for r in tied_rows}
    warnings: list[str] = []
    candidates = tied_rows
    winner: Optional[ReportRow] = None
    decided_by: Optional[str] = None

    if len(candidates) == 1:
        winner, decided_by = candidates[0], "avg"
    if winner is None:
        values = [inputs_of(r.run_id).shift_psnr_cc for r in candidates]
        if all(v is not None for v in values):
            top = max(values)
            candidates = [r for r, v in zip(candidates, values) if top - v < threshold_db]
            if len(candidates) == 1:
                winner, decided_by = candidates[0], "r2"
    if winner is None:
        gaps = [inputs_of(r.run_id) for r in candidates]
        if all(g.export_applicable and g.export_gap_db is not None for g in gaps):
            lowest = min(g.export_gap_db for g in gaps)
            candidates = [
                r for r, g in zip(candidates, gaps) if g.export_gap_db - lowest < EQUAL_GAP_DB
            ]
            if len(candidates) == 1:
                winner, decided_by = candidates[0], "r3"
    if winner is None:
        gpus = {r.gpu for r in candidates}
        if (
            len(gpus) == 1
            and None not in gpus
            and all(r.train_time_s is not None for r in candidates)
        ):
            winner = min(candidates, key=lambda r: r.train_time_s)
            decided_by = "train_time"
        else:
            warnings.append(
                "các run ngang nhau, khác GPU hoặc thiếu thời gian train nên không chọn được "
                "người thắng: " + ", ".join(r.run_id for r in candidates)
            )

    ordered = [r.run_id for r in measured if r.run_id in tied]
    if winner is not None:
        ordered.remove(winner.run_id)
        ordered.insert(0, winner.run_id)
    rest = [run_id for run_id in by_avg if run_id not in tied]
    return RankOrder(
        run_ids=ordered + rest + unmeasured_ids,
        tied=tied,
        winner=winner.run_id if winner else None,
        decided_by=decided_by,
        warnings=warnings,
    )


def export_applicable(stage_ids: Iterable[str]) -> bool:
    """Approach có xuất được PLY nền tĩnh không, theo danh sách stage của nó.

    Args:
        stage_ids: ``stage_id`` các stage của run.

    Returns:
        ``True`` nếu có ít nhất một stage trong ``EXPORT_PLY_STAGES``.
    """
    return any(stage_id in EXPORT_PLY_STAGES for stage_id in stage_ids)


def run_stage_ids(run_dir: Path) -> Optional[list[str]]:
    """Danh sách ``stage_id`` của run, đọc từ ``d2s/run.json``.

    Lấy từ run (đã thay chỗ giữ chỗ ``${options.*}``) thay vì từ file approach, vì file approach
    có thể đã đổi sau khi run được tạo.

    Args:
        run_dir: Thư mục run.

    Returns:
        Danh sách stage theo thứ tự chạy; ``None`` nếu không đọc được ``run.json``.
    """
    try:
        record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    except (OSError, ValueError):
        return None
    return [stage.stage_id for stage in record.stages]


def run_export_applicable(run_dir: Path) -> bool:
    """``export_applicable`` của run; không đọc được danh sách stage thì coi là áp dụng."""
    stage_ids = run_stage_ids(run_dir)
    return True if stage_ids is None else export_applicable(stage_ids)


@dataclass
class GroupContext:
    """Những điều về cảnh và nhóm mà mọi dòng của một nhóm cùng cần.

    Attributes:
        bench_track: Track benchmark (``A``, ``B``, ``C``); chỉ track ``B`` tính chi phí mỗi 100 m.
        fisheye_ids: ``camera_id`` của các camera fisheye của cảnh.
        fisheye_indices: Vị trí các camera fisheye trong manifest (khớp ``camera_index``).
        test_only_indices: Vị trí các camera chỉ test của split trong manifest.
    """

    bench_track: str
    fisheye_ids: set[str] = field(default_factory=set)
    fisheye_indices: set[int] = field(default_factory=set)
    test_only_indices: set[int] = field(default_factory=set)


def _camera_psnr(
    summary: Optional[Summary], fisheye_ids: set[str], want_fisheye: bool
) -> Optional[float]:
    """PSNR trung bình các camera pinhole hoặc fisheye của run (R5); ``None`` nếu không có."""
    if summary is None:
        return None
    values = [
        metrics.psnr
        for camera, metrics in summary.per_camera.items()
        if (camera in fisheye_ids) == want_fisheye
    ]
    return statistics.fmean(values) if values else None


def _export_gap(run_dir: Path) -> Optional[float]:
    """Export gap (dB) trong ``eval.ply/ply_gap.json`` của run; ``None`` nếu chưa đo."""
    path = run_dir / D2S_DIR / "stages" / PLY_GAP_STAGE / PLY_GAP_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("export_gap_db")
    except (OSError, ValueError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _ate_rmse_m(stage_dir: Path) -> Optional[float]:
    """ATE (m) do method tự báo trong ``mcgs_eval.json``; ``None`` nếu không có.

    File nằm cạnh ``summary.json``. Chỉ đọc số method báo, không tính lại (yêu cầu R6).
    """
    try:
        value = json.loads((stage_dir / MCGS_NOTE_FILE).read_text(encoding="utf-8")).get(
            "ate_rmse_m"
        )
    except (OSError, ValueError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _per_100m(value: Optional[float], distance_m: Optional[float]) -> Optional[float]:
    """Chia ``value`` cho ``distance_m / 100``; ``None`` khi thiếu số hoặc không có quãng đường."""
    if value is None or not distance_m or distance_m <= 0:
        return None
    return value / (distance_m / 100.0)


def build_row(
    row: ReportRow,
    run_dir: Path,
    context: GroupContext,
    rank: int,
    tied_with_best: bool,
) -> BenchmarkRunRow:
    """Dựng một dòng benchmark từ dòng báo cáo và các file kết quả của run.

    Ảnh của camera chỉ test không tính vào đoạn hỏng (R7), quãng đường (R6) và ``frames_measured``:
    camera chưa train có PSNR thấp hơn hẳn và vị trí lệch khỏi quỹ đạo, sẽ làm lệch cả ba.

    Args:
        row: Dòng của ``report_rows`` cho run.
        run_dir: Thư mục run.
        context: Camera của cảnh và track của nhóm.
        rank: Thứ hạng trong nhóm.
        tied_with_best: Run có ngang run Avg tốt nhất không.

    Returns:
        Dòng kết quả; trường chưa đo để trống.
    """
    summary = load_summary(run_dir)
    # frames.jsonl nằm cạnh summary.json của stage eval.* đã ghi nó (eval.standard, eval.mcgs)
    stage_dir = summary_path(run_dir).parent
    frames = read_frames(stage_dir / FRAMES_FILE)
    trained = [r for r in frames if r["camera_index"] not in context.test_only_indices]
    per_frame = _distance_by_frame(trained)
    distance_m = max((distance for distance, _ in per_frame.values()), default=None)
    shifted = summary.shifted if summary else None
    per_100m = context.bench_track == "B"
    return BenchmarkRunRow(
        run_id=row.run_id,
        approach_id=row.approach_id,
        rank=rank,
        tied_with_best=tied_with_best,
        psnr=row.psnr,
        ssim=row.ssim,
        lpips=row.lpips,
        avg_error=row.avg_error,
        psnr_pinhole=_camera_psnr(summary, context.fisheye_ids, False),
        psnr_fisheye=_camera_psnr(summary, context.fisheye_ids, True),
        shift_psnr=shifted.psnr if shifted else None,
        shift_lpips=shifted.lpips if shifted else None,
        shift_psnr_cc=shifted.psnr_cc if shifted else None,
        shift_lpips_cc=shifted.lpips_cc if shifted else None,
        shift_cameras=list(shifted.cameras) if shifted else [],
        depth=summary.depth if summary else None,
        export_gap_db=_export_gap(run_dir),
        export_applicable=run_export_applicable(run_dir),
        ply_gaussians=row.ply_gaussians,
        ply_size_mb=row.ply_size_mb,
        train_time_s=row.train_time_s,
        recon_time_s=row.recon_time_s,
        peak_vram_mb=row.peak_vram_mb,
        render_fps=row.render_fps,
        gpu=row.gpu,
        distance_m=distance_m,
        train_s_per_100m=_per_100m(row.train_time_s, distance_m) if per_100m else None,
        ply_mb_per_100m=_per_100m(row.ply_size_mb, distance_m) if per_100m else None,
        ate_rmse_m=_ate_rmse_m(stage_dir),
        bad_segments=bad_segments_by_group(
            frames, context.fisheye_indices, context.test_only_indices
        ),
        frames_measured=len({(r["frame_index"], r["camera_index"]) for r in trained}),
    )


def _group_warnings(rows: Sequence[ReportRow]) -> list[str]:
    """Cảnh báo khi các run của một nhóm không so được trực tiếp (run thiếu kết quả, khác GPU).

    Cảnh báo protocol khác nhau và không chọn được người thắng do ``order_group`` trả về.
    """
    warnings = []
    missing = [r.run_id for r in rows if r.psnr is None]
    if missing:
        warnings.append(f"chưa có kết quả đánh giá: {', '.join(missing)}")
    gpus = {r.gpu for r in rows if r.gpu}
    if len(gpus) > 1:
        warnings.append(
            "các run chạy trên GPU khác nhau (" + "; ".join(sorted(gpus)) + "): thời gian, VRAM "
            "và FPS không so được"
        )
    return warnings


def run_group(workdir: Workdir, entry: BenchmarkEntry) -> BenchmarkGroup:
    """Xếp hạng các run của một nhóm benchmark.

    Args:
        workdir: Thư mục làm việc.
        entry: Nhóm trong file định nghĩa benchmark.

    Returns:
        Kết quả của nhóm.

    Raises:
        ValueError: Nếu một run không tồn tại hoặc không thuộc cảnh, split của nhóm.
    """
    rows = report_rows(workdir, entry.scene_id, entry.split_id, list(entry.runs))
    found = {r.run_id for r in rows}
    missing = [run_id for run_id in entry.runs if run_id not in found]
    if missing:
        raise ValueError(
            f"run không có hoặc không thuộc cảnh {entry.scene_id}, split {entry.split_id}: "
            + ", ".join(missing)
        )
    scene = workdir.load_scene(entry.scene_id)
    split = workdir.load_split(entry.scene_id, entry.split_id)
    cameras = scene.manifest.cameras
    context = GroupContext(
        bench_track=entry.bench_track,
        fisheye_ids={c.camera_id for c in cameras if c.model in FISHEYE_MODELS},
        fisheye_indices={i for i, c in enumerate(cameras) if c.model in FISHEYE_MODELS},
        test_only_indices={
            i for i, c in enumerate(cameras) if c.camera_id in split.test_only_cameras
        },
    )
    run_dirs = {r.run_id: workdir.run_dir(r.scene_id, r.approach_id, r.run_id) for r in rows}
    summaries = {run_id: load_summary(path) for run_id, path in run_dirs.items()}
    inputs = {
        run_id: TieInputs(
            shift_psnr_cc=summary.shifted.psnr_cc if summary and summary.shifted else None,
            export_gap_db=_export_gap(run_dirs[run_id]),
            export_applicable=run_export_applicable(run_dirs[run_id]),
            protocol=summary.protocol.version if summary is not None else None,
        )
        for run_id, summary in summaries.items()
    }
    order = order_group(rows, inputs)
    by_id = {r.run_id: r for r in rows}
    result_rows = [
        build_row(by_id[run_id], run_dirs[run_id], context, rank, run_id in order.tied)
        for rank, run_id in enumerate(order.run_ids, start=1)
    ]
    return BenchmarkGroup(
        bench_track=entry.bench_track,
        cell=entry.cell,
        scene_id=entry.scene_id,
        split_id=entry.split_id,
        winner_run_id=order.winner,
        decided_by=order.decided_by,
        rows=result_rows,
        warnings=_group_warnings(rows) + order.warnings,
    )


def run_benchmark(workdir: Workdir, definition: Path) -> BenchmarkResult:
    """Đọc file định nghĩa benchmark và xếp hạng mọi nhóm.

    Args:
        workdir: Thư mục làm việc.
        definition: File ``benchmarks/<tên>.yaml``.

    Returns:
        Kết quả mọi nhóm, theo thứ tự trong file.

    Raises:
        ValueError: Nếu file sai schema hoặc một run không hợp lệ.
        OSError: Nếu không đọc được file.
    """
    parsed = files.load(definition, BenchmarkFile)
    return BenchmarkResult(
        schema_version=SCHEMA_VERSIONS["benchresult"],
        tie_threshold_db=TIE_THRESHOLD_DB,
        groups=[run_group(workdir, entry) for entry in parsed.entries],
    )
