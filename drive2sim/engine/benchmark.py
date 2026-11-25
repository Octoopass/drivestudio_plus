"""Gom kết quả các run thành bảng benchmark và chọn method thắng trong từng nhóm.

File này chứa:
- ``bad_segments``: tìm các đoạn đường hỏng (yêu cầu R7) từ metric từng ảnh test.
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
from collections.abc import Sequence
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
    Summary,
)
from drive2sim.engine.report import ReportRow, report_rows
from drive2sim.engine.run import D2S_DIR, load_summary, summary_path
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


def build_row(
    row: ReportRow,
    run_dir: Path,
    fisheye_ids: set[str],
    rank: int,
    tied_with_best: bool,
) -> BenchmarkRunRow:
    """Dựng một dòng benchmark từ dòng báo cáo và các file kết quả của run.

    Args:
        row: Dòng của ``report_rows`` cho run.
        run_dir: Thư mục run.
        fisheye_ids: ``camera_id`` của các camera fisheye trong cảnh.
        rank: Thứ hạng trong nhóm.
        tied_with_best: Run có ngang run tốt nhất không.

    Returns:
        Dòng kết quả; trường chưa đo để trống.
    """
    summary = load_summary(run_dir)
    # frames.jsonl nằm cạnh summary.json của stage eval.* đã ghi nó (eval.standard, eval.mcgs)
    frames = read_frames(summary_path(run_dir).parent / FRAMES_FILE)
    return BenchmarkRunRow(
        run_id=row.run_id,
        approach_id=row.approach_id,
        rank=rank,
        tied_with_best=tied_with_best,
        psnr=row.psnr,
        ssim=row.ssim,
        lpips=row.lpips,
        avg_error=row.avg_error,
        psnr_pinhole=_camera_psnr(summary, fisheye_ids, False),
        psnr_fisheye=_camera_psnr(summary, fisheye_ids, True),
        depth=summary.depth if summary else None,
        export_gap_db=_export_gap(run_dir),
        ply_gaussians=row.ply_gaussians,
        ply_size_mb=row.ply_size_mb,
        train_time_s=row.train_time_s,
        peak_vram_mb=row.peak_vram_mb,
        render_fps=row.render_fps,
        gpu=row.gpu,
        bad_segments=bad_segments(frames),
        frames_measured=len({(r["frame_index"], r["camera_index"]) for r in frames}),
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
    fisheye_ids = {c.camera_id for c in scene.manifest.cameras if c.model in FISHEYE_MODELS}
    run_dirs = {r.run_id: workdir.run_dir(r.scene_id, r.approach_id, r.run_id) for r in rows}
    summaries = {run_id: load_summary(path) for run_id, path in run_dirs.items()}
    inputs = {
        run_id: TieInputs(
            export_gap_db=_export_gap(run_dirs[run_id]),
            protocol=summary.protocol.version if summary is not None else None,
        )
        for run_id, summary in summaries.items()
    }
    order = order_group(rows, inputs)
    by_id = {r.run_id: r for r in rows}
    result_rows = [
        build_row(by_id[run_id], run_dirs[run_id], fisheye_ids, rank, run_id in order.tied)
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
