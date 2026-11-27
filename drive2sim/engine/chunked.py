"""Gộp kết quả các chunk của một lần dựng cảnh dài thành một dòng benchmark (R6, track B).

Một lần dựng chia chunk là nhiều run thường trên các split có ``frame_range`` chồng nhau của cùng
cảnh (ARCHITECTURE mục 2.17). File này gộp ``frames.jsonl`` của các run đó:

- đưa chỉ số khung của từng ảnh về chỉ số trong cảnh: ``eval.standard`` của DriveStudio ghi chỉ
  số tính từ khung đầu của split, nên cộng ``frame_range[0]``; bản ghi đã theo chỉ số của cảnh
  (khớp tập test của split) thì giữ nguyên;
- đặt vị trí của mỗi khung theo ``ego_pose/`` của cảnh, vì vị trí camera trong ``frames.jsonl`` của
  DriveStudio tính trong hệ lấy gốc ở khung đầu của từng chunk, không so được giữa các chunk;
- trong đoạn chồng của hai chunk kề nhau, khung thuộc chunk mà nó xa mép hơn (nửa đầu cho chunk
  trước, nửa sau cho chunk sau), để metric gộp đếm mỗi ảnh một lần;
- đo PSNR quanh từng chỗ nối (``ChunkSeam``) và so hai chunk trên khung test chung của đoạn chồng.

``engine/benchmark.py`` gọi ``run_chunked`` cho mỗi mục ``chunked`` của file định nghĩa. Chỉ đọc
file kết quả đã có, không chạy GPU. Xem CONTRACTS mục 6.5.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from drive2sim.core.schema import (
    BenchmarkChunkedEntry,
    BenchmarkChunkedGroup,
    ChunkRow,
    ChunkSeam,
    Split,
)
from drive2sim.engine.report import ReportRow, avg_error, report_rows
from drive2sim.engine.run import summary_path
from drive2sim.engine.workdir import Workdir

SEAM_WINDOW_FRAMES = 24
"""Nửa độ rộng vùng quanh chỗ nối, khung của cảnh: với split ``every_nth_frame:8`` là khoảng ba
khung test mỗi bên."""
FRAMES_FILE = "frames.jsonl"
FISHEYE_MODELS = frozenset({"opencv_fisheye", "ftheta"})


@dataclass
class LoadedChunk:
    """Một chunk đã đọc: dòng báo cáo, split và bản ghi từng ảnh theo chỉ số khung của cảnh."""

    row: ReportRow
    split: Split
    records: list[dict]


def to_scene_frames(records: Sequence[dict], split: Split) -> tuple[list[dict], Optional[str]]:
    """Đổi ``frame_index`` của các bản ghi về chỉ số khung trong cảnh.

    Chọn cách đọc khớp tập test của split: chỉ số tính từ khung đầu của split (DriveStudio) hay
    chỉ số của cảnh. Khớp cả hai (chunk bắt đầu ở khung 0) thì như nhau.

    Args:
        records: Bản ghi của ``frames.jsonl``.
        split: Split của run.

    Returns:
        ``(bản ghi đã đổi, cảnh báo)``; cảnh báo khác ``None`` khi không cách nào khớp tập test
        (vẫn đổi theo cách tương đối).
    """
    first = split.frame_range[0]
    test_frames = {frame for frame, _ in split.test}
    indices = {int(record["frame_index"]) for record in records}
    relative = {index + first for index in indices}
    warning = None
    if relative <= test_frames:
        offset = first
    elif indices <= test_frames:
        offset = 0
    else:
        offset = first
        warning = (
            f"{split.split_id}: chỉ số khung trong frames.jsonl không khớp tập test của split, "
            "coi là tính từ khung đầu của split"
        )
    shifted = [
        {**record, "frame_index": int(record["frame_index"]) + offset} for record in records
    ]
    return shifted, warning


def owner_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Đoạn khung mà mỗi chunk "sở hữu" sau khi chia đôi đoạn chồng với chunk kề.

    Args:
        ranges: ``frame_range`` của các chunk, đã xếp theo khung đầu.

    Returns:
        Đoạn sở hữu của từng chunk, tính cả hai đầu; chunk đầu bắt đầu ở khung đầu của nó, chunk
        cuối kết thúc ở khung cuối của nó.
    """
    owned = []
    for index, (start, end) in enumerate(ranges):
        low, high = start, end
        if index > 0:
            prev_end = ranges[index - 1][1]
            low = (start + prev_end) // 2 + 1 if prev_end >= start else start
        if index + 1 < len(ranges):
            next_start = ranges[index + 1][0]
            high = (next_start + end) // 2 if end >= next_start else end
        owned.append((low, high))
    return owned


def ego_positions(scene_dir: Path) -> dict[int, list[float]]:
    """Tâm xe ở từng khung của cảnh (``ego_pose/<khung>.txt``, cột tịnh tiến); rỗng nếu thiếu."""
    positions = {}
    for path in (scene_dir / "ego_pose").glob("*.txt"):
        try:
            rows = [[float(v) for v in line.split()] for line in path.read_text().splitlines()]
            positions[int(path.stem)] = [rows[0][3], rows[1][3], rows[2][3]]
        except (OSError, ValueError, IndexError):
            continue
    return positions


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Trung bình các giá trị khác ``None``; ``None`` nếu không còn giá trị nào."""
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def _per_frame_psnr(records: Sequence[dict]) -> dict[int, float]:
    """PSNR trung bình các camera của từng khung; bỏ ảnh có ``psnr`` là ``None``."""
    by_frame: dict[int, list[float]] = {}
    for record in records:
        if record.get("psnr") is not None:
            by_frame.setdefault(int(record["frame_index"]), []).append(float(record["psnr"]))
    return {frame: statistics.fmean(values) for frame, values in by_frame.items()}


def _sum_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    """Tổng; ``None`` nếu thiếu một giá trị (tổng thiếu một chunk sẽ nhỏ sai)."""
    return None if any(v is None for v in values) else float(sum(values))  # type: ignore[arg-type]


def build_seams(
    ranges: Sequence[tuple[int, int]],
    owned_frames: dict[int, float],
    trained_by_chunk: Sequence[Sequence[dict]],
    distance_at: dict[int, float],
    window: Optional[int] = None,
) -> list[ChunkSeam]:
    """Đo PSNR quanh từng chỗ nối giữa hai chunk kề nhau.

    Args:
        ranges: ``frame_range`` của các chunk, đã xếp.
        owned_frames: PSNR theo khung sau khi gộp (mỗi khung từ chunk sở hữu nó).
        trained_by_chunk: Bản ghi camera đã train của từng chunk (chỉ số khung của cảnh), trước
            khi chia đoạn chồng.
        distance_at: Quãng đường tới từng khung test đã gộp, mét.
        window: Nửa độ rộng vùng quanh chỗ nối, khung; ``None`` là ``SEAM_WINDOW_FRAMES``.

    Returns:
        Một ``ChunkSeam`` cho mỗi cặp chunk kề nhau.
    """
    window = SEAM_WINDOW_FRAMES if window is None else window
    median = statistics.median(owned_frames.values()) if owned_frames else None
    seams = []
    for index in range(len(ranges) - 1):
        end_before, start_after = ranges[index][1], ranges[index + 1][0]
        overlap = (start_after, end_before) if end_before >= start_after else None
        middle = (start_after + end_before) / 2
        near = [psnr for frame, psnr in owned_frames.items() if abs(frame - middle) <= window]
        psnr_near = statistics.fmean(near) if near else None
        before_frames, after_frames = (
            _per_frame_psnr(trained_by_chunk[index]),
            _per_frame_psnr(trained_by_chunk[index + 1]),
        )
        shared = []
        if overlap is not None:
            shared = [
                frame
                for frame in before_frames
                if overlap[0] <= frame <= overlap[1] and frame in after_frames
            ]
        nearest = min(distance_at, key=lambda frame: abs(frame - middle), default=None)
        seams.append(
            ChunkSeam(
                index=index + 1,
                overlap=overlap,
                at_m=round(distance_at[nearest], 2) if nearest is not None else None,
                window_frames=window,
                num_frames=len(near),
                psnr_near=psnr_near,
                drop_db=(median - psnr_near)
                if median is not None and psnr_near is not None
                else None,
                overlap_frames=len(shared),
                overlap_psnr_before=_mean([before_frames[f] for f in shared]),
                overlap_psnr_after=_mean([after_frames[f] for f in shared]),
            )
        )
    return seams


def run_chunked(workdir: Workdir, entry: BenchmarkChunkedEntry) -> BenchmarkChunkedGroup:
    """Gộp các chunk của một lần dựng thành một kết quả.

    Args:
        workdir: Thư mục làm việc.
        entry: Mục ``chunked`` của file định nghĩa.

    Returns:
        Kết quả gộp; trường chưa đo để trống.

    Raises:
        ValueError: Nếu một run không có hoặc không thuộc cảnh, split đã khai báo.
    """
    from drive2sim.engine.benchmark import (  # tránh nạp vòng: benchmark gọi file này
        bad_segments_by_group,
        read_frames,
    )

    warnings: list[str] = []
    loaded: list[LoadedChunk] = []
    for chunk in entry.chunks:
        rows = report_rows(workdir, entry.scene_id, chunk.split_id, [chunk.run_id])
        if not rows:
            raise ValueError(
                f"run không có hoặc không thuộc cảnh {entry.scene_id}, split {chunk.split_id}: "
                f"{chunk.run_id}"
            )
        row = rows[0]
        split = workdir.load_split(entry.scene_id, chunk.split_id)
        run_dir = workdir.run_dir(row.scene_id, row.approach_id, row.run_id)
        summary_file = summary_path(run_dir)
        records = read_frames(summary_file.parent / FRAMES_FILE) if summary_file else []
        if not records:
            warnings.append(f"{chunk.run_id}: chưa có frames.jsonl, chunk không vào số gộp")
        records, warning = to_scene_frames(records, split)
        if warning:
            warnings.append(warning)
        loaded.append(LoadedChunk(row, split, records))
    loaded.sort(key=lambda chunk: chunk.split.frame_range)
    ranges = [tuple(chunk.split.frame_range) for chunk in loaded]
    for (_, end_before), (start_after, _) in zip(ranges, ranges[1:]):
        if start_after > end_before + 1:
            warnings.append(f"hai chunk kề nhau hở khung {end_before + 1}–{start_after - 1}")

    scene = workdir.load_scene(entry.scene_id)
    cameras = scene.manifest.cameras
    fisheye_indices = {i for i, c in enumerate(cameras) if c.model in FISHEYE_MODELS}
    test_only_ids = set().union(*(chunk.split.test_only_cameras for chunk in loaded))
    test_only_indices = {i for i, c in enumerate(cameras) if c.camera_id in test_only_ids}
    positions = ego_positions(workdir.scene_dir(entry.scene_id))

    merged: list[dict] = []
    trained_by_chunk: list[list[dict]] = []
    for chunk, (low, high) in zip(loaded, owner_ranges(ranges)):
        trained_by_chunk.append(
            [r for r in chunk.records if r["camera_index"] not in test_only_indices]
        )
        merged += [r for r in chunk.records if low <= r["frame_index"] <= high]
    missing_pose = sorted({r["frame_index"] for r in merged if r["frame_index"] not in positions})
    if missing_pose:
        warnings.append(
            f"cảnh thiếu ego_pose ở {len(missing_pose)} khung test: không tính quãng đường, "
            "đoạn hỏng và vị trí chỗ nối"
        )
    for record in merged:
        record["position"] = positions.get(record["frame_index"], [math.nan] * 3)

    trained = [r for r in merged if r["camera_index"] not in test_only_indices]
    shifted = [r for r in merged if r["camera_index"] in test_only_indices]
    scored = [r for r in trained if r.get("psnr") is not None]
    psnr = _mean([r["psnr"] for r in scored])
    ssim = _mean([r.get("ssim") for r in scored])
    lpips_values = [r.get("lpips") for r in scored]
    lpips = None if any(v is None for v in lpips_values) else _mean(lpips_values)

    distance_at: dict[int, float] = {}
    if not missing_pose:
        travelled, previous = 0.0, None
        for frame in sorted({int(r["frame_index"]) for r in scored}):
            if previous is not None:
                travelled += math.dist(positions[previous], positions[frame])
            distance_at[frame] = travelled
            previous = frame
    distance = max(distance_at.values(), default=None)

    rows = [chunk.row for chunk in loaded]
    train_time = _sum_or_none([r.train_time_s for r in rows])
    ply_size = _sum_or_none([r.ply_size_mb for r in rows])
    gpus = {r.gpu for r in rows}
    gpu = next(iter(gpus)) if len(gpus) == 1 else None
    if len(gpus - {None}) > 1:
        warnings.append("các chunk chạy trên GPU khác nhau: tổng thời gian train không so được")
    approaches = sorted({r.approach_id for r in rows})
    if len(approaches) > 1:
        warnings.append("các chunk dùng approach khác nhau: " + ", ".join(approaches))

    merged_avg = (
        avg_error(psnr, ssim, lpips)
        if psnr is not None and ssim is not None and lpips is not None
        else None
    )

    def per_100m(value: Optional[float]) -> Optional[float]:
        if value is None or not distance:
            return None
        return value / (distance / 100.0)

    return BenchmarkChunkedGroup(
        bench_track=entry.bench_track,
        label=entry.label,
        scene_id=entry.scene_id,
        approach_ids=approaches,
        chunks=[
            ChunkRow(
                split_id=chunk.split.split_id,
                run_id=chunk.row.run_id,
                approach_id=chunk.row.approach_id,
                frame_range=chunk.split.frame_range,
                psnr=chunk.row.psnr,
                ssim=chunk.row.ssim,
                lpips=chunk.row.lpips,
                train_time_s=chunk.row.train_time_s,
                peak_vram_mb=chunk.row.peak_vram_mb,
                ply_size_mb=chunk.row.ply_size_mb,
                gpu=chunk.row.gpu,
            )
            for chunk in loaded
        ],
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        avg_error=merged_avg,
        shift_psnr=_mean([r.get("psnr") for r in shifted]),
        shift_psnr_cc=_mean([r.get("psnr_cc") for r in shifted]),
        distance_m=distance,
        train_time_s=train_time,
        peak_vram_mb=max((r.peak_vram_mb for r in rows if r.peak_vram_mb), default=None),
        ply_size_mb=ply_size,
        gpu=gpu,
        train_s_per_100m=per_100m(train_time),
        ply_mb_per_100m=per_100m(ply_size),
        frames_measured=len({(r["frame_index"], r["camera_index"]) for r in scored}),
        bad_segments=bad_segments_by_group(merged, fisheye_indices, test_only_indices)
        if not missing_pose
        else {},
        seams=build_seams(ranges, _per_frame_psnr(trained), trained_by_chunk, distance_at),
        warnings=warnings,
    )
