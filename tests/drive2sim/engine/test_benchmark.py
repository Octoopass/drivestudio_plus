"""Kiểm tra ``drive2sim/engine/benchmark.py``: đoạn đường hỏng, thứ hạng và gom kết quả nhóm.

- ``bad_segments``: ít nhất 5 khung test liền nhau thấp hơn trung vị quá 2 dB mới là một đoạn.
- ``order_group``: xếp theo Avg, nhưng run kém run tốt nhất dưới 0,3 dB PSNR xét thêm export gap
  rồi chi phí.
- ``run_group``: ghép dòng báo cáo, ``summary.json`` và ``frames.jsonl`` thành kết quả của nhóm.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from drive2sim.core.schema import BenchmarkEntry, DepthMetrics
from drive2sim.engine import benchmark
from drive2sim.engine.benchmark import bad_segments, order_group
from drive2sim.engine.report import ReportRow, avg_error


def frames_with(psnrs: list[float]) -> list[dict]:
    """Mỗi khung một camera, đi thẳng 10 m giữa hai khung liền nhau."""
    return [
        {
            "frame_index": index * 10,
            "camera_index": 0,
            "position": [10.0 * index, 0.0, 0.0],
            "psnr": psnr,
        }
        for index, psnr in enumerate(psnrs)
    ]


def test_bad_segment_needs_five_consecutive_frames() -> None:
    good = [25.0] * 6
    four_bad = good + [20.0] * 4 + good
    assert bad_segments(frames_with(four_bad)) == []
    five_bad = good + [20.0, 19.5, 20.0, 20.5, 20.0] + good
    segments = bad_segments(frames_with(five_bad))
    assert len(segments) == 1
    assert (segments[0].start_m, segments[0].end_m) == (60.0, 100.0)
    assert segments[0].num_frames == 5 and segments[0].min_psnr == 19.5


def test_bad_segment_threshold_is_strict_and_uses_median() -> None:
    # đúng 2 dB dưới trung vị (25) thì chưa tính là hỏng
    assert bad_segments(frames_with([25.0] * 6 + [23.0] * 5 + [25.0] * 6)) == []
    # nhiều đoạn hỏng
    psnrs = [25.0] * 8 + [21.0] * 5 + [25.0] * 8 + [22.0] * 6 + [25.0] * 8
    assert [s.num_frames for s in bad_segments(frames_with(psnrs))] == [5, 6]


def test_bad_segments_average_cameras_of_a_frame_and_handle_empty() -> None:
    assert bad_segments([]) == []
    records = []
    for index in range(12):
        for camera, delta in ((0, 3.0), (1, -3.0)):  # trung bình hai camera là 25
            records.append(
                {
                    "frame_index": index,
                    "camera_index": camera,
                    "position": [float(index), 0.0, 0.0],
                    "psnr": 25.0 + delta,
                }
            )
    assert bad_segments(records) == []


def make_row(
    run_id: str, psnr: float, train_time_s: Optional[float] = 100.0, ssim: float = 0.8
) -> ReportRow:
    lpips = 0.2
    return ReportRow(
        run_id=run_id,
        scene_id="s",
        split_id="x",
        approach_id="omnire",
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        avg_error=avg_error(psnr, ssim, lpips),
        train_time_s=train_time_s,
    )


def test_order_group_by_avg_when_not_tied() -> None:
    rows = [make_row("b", 24.0), make_row("a", 25.0), make_row("c", 23.0)]
    assert order_group(rows, {}) == ["a", "b", "c"]


def test_order_group_tie_uses_export_gap_then_cost() -> None:
    rows = [
        make_row("best", 25.20, train_time_s=3000),
        make_row("small_gap", 25.01, train_time_s=9000),
        make_row("cheap", 25.05, train_time_s=1000),
        make_row("far", 24.50, train_time_s=100),  # kém 0,70 dB: không ngang
    ]
    gaps = {"best": 1.5, "small_gap": 0.2, "cheap": None, "far": 0.0}
    # ngang nhau: export gap nhỏ trước, chưa đo sau; "far" không ngang nên xếp sau cùng
    assert order_group(rows, gaps) == ["small_gap", "best", "cheap", "far"]
    # cùng chưa đo gap: chi phí thấp trước
    assert order_group(rows[:3], {})[0] == "cheap"


def test_order_group_unmeasured_last_and_all_unmeasured() -> None:
    nothing = ReportRow(run_id="n", scene_id="s", split_id="x", approach_id="omnire")
    assert order_group([nothing, make_row("a", 25.0)], {}) == ["a", "n"]
    assert order_group([nothing], {}) == ["n"]


def test_benchmark_entry_requires_cell_only_for_track_a() -> None:
    base = {"scene_id": "s", "split_id": "x", "runs": ["20261101-000000-omnire-abcd"]}
    BenchmarkEntry(bench_track="A", cell="A-LB", **base)
    BenchmarkEntry(bench_track="C", **base)
    with pytest.raises(ValueError, match="cell"):
        BenchmarkEntry(bench_track="A", **base)
    with pytest.raises(ValueError, match="cell"):
        BenchmarkEntry(bench_track="B", cell="A-L", **base)


def test_run_group_collects_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_a, run_b = "20261101-000000-omnire-aaaa", "20261101-000001-pvg-bbbb"
    rows = [make_row(run_a, 25.0), make_row(run_b, 25.1, train_time_s=50.0)]
    rows[0].gpu = "RTX 4090"
    rows[1].gpu = "Tesla T4"
    monkeypatch.setattr(benchmark, "report_rows", lambda *args, **kwargs: rows)
    summary = SimpleNamespace(
        per_camera={
            "image_00": SimpleNamespace(psnr=26.0),
            "image_02": SimpleNamespace(psnr=22.0),
        },
        depth=DepthMetrics(abs_rel=0.1, rmse=2.0, delta_1_25=0.9),
        protocol=SimpleNamespace(version="0.4"),
    )
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: summary)
    frames_dir = tmp_path / "d2s" / "stages" / "eval.standard"
    frames_dir.mkdir(parents=True)
    psnrs = [25.0] * 6 + [20.0] * 5 + [25.0] * 6
    (frames_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(r) for r in frames_with(psnrs)) + "\n"
    )
    cameras = [
        SimpleNamespace(camera_id="image_00", model="opencv_pinhole"),
        SimpleNamespace(camera_id="image_02", model="opencv_fisheye"),
    ]
    workdir = SimpleNamespace(
        load_scene=lambda scene_id: SimpleNamespace(manifest=SimpleNamespace(cameras=cameras)),
        run_dir=lambda scene_id, approach_id, run_id: tmp_path,
    )
    entry = BenchmarkEntry(
        bench_track="A", cell="A-L", scene_id="s", split_id="x", runs=[run_a, run_b]
    )
    group = benchmark.run_group(workdir, entry)  # type: ignore[arg-type]
    assert group.winner_run_id == run_b  # ngang nhau, cùng chưa đo gap: chi phí thấp hơn
    first = group.rows[0]
    assert first.rank == 1 and first.tied_with_best
    assert (first.psnr_pinhole, first.psnr_fisheye) == (26.0, 22.0)
    assert first.depth is not None and first.frames_measured == 17
    assert len(first.bad_segments) == 1
    assert any("GPU khác nhau" in w for w in group.warnings)
    with pytest.raises(ValueError, match="không có hoặc không thuộc"):
        benchmark.run_group(
            workdir,  # type: ignore[arg-type]
            entry.model_copy(update={"runs": [run_a, "20261101-000002-omnire-cccc"]}),
        )
