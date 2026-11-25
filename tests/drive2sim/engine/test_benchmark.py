"""Kiểm tra ``drive2sim/engine/benchmark.py``: đoạn đường hỏng, thứ hạng và gom kết quả nhóm.

- ``bad_segments``: ít nhất 5 khung test liền nhau thấp hơn trung vị quá 2 dB mới là một đoạn.
- ``order_group``: xếp theo Avg; run kém run tốt nhất dưới ngưỡng PSNR là ngang và được phân xử
  lần lượt bằng R2, R3, rồi thời gian train (chỉ khi cùng GPU).
- ``run_group``: ghép dòng báo cáo, ``summary.json`` và ``frames.jsonl`` thành kết quả của nhóm.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from drive2sim.core.schema import BenchmarkEntry, DepthMetrics
from drive2sim.engine import benchmark
from drive2sim.engine.benchmark import TieInputs, bad_segments, order_group
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
    run_id: str,
    psnr: float,
    train_time_s: Optional[float] = 100.0,
    ssim: float = 0.8,
    gpu: Optional[str] = "RTX 4090",
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
        gpu=gpu,
    )


def tie_inputs(**by_run: dict) -> dict[str, TieInputs]:
    """``TieInputs`` theo run: ``tie_inputs(a={"shift_psnr_cc": 22.0})``."""
    return {run_id: TieInputs(**fields) for run_id, fields in by_run.items()}


def abc_rows() -> list[ReportRow]:
    # A 25.0/5000 s, B 24.8/3000 s, C 24.6/1000 s, cùng GPU
    return [
        make_row("A", 25.0, train_time_s=5000),
        make_row("B", 24.8, train_time_s=3000),
        make_row("C", 24.6, train_time_s=1000),
    ]


def test_order_group_by_avg_when_not_tied() -> None:
    rows = [make_row("b", 24.0), make_row("a", 25.0), make_row("c", 23.0)]
    order = order_group(rows, {})
    assert order.run_ids == ["a", "b", "c"]
    assert order.winner == "a" and order.decided_by == "avg" and order.tied == {"a"}


def test_tied_flag_is_relative_to_avg_best() -> None:
    # C kém A 0,4 dB nên không ngang dù rẻ nhất; trước đây cờ ngang tính theo run đứng đầu sau
    # khi xếp phụ nên có thể lệch.
    order = order_group(abc_rows(), {})
    assert order.tied == {"A", "B"}
    assert order.winner == "B" and order.decided_by == "train_time"
    assert order.run_ids == ["B", "A", "C"]


def test_r2_decides_before_cost() -> None:
    rows = [make_row("A", 25.0, train_time_s=100), make_row("B", 24.9, train_time_s=900)]
    inputs = tie_inputs(A={"shift_psnr_cc": 22.0}, B={"shift_psnr_cc": 23.0})
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "r2"


def test_r2_within_threshold_falls_through_to_r3() -> None:
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(
        A={"shift_psnr_cc": 22.2, "export_gap_db": 0.1},
        B={"shift_psnr_cc": 22.0, "export_gap_db": 0.9},
    )
    order = order_group(rows, inputs)
    assert order.winner == "A" and order.decided_by == "r3"


def test_r3_gap_within_equal_threshold_falls_through_to_cost() -> None:
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(A={"export_gap_db": 0.5}, B={"export_gap_db": 0.7})
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "train_time"


def test_step_skipped_when_a_tied_run_lacks_value() -> None:
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(A={"shift_psnr_cc": 23.5}, B={})  # B chưa đo R2: bỏ bước cho cả tập
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "train_time"


def test_r3_not_applicable_is_skipped() -> None:
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(
        A={"export_gap_db": 0.1}, B={"export_gap_db": 0.9, "export_applicable": False}
    )
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "train_time"


def test_cost_step_needs_same_gpu() -> None:
    rows = [
        make_row("A", 25.0, train_time_s=900, gpu="RTX 4090"),
        make_row("B", 24.9, train_time_s=100, gpu="Tesla T4"),
    ]
    order = order_group(rows, {})
    assert order.winner is None and order.decided_by is None
    assert order.tied == {"A", "B"} and order.run_ids == ["A", "B"]
    assert any("GPU" in warning for warning in order.warnings)


def test_cost_step_needs_train_time_and_gpu() -> None:
    rows = [make_row("A", 25.0, train_time_s=None), make_row("B", 24.9, train_time_s=100)]
    assert order_group(rows, {}).winner is None
    rows = [make_row("A", 25.0, gpu=None), make_row("B", 24.9, gpu=None)]
    assert order_group(rows, {}).winner is None


def test_mixed_protocol_has_no_winner() -> None:
    rows = [make_row("A", 25.0), make_row("B", 24.9)]
    inputs = tie_inputs(A={"protocol": "0.5"}, B={"protocol": "ncore-0.3"})
    order = order_group(rows, inputs)
    assert order.winner is None and order.tied == set() and order.decided_by is None
    assert order.run_ids == ["A", "B"]
    assert any("protocol" in warning for warning in order.warnings)


def test_drivestudio_protocols_0_4_and_0_5_rank_together() -> None:
    # 0.4 và 0.5 cùng cách đo R1 (CONTRACTS 6.3): xếp hạng được; run 0.4 chưa có R2 nên bỏ bước đó
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(
        A={"protocol": "0.4"}, B={"protocol": "0.5", "shift_psnr_cc": 23.0, "export_gap_db": 0.2}
    )
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "train_time"
    assert order.warnings == []


def test_old_runs_without_new_fields_rank_by_avg_and_cost() -> None:
    inputs = {run_id: TieInputs(protocol="0.4") for run_id in "ABC"}
    order = order_group(abc_rows(), inputs)
    assert order.winner == "B" and order.decided_by == "train_time"


def test_custom_threshold_changes_who_is_tied() -> None:
    order = order_group(abc_rows(), {}, threshold_db=0.5)
    assert order.tied == {"A", "B", "C"} and order.winner == "C"


def test_order_group_unmeasured_last_and_all_unmeasured() -> None:
    nothing = ReportRow(run_id="n", scene_id="s", split_id="x", approach_id="omnire")
    assert order_group([nothing, make_row("a", 25.0)], {}).run_ids == ["a", "n"]
    only = order_group([nothing], {})
    assert only.run_ids == ["n"] and only.winner is None and only.tied == set()


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
    # ngang nhau nhưng khác GPU: thời gian train không so được nên không có người thắng
    assert group.winner_run_id is None and group.decided_by is None
    first = group.rows[0]
    assert first.run_id == run_b and first.rank == 1 and first.tied_with_best
    assert group.rows[1].tied_with_best
    assert (first.psnr_pinhole, first.psnr_fisheye) == (26.0, 22.0)
    assert first.depth is not None and first.frames_measured == 17
    assert len(first.bad_segments) == 1
    assert any("GPU khác nhau" in w for w in group.warnings)
    with pytest.raises(ValueError, match="không có hoặc không thuộc"):
        benchmark.run_group(
            workdir,  # type: ignore[arg-type]
            entry.model_copy(update={"runs": [run_a, "20261101-000002-omnire-cccc"]}),
        )
