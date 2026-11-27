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

from drive2sim.core import files
from drive2sim.core.schema import (
    BenchmarkEntry,
    BenchmarkFile,
    BenchmarkGroup,
    BenchmarkRunRow,
    DepthMetrics,
    RunApproach,
    RunRecord,
    ShiftedMetrics,
    StageRecord,
)
from drive2sim.engine import benchmark
from drive2sim.engine.benchmark import (
    GroupContext,
    TieInputs,
    bad_segments,
    bad_segments_by_group,
    order_group,
    tie_threshold_from_psnrs,
)
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


def test_bad_segments_skip_views_without_valid_pixels() -> None:
    frames = frames_with([25.0] * 12)
    frames[3]["psnr"] = None  # view không có điểm hợp lệ: eval ghi null
    assert bad_segments(frames) == []  # không TypeError, không NaN
    only_null = frames_with([25.0])
    only_null[0]["psnr"] = None
    assert bad_segments(only_null) == []


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
        A={"shift_psnr_cc": 22.2, "ply_psnr": 23.9},
        B={"shift_psnr_cc": 22.0, "ply_psnr": 23.1},
    )
    order = order_group(rows, inputs)
    assert order.winner == "A" and order.decided_by == "r3"


def test_r3_uses_ply_psnr() -> None:
    # A render gốc tốt hơn nên gap lớn hơn, nhưng .ply của A vẫn tốt hơn: R3 chọn A theo ply_psnr
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(A={"ply_psnr": 24.0}, B={"ply_psnr": 23.5})
    order = order_group(rows, inputs)
    assert order.winner == "A" and order.decided_by == "r3"


def test_r3_skipped_when_ply_is_not_static_background() -> None:
    # PVG, DeformableGS: .ply là tham số gốc, chưa tính theo thời gian; R3 không phân xử
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(
        A={"ply_psnr": 24.0, "export_static": False}, B={"ply_psnr": 22.0, "export_static": False}
    )
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "train_time"
    assert any("R3" in warning for warning in order.warnings)


def test_r3_ply_psnr_within_equal_threshold_falls_through_to_cost() -> None:
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(A={"ply_psnr": 23.5}, B={"ply_psnr": 23.3})
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "train_time"


def test_step_skipped_when_a_tied_run_lacks_value() -> None:
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(A={"shift_psnr_cc": 23.5}, B={})  # B chưa đo R2: bỏ bước cho cả tập
    order = order_group(rows, inputs)
    assert order.winner == "B" and order.decided_by == "train_time"


def test_r3_not_applicable_is_skipped() -> None:
    rows = [make_row("A", 25.0, train_time_s=900), make_row("B", 24.9, train_time_s=100)]
    inputs = tie_inputs(A={"ply_psnr": 24.0}, B={"ply_psnr": 22.0, "export_applicable": False})
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
        A={"protocol": "0.4"}, B={"protocol": "0.5", "shift_psnr_cc": 23.0, "ply_psnr": 23.2}
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
        shifted=None,
        coverage=0.5,
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
        load_split=lambda scene_id, split_id: SimpleNamespace(test_only_cameras=[]),
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
    assert first.coverage == 0.5
    assert len(first.bad_segments["pinhole"]) == 1
    assert any("GPU khác nhau" in w for w in group.warnings)
    with pytest.raises(ValueError, match="không có hoặc không thuộc"):
        benchmark.run_group(
            workdir,  # type: ignore[arg-type]
            entry.model_copy(update={"runs": [run_a, "20261101-000002-omnire-cccc"]}),
        )


# ---------------------------------------------------------------- R2, R3, R6, R7, R8 theo dòng


def frames_for_cameras(num_frames: int, per_camera: dict[int, dict]) -> list[dict]:
    """Bản ghi từng ảnh cho nhiều camera; ``per_camera[i]`` có ``psnr``, ``x`` (hàm của khung)."""
    records = []
    for index in range(num_frames):
        for camera, spec in per_camera.items():
            records.append(
                {
                    "frame_index": index,
                    "camera_index": camera,
                    "position": [spec["x"](index), 0.0, 0.0],
                    "psnr": spec["psnr"](index),
                }
            )
    return records


def test_bad_segments_are_split_by_camera_group_and_skip_test_only() -> None:
    frames = frames_for_cameras(
        17,
        {
            0: {"x": lambda i: 10.0 * i, "psnr": lambda i: 25.0},  # pinhole tốt
            1: {"x": lambda i: 10.0 * i, "psnr": lambda i: 10.0 if i < 5 else 25.0},  # chỉ test
            2: {"x": lambda i: 10.0 * i, "psnr": lambda i: 17.0 if 6 <= i <= 10 else 22.0},
        },
    )
    groups = bad_segments_by_group(frames, fisheye_indices={2}, exclude_indices={1})
    assert groups["pinhole"] == [] and len(groups["fisheye"]) == 1
    assert groups["fisheye"][0].num_frames == 5
    # không loại camera chỉ test thì nó làm lộ một đoạn hỏng ở nhóm pinhole
    leaked = bad_segments_by_group(frames, fisheye_indices={2}, exclude_indices=set())
    assert len(leaked["pinhole"]) == 1


def test_bad_segments_by_group_omits_groups_without_images() -> None:
    frames = frames_with([25.0] * 6)
    groups = bad_segments_by_group(frames, fisheye_indices={2}, exclude_indices=set())
    assert set(groups) == {"pinhole"}


def stage_summary(**fields: object) -> SimpleNamespace:
    """``summary.json`` giả với đủ trường mà ``build_row`` đọc."""
    base = {
        "per_camera": {},
        "depth": None,
        "protocol": SimpleNamespace(version="0.5"),
        "shifted": None,
        "coverage": None,
    }
    return SimpleNamespace(**{**base, **fields})


def write_frames(run_dir: Path, records: list[dict]) -> None:
    frames_dir = run_dir / "d2s" / "stages" / "eval.standard"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_camera_group_psnr_skips_test_only_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # per_camera của summary giữ số DriveStudio, gồm cả camera chỉ test; R1 theo loại camera chỉ
    # tính trên camera đã train
    per_camera = {
        "image_00": SimpleNamespace(psnr=26.0),
        "image_01": SimpleNamespace(psnr=20.0),
        "image_02": SimpleNamespace(psnr=22.0),
    }
    monkeypatch.setattr(
        benchmark, "load_summary", lambda run_dir: stage_summary(per_camera=per_camera)
    )
    context = GroupContext(
        "A", fisheye_ids={"image_02"}, test_only_ids={"image_01"}, test_only_indices={1}
    )
    row = benchmark.build_row(make_row("a", 25.0), tmp_path, context, 1, True)
    assert (row.psnr_pinhole, row.psnr_fisheye) == (26.0, 22.0)


def test_row_reads_shifted_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shifted = ShiftedMetrics(
        cameras=["image_01"],
        num_images=2,
        psnr=21.0,
        ssim=0.7,
        lpips=0.3,
        psnr_cc=23.1,
        lpips_cc=0.27,
    )
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: stage_summary(shifted=shifted))
    row = benchmark.build_row(make_row("a", 25.0), tmp_path, GroupContext("A"), 1, True)
    assert (row.shift_psnr, row.shift_lpips) == (21.0, 0.3)
    assert (row.shift_psnr_cc, row.shift_lpips_cc) == (23.1, 0.27)
    assert row.shift_cameras == ["image_01"]


def test_row_without_shifted_leaves_r2_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: stage_summary())
    row = benchmark.build_row(make_row("a", 25.0), tmp_path, GroupContext("A"), 1, True)
    assert row.shift_psnr_cc is None and row.shift_psnr is None and row.shift_cameras == []
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: None)
    bare = benchmark.build_row(make_row("a", 25.0), tmp_path, GroupContext("A"), 1, True)
    assert bare.shift_psnr_cc is None and bare.bad_segments == {}


def test_export_applicable_depends_on_stage_list(tmp_path: Path) -> None:
    assert benchmark.export_applicable(["train.drivestudio", "export.ply_world"])
    assert benchmark.export_applicable(["train.nurec", "export.nurec_ply"])
    assert not benchmark.export_applicable(["train.mcgs", "eval.mcgs"])
    run = RunRecord(
        schema_version="d2s-run/0.3",
        run_id="20261112-081000-omnire-a1b2",
        scene_id="nuscenes_scene-0061",
        split_id="every8_allcams",
        approach=RunApproach(approach_id="omnire", version="0.1.0"),
        level="L1",
        executor="local",
        status="succeeded",
        stages=[StageRecord(stage_id="train.mcgs", status="succeeded")],
    )
    files.dump(run, tmp_path / "d2s" / "run.json")
    assert benchmark.run_stage_ids(tmp_path) == ["train.mcgs"]
    assert benchmark.run_stage_ids(tmp_path / "missing") is None


@pytest.mark.parametrize(
    ("method", "static"), [("pvg", False), ("deformablegs", False), ("omnire", True)]
)
def test_export_static_depends_on_trained_method(
    tmp_path: Path, method: str, static: bool
) -> None:
    run = RunRecord(
        schema_version="d2s-run/0.3",
        run_id=f"20261112-081000-{method}-a1b2",
        scene_id="nuscenes_scene-0061",
        split_id="every8_allcams",
        approach=RunApproach(approach_id=method, version="0.1.0"),
        level="L1",
        executor="local",
        status="succeeded",
        stages=[
            StageRecord(
                stage_id="train.drivestudio", status="succeeded", params={"method": method}
            )
        ],
    )
    files.dump(run, tmp_path / "d2s" / "run.json")
    assert benchmark.run_export_static(tmp_path) is static
    assert benchmark.run_export_static(tmp_path / "missing")


def test_row_marks_export_not_applicable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: stage_summary())
    stages = [StageRecord(stage_id="train.mcgs", status="succeeded")]
    run = RunRecord(
        schema_version="d2s-run/0.3",
        run_id="20261112-081000-omnire-a1b2",
        scene_id="nuscenes_scene-0061",
        split_id="every8_allcams",
        approach=RunApproach(approach_id="omnire", version="0.1.0"),
        level="L1",
        executor="local",
        status="succeeded",
        stages=stages,
    )
    files.dump(run, tmp_path / "d2s" / "run.json")
    row = benchmark.build_row(make_row("a", 25.0), tmp_path, GroupContext("A"), 1, True)
    assert row.export_applicable is False
    plain = tmp_path / "other"
    plain.mkdir()
    assert benchmark.build_row(
        make_row("a", 25.0), plain, GroupContext("A"), 1, True
    ).export_applicable


def test_per_100m_only_for_track_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: stage_summary())
    write_frames(tmp_path, frames_with([25.0] * 51))  # 50 khoảng 10 m = 500 m
    run = make_row("a", 25.0, train_time_s=3600.0)
    run.ply_size_mb = 200.0
    run.recon_time_s = 4000.0
    track_b = benchmark.build_row(run, tmp_path, GroupContext("B"), 1, True)
    assert track_b.distance_m == pytest.approx(500.0)
    assert track_b.train_s_per_100m == pytest.approx(720.0)
    assert track_b.ply_mb_per_100m == pytest.approx(40.0)
    assert track_b.recon_time_s == 4000.0
    track_a = benchmark.build_row(run, tmp_path, GroupContext("A"), 1, True)
    assert track_a.distance_m == pytest.approx(500.0)
    assert track_a.train_s_per_100m is None and track_a.ply_mb_per_100m is None


def test_distance_ignores_test_only_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: stage_summary())
    frames = frames_for_cameras(
        6,
        {
            0: {"x": lambda i: 10.0 * i, "psnr": lambda i: 25.0},
            1: {"x": lambda i: 0.0, "psnr": lambda i: 5.0},  # camera chỉ test, vị trí lệch hẳn
        },
    )
    write_frames(tmp_path, frames)
    context = GroupContext("B", test_only_indices={1})
    row = benchmark.build_row(make_row("a", 25.0), tmp_path, context, 1, True)
    assert row.distance_m == pytest.approx(50.0)
    assert row.frames_measured == 6  # chỉ đếm ảnh của camera đã train


def test_ate_is_read_from_mcgs_eval_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: stage_summary())
    write_frames(tmp_path, frames_with([25.0] * 6))
    stage_dir = tmp_path / "d2s" / "stages" / "eval.standard"
    (stage_dir / "mcgs_eval.json").write_text(json.dumps({"ate_rmse_m": 0.42}))
    row = benchmark.build_row(make_row("a", 25.0), tmp_path, GroupContext("A"), 1, True)
    assert row.ate_rmse_m == pytest.approx(0.42)
    (stage_dir / "mcgs_eval.json").unlink()
    again = benchmark.build_row(make_row("a", 25.0), tmp_path, GroupContext("A"), 1, True)
    assert again.ate_rmse_m is None


def test_run_group_settles_a_tie_with_r2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_a, run_b = "20261101-000000-omnire-aaaa", "20261101-000001-pvg-bbbb"
    rows = [make_row(run_a, 25.0, train_time_s=10.0), make_row(run_b, 24.9, train_time_s=900.0)]
    monkeypatch.setattr(benchmark, "report_rows", lambda *args, **kwargs: rows)
    shifted = {run_a: 22.0, run_b: 23.0}

    def fake_summary(run_dir: Path) -> SimpleNamespace:
        value = shifted[run_dir.name]
        return stage_summary(
            shifted=ShiftedMetrics(
                cameras=["image_01"],
                num_images=1,
                psnr=value - 1,
                ssim=0.7,
                lpips=0.3,
                psnr_cc=value,
            )
        )

    monkeypatch.setattr(benchmark, "load_summary", fake_summary)
    cameras = [
        SimpleNamespace(camera_id="image_00", model="opencv_pinhole"),
        SimpleNamespace(camera_id="image_01", model="opencv_pinhole"),
    ]
    split = SimpleNamespace(test_only_cameras=["image_01"])
    workdir = SimpleNamespace(
        load_scene=lambda scene_id: SimpleNamespace(manifest=SimpleNamespace(cameras=cameras)),
        load_split=lambda scene_id, split_id: split,
        run_dir=lambda scene_id, approach_id, run_id: tmp_path / run_id,
    )
    entry = BenchmarkEntry(
        bench_track="A", cell="A-L", scene_id="s", split_id="x", runs=[run_a, run_b]
    )
    group = benchmark.run_group(workdir, entry)  # type: ignore[arg-type]
    assert group.winner_run_id == run_b and group.decided_by == "r2"
    assert [row.run_id for row in group.rows] == [run_b, run_a]
    assert group.rows[0].shift_psnr_cc == 23.0


# ---------------------------------------------------------------- ngưỡng hoà hiệu chỉnh bằng seed


def test_threshold_floor_is_0_3() -> None:
    assert tie_threshold_from_psnrs([25.0, 25.1, 25.2]) == pytest.approx(0.3)


def test_threshold_is_two_sigma() -> None:
    assert tie_threshold_from_psnrs([25.0, 25.5, 26.0]) == pytest.approx(1.0)


def test_threshold_needs_two_values() -> None:
    assert tie_threshold_from_psnrs([25.0]) is None
    assert tie_threshold_from_psnrs([]) is None


def test_benchmark_file_0_1_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(
        "schema_version: d2s-benchmark/0.1\n"
        "entries:\n"
        "  - {bench_track: C, scene_id: s, split_id: x, runs: [r1]}\n"
    )
    assert files.load(path, BenchmarkFile).calibration is None


CALIBRATION_YAML = (
    "schema_version: d2s-benchmark/0.2\n"
    "calibration: {scene_id: s, split_id: x, runs: [%s]}\n"
    "entries:\n"
    "  - {bench_track: A, cell: A-L, scene_id: s, split_id: x, runs: [r9]}\n"
)


def run_benchmark_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    psnrs: dict[str, Optional[float]],
    yaml_text: Optional[str] = None,
) -> tuple:
    """Chạy ``run_benchmark`` với ``report_rows`` và ``run_group`` giả; trả (kết quả, ngưỡng)."""
    definition = tmp_path / "bench.yaml"
    definition.write_text(yaml_text or CALIBRATION_YAML % ", ".join(psnrs))
    rows = [
        make_row(run_id, psnr)
        if psnr is not None
        else ReportRow(run_id=run_id, scene_id="s", split_id="x", approach_id="omnire")
        for run_id, psnr in psnrs.items()
    ]
    monkeypatch.setattr(benchmark, "report_rows", lambda *args, **kwargs: rows)
    used: list[float] = []

    def fake_group(workdir: object, entry: BenchmarkEntry, threshold_db: float) -> BenchmarkGroup:
        used.append(threshold_db)
        return BenchmarkGroup(
            bench_track="A", cell="A-L", scene_id="s", split_id="x", rows=[], warnings=[]
        )

    monkeypatch.setattr(benchmark, "run_group", fake_group)
    result = benchmark.run_benchmark(SimpleNamespace(), definition)  # type: ignore[arg-type]
    return result, used


def test_calibration_sets_the_threshold_for_every_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, used = run_benchmark_with(tmp_path, monkeypatch, {"r1": 25.0, "r2": 25.5, "r3": 26.0})
    assert result.tie_threshold_db == pytest.approx(1.0)
    assert result.tie_threshold_source == "calibration"
    assert used == [pytest.approx(1.0)] and result.warnings == []


def test_calibration_with_one_measured_run_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, used = run_benchmark_with(tmp_path, monkeypatch, {"r1": 25.0, "r2": None})
    assert result.tie_threshold_db == 0.3 and used == [0.3]
    assert result.tie_threshold_source == "default"
    assert len(result.warnings) == 1 and "hiệu chỉnh" in result.warnings[0]


def test_no_calibration_uses_default_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = (
        "schema_version: d2s-benchmark/0.2\n"
        "entries:\n"
        "  - {bench_track: C, scene_id: s, split_id: x, runs: [r9]}\n"
    )
    result, used = run_benchmark_with(tmp_path, monkeypatch, {"r9": 25.0}, text)
    assert result.tie_threshold_db == 0.3 and used == [0.3]
    assert result.tie_threshold_source == "default" and result.warnings == []


def test_calibration_with_unknown_run_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = CALIBRATION_YAML % "r1, r_missing"
    with pytest.raises(ValueError, match="r_missing"):
        run_benchmark_with(tmp_path, monkeypatch, {"r1": 25.0, "r2": 25.1}, text)


def test_legacy_bad_segments_list_maps_to_all_group() -> None:
    # d2s-benchresult/0.1 ghi một danh sách chung; từ 0.2 là dict theo nhóm camera. Đọc được file
    # cũ (cùng major) bằng khoá "all", không phải khoá của nhóm camera nào.
    legacy = {
        "run_id": "20261101-000000-omnire-aaaa",
        "approach_id": "omnire",
        "rank": 1,
        "tied_with_best": True,
        "bad_segments": [{"start_m": 0.0, "end_m": 40.0, "num_frames": 5, "min_psnr": 19.5}],
    }
    row = BenchmarkRunRow.model_validate(legacy)
    assert list(row.bad_segments) == ["all"] and row.bad_segments["all"][0].num_frames == 5
    assert BenchmarkRunRow.model_validate({**legacy, "bad_segments": []}).bad_segments == {}


def test_legacy_ncore_protocol_0_5_is_separated_when_reading_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_a, run_b = "20261101-000000-nurec-aaaa", "20261101-000001-pvg-bbbb"
    rows = [make_row(run_a, 25.0), make_row(run_b, 24.9)]
    monkeypatch.setattr(benchmark, "report_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(benchmark, "load_summary", lambda run_dir: stage_summary())
    monkeypatch.setattr(
        benchmark,
        "run_stage_ids",
        lambda run_dir: ["eval.ncore"] if run_dir.name == run_a else ["eval.standard"],
    )
    cameras = [SimpleNamespace(camera_id="image_00", model="opencv_pinhole")]
    workdir = SimpleNamespace(
        load_scene=lambda _: SimpleNamespace(manifest=SimpleNamespace(cameras=cameras)),
        load_split=lambda *args: SimpleNamespace(test_only_cameras=[]),
        run_dir=lambda scene, approach, run: tmp_path / run,
    )
    entry = BenchmarkEntry(
        bench_track="A", cell="A-L", scene_id="s", split_id="x", runs=[run_a, run_b]
    )
    group = benchmark.run_group(workdir, entry)
    assert group.winner_run_id is None
    assert any("protocol" in warning for warning in group.warnings)
