"""Kiểm tra ``drive2sim/scripts/setup_benchmark_week04.py``: chia chunk, đổi khung gốc, xếp run.

- ``chunk_ranges``: 6 đoạn phủ cả cảnh, hai đoạn kề chồng đúng 10 khung.
- ``source_to_index_range``: đổi số hiệu khung gốc (có chỗ thiếu khung) sang chỉ số của cảnh.
- ``heading_change_deg``: đường thẳng gần 0 độ, rẽ vuông góc khoảng 90 độ.
- ``build_definition``: run được xếp đúng nhóm; bỏ run chạy thử, run lỗi; nhiều run khớp thì lấy
  run mới nhất; nhóm hiệu chỉnh cần ít nhất hai seed.
"""

import math
from pathlib import Path
from typing import Any, Optional

import pytest

from drive2sim.core.schema import RunApproach, RunRecord, StageRecord
from drive2sim.scripts import setup_benchmark_week04 as week04

A1 = week04.SHORT_SCENE


def make_run(
    run_id: str,
    approach_id: str,
    split_id: str = "every8_stereo",
    scene_id: str = A1,
    options: Optional[dict[str, Any]] = None,
    train: Optional[dict[str, Any]] = None,
    status: str = "succeeded",
    quick: bool = False,
) -> RunRecord:
    """Run giả với một stage ``train.drivestudio`` (khi ``train`` khác ``None``)."""
    stages = []
    if train is not None:
        stages.append(StageRecord(stage_id="train.drivestudio", status=status, params=train))
    return RunRecord(
        schema_version="d2s-run/0.3",
        run_id=run_id,
        scene_id=scene_id,
        split_id=split_id,
        approach=RunApproach(
            approach_id=approach_id, version="0.1.0", options=options or {}, quick=quick
        ),
        level="L1",
        executor="local",
        status=status,
        stages=stages,
    )


def test_chunks_cover_the_scene_with_overlap() -> None:
    ranges = week04.chunk_ranges(864)
    assert len(ranges) == 6
    assert ranges[0][0] == 0 and ranges[-1][1] == 863
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert end - start + 1 == week04.CHUNK_OVERLAP
    assert max(end - start + 1 for start, end in ranges) <= 153


def test_chunks_reject_a_scene_too_short() -> None:
    with pytest.raises(ValueError):
        week04.chunk_ranges(5, count=6, overlap=10)


def test_source_range_maps_to_scene_indices() -> None:
    frame_ids = [167, 168, 170, 175, 176, 180]
    assert week04.source_to_index_range(frame_ids, (169, 176)) == (2, 4)
    with pytest.raises(ValueError):
        week04.source_to_index_range(frame_ids, (1, 100))


def write_poses(scene_dir: Path, headings_deg: list[float]) -> None:
    """Ghi ``ego_pose/<t>.txt`` cho xe quay quanh trục z theo các góc cho trước."""
    pose_dir = scene_dir / "ego_pose"
    pose_dir.mkdir(parents=True)
    for index, degrees in enumerate(headings_deg):
        angle = math.radians(degrees)
        rows = [
            [math.cos(angle), -math.sin(angle), 0.0, float(index)],
            [math.sin(angle), math.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        text = "\n".join(" ".join(str(value) for value in row) for row in rows)
        (pose_dir / f"{index:03d}.txt").write_text(text)


def test_heading_change_detects_a_turn(tmp_path: Path) -> None:
    # đi thẳng rồi rẽ 90 độ, góc đi qua ±180 độ để kiểm phép nối góc
    write_poses(tmp_path, [0.0] * 5 + [170.0, 180.0, -170.0, -100.0])
    assert week04.heading_change_deg(tmp_path, (0, 4)) == pytest.approx(0.0, abs=1e-6)
    assert week04.heading_change_deg(tmp_path, (5, 8)) == pytest.approx(90.0, abs=1e-6)


def test_definition_groups_runs_by_matrix_cell() -> None:
    omnire = {"renderer": "3dgs", "fastgs": True}
    pinhole, fisheye = (
        {"renderer": "3dgs", "campose": True},
        {"renderer": "3dgut", "campose": False},
    )
    runs = [  # mới nhất trước, như list_runs
        make_run("20261127-120000-pvg-aaaa", "pvg", options=pinhole, train={"lidar": False}),
        make_run("20261127-110000-pvg-bbbb", "pvg", options=pinhole, train={}),
        make_run("20261127-100000-pvg-cccc", "pvg", options=pinhole, train={}),
        make_run("20261127-095000-pvg-dddd", "pvg", options=pinhole, train={}, quick=True),
        make_run("20261127-090000-deformablegs-eeee", "deformablegs", train={}, status="failed"),
        make_run("20261127-080000-omnire-0002", "omnire", options=omnire, train={"seed": 2}),
        make_run("20261127-070000-omnire-0001", "omnire", options=omnire, train={"seed": 1}),
        make_run("20261127-060000-omnire-0000", "omnire", options=omnire, train={"seed": 0}),
        make_run(
            "20261127-050000-omnire-a3d0",
            "omnire",
            split_id="every8_stereo_fisheye2",
            options={"renderer": "3dgut", "fastgs": True},
            train={},
        ),
        make_run("20261127-040000-nurec-c0d0", "nurec", split_id="every8_mixed3"),
        make_run(
            "20261127-030000-pvg-f1e2",
            "pvg",
            split_id="every8_stereo_fisheye2",
            options=fisheye,
            train={},
        ),
    ]
    result = week04.build_definition(runs)
    assert result.benchmark is not None
    groups = {
        (entry.split_id, entry.cell or entry.bench_track): entry.runs
        for entry in result.benchmark.entries
    }
    assert groups[("every8_stereo", "A-LB")] == ["20261127-060000-omnire-0000"]
    assert groups[("every8_stereo", "A-L")] == ["20261127-110000-pvg-bbbb"]
    assert groups[("every8_stereo", "A-0")] == ["20261127-120000-pvg-aaaa"]
    assert groups[("every8_stereo_fisheye2", "A-LB")] == ["20261127-050000-omnire-a3d0"]
    assert groups[("every8_mixed3", "C")] == ["20261127-040000-nurec-c0d0"]
    assert groups[("every8_stereo_fisheye2", "A-L")] == ["20261127-030000-pvg-f1e2"]
    assert ("every8_pinhole2", "B") not in groups
    assert result.benchmark.calibration is not None
    assert result.benchmark.calibration.runs == [
        "20261127-060000-omnire-0000",
        "20261127-070000-omnire-0001",
        "20261127-080000-omnire-0002",
    ]
    assert any("bỏ 20261127-100000-pvg-cccc" in note for note in result.notes)


def test_definition_without_runs_writes_nothing() -> None:
    result = week04.build_definition([])
    assert result.benchmark is None
    assert any("hiệu chỉnh seed" in note for note in result.notes)


def test_calibration_needs_two_seeds() -> None:
    omnire = {"renderer": "3dgs", "fastgs": True}
    runs = [make_run("20261127-060000-omnire-0000", "omnire", options=omnire, train={})]
    result = week04.build_definition(runs)
    assert result.benchmark is not None and result.benchmark.calibration is None
