"""Kiểm tra ``drive2sim/scripts/setup_benchmark_week04.py``: chia chunk, đổi khung gốc, xếp run.

- ``chunk_ranges``: 6 đoạn phủ cả cảnh, hai đoạn kề chồng đúng 10 khung.
- ``source_to_index_range``: đổi số hiệu khung gốc (có chỗ thiếu khung) sang chỉ số của cảnh.
- ``heading_change_deg``: đường thẳng gần 0 độ, rẽ vuông góc khoảng 90 độ.
- ``build_definition``: run được xếp đúng nhóm (pinhole và MCGS-SLAM trên W1, fisheye và track C
  trên A1); bỏ run chạy thử, run lỗi; nhiều run khớp thì lấy run mới nhất; nhóm hiệu chỉnh cần ít
  nhất hai seed; các chunk của #11 thành một mục ``chunked``.
- ``build_queue``: run thắng đã chốt 27/11 là mặc định; đợt 4 chạy lại các run W1.
- ``scene_stats``: quãng đường, tốc độ và số vật thể chạy của cảnh ứng viên W1.
"""

import json
import math
from pathlib import Path
from typing import Any, Optional

import pytest

from drive2sim.cli.output import parse_set
from drive2sim.core.schema import RunApproach, RunRecord, StageRecord
from drive2sim.scripts import setup_benchmark_week04 as week04

A1 = week04.SHORT_SCENE
W1 = week04.WAYMO_SCENE
W1_SPLIT = week04.WAYMO_SPLIT


def make_run(
    run_id: str,
    approach_id: str,
    split_id: str = W1_SPLIT,
    scene_id: str = W1,
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
        {"renderer": "3dgeer", "campose": False},
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
            scene_id=A1,
            options={"renderer": "3dgut", "fastgs": True},
            train={},
        ),
        make_run(
            "20261127-040000-nurec-c0d0",
            "nurec",
            split_id="every8_mixed3",
            scene_id=week04.LONG_SCENE,
        ),
        make_run(
            "20261127-030000-pvg-f1e2",
            "pvg",
            split_id="every8_stereo_fisheye2",
            scene_id=A1,
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
    assert groups[(W1_SPLIT, "A-LB")] == ["20261127-060000-omnire-0000"]
    assert groups[(W1_SPLIT, "A-L")] == ["20261127-110000-pvg-bbbb"]
    assert groups[(W1_SPLIT, "A-0")] == ["20261127-120000-pvg-aaaa"]
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


def test_wave_one_needs_no_winner_and_runs_long_jobs_first() -> None:
    queue = week04.build_queue("1")
    assert [c.number for c in queue] == ["5", "1", "2", "3", "6", "8"]  # run dài trước
    hours = [week04.EST_GPU_HOURS[c.number] for c in queue]
    assert hours == sorted(hours, reverse=True)
    assert next(c for c in queue if c.number == "1").argv() == [
        "omnire",
        "--scene",
        week04.WAYMO_SCENE,
        "--split",
        week04.WAYMO_SPLIT,
        "--set",
        "fastgs=true",
    ]
    assert "9" in [c.number for c in week04.build_queue("1", nurec=True)]
    assert "13" in [c.number for c in week04.build_queue("1", p2=True)]


def test_wave_two_fills_the_a_l_winner_and_reruns_six_and_eight() -> None:
    with pytest.raises(week04.QueueError):
        week04.build_queue("2", al_winner=None)
    assert {c.approach for c in week04.build_queue("2") if c.number == "4"} == {"pvg"}
    queue = week04.build_queue("2", al_winner="pvg", p2=True)
    by_number = {c.number: c for c in queue}
    assert sorted(by_number) == ["12", "4", "6", "7", "8"]
    assert {by_number[n].approach for n in ("4", "7", "12")} == {"pvg"}
    # chạy lại sau khi sửa: giữ approach của chính run, không bị thay bằng run thắng A-L
    assert by_number["6"].approach == "omnire" and by_number["8"].approach == "mcgs"
    assert by_number["6"].sets == ("renderer=3dgeer", "fastgs=true")
    assert by_number["7"].sets == ("renderer=3dgeer", "campose=false")


def test_wave_three_adds_one_run_per_chunk() -> None:
    with pytest.raises(week04.QueueError):
        week04.build_queue("3", alb_winner="pvg fastgs=true")
    with pytest.raises(week04.QueueError):
        week04.build_queue("3", alb_winner="omnire fastgs")
    queue = week04.build_queue("3", alb_winner="omnire renderer=3dgs fastgs=true", num_chunks=6)
    assert [c.number for c in queue] == ["10"] + ["11"] * 6
    assert queue[-1].split_id == "every8_stereo_c6"
    assert all(c.sets == ("renderer=3dgs", "fastgs=true") for c in queue)


def test_queue_file_lines_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "wave3.tsv"
    status = week04.main(
        [
            "--workdir",
            str(tmp_path),
            "queue",
            "--wave",
            "3",
            "--alb-winner",
            "omnire fastgs=true",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + week04.CHUNK_COUNT
    number, args = lines[0].split("\t")
    assert number == "10"
    assert args.split()[:1] == ["omnire"]
    # run thắng đã chốt là mặc định nên đợt 2 không cần --al-winner nữa
    assert week04.main(["--workdir", str(tmp_path), "queue", "--wave", "2"]) == 0


def test_wave_three_defaults_to_the_chosen_a_lb_winner() -> None:
    queue = week04.build_queue("3", num_chunks=6)
    assert {c.approach for c in queue} == {"omnire"}
    assert all(c.sets == ("renderer=3dgeer", "fastgs=true") for c in queue)


def test_wave_four_reruns_every_w1_run_with_mcgs_on_w1() -> None:
    queue = week04.build_queue("4", p2=True)
    by_number = {c.number: c for c in queue}
    assert sorted(by_number, key=int) == ["1", "2", "3", "4", "8", "13"]
    assert all(c.scene_id == week04.WAYMO_SCENE for c in queue)
    assert by_number["4"].approach == "pvg"
    assert by_number["4"].sets == ("train.drivestudio.lidar=false",)
    assert by_number["8"].approach == "mcgs"
    assert by_number["8"].sets == ("prep.mcgs.cameras=FRONT,FRONT_LEFT,FRONT_RIGHT",)
    _, stage_params = parse_set(list(by_number["8"].sets))
    assert stage_params == {"prep.mcgs": {"cameras": ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]}}


def test_commands_fill_the_chosen_winners(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert week04.main(["--workdir", str(tmp_path), "commands"]) == 0
    out = capsys.readouterr().out
    assert "<thắng" not in out
    assert "--split every8_stereo_a3 --set renderer=3dgeer --set fastgs=true" in out
    assert "d2s run start mcgs --scene " + week04.WAYMO_SCENE in out


def test_waymo_segment_reads_the_train_list() -> None:
    assert week04.waymo_segment(week04.WAYMO_W1_PREVIOUS).startswith(
        "segment-10455472356147194054"
    )
    assert week04.WAYMO_SEGMENT == week04.waymo_segment(week04.WAYMO_INDEX)
    with pytest.raises(ValueError):
        week04.waymo_segment(100000)


def test_scene_stats_measures_distance_speed_and_moving_objects(tmp_path: Path) -> None:
    write_poses(tmp_path, [0.0] * 11)  # 11 khung, mỗi khung 1 m: 10 m trong 1 giây
    instances = {
        "0": {
            "class_name": "Vehicle",
            "frame_annotations": {
                "obj_to_world": [
                    [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for x in (0, 5)
                ]
            },
        },
        "1": {
            "class_name": "Pedestrian",
            "frame_annotations": {
                "obj_to_world": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * 2
            },
        },
    }
    (tmp_path / "instances").mkdir()
    (tmp_path / "instances" / "instances_info.json").write_text(json.dumps(instances))
    stats = week04.scene_stats(tmp_path)
    assert stats.num_frames == 11
    assert stats.distance_m == pytest.approx(10.0)
    assert stats.speed_mps == pytest.approx(10.0)
    assert (stats.vehicles, stats.pedestrians, stats.moving) == (1, 1, 1)


def test_definition_puts_mcgs_on_w1_and_chunks_into_one_chunked_item() -> None:
    chunk = {"renderer": "3dgeer", "fastgs": True}
    runs = [
        make_run("20261128-000000-mcgs-a0a0", "mcgs"),
        *[
            make_run(
                f"20261128-0{index}0000-omnire-c{index}c{index}",
                "omnire",
                split_id=f"every8_stereo_c{index}",
                scene_id=week04.LONG_SCENE,
                options=chunk,
                train={},
            )
            for index in range(1, 5)
        ],
        make_run(  # cấu hình khác thắng A-LB: không vào #11
            "20261128-050000-omnire-c5c5",
            "omnire",
            split_id="every8_stereo_c5",
            scene_id=week04.LONG_SCENE,
            options={"renderer": "3dgut", "fastgs": True},
            train={},
        ),
    ]
    result = week04.build_definition(runs)
    assert result.benchmark is not None
    track_b = [e for e in result.benchmark.entries if e.bench_track == "B"]
    assert [(e.scene_id, e.split_id, e.runs) for e in track_b] == [
        (W1, W1_SPLIT, ["20261128-000000-mcgs-a0a0"])
    ]
    (item,) = result.benchmark.chunked
    assert item.scene_id == week04.LONG_SCENE
    assert [c.split_id for c in item.chunks] == [f"every8_stereo_c{i}" for i in range(1, 5)]
    assert any("thiếu chunk every8_stereo_c5, every8_stereo_c6" in n for n in result.notes)


def test_definition_skips_chunked_with_fewer_than_two_chunks() -> None:
    runs = [
        make_run(
            "20261128-010000-omnire-c1c1",
            "omnire",
            split_id="every8_stereo_c1",
            scene_id=week04.LONG_SCENE,
            options={"renderer": "3dgeer", "fastgs": True},
            train={},
        )
    ]
    result = week04.build_definition(runs)
    assert result.benchmark is None
    assert any("bỏ mục chunked" in note for note in result.notes)
