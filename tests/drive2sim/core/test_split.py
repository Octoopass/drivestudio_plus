"""Test cách chia split trong ``drive2sim.core.split``."""

import json
from pathlib import Path

import pytest

from drive2sim.core import files
from drive2sim.core.schema import Split
from drive2sim.core.split import every_nth_frame_pairs, make_split, matches_strategy


def test_every_nth_frame_matches_drivestudio_rule() -> None:
    # split_train_test của DriveStudio: khung test là arange(N, num_timesteps, N),
    # tính từ khung đầu của đoạn.
    train, test = every_nth_frame_pairs((0, 20), ["A"], 8)
    assert [frame for frame, _ in test] == [8, 16]
    train_frames = [frame for frame, _ in train]
    assert 0 in train_frames and 8 not in train_frames


def test_every_nth_frame_with_offset_range() -> None:
    _, test = every_nth_frame_pairs((5, 25), ["A", "B"], 10)
    assert test == [(15, "A"), (15, "B"), (25, "A"), (25, "B")]


def test_make_split_is_valid_and_complete() -> None:
    split = make_split(
        "nuscenes_scene-0061", ["A", "B", "C"], (0, 38), "every_nth_frame:8", "every8_allcams"
    )
    assert len(split.train) + len(split.test) == 39 * 3
    assert matches_strategy(split)


def test_test_only_camera_is_never_trained() -> None:
    train, test = every_nth_frame_pairs((0, 16), ["image_00", "image_01"], 8, ["image_01"])
    assert all(camera == "image_00" for _, camera in train)
    assert len(train) == 15  # 17 khung − 2 khung test, chỉ image_00
    assert test == [(8, "image_00"), (8, "image_01"), (16, "image_00"), (16, "image_01")]


def test_matches_strategy_with_test_only_camera() -> None:
    split = make_split(
        "kitti360_drive-0003",
        ["image_00", "image_01"],
        (0, 199),
        "every_nth_frame:8",
        "every8_stereo",
        ["image_01"],
    )
    assert split.test_only_cameras == ["image_01"]
    assert matches_strategy(split)


@pytest.mark.parametrize("test_only", [["image_09"], ["image_00", "image_01"]])
def test_rejects_unknown_or_all_cameras_test_only(test_only: list[str]) -> None:
    with pytest.raises(ValueError):
        make_split(
            "kitti360_drive-0003",
            ["image_00", "image_01"],
            (0, 16),
            "every_nth_frame:8",
            "x",
            test_only,
        )


def test_split_0_1_file_still_loads(tmp_path: Path) -> None:
    # file d2s-split/0.1 không có test_only_cameras: nạp được, trường mặc định rỗng
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "d2s-split/0.1",
                "split_id": "every8_allcams",
                "scene_id": "nuscenes_scene-0061",
                "strategy": "every_nth_frame:8",
                "frame_range": [0, 16],
                "cameras": ["CAM_FRONT"],
                "train": [[0, "CAM_FRONT"], [1, "CAM_FRONT"]],
                "test": [[8, "CAM_FRONT"], [16, "CAM_FRONT"]],
            }
        )
    )
    split = files.load(path, Split)
    assert split.test_only_cameras == []
