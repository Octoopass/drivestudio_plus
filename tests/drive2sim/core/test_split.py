"""Test cách chia split trong ``drive2sim.core.split``."""

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
