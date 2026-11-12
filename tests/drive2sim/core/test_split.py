from drive2sim.core.split import every_nth_frame_pairs, make_split, matches_strategy


def test_every_nth_frame_matches_drivestudio_rule():
    # DriveStudio split_train_test: test timesteps = arange(N, num_timesteps, N) relative to the start
    train, test = every_nth_frame_pairs((0, 20), ["A"], 8)
    assert [f for f, _ in test] == [8, 16]
    assert 0 in [f for f, _ in train] and 8 not in [f for f, _ in train]


def test_every_nth_frame_with_offset_range():
    _, test = every_nth_frame_pairs((5, 25), ["A", "B"], 10)
    assert test == [(15, "A"), (15, "B"), (25, "A"), (25, "B")]


def test_make_split_is_valid_and_complete():
    s = make_split("nuscenes_scene-0061", ["A", "B", "C"], (0, 38), "every_nth_frame:8", "every8_allcams")
    assert len(s.train) + len(s.test) == 39 * 3
    assert matches_strategy(s)
