"""Kiểm tra ``split_indices``: chỉ số ảnh train/test của DriveStudio, có camera chỉ test."""

from drive2sim.recon.data.split_indices import split_indices


def test_matches_drivestudio_without_test_only() -> None:
    train_t, test_t, train_i, test_i = split_indices(17, 2, 8)
    assert test_t == [8, 16] and len(train_t) == 15
    assert test_i == [16, 17, 32, 33] and 0 in train_i and 1 in train_i
    assert len(train_i) == 15 * 2


def test_test_only_camera_only_in_test_indices() -> None:
    _, _, train_i, test_i = split_indices(17, 2, 8, [1])
    assert all(i % 2 == 0 for i in train_i)
    assert len(train_i) == 15
    assert test_i == [16, 17, 32, 33]


def test_zero_stride_has_no_test_images() -> None:
    train_t, test_t, train_i, test_i = split_indices(5, 2, 0)
    assert test_t == [] and test_i == []
    assert train_t == [0, 1, 2, 3, 4] and len(train_i) == 10


def test_test_only_camera_never_trains_even_without_test_frames() -> None:
    _, _, train_i, test_i = split_indices(5, 2, 0, [1])
    assert all(i % 2 == 0 for i in train_i) and test_i == []
