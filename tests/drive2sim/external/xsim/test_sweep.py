"""Kiểm tra phép chiếu XYZI tại seam, biên cao độ và ô trùng của lưới XSIM."""

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from drive2sim.external.xsim.sweep import read_sweep_grid  # noqa: E402


def test_nearest_return_and_seam(tmp_path: Path) -> None:
    path = tmp_path / "000.bin"
    np.array(
        [
            [0, 3, 0, 1],
            [0, 2, 0, 2],
            [0, 2, 0, 9],
            [-0.01, 2, 0, 3],
            [0, 0, -1, 4],
            [0, 0, 0, 5],
            [np.nan, 0, 0, 6],
        ],
        dtype="<f4",
    ).tofile(path)
    grid, mask = read_sweep_grid(path, 4, 8, math.pi / 2, -math.pi / 2)
    assert mask.sum() == 3
    assert grid[2, 0].tolist() == [0, 2, 0, 2]
    assert grid[2, 7, 3] == 3
    assert grid[3, 0, 3] == 4
    assert not grid[~mask].any()


def test_truncated_sweep_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "001.bin"
    path.write_bytes(b"1234")
    with pytest.raises(ValueError, match="thiếu byte"):
        read_sweep_grid(path, 4, 8, 0.5, -0.5)
