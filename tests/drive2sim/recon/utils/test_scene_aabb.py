"""Test ``drive2sim.recon.utils.scene_aabb``: viewer dùng đúng khung bao cảnh lúc train."""

from pathlib import Path

import pytest

from drive2sim.recon.utils.scene_aabb import save_scene_aabb, training_aabb

LOG = """\
I20261115 08:46:44 root lidar_source.py:118] [Lidar] Computing auto AABB ...
I20261115 08:46:44 root lidar_source.py:138] [Lidar] Auto AABB from LiDAR: tensor([-70.0000, \
-10.0000,   9.5000,  32.0000,   1.0000, 104.0000])
I20261115 08:47:01 root pixel_source.py:798] [Pixel] Auto AABB from camera: tensor([-1., -2., \
-3., 4., 5., 6.])
"""


def test_reads_last_aabb_from_old_log(tmp_path: Path) -> None:
    assert training_aabb(tmp_path) is None
    (tmp_path / "log.txt").write_text("không có khung\n")
    assert training_aabb(tmp_path) is None
    (tmp_path / "log.txt").write_text(LOG.replace("\\\n", ""))
    assert training_aabb(tmp_path) == [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0]


def test_saved_file_wins_over_log(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text(LOG.replace("\\\n", ""))
    save_scene_aabb(tmp_path, [-70, -10, 9.5, 32, 1, 104])
    assert training_aabb(tmp_path) == [-70.0, -10.0, 9.5, 32.0, 1.0, 104.0]
    with pytest.raises(ValueError, match="6 số"):
        save_scene_aabb(tmp_path, [1, 2, 3])
