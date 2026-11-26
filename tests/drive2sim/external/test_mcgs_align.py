"""Kiểm tra ``drive2sim/external/mcgs/align.py``: căn quỹ đạo SLAM về hệ của cảnh.

- ``umeyama``: tìm lại đúng phép đồng dạng đã dùng để tạo dữ liệu, có và không có tỉ lệ; từ chối
  đầu vào suy biến.
- ``ate_rmse``: bằng 0 khi hai quỹ đạo chỉ khác một phép đồng dạng, và đo đúng nhiễu thêm vào.
- ``read_tum``, ``match_by_time``, ``pose_to_matrix``: đọc file quỹ đạo và ghép theo thời gian.
"""

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from drive2sim.external.mcgs import align  # noqa: E402


def make_rotation(angle: float) -> "np.ndarray":
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def make_track() -> "np.ndarray":
    rng = np.random.default_rng(0)
    steps = rng.normal(size=(40, 3)) * [1.0, 0.3, 0.05]
    return np.cumsum(steps, axis=0)


def test_umeyama_recovers_similarity() -> None:
    source = make_track()
    rotation, translation, scale = make_rotation(0.7), np.array([3.0, -2.0, 1.0]), 4.2
    target = scale * source @ rotation.T + translation
    found_scale, found_rotation, found_translation = align.umeyama(source, target)
    assert found_scale == pytest.approx(scale)
    assert found_rotation == pytest.approx(rotation, abs=1e-9)
    assert found_translation == pytest.approx(translation, abs=1e-9)
    assert np.linalg.det(found_rotation) == pytest.approx(1.0)


def test_umeyama_without_scale_keeps_unit_scale() -> None:
    source = make_track()
    target = source @ make_rotation(-0.4).T + np.array([1.0, 2.0, 3.0])
    scale, rotation, translation = align.umeyama(source, target, with_scale=False)
    assert scale == 1.0
    assert source @ rotation.T + translation == pytest.approx(target, abs=1e-9)


def test_umeyama_rejects_bad_input() -> None:
    track = make_track()
    with pytest.raises(ValueError, match="cùng dạng"):
        align.umeyama(track, track[:-1])
    with pytest.raises(ValueError, match="ít nhất 3"):
        align.umeyama(track[:2], track[:2])
    with pytest.raises(ValueError, match="trùng nhau"):
        align.umeyama(np.zeros((5, 3)), track[:5])


def test_ate_is_zero_for_a_similarity_and_measures_noise() -> None:
    truth = make_track()
    estimated = (truth - [3.0, -2.0, 1.0]) @ make_rotation(-0.7).T / 4.2
    rmse, _ = align.ate_rmse(estimated, truth)
    assert rmse == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(1)
    noisy = estimated + rng.normal(scale=0.01, size=estimated.shape)
    rmse_noisy, _ = align.ate_rmse(noisy, truth)
    assert 0.0 < rmse_noisy < 0.2  # nhiễu 1 cm ở hệ SLAM, nhân tỉ lệ 4,2 thì cỡ vài cm


def test_read_tum_and_match_by_time(tmp_path: Path) -> None:
    path = tmp_path / "traj.txt"
    path.write_text("# comment\n0.0 1 2 3 0 0 0 1\n\n0.5 4 5 6 0 0 0 1\n1.0 7 8 9 0 0 0 1\n")
    stamps, poses = align.read_tum(path)
    assert stamps.tolist() == [0.0, 0.5, 1.0] and poses.shape == (3, 7)
    assert align.positions(poses, [1]).tolist() == [[4.0, 5.0, 6.0]]
    reference = np.array([0.0, 0.1, 0.5001, 2.0])
    assert align.match_by_time(stamps, reference, tolerance=1e-3) == [(0, 0), (1, 2)]
    path.write_text("0.0 1 2 3\n")
    with pytest.raises(ValueError, match="8 số"):
        align.read_tum(path)


def test_pose_to_matrix_quaternion_order() -> None:
    quarter_turn_z = [1.0, 2.0, 3.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    matrix = align.pose_to_matrix(np.array(quarter_turn_z))
    assert matrix[:3, :3] @ [1.0, 0.0, 0.0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert matrix[:3, 3].tolist() == [1.0, 2.0, 3.0]
    # qw đứng đầu: cùng phép quay khi truyền đúng cờ
    wxyz = [1.0, 2.0, 3.0, math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    assert align.pose_to_matrix(np.array(wxyz), quaternion_last=False) == pytest.approx(matrix)
