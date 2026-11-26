"""Kiểm tra ``external/mcgs/calib.py``: thứ tự thư mục ảnh và ``calib.yml`` cho MCGS-SLAM.

- ``se3_vector`` đổi ma trận 4×4 sang bảy số ``tx ty tz qx qy qz qw`` của lietorch (``qw`` >= 0).
- ``rig_layout`` xếp camera như mẫu Waymo của MCGS-SLAM (camera đối tác stereo lặp ở cuối).
- ``calib_document`` dựng ``baseline`` và ``T_cami_cam0`` đúng quy ước đọc được trong mã MCGS.
- ``check_cameras`` từ chối camera không phải pinhole hoặc rig dưới hai camera.
- ``slam_frames`` bỏ khung test khỏi đầu vào của SLAM.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from drive2sim.external.mcgs.calib import (  # noqa: E402
    calib_document,
    check_cameras,
    relative_to_reference,
    rig_layout,
    se3_vector,
    slam_frames,
)


def offset(x: float, y: float = 0.0) -> "np.ndarray":
    transform = np.eye(4)
    transform[0, 3], transform[1, 3] = x, y
    return transform


def rotation_y(angle: float) -> "np.ndarray":
    cos, sin = math.cos(angle), math.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]]
    return transform


def test_kitti_stereo_baseline() -> None:
    rig = {"image_00": offset(0.0), "image_01": offset(0.59)}  # image_01 lệch phải 0,59 m
    layout = rig_layout(["image_00", "image_01"], rig, "image_00")
    intrinsics = {
        "image_00": [552.6, 552.6, 682.0, 238.8],
        "image_01": [552.6, 552.6, 682.0, 238.8],
    }
    doc = calib_document(layout, intrinsics, rig)
    assert layout == ["image_00", "image_01"]
    assert doc["baseline"] == pytest.approx([-0.59, 0, 0, 0, 0, 0, 1])
    assert doc["T_cami_cam0"] == [[0, 0, 0, 0, 0, 0, 1]] and doc["timescale"] == 1
    assert doc["camera"] == "pinhole"
    assert doc["intrinsic"][0] == [552.6, 552.6, 682.0, 238.8]  # không hệ số méo → 4 cột


def test_three_camera_layout_matches_upstream_sample() -> None:
    rig = {"front": offset(0.0), "front_left": offset(-1.0), "front_right": offset(0.98)}
    cameras = ["front", "front_left", "front_right"]
    layout = rig_layout(cameras, rig, "front", partner="front_right")
    assert layout == ["front", "front_right", "front_left", "front_right"]
    doc = calib_document(layout, {c: [1, 1, 0, 0] for c in rig}, rig)
    assert doc["baseline"] == doc["T_cami_cam0"][-1]
    assert len(doc["intrinsic"]) == 4 and len(doc["T_cami_cam0"]) == 3
    assert doc["T_cami_cam0"][0] == [0, 0, 0, 0, 0, 0, 1]
    assert doc["T_cami_cam0"][1][0] == pytest.approx(1.0)  # front_left: tx +1,0


def test_default_partner_is_the_nearest_camera() -> None:
    rig = {"a": offset(0.0), "far": offset(3.0), "near": offset(0.5, 0.5), "mid": offset(1.0)}
    layout = rig_layout(["a", "far", "near", "mid"], rig, "a")
    assert layout == ["a", "near", "far", "mid", "near"]
    tie = {"a": offset(0.0), "b": offset(1.0), "c": offset(-1.0)}
    assert rig_layout(["a", "b", "c"], tie, "a")[1] == "b"  # hoà thì theo thứ tự cameras


def test_rig_layout_rejects_unknown_reference_or_partner() -> None:
    rig = {"a": offset(0.0), "b": offset(1.0)}
    with pytest.raises(ValueError, match="reference"):
        rig_layout(["a", "b"], rig, "zzz")
    with pytest.raises(ValueError, match="stereo_partner"):
        rig_layout(["a", "b"], rig, "a", partner="a")


def test_relative_to_reference_maps_reference_points_into_the_camera() -> None:
    rig = {"a": offset(0.0), "b": offset(0.59)}
    transform = relative_to_reference(rig, "a", "b")
    assert (transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3] == pytest.approx([-0.59, 0, 0])


def test_se3_vector_rotation_about_y() -> None:
    # quay 45° quanh y → qy = sin(22,5°), qw = cos(22,5°)
    transform = rotation_y(math.radians(45.0))
    transform[:3, 3] = [1.0, 2.0, 3.0]
    expected = [
        1.0,
        2.0,
        3.0,
        0.0,
        math.sin(math.radians(22.5)),
        0.0,
        math.cos(math.radians(22.5)),
    ]
    assert se3_vector(transform) == pytest.approx(expected)


def test_se3_vector_keeps_qw_non_negative() -> None:
    # quay 270° quanh y: q = (0, sin 135°, 0, cos 135°) có qw < 0 → đổi dấu cả bốn số
    vector = se3_vector(rotation_y(math.radians(270.0)))
    sin45, cos45 = math.sin(math.radians(45.0)), math.cos(math.radians(45.0))
    assert vector[6] >= 0
    assert vector[3:] == pytest.approx([0.0, -sin45, 0.0, cos45])


def test_calib_document_appends_distortion_only_when_nonzero() -> None:
    rig = {"a": offset(0.0), "b": offset(0.5)}
    plain = [10.0, 10.0, 5.0, 5.0, 0, 0, 0, 0, 0]
    distorted = [10.0, 10.0, 5.0, 5.0, -0.1, 0.02, 0.001, 0.0, 0.003]
    doc = calib_document(["a", "b"], {"a": plain, "b": distorted}, rig)
    assert doc["intrinsic"] == [[10.0, 10.0, 5.0, 5.0], distorted]


@pytest.mark.parametrize(
    "models",
    [
        {"image_00": "opencv_pinhole"},
        {"image_00": "opencv_pinhole", "image_02": "opencv_fisheye"},
    ],
)
def test_check_cameras_rejects(models: dict) -> None:
    with pytest.raises(ValueError):
        check_cameras(models)


def test_check_cameras_accepts_two_pinhole_cameras() -> None:
    check_cameras({"image_00": "opencv_pinhole", "image_01": "ideal_pinhole"})


def test_slam_frames_drops_test_frames_only_when_held_out() -> None:
    assert slam_frames((0, 9), {3, 8}, True) == [0, 1, 2, 4, 5, 6, 7, 9]
    assert slam_frames((0, 4), {3}, False) == [0, 1, 2, 3, 4]
