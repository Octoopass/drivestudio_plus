"""Sơ đồ camera và quỹ đạo xe (``recon/tools/scene_rig.py``)."""

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from drive2sim.recon.tools.scene_rig import _angles, topdown_frame  # noqa: E402


def _pose(position: "np.ndarray", heading: float, mount_yaw: float) -> "np.ndarray":
    """Pose camera OpenCV (z nhìn, y xuống) gắn trên xe.

    Xe ở ``position``, hướng ``heading`` (rad) trong world z lên; camera lệch ``mount_yaw`` (rad,
    dương sang trái) so với đầu xe.
    """
    yaw = heading + mount_yaw
    look = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    right = np.cross(down, look)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, down, look, position
    return pose


def _drive(mount_yaws: dict[str, float], frames: int = 60) -> dict[str, "np.ndarray"]:
    """Xe chạy theo cung tròn bán kính 30 m, rẽ trái 90 độ, bắt đầu hướng về +world_y."""
    poses = {name: [] for name in mount_yaws}
    for i in range(frames):
        angle = (math.pi / 2) * i / (frames - 1)
        centre = np.array([30 * (1 - math.cos(angle)) * -1, 30 * math.sin(angle), 0.0])
        heading = math.pi / 2 + angle
        for name, mount in mount_yaws.items():
            poses[name].append(_pose(centre, heading, mount))
    return {name: np.stack(p) for name, p in poses.items()}


def test_heading_comes_from_motion_even_without_a_front_camera() -> None:
    # Chỉ hai camera nhìn ngang (như cảnh KITTI-360 chỉ có hai camera mắt cá), xe đang vào cua.
    c2ws = _drive({"left": math.pi / 2, "right": -math.pi / 2})
    world_to_rig, source = topdown_frame(c2ws, "left")
    assert source == "motion"
    rotation = world_to_rig[:3, :3]
    yaw_left, _ = _angles(rotation @ c2ws["left"][0, :3, 2])
    yaw_right, _ = _angles(rotation @ c2ws["right"][0, :3, 2])
    # Trung bình trên cả cung: lệch nhỏ (cung đều nên dây cung lệch nửa bước góc).
    assert yaw_left == pytest.approx(90, abs=2)
    assert yaw_right == pytest.approx(-90, abs=2)


def test_front_camera_points_forward_and_up_is_world_up() -> None:
    c2ws = _drive({"front": 0.0, "back": math.pi})
    world_to_rig, _ = topdown_frame(c2ws, "front")
    rotation = world_to_rig[:3, :3]
    assert _angles(rotation @ c2ws["front"][0, :3, 2])[0] == pytest.approx(0, abs=2)
    assert abs(_angles(rotation @ c2ws["back"][0, :3, 2])[0]) == pytest.approx(180, abs=2)
    assert np.allclose(rotation[2], [0, 0, 1], atol=1e-6)


def test_still_vehicle_uses_the_reference_camera() -> None:
    pose = _pose(np.zeros(3), 0.3, 0.0)
    c2ws = {"front": np.stack([pose] * 5)}
    world_to_rig, source = topdown_frame(c2ws, "front")
    assert source == "camera"
    assert _angles(world_to_rig[:3, :3] @ pose[:3, 2])[0] == pytest.approx(0, abs=1e-6)


def test_compute_rig_reads_poses_through_the_dataset_loader(tmp_path) -> None:  # noqa: ANN001
    """Cảnh KITTI-360 giả (một camera pinhole, một mắt cá) đi qua loader thật ở chế độ viewer."""
    pytest.importorskip("cv2")
    from drive2sim.recon.tools.scene_rig import compute_rig
    from tests.drive2sim.recon.data.loaders.test_kitti360_loader import make_scene

    rig = compute_rig(make_scene(tmp_path))
    cameras = {c["camera_id"]: c for c in rig["cameras"]}
    assert set(cameras) == {"image_00", "image_02"}
    # cảnh giả đứng yên: hướng tới theo camera tham chiếu (image_00 là reference_sensor)
    assert rig["forward_source"] == "camera"
    assert cameras["image_00"]["yaw_deg"] == pytest.approx(0, abs=1e-6)
    # pinhole fx 20 trên ảnh rộng 32: 2·atan(16/20)
    expected = math.degrees(2 * math.atan(16 / 20))
    assert cameras["image_00"]["hfov_deg"] == pytest.approx(expected, abs=0.5)
    # mắt cá: góc nhìn trong giới hạn che của loader, rộng hơn pinhole
    assert cameras["image_00"]["hfov_deg"] < cameras["image_02"]["hfov_deg"] <= 360
    assert len(rig["trajectory"]) == 2
