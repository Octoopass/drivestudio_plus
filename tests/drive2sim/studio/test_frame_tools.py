"""Test ảnh toàn cảnh và LiDAR chồng ảnh (``studio/api/frame_tools.py``) trên cảnh giả."""

from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from drive2sim.studio.api import frame_tools  # noqa: E402

# camera OpenCV nhìn về trục tới (x) của xe: z camera → x xe, x camera → -y xe, y camera → -z xe
FRONT = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def make_scene(root: Path, dataset: str = "kitti360", model: str = "opencv_pinhole") -> object:
    for folder in ("images", "intrinsics", "extrinsics", "lidar"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    np.savetxt(root / "extrinsics" / "0.txt", FRONT)
    values = [100, 100, 100, 50] + ([0] * 5 if model == "opencv_pinhole" else [0] * 4)
    np.savetxt(root / "intrinsics" / "0.txt", np.array(values, dtype=np.float64))
    cv2.imwrite(str(root / "images" / "000_0.png"), np.full((100, 200, 3), 200, np.uint8))
    camera = SimpleNamespace(camera_id="cam0", model=model, resolution=(200, 100), max_angle=None)
    return SimpleNamespace(cameras=[camera], source=SimpleNamespace(dataset=dataset))


def decode(png: bytes) -> "np.ndarray":
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def test_lidar_point_ahead_lands_on_the_principal_point(tmp_path: Path) -> None:
    manifest = make_scene(tmp_path)
    np.array([[10, 0, 0, 1], [-10, 0, 0, 1]], np.float32).tofile(tmp_path / "lidar" / "000.bin")
    png, points = frame_tools.lidar_overlay(tmp_path, manifest, 0, "cam0")
    image = decode(png)
    assert points == 1  # điểm phía sau xe không vẽ
    assert not np.array_equal(image[50, 100], [200, 200, 200])  # tâm ảnh (cx=100, cy=50) có điểm
    with pytest.raises(frame_tools.FrameToolError, match="cảnh không có camera"):
        frame_tools.lidar_overlay(tmp_path, manifest, 0, "nope")


def test_lidar_refuses_other_layouts(tmp_path: Path) -> None:
    manifest = make_scene(tmp_path, dataset="nuscenes")
    with pytest.raises(frame_tools.FrameToolError, match="dạng khác"):
        frame_tools.lidar_overlay(tmp_path, manifest, 0, "cam0")


@pytest.mark.parametrize("model", ["opencv_pinhole", "opencv_fisheye"])
def test_panorama_puts_the_front_camera_in_the_middle(tmp_path: Path, model: str) -> None:
    manifest = make_scene(tmp_path, model=model)
    png, coverage = frame_tools.panorama(tmp_path, manifest, 0, width=512)
    image = decode(png)
    assert image.shape == (256, 512, 3)
    assert image[128, 256].sum() > 0  # giữa ảnh: phía trước xe
    assert image[128, 2].sum() == 0  # mép ảnh: phía sau xe, không camera nào
    assert 0 < coverage < 0.5


def test_needs_fixed_calibration(tmp_path: Path) -> None:
    manifest = make_scene(tmp_path)
    (tmp_path / "extrinsics" / "0.txt").unlink()
    with pytest.raises(frame_tools.FrameToolError, match="cố định"):
        frame_tools.panorama(tmp_path, manifest, 0, width=512)
