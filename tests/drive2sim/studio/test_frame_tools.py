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


def test_depth_colours_match_the_loader_scale() -> None:
    """Cùng thang màu với depth_visualizer (video độ sâu, ảnh chồng của tab Kiểm tra)."""
    pytest.importorskip("torch")
    from drive2sim.recon.utils.visualization import depth_visualizer

    depth = np.array([4.0, 8.0, 15.0, 30.0, 60.0, 120.0], np.float32)
    expected = depth_visualizer(depth[None], np.ones((1, len(depth)), np.float32))[0] * 255
    ours = frame_tools.depth_colors(depth)[:, ::-1]  # BGR -> RGB
    assert np.abs(ours.astype(int) - expected.round().astype(int)).max() <= 16


def test_points_on_the_ego_car_are_grey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = make_scene(tmp_path)
    masks = tmp_path / "ego_masks"
    (masks / "kitti360").mkdir(parents=True)
    cv2.imwrite(str(masks / "kitti360" / "cam0.png"), np.full((100, 200), 255, np.uint8))
    monkeypatch.setattr("drive2sim.paths.EGO_MASKS_DIR", masks)
    np.array([[10, 0, 0, 1]], np.float32).tofile(tmp_path / "lidar" / "000.bin")
    image = decode(frame_tools.lidar_overlay(tmp_path, manifest, 0, "cam0")[0])
    assert np.array_equal(image[50, 100], frame_tools.EGO_GREY)


def test_fast_projection_matches_the_kitti360_loader(tmp_path: Path) -> None:
    """Phép chiếu nhanh và phép chiếu của loader (dữ liệu train) cho cùng pixel."""
    pytest.importorskip("torch")
    import math

    import torch

    from drive2sim.core.schema import Manifest
    from tests.drive2sim.recon.data.loaders.test_kitti360_loader import make_scene as kitti_scene
    from tests.drive2sim.recon.data.loaders.test_kitti360_loader import make_source

    scene = kitti_scene(tmp_path / "k")
    source = make_source(scene)
    manifest = Manifest.model_validate(
        __import__("yaml").safe_load((scene / "manifest.yaml").read_text())
    )
    for position, cam in source.camera_data.items():
        index = [c.camera_id for c in manifest.cameras].index(str(cam.cam_name))
        camera = frame_tools._camera(scene, manifest, index)
        angles = [math.radians(a) for a in (0, 20, 40, 60, 75)]
        points = np.array(
            [
                [5 * math.sin(a) * math.cos(b), 5 * math.sin(a) * math.sin(b), 5 * math.cos(a)]
                for a in angles
                for b in (0.3, 1.9, 4.0)
            ]
        )
        # điểm trong hệ camera -> hệ xe (extrinsics camera -> xe) cho phép chiếu nhanh
        ego = points @ camera["T"][:3, :3].T + camera["T"][:3, 3]
        uv_fast, ok_fast = frame_tools._project(
            (ego - camera["T"][:3, 3]) @ camera["T"][:3, :3], camera
        )
        pixels, _, ok_loader = cam.project_points(torch.from_numpy(ego).float(), 0)
        scale = np.array([cam.WIDTH / camera["size"][0], cam.HEIGHT / camera["size"][1]])
        both = ok_fast & ok_loader.numpy()
        assert both.sum() >= 3, (position, both)
        assert np.allclose(uv_fast[both] * scale, pixels.numpy()[both], atol=0.05)
