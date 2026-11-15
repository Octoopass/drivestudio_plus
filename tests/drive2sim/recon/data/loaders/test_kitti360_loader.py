"""Test loader DriveStudio cho cảnh KITTI-360 (``drive2sim/recon/data/loaders/kitti360.py``).

Dựng một cảnh giả rất nhỏ theo khung KITTI của converter KITTI-360 (một camera pinhole, một camera
fisheye, hai khung, pose đơn vị) rồi kiểm tra:
- mô hình ``fisheye`` và ``radial_coeffs`` tới được ``cam_infos``, intrinsics được nhân theo kích
  thước nạp, còn hệ số méo không đổi;
- pixel ở điểm chính cho tia trùng trục quang, pixel lệch trục có góc tia đúng như
  ``cv2.fisheye`` tính; pixel có góc lệch quá ``MEI_FIT_MAX_ANGLE`` bị loại khỏi loss;
- phép chiếu điểm world theo mô hình fisheye;
- camera pinhole đi đúng đường cũ (tia pinhole của ``CameraData.get_image``);
- ảnh ghép nhiều camera của KITTI-360.

Cần PyTorch, PyTorch3D, OpenCV (môi trường GPU của DriveStudio, chạy được trên CPU); môi trường
nhẹ ``.venv`` không có nên test bị bỏ qua ở đó.
"""

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from drive2sim.core.camera import MEI_FIT_MAX_ANGLE  # noqa: E402
from drive2sim.recon.data.base.pixel_source import get_rays  # noqa: E402
from drive2sim.recon.data.loaders.kitti import KITTICameraData, KITTIPixelSource  # noqa: E402
from drive2sim.recon.data.loaders.kitti360 import KITTI360PixelSource  # noqa: E402
from drive2sim.recon.utils.visualization import get_layout  # noqa: E402
from tests.drive2sim.samples import MANIFEST, NUMPY_COEFFS_02  # noqa: E402

PINHOLE_SIZE = (32, 16)  # rộng, cao
FISHEYE_SIZE = (40, 40)
PINHOLE_INTRINSICS = [20.0, 20.0, 16.0, 8.0, 0, 0, 0, 0, 0]
# Ảnh fisheye nạp ở một nửa kích thước (20x20): fx = 6, điểm chính (10.5, 10.5) là tâm
# pixel (10, 10).
FISHEYE_INTRINSICS = [12.0, 12.0, 21.0, 21.0, *NUMPY_COEFFS_02]


def make_scene(root: Path) -> Path:
    """Ghi cảnh giả: camera 0 ``image_00`` pinhole, camera 1 ``image_02`` fisheye, hai khung."""
    scene = root / "000"
    for name in ("images", "ego_pose", "extrinsics", "intrinsics"):
        (scene / name).mkdir(parents=True)
    manifest = {
        **MANIFEST,
        "scene_id": "kitti360_2013_05_28_drive_0003_sync",
        "source": {**MANIFEST["source"], "dataset": "kitti360", "converter": "kitti360"},
        "track": "fisheye",
        "cameras": [
            {
                "camera_id": "image_00",
                "model": "opencv_pinhole",
                "resolution": list(PINHOLE_SIZE),
                "shutter": "GLOBAL",
            },
            {
                "camera_id": "image_02",
                "model": "opencv_fisheye",
                "resolution": list(FISHEYE_SIZE),
                "shutter": "GLOBAL",
            },
        ],
        "timeline": {"reference_sensor": "image_00", "num_frames": 2},
    }
    (scene / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    for index, size in enumerate((PINHOLE_SIZE, FISHEYE_SIZE)):
        np.savetxt(scene / "extrinsics" / f"{index}.txt", np.eye(4))
        for frame in range(2):
            pixels = np.full((size[1], size[0], 3), 40 * (frame + 1), dtype=np.uint8)
            Image.fromarray(pixels).save(scene / "images" / f"{frame:03d}_{index}.png")
    for frame in range(2):
        np.savetxt(scene / "ego_pose" / f"{frame:03d}.txt", np.eye(4))
    np.savetxt(scene / "intrinsics" / "0.txt", [PINHOLE_INTRINSICS])
    np.savetxt(scene / "intrinsics" / "1.txt", [FISHEYE_INTRINSICS])
    return scene


def make_source(scene: Path, undistort: bool = False) -> KITTI360PixelSource:
    """Nạp nguồn ảnh của cảnh giả trên CPU: camera pinhole nạp nguyên, fisheye nạp một nửa."""
    config = OmegaConf.create(
        {
            "cameras": [0, 1],
            "downscale_when_loading": [1, 2],
            "downscale": 1,
            "undistort": undistort,
            "test_image_stride": 0,
            "load_sky_mask": False,
            "load_dynamic_mask": False,
            "load_objects": False,
            "load_smpl": False,
            "sampler": {"buffer_downscale": 1, "buffer_ratio": 0.0, "start_enhance_weight": 1},
        }
    )
    return KITTI360PixelSource("kitti360", config, str(scene), 0, 2)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> KITTI360PixelSource:
    return make_source(make_scene(tmp_path_factory.mktemp("kitti360")))


def test_fisheye_model_and_radial_coeffs_reach_camera_infos(source: KITTI360PixelSource) -> None:
    camera = source.camera_data[1]
    assert camera.cam_name == "image_02"
    assert (camera.HEIGHT, camera.WIDTH) == (20, 20)
    image, infos = camera.get_image(0)
    assert infos["camera_model"] == "fisheye"
    assert torch.allclose(infos["radial_coeffs"], torch.tensor(NUMPY_COEFFS_02).float())
    # fx, fy, cx, cy nhân theo kích thước nạp; k1..k4 không đổi
    expected = torch.tensor([[6.0, 0, 10.5], [0, 6.0, 10.5], [0, 0, 1]])
    assert torch.allclose(infos["intrinsics"], expected)
    assert image["viewdirs"].shape == (20, 20, 3)
    assert torch.allclose(image["direction_norm"], torch.ones(20, 20, 1))


def test_principal_point_ray_is_optical_axis(source: KITTI360PixelSource) -> None:
    image, _ = source.camera_data[1].get_image(0)
    assert torch.allclose(image["viewdirs"][10, 10], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_off_axis_ray_angle_matches_opencv_fisheye(source: KITTI360PixelSource) -> None:
    image, infos = source.camera_data[1].get_image(0)
    K = infos["intrinsics"].double().numpy()
    D = np.array(NUMPY_COEFFS_02, dtype=np.float64)
    for row, col in ((10, 14), (13, 7), (4, 15)):
        # tâm pixel (col, row) nằm ở toạ độ (col + 0.5, row + 0.5)
        point = np.array([[[col + 0.5, row + 0.5]]], dtype=np.float64)
        undistorted = cv2.fisheye.undistortPoints(point, K, D)[0, 0]
        expected = math.atan(float(np.linalg.norm(undistorted)))
        ray = image["viewdirs"][row, col].double()
        assert math.acos(float(ray[2])) == pytest.approx(expected, abs=1e-5)
        # hướng của tia trên mặt phẳng ảnh cùng hướng với điểm đã khử méo
        direction = ray[:2].numpy() / np.linalg.norm(ray[:2].numpy())
        assert np.allclose(direction, undistorted / np.linalg.norm(undistorted), atol=1e-5)


def test_pixels_beyond_fit_angle_are_excluded_from_loss(source: KITTI360PixelSource) -> None:
    image, _ = source.camera_data[1].get_image(0)
    angle = torch.acos(image["viewdirs"][..., 2].clamp(-1, 1))
    valid = image["valid_masks"]
    assert valid[10, 10] and not valid[0, 0]
    assert bool((angle[valid] <= MEI_FIT_MAX_ANGLE + 1e-6).all())
    assert torch.equal(image["egocar_masks"], (~valid).float())


def test_fisheye_project_points(source: KITTI360PixelSource) -> None:
    camera = source.camera_data[1]
    off_axis = math.radians(30)
    beyond = MEI_FIT_MAX_ANGLE + 0.05
    points = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [5 * math.sin(off_axis), 0.0, 5 * math.cos(off_axis)],
            [math.sin(beyond), 0.0, math.cos(beyond)],
            [0.0, 0.0, -5.0],
        ]
    )
    pixels, depth, valid = camera.project_points(points, 0)
    assert valid.tolist() == [True, True, False, False]
    assert torch.allclose(pixels[0], torch.tensor([10.5, 10.5]))
    assert depth[0] == pytest.approx(5.0)
    # pixel của điểm lệch 30 độ quay ngược lại đúng tia lệch 30 độ
    expected, _ = cv2.fisheye.projectPoints(
        points[1:2].double().numpy()[None],
        np.zeros(3),
        np.zeros(3),
        camera.intrinsics[0].double().numpy(),
        np.array(NUMPY_COEFFS_02, dtype=np.float64),
    )
    assert np.allclose(pixels[1].numpy(), expected[0, 0], atol=1e-4)


def test_pinhole_camera_keeps_kitti_path(source: KITTI360PixelSource) -> None:
    camera = source.camera_data[0]
    assert camera.cam_name == "image_00" and camera.camera_model == "pinhole"
    assert camera.radial_coeffs is None
    image, infos = camera.get_image(0)
    assert "camera_model" not in infos and "radial_coeffs" not in infos
    assert "valid_masks" not in image and "egocar_masks" not in image
    K = torch.tensor([[20.0, 0, 16.0], [0, 20.0, 8.0], [0, 0, 1]])
    assert torch.allclose(infos["intrinsics"], K)
    x, y = torch.meshgrid(torch.arange(32), torch.arange(16), indexing="xy")
    _, viewdirs, _ = get_rays(x.flatten(), y.flatten(), torch.eye(4), K)
    assert torch.allclose(image["viewdirs"], viewdirs.reshape(16, 32, 3))
    assert float(image["pixels"][0, 0, 0]) == pytest.approx(40 / 255)


def test_kitti_loader_still_uses_kitti_cameras() -> None:
    assert KITTIPixelSource.CAMERA_DATA_CLASS is KITTICameraData
    assert KITTIPixelSource.SMPL_CAMERA_LIST == [0, 1]


def test_fisheye_refuses_undistort(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="undistort"):
        make_source(make_scene(tmp_path), undistort=True)


def test_layout_tiles_pinhole_and_fisheye_cameras() -> None:
    layout = get_layout("kitti360")
    pinhole = np.ones((376, 1408, 3), np.float32)
    fisheye = np.ones((1400, 1400, 3), np.float32)
    tiled = layout([pinhole, fisheye, fisheye], ["image_00", "image_02", "image_03"])
    assert tiled.shape == (376 + 1408, 2 * 1408, 3)
