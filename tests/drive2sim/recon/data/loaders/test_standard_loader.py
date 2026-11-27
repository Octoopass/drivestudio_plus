"""Test loader chung (``drive2sim/recon/data/loaders/standard.py``) trên cảnh kiểu KITTI-360.

Dựng một cảnh giả rất nhỏ theo khung KITTI của converter KITTI-360 (một camera pinhole, một camera
fisheye, hai khung, pose đơn vị) rồi kiểm tra:
- mô hình ``fisheye`` và ``radial_coeffs`` tới được ``cam_infos``, intrinsics được nhân theo kích
  thước nạp, còn hệ số méo không đổi;
- pixel ở điểm chính cho tia trùng trục quang, pixel lệch trục có góc tia đúng như
  ``cv2.fisheye`` tính; pixel có góc lệch quá giới hạn của camera bị loại khỏi loss;
- phép chiếu điểm world theo mô hình fisheye;
- camera pinhole đi đúng đường cũ (tia pinhole của ``CameraData.get_image``);
- ảnh ghép nhiều camera của KITTI-360.

Cần PyTorch, PyTorch3D, OpenCV (môi trường GPU của DriveStudio, chạy được trên CPU); môi trường
nhẹ ``.venv`` không có nên test bị bỏ qua ở đó.
"""

import math
from pathlib import Path
from typing import Optional

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
from drive2sim.recon.data.loaders.standard import StandardPixelSource  # noqa: E402
from drive2sim.recon.utils.visualization import get_layout  # noqa: E402
from tests.drive2sim.samples import FISHEYE_COEFFS_02, MANIFEST  # noqa: E402

PINHOLE_SIZE = (32, 16)  # rộng, cao
FISHEYE_SIZE = (40, 40)
PINHOLE_INTRINSICS = [20.0, 20.0, 16.0, 8.0, 0, 0, 0, 0, 0]
# Ảnh fisheye nạp ở một nửa kích thước (20x20): fx = 6, điểm chính (10.5, 10.5) là tâm
# pixel (10, 10).
FISHEYE_INTRINSICS = [12.0, 12.0, 21.0, 21.0, *FISHEYE_COEFFS_02]


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


def make_source(
    scene: Path,
    undistort: bool = False,
    fisheye_max_angle: Optional[float] = None,
    renderer_max_angle: Optional[float] = None,
) -> StandardPixelSource:
    """Nạp nguồn ảnh của cảnh giả trên CPU: camera pinhole nạp nguyên, fisheye nạp một nửa."""
    config = OmegaConf.create(
        {
            "fisheye_max_angle": fisheye_max_angle,
            "fisheye_renderer_max_angle": renderer_max_angle,
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
    return StandardPixelSource("kitti360", config, str(scene), 0, 2)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> StandardPixelSource:
    return make_source(make_scene(tmp_path_factory.mktemp("kitti360")))


def test_fisheye_model_and_radial_coeffs_reach_camera_infos(source: StandardPixelSource) -> None:
    camera = source.camera_data[1]
    assert camera.cam_name == "image_02"
    assert (camera.HEIGHT, camera.WIDTH) == (20, 20)
    image, infos = camera.get_image(0)
    assert infos["camera_model"] == "fisheye"
    assert torch.allclose(infos["radial_coeffs"], torch.tensor(FISHEYE_COEFFS_02).float())
    # fx, fy, cx, cy nhân theo kích thước nạp; k1..k4 không đổi
    expected = torch.tensor([[6.0, 0, 10.5], [0, 6.0, 10.5], [0, 0, 1]])
    assert torch.allclose(infos["intrinsics"], expected)
    assert image["viewdirs"].shape == (20, 20, 3)
    assert torch.allclose(image["direction_norm"], torch.ones(20, 20, 1))


def test_principal_point_ray_is_optical_axis(source: StandardPixelSource) -> None:
    image, _ = source.camera_data[1].get_image(0)
    assert torch.allclose(image["viewdirs"][10, 10], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_off_axis_ray_angle_matches_opencv_fisheye(source: StandardPixelSource) -> None:
    image, infos = source.camera_data[1].get_image(0)
    K = infos["intrinsics"].double().numpy()
    D = np.array(FISHEYE_COEFFS_02, dtype=np.float64)
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


def test_pixels_beyond_fit_angle_are_excluded_from_loss(source: StandardPixelSource) -> None:
    image, _ = source.camera_data[1].get_image(0)
    angle = torch.acos(image["viewdirs"][..., 2].clamp(-1, 1))
    valid = image["valid_masks"]
    assert valid[10, 10] and not valid[0, 0]
    assert bool((angle[valid] <= MEI_FIT_MAX_ANGLE + 1e-6).all())
    assert torch.equal(image["egocar_masks"], (~valid).float())


def test_fisheye_max_angle_from_config_overrides_default(tmp_path: Path) -> None:
    # Tham số thử nghiệm (data.pixel_source.fisheye_max_angle) đổi được ngưỡng che mặc định.
    scene = make_scene(tmp_path)
    default = make_source(scene).camera_data[1].get_image(0)[0]["valid_masks"]
    narrow = make_source(scene, fisheye_max_angle=1.2).camera_data[1].get_image(0)[0]
    angle = torch.acos(narrow["viewdirs"][..., 2].clamp(-1, 1))
    assert int(narrow["valid_masks"].sum()) < int(default.sum())
    assert bool((angle[narrow["valid_masks"]] <= 1.2 + 1e-6).all())


def test_fisheye_max_angle_from_manifest(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    manifest = yaml.safe_load((scene / "manifest.yaml").read_text())
    manifest["cameras"][1]["max_angle"] = 1.2
    (scene / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    image = make_source(scene).camera_data[1].get_image(0)[0]
    angle = torch.acos(image["viewdirs"][..., 2].clamp(-1, 1))
    assert bool((angle[image["valid_masks"]] <= 1.2 + 1e-6).all())


def test_mask_angle_is_smaller_of_lens_and_renderer_limits(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    manifest = yaml.safe_load((scene / "manifest.yaml").read_text())
    manifest["cameras"][1]["max_angle"] = 1.3
    (scene / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    lens_only = make_source(scene).camera_data[1]
    assert lens_only.max_angle == 1.3  # 3DGEER: chỉ giới hạn của ống kính, ảnh
    tighter = make_source(scene, renderer_max_angle=1.2).camera_data[1]
    assert tighter.max_angle == 1.2  # 3DGUT nhỏ hơn: dùng giới hạn renderer
    looser = make_source(scene, renderer_max_angle=1.5).camera_data[1]
    assert looser.max_angle == 1.3
    forced = make_source(scene, fisheye_max_angle=1.4, renderer_max_angle=1.2).camera_data[1]
    assert forced.max_angle == 1.4  # tham số thử nghiệm đè lên cả hai


def test_scene_without_manifest_limit_keeps_old_limit(tmp_path: Path) -> None:
    # Cảnh ingest trước khi manifest có max_angle (kitti360_drive0003): không có calib Mei để tính,
    # giữ giới hạn cũ 1,45 rad cho an toàn.
    source = make_source(make_scene(tmp_path))
    assert source.camera_data[1].max_angle == MEI_FIT_MAX_ANGLE


def test_dark_border_outside_lens_circle_is_masked(tmp_path: Path) -> None:
    # Ảnh đen ngoài hình tròn bán kính 13 px quanh điểm chính (ảnh gốc 40x40), ở mọi khung, như
    # vùng ống kính không chiếu tới; trong hình tròn có một chấm tối (cảnh tối thật) phải giữ.
    scene = make_scene(tmp_path)
    v, u = np.mgrid[0:40, 0:40]
    outside = np.hypot(u + 0.5 - 21.0, v + 0.5 - 21.0) > 13
    for path in (scene / "images").glob("*_1.png"):
        image = np.asarray(Image.open(path).convert("RGB")).copy()
        image[outside] = 0
        image[20:22, 20:22] = 0
        Image.fromarray(image).save(path)
    camera = make_source(scene, fisheye_max_angle=3.0).camera_data[1]
    assert camera.dark_angle is not None and camera.dark_angle < 3.0
    assert camera.max_angle == camera.dark_angle
    image = camera.get_image(0)[0]
    valid, rgb = image["valid_masks"], image["pixels"]
    dark = rgb.mean(-1) < 0.05
    assert bool(valid[10, 10])  # chấm tối ở giữa vẫn được dùng
    # mọi pixel còn hợp lệ ngoài chấm giữa đều sáng: vùng đen ngoài vòng đã bị che
    border = torch.ones_like(valid)
    border[9:12, 9:12] = False
    assert not bool((valid & dark & border).any())


def test_uniform_images_have_no_dark_border(source: StandardPixelSource) -> None:
    assert source.camera_data[1].dark_angle == math.pi


def test_fisheye_project_points(source: StandardPixelSource) -> None:
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
        np.array(FISHEYE_COEFFS_02, dtype=np.float64),
    )
    assert np.allclose(pixels[1].numpy(), expected[0, 0], atol=1e-4)


def test_pinhole_camera_keeps_kitti_path(source: StandardPixelSource) -> None:
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


@pytest.mark.parametrize("camera_name", ["image_02", "image_03"])
def test_ego_mask_follows_manifest_name_after_camera_reindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, camera_name: str
) -> None:
    from drive2sim import paths

    scene = make_scene(tmp_path / "scene")
    manifest = yaml.safe_load((scene / "manifest.yaml").read_text())
    manifest["cameras"][1]["camera_id"] = camera_name
    (scene / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    mask_dir = tmp_path / "assets" / "kitti360"
    mask_dir.mkdir(parents=True)
    for name, row in (("image_02", 12), ("image_03", 22)):
        pixels = np.zeros((40, 40), dtype=np.uint8)
        pixels[row : row + 4, 18:22] = 255
        Image.fromarray(pixels).save(mask_dir / f"{name}.png")
    monkeypatch.setattr(paths, "EGO_MASKS_DIR", mask_dir.parent)
    camera = make_source(scene).camera_data[1]
    expected = torch.from_numpy(
        np.array(
            Image.open(mask_dir / f"{camera_name}.png").resize((20, 20), Image.Resampling.NEAREST)
        )
        > 0
    )
    assert torch.equal(camera.egocar_mask.bool(), expected)
    for factor in (1.0, 0.5):
        camera.set_downscale_factor(factor)
        image, infos = camera.get_image(0)
        height, width = int(infos["height"]), int(infos["width"])
        _, angular_valid = camera.camera_rays(infos["intrinsics"], height, width)
        ego = torch.nn.functional.interpolate(
            expected[None, None].float(), size=(height, width), mode="nearest"
        )[0, 0].bool()
        assert bool((ego & angular_valid).any())
        assert torch.equal(image["valid_masks"], angular_valid & ~ego)
        assert torch.equal(image["egocar_masks"].bool(), ~image["valid_masks"])

    camera.set_downscale_factor(1.0)
    # Ray through the centre of the excluded patch: LiDAR must not use it for colour/depth.
    row = 6 if camera_name == "image_02" else 11
    rays, _ = camera.camera_rays(camera.intrinsics[0], 20, 20)
    _, _, visible = camera.project_points(rays[row, 9][None] * 5, 0)
    assert not bool(visible[0])


def test_packaged_masks_keep_road_and_exclude_hardware() -> None:
    from drive2sim import paths

    for camera, gps_x, button_x in (("image_02", 100, 325), ("image_03", 1299, 1074)):
        with Image.open(paths.EGO_MASKS_DIR / "kitti360" / f"{camera}.png") as mask:
            pixels = np.array(mask)
        assert pixels.shape == (1400, 1400)
        assert set(np.unique(pixels)) == {0, 255}
        assert pixels[930, gps_x] == 255
        assert pixels[1080, button_x] == 255
        assert pixels[1250, 700] == 255
        assert pixels[1050, 700] == 0  # road, even when the vehicle casts a shadow here
        assert pixels[410, 700] == 0  # fixed sun streak is not vehicle hardware


def test_masked_rgb_does_not_contribute_to_training_loss(source: StandardPixelSource) -> None:
    from types import SimpleNamespace

    from drive2sim.recon.trainers.base import BasicTrainer

    image, infos = source.camera_data[1].get_image(0)
    image["sky_masks"] = torch.zeros_like(image["valid_masks"]).float()
    trainer = SimpleNamespace(
        losses_dict=OmegaConf.create({"rgb": {"w": 0.8}, "ssim": {"w": 0.2}}),
        ssim=lambda a, b: 1 - (a - b).abs().mean(),
        sky_opacity_loss_fn=None,
        use_depth_supervision=False,
        gaussian_classes={},
        models={},
    )
    rgb = image["pixels"].clone()
    rgb[~image["valid_masks"]] = 1
    outputs = {"rgb": rgb, "opacity": torch.ones_like(image["valid_masks"]).float()}
    losses = BasicTrainer.compute_losses(trainer, outputs, image, infos)
    assert sum(losses.values()).item() == pytest.approx(0)
    rgb[image["valid_masks"]] += 0.1
    assert sum(BasicTrainer.compute_losses(trainer, outputs, image, infos).values()).item() > 0


def test_evaluation_ignores_rgb_errors_inside_vehicle_mask(
    source: StandardPixelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drive2sim.recon.rendering.videos import render

    camera = source.camera_data[1]
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class Images:
        split = "test"

        def __len__(self) -> int:
            return 1

        def get_image(self, index: int, downscale: float) -> tuple:
            return camera.get_image(index)

    class Trainer:
        changed = False

        def _get_downscale_factor(self) -> int:
            return 1

        def __call__(self, image: dict, camera: dict) -> dict:
            rgb = image["pixels"].clone() + 0.1
            if self.changed:
                rgb[~image["valid_masks"]] = 1
            return {"rgb": rgb}

        def lpips(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Do not download a network: verify the tensors passed to the metric consumer.
            return (prediction - target).abs().mean()

    trainer = Trainer()
    before = render(Images(), trainer, compute_metrics=True)
    trainer.changed = True
    after = render(Images(), trainer, compute_metrics=True)
    for key in ("psnr", "ssim", "lpips"):
        assert before[key] == pytest.approx(after[key])
    image, _ = camera.get_image(0)
    assert after["psnr"] == pytest.approx(
        -20 * math.log10(round((40 / 255 + 0.1) * 255) / 255 - 40 / 255)
    )


def test_vehicle_mask_survives_reversed_camera_order(tmp_path: Path) -> None:
    from drive2sim import paths

    scene = make_scene(tmp_path)
    manifest = yaml.safe_load((scene / "manifest.yaml").read_text())
    manifest["cameras"].reverse()
    (scene / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    for directory, suffix, prefixes in (
        ("images", ".png", ["000_", "001_"]),
        ("intrinsics", ".txt", [""]),
        ("extrinsics", ".txt", [""]),
    ):
        for prefix in prefixes:
            first = scene / directory / f"{prefix}0{suffix}"
            second = scene / directory / f"{prefix}1{suffix}"
            a, b = first.read_bytes(), second.read_bytes()
            first.write_bytes(b)
            second.write_bytes(a)
    source = make_source(scene)
    camera = source.camera_data[0]
    assert camera.cam_name == "image_02"
    with Image.open(paths.EGO_MASKS_DIR / "kitti360" / "image_02.png") as mask:
        expected = np.array(mask.resize((40, 40), Image.Resampling.NEAREST)) > 0
    assert torch.equal(camera.egocar_mask.bool(), torch.from_numpy(expected))
    assert source.camera_data[1].egocar_mask is None


def test_pixel_sources_do_not_share_cameras(tmp_path: Path) -> None:
    first = make_source(make_scene(tmp_path / "a"))
    second = make_source(make_scene(tmp_path / "b"))
    assert first.camera_data is not second.camera_data
    assert first.camera_data[1] is not second.camera_data[1]


def test_lidar_points_follow_the_camera_mask_angle() -> None:
    """Điểm LiDAR được chiếu tới đúng ngưỡng che ``max_angle`` của camera.

    Không truyền ``max_angle`` thì giữ giới hạn 90 độ cũ.
    """
    from drive2sim.recon.utils.geometry import project_world_camera

    K = torch.tensor([[400.0, 0, 700], [0, 400.0, 700], [0, 0, 1]])
    angles = torch.tensor([math.radians(a) for a in (80, 87, 89)])
    points = torch.stack((4 * angles.sin(), torch.zeros(3), 4 * angles.cos()), -1)
    pose = torch.eye(4)
    args = (points, pose, pose, K, 1400, 1400, "fisheye", FISHEYE_COEFFS_02)
    _, depth, valid = project_world_camera(*args, max_angle=math.radians(88))
    assert valid.tolist() == [True, True, False]
    assert torch.allclose(depth, points[:, 2])
    assert project_world_camera(*args)[2].tolist() == [True, True, True]


def test_lidar_points_behind_the_lens_fold_are_rejected(source: StandardPixelSource) -> None:
    """Điểm LiDAR phía sau, quá đỉnh của đa thức ống kính, bị loại.

    Sau đỉnh, bán kính giảm lại: điểm phía sau rơi đúng vào pixel của một điểm trong vùng dùng
    được, nên phải loại theo góc, không theo pixel.
    """
    camera = source.camera_data[1]
    k = FISHEYE_COEFFS_02
    theta = torch.linspace(0, math.pi, 100001, dtype=torch.float64)
    radius = theta * (1 + sum(c * theta ** (2 * i + 2) for i, c in enumerate(k)))
    peak = int(radius.argmax())
    edge = float(radius[theta <= camera.max_angle].max())
    behind = float(theta[peak:][(radius[peak:] - edge * 0.98).abs().argmin()])
    assert behind > math.pi / 2
    point = torch.tensor([[5 * math.sin(behind), 0.0, 5 * math.cos(behind)]])
    pixels, _, valid = camera.project_points(point, 0)
    inside = 0 <= pixels[0, 0] < camera.WIDTH and 0 <= pixels[0, 1] < camera.HEIGHT
    assert inside
    assert valid.tolist() == [False]


def test_depth_map_keeps_nearest_point_per_pixel() -> None:
    from drive2sim.recon.utils.geometry import nearest_depth_map

    pixels = torch.tensor([[1.2, 0.5], [1.7, 0.1], [0.0, 1.0]])
    depth_map = nearest_depth_map(pixels, torch.tensor([9.0, 3.0, 5.0]), 2, 3)
    assert depth_map.tolist() == [[0.0, 3.0, 0.0], [5.0, 0.0, 0.0]]


def test_instance_num_is_zero_without_boxes() -> None:
    """Cảnh không có box (hoặc load_objects tắt): instance_num là 0 thay vì lỗi khi khởi tạo."""
    from types import SimpleNamespace

    from drive2sim.recon.data.driving_dataset import DrivingDataset

    no_boxes = SimpleNamespace(pixel_source=SimpleNamespace(instances_pose=None))
    assert DrivingDataset.instance_num.fget(no_boxes) == 0
    boxes = SimpleNamespace(pixel_source=SimpleNamespace(instances_pose=torch.zeros(2, 3, 4, 4)))
    assert DrivingDataset.instance_num.fget(boxes) == 3


@pytest.mark.parametrize(
    "loader_path",
    [
        "drive2sim.recon.data.loaders.kitti360.KITTI360PixelSource",
        "drive2sim.recon.data.loaders.phenikaa.PhenikaaPixelSource",
        "drive2sim.recon.data.loaders.standard.StandardPixelSource",
    ],
)
def test_saved_loader_paths_build_dataset_sources(tmp_path: Path, loader_path: str) -> None:
    from drive2sim.recon.data.driving_dataset import DrivingDataset
    from drive2sim.recon.utils.misc import import_str

    scene = make_scene(tmp_path)
    reference = make_source(scene)
    dataset = DrivingDataset.__new__(DrivingDataset)
    dataset.data_cfg = OmegaConf.create(
        {
            "dataset": "kitti360",
            "preload_device": "cpu",
            "pixel_source": {**OmegaConf.to_container(reference.data_cfg), "type": loader_path},
            "lidar_source": {"load_lidar": False},
        }
    )
    dataset.data_path = str(scene)
    dataset.start_timestep = 0
    dataset.end_timestep = 2
    pixels, lidar = dataset.build_data_source()
    assert import_str(loader_path) is StandardPixelSource
    assert isinstance(pixels, StandardPixelSource)
    assert lidar is None
    for camera_id, camera in pixels.camera_data.items():
        expected = reference.camera_data[camera_id]
        torch.testing.assert_close(camera.images, expected.images)
        torch.testing.assert_close(camera.cam_to_worlds, expected.cam_to_worlds)
        torch.testing.assert_close(camera.intrinsics, expected.intrinsics)


def test_physicalai_body_override_preserves_per_frame_sensor_mask(tmp_path: Path) -> None:
    from drive2sim.ingest.ego_masks import set_state
    from drive2sim.recon.data.loaders.physicalai import PhysicalAICameraData

    camera = PhysicalAICameraData.__new__(PhysicalAICameraData)
    camera.data_path = str(tmp_path)
    camera.dataset_name = "physicalai"
    camera.cam_name = "front"
    camera.cam_id = 0
    camera.start_timestep, camera.end_timestep = 0, 2
    camera.rectification = None
    sensor = torch.zeros(16, 32, dtype=torch.uint8)
    sensor[0, 0] = 255
    body = torch.zeros_like(sensor)
    body[12:] = 255
    (tmp_path / "ego_masks").mkdir()
    Image.fromarray(body.numpy()).save(tmp_path / "ego_masks" / "front.png")
    camera._read = lambda path, mask: body if path.parent.name == "ego_masks" else sensor
    camera.load_egocar_mask()
    assert camera.invalid_masks[:, 0, 0].all() and camera.invalid_masks[:, 15, 5].all()
    (tmp_path / "ego_masks" / "front.png").unlink()
    set_state(tmp_path, "front", "none")
    camera.load_egocar_mask()
    assert camera.invalid_masks[:, 0, 0].all() and not camera.invalid_masks[:, 15, 5].any()
