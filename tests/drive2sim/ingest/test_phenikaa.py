"""Kiểm tra converter PHENIKAA trên một chuỗi giả nhỏ đúng hình dạng dữ liệu Phenikaa-X.

Chuỗi giả có ba mốc thời gian, một camera pinhole (méo Brown–Conrady) và một camera mắt cá
(Kannala–Brandt, có skew), LiDAR ``.laz`` ghi bằng ``laspy``, ego pose toàn cục (như COMPLEX URBAN)
và calib JSON như bản Phenikaa-X gửi. Kiểm tra: nhận dạng, gợi ý lệnh, ingest (khử méo pinhole,
intrinsics, extrinsics nghịch đảo, world ở khung đầu, LiDAR ``.bin``, manifest) và các lỗi.
"""

import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
laspy = pytest.importorskip("laspy")
pytest.importorskip("lazrs")

from drive2sim.core import files  # noqa: E402
from drive2sim.core.schema import Manifest  # noqa: E402
from drive2sim.engine.workdir import Workdir  # noqa: E402
from drive2sim.ingest import phenikaa  # noqa: E402
from drive2sim.ingest.detect import inspect_source  # noqa: E402
from drive2sim.ingest.dispatch import plan_ingest, run_ingest  # noqa: E402
from drive2sim.ingest.drivestudio import IngestError  # noqa: E402

STAMPS = ["1758620947-799749851", "1758620947-899739742", "1758620947-999741793"]
PINHOLE_K = [[200.0, 0.0, 64.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]]
FISHEYE_K = [[60.0, 0.4, 64.0], [0.0, 60.0, 36.0], [0.0, 0.0, 1.0]]
VEHICLE_TO_PINHOLE = [[0, -1, 0, 0.1], [0, 0, -1, 1.8], [1, 0, 0, -1.3], [0, 0, 0, 1]]
VEHICLE_TO_FISHEYE = [[0, 1, 0, 0.0], [0, 0, -1, 1.3], [-1, 0, 0, -2.0], [0, 0, 0, 1]]


@pytest.fixture(autouse=True)
def no_tool_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Asset mask thân xe trỏ tới thư mục trống: mask thật là ảnh 1920 px, chuỗi giả nhỏ hơn."""
    from drive2sim import paths

    folder = tmp_path / "assets"
    folder.mkdir()
    monkeypatch.setattr(paths, "EGO_MASKS_DIR", folder)
    return folder


def pose(yaw: float, x: float) -> list[float]:
    """Ego pose 3x4 dạng một dòng như file ``ego_pose/<mốc>.txt``, toạ độ toàn cục lớn."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [c, -s, 0, 1000.0 + x, s, c, 0, -9000.0, 0, 0, 1, 600.0]


def make_source(root: Path, fisheye_frames: int = 3) -> Path:
    """Thư mục gốc gồm ``calib/`` và chuỗi ``TEST SEQ`` (tên có dấu cách như dữ liệu thật)."""
    sequence = root / "TEST SEQ"
    for name in ("ego_pose", "label", "lidar", "camera/CAM_P_F", "camera/CAM_F_L"):
        (sequence / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for t, stamp in enumerate(STAMPS):
        (sequence / "ego_pose" / f"{stamp}.txt").write_text(
            " ".join(map(str, pose(0.1 * t, 2.0 * t)))
        )
        (sequence / "label" / f"{stamp}.txt").write_text("1 2 0.8 4.5 2 1.5 0.0 Car\n")
        cv2.imwrite(
            str(sequence / "camera/CAM_P_F" / f"{stamp}.jpg"), np.full((96, 128, 3), 80, np.uint8)
        )
        if t < fisheye_frames:
            cv2.imwrite(
                str(sequence / "camera/CAM_F_L" / f"{stamp}.jpg"),
                np.full((72, 128, 3), 160, np.uint8),
            )
        las = laspy.create(point_format=0, file_version="1.2")
        las.header.scales = [0.001, 0.001, 0.001]
        xyz = rng.uniform(-20, 20, (50, 3))
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.intensity = np.full(50, 255, np.uint16)
        las.write(str(sequence / "lidar" / f"{stamp}.laz"))
    calib = root / "calib"
    calib.mkdir()
    (calib / phenikaa.INTRINSICS_FILE).write_text(
        json.dumps(
            {
                "CAM_P_F": {
                    "image_height": 96, "image_width": 128, "camera_matrix": PINHOLE_K,
                    "distortion_coefficients": [-0.3, 0.1, 0.0, 0.0, 0.0],
                    "camera_model": "pinhole + 5D brown_conrady",
                },
                "CAM_F_L": {
                    "image_height": 72, "image_width": 128, "camera_matrix": FISHEYE_K,
                    "distortion_coefficients": [-0.02, 0.003, -0.003, 0.0007],
                    "camera_model": "kannala_brandt",
                },
            }
        )
    )  # fmt: skip
    (calib / phenikaa.EXTRINSICS_FILE).write_text(
        json.dumps({"CAM_P_F": VEHICLE_TO_PINHOLE, "CAM_F_L": VEHICLE_TO_FISHEYE})
    )
    return root


def test_detects_sequences_and_quotes_suggestion(tmp_path: Path) -> None:
    from drive2sim.cli.source import _suggestion

    report = inspect_source(make_source(tmp_path / "raw"))
    assert report.dataset == "phenikaa" and report.versions == {"": ["TEST SEQ"]}
    assert report.has_lidar is True and report.has_cuboids is True and not report.errors
    assert "--sequence 'TEST SEQ'" in _suggestion(report)


def test_ingest_writes_kitti_layout_scene(tmp_path: Path) -> None:
    raw = make_source(tmp_path / "raw")
    workdir = Workdir(tmp_path / "work")
    target = plan_ingest(inspect_source(raw))
    scene = run_ingest(raw, workdir, target)
    assert scene.name == "phenikaa_test-seq"
    manifest = files.load(scene / "manifest.yaml", Manifest)
    assert [c.camera_id for c in manifest.cameras] == ["CAM_P_F", "CAM_F_L"]
    assert [c.model for c in manifest.cameras] == ["opencv_pinhole", "opencv_fisheye"]
    assert manifest.cameras[1].max_angle == pytest.approx(math.pi / 2, abs=2e-3)
    assert manifest.available.cuboids == "none" and manifest.timeline.num_frames == 3
    # Pinhole: khử méo về camera lý tưởng, méo bằng 0; mắt cá: 4 hệ số giữ nguyên.
    pinhole = np.loadtxt(scene / "intrinsics" / "0.txt")
    assert pinhole.shape == (9,) and not pinhole[4:].any()
    fisheye = np.loadtxt(scene / "intrinsics" / "1.txt")
    assert np.allclose(fisheye, [60, 60, 64, 36, -0.02, 0.003, -0.003, 0.0007])
    # Extrinsics: camera → xe, nghịch đảo của calib xe → camera.
    cam_to_vehicle = np.loadtxt(scene / "extrinsics" / "0.txt")
    assert np.allclose(cam_to_vehicle @ np.array(VEHICLE_TO_PINHOLE), np.eye(4))
    # World là hệ xe ở khung đầu: pose đầu là đơn vị, pose sau tịnh tiến 2 m dọc trục x toàn cục.
    assert np.allclose(np.loadtxt(scene / "ego_pose" / "000.txt"), np.eye(4))
    second = np.loadtxt(scene / "ego_pose" / "001.txt")
    assert np.allclose(second[:3, 3], [2, 0, 0], atol=1e-6)
    lidar = np.fromfile(scene / "lidar" / "002.bin", dtype=np.float32).reshape(-1, 4)
    assert lidar.shape == (50, 4) and np.allclose(lidar[:, 3], 1.0)
    assert (scene / "images" / "000_0.jpg").is_file() and (
        scene / "images" / "002_1.jpg"
    ).is_file()
    assert (scene / "timestamps.txt").read_text().split()[0] == "1758620947799749"
    report = json.loads((scene / "reports" / "ingest.json").read_text())
    assert report["cameras"]["CAM_F_L"]["skew_dropped"] == pytest.approx(0.4)
    with pytest.raises(IngestError, match="đã có"):
        run_ingest(raw, workdir, target)


def test_partial_camera_is_left_out_or_refused(tmp_path: Path) -> None:
    raw = make_source(tmp_path / "raw", fisheye_frames=2)
    workdir = Workdir(tmp_path / "work")
    target = plan_ingest(inspect_source(raw))
    scene = run_ingest(raw, workdir, target, scene_id="phenikaa_partial")
    manifest = files.load(scene / "manifest.yaml", Manifest)
    assert [c.camera_id for c in manifest.cameras] == ["CAM_P_F"]
    with pytest.raises(IngestError, match="thiếu ảnh"):
        run_ingest(
            raw,
            workdir,
            target,
            scene_id="phenikaa_forced",
            phenikaa_options=phenikaa.PhenikaaOptions(cameras=("CAM_F_L",)),
        )


def test_missing_calib_and_fisheye_fold(tmp_path: Path) -> None:
    raw = make_source(tmp_path / "raw")
    (raw / "calib" / phenikaa.EXTRINSICS_FILE).unlink()
    assert inspect_source(raw).errors
    with pytest.raises(IngestError, match="không thấy"):
        run_ingest(raw, Workdir(tmp_path / "work"), plan_ingest(inspect_source(raw)))
    # Hệ số làm đa thức gập trước 90 độ thì góc dùng được dừng ở điểm gập.
    assert phenikaa.fisheye_max_angle([-0.5, 0.0, 0.0, 0.0]) < 1.0


def test_ego_masks_reach_scene_and_loader(tmp_path: Path, no_tool_masks: Path) -> None:
    """Mask của tool được chép vào ``ego_masks/``; loader dùng nó cho loss, chấm điểm và LiDAR."""
    torch = pytest.importorskip("torch")
    from drive2sim.recon.data.base.pixel_source import scene_ego_mask

    assets = no_tool_masks
    (assets / "phenikaa").mkdir()
    mask = np.zeros((96, 128), np.uint8)
    mask[80:, :] = 255  # nắp capo ở dưới ảnh
    cv2.imwrite(str(assets / "phenikaa" / "CAM_P_F.png"), mask)
    cv2.imwrite(str(assets / "phenikaa" / "CAM_F_L.png"), np.zeros((10, 10), np.uint8))
    raw = make_source(tmp_path / "raw")
    target = plan_ingest(inspect_source(raw))
    # Mask sai kích thước (bộ camera khác) bị từ chối, không co giãn.
    with pytest.raises(IngestError, match="không khớp"):
        run_ingest(raw, Workdir(tmp_path / "bad"), target)
    (assets / "phenikaa" / "CAM_F_L.png").unlink()
    scene = run_ingest(raw, Workdir(tmp_path / "work"), target)
    assert sorted(p.name for p in (scene / "ego_masks").iterdir()) == ["CAM_P_F.png"]
    loaded = scene_ego_mask(str(scene), "CAM_P_F", (96, 128))
    assert loaded is not None and np.array(loaded)[90, 5] == 255
    assert scene_ego_mask(str(scene), "CAM_F_L", (72, 128)) is None
    with pytest.raises(ValueError, match="khác ảnh"):
        scene_ego_mask(str(scene), "CAM_P_F", (72, 128))

    from drive2sim.recon.data.loaders.standard import StandardCameraData

    camera = StandardCameraData(
        dataset_name="phenikaa", data_path=str(scene), cam_id=0,
        start_timestep=0, end_timestep=3, device=torch.device("cpu"),
    )  # fmt: skip
    assert camera.egocar_mask is not None and camera.egocar_mask[90].all()
    camera.normalized_time = None  # pixel source đặt hai trường này khi dựng cả cảnh
    camera.unique_img_idx = torch.arange(3)
    image, _ = camera.get_image(0)
    assert not image["valid_masks"][90].any() and image["valid_masks"][10].all()
    assert image["egocar_masks"][90].all()


def test_existing_scene_is_rejected_before_reading_source(tmp_path: Path) -> None:
    from drive2sim.engine.workdir import Workdir
    from drive2sim.ingest.drivestudio import IngestError
    from drive2sim.ingest.phenikaa import ingest_phenikaa

    workdir = Workdir(tmp_path / "work")
    workdir.scene_dir("phenikaa_complex-city").mkdir(parents=True)
    with pytest.raises(IngestError, match="đã có"):
        ingest_phenikaa(tmp_path / "source-does-not-exist", workdir, "COMPLEX CITY")
