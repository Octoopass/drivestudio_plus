"""Kiểm tra converter KITTI-360 và phép khớp Mei → OpenCV fisheye.

Không có dữ liệu KITTI-360 thật trên máy test, nên file này dựng một thư mục gốc giả đúng hình
dạng KITTI-360: ``calibration/`` với số liệu giả nhưng đúng định dạng, ``data_poses.zip`` có
``poses.txt`` thiếu pose ở vài khung (để thử chỗ hở) và ``cam0_to_world.txt`` tính độc lập bằng
công thức của ``kitti360Scripts``, ảnh PNG chỉ có phần đầu, LiDAR vài byte, đóng gói trong ``.tar``
như bản tải về (hoặc để ở thư mục đã giải nén). Kiểm tra: nhận dạng, chọn khung, manifest (mô hình
camera, track), intrinsics và extrinsics, kiểm tra chéo với ``cam0_to_world.txt``, CLI. Độ chính
xác của phép khớp Mei kiểm ở ``tests/drive2sim/core/test_camera.py``.
"""

import json
import math
import pickle
import struct
import tarfile
import zipfile
from pathlib import Path

import pytest

from drive2sim.cli.main import main
from drive2sim.core import files
from drive2sim.core.schema import Manifest, ValidationReport
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest import kitti360
from drive2sim.ingest.detect import inspect_source
from drive2sim.ingest.dispatch import plan_ingest, run_ingest
from drive2sim.ingest.drivestudio import IngestError
from tests.drive2sim.samples import GAMMA_02, MEI_02, NUMPY_COEFFS_02

SEQ = "2013_05_28_drive_0003_sync"
# Khung có ảnh và LiDAR: 0..13; khung có pose: thiếu 4..7 (chỗ hở) và 13.
IMAGE_FRAMES = list(range(14))
POSE_FRAMES = [0, 1, 2, 3, 8, 9, 10, 11, 12]


def mat(rows: list[list[float]]) -> list[list[float]]:
    return [list(map(float, row)) for row in rows] + [[0.0, 0.0, 0.0, 1.0]]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def rigid_inverse(m: list[list[float]]) -> list[list[float]]:
    rotation_t = [[m[j][i] for j in range(3)] for i in range(3)]
    t = [-sum(rotation_t[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return [rotation_t[i] + [t[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def flat12(m: list[list[float]]) -> str:
    return " ".join(f"{v:.12g}" for row in m[:3] for v in row)


# Hệ pose: x tới trước, y phải, z xuống; camera: x phải, y xuống, z tới trước.
CAM_TO_POSE = {
    "image_00": mat([[0, 0, 1, 1.5], [1, 0, 0, 0.0], [0, 1, 0, -0.6]]),
    "image_01": mat([[0, 0, 1, 1.5], [1, 0, 0, 0.6], [0, 1, 0, -0.6]]),
    "image_02": mat([[1, 0, 0, 0.7], [0, 0, -1, -0.2], [0, 1, 0, -1.0]]),
    "image_03": mat([[-1, 0, 0, 0.7], [0, 0, 1, 0.8], [0, 1, 0, -1.0]]),
}
# Velodyne: x tới trước, y trái, z lên; camera 0 → Velodyne.
CAM0_TO_VELO = mat([[0, 0, 1, 0.8], [-1, 0, 0, 0.3], [0, -1, 0, -0.2]])
# Xoay nhỏ quanh trục y cho R_rect, để phép rectify không phải ma trận đơn vị.
_A = 0.01
R_RECT_00 = mat(
    [[math.cos(_A), 0, math.sin(_A), 0], [0, 1, 0, 0], [-math.sin(_A), 0, math.cos(_A), 0]]
)


def pose_of(frame: int) -> list[list[float]]:
    """Pose → world: xe chạy theo trục x của world, quay đầu dần; z của pose hướng xuống."""
    yaw = 0.02 * frame
    c, s = math.cos(yaw), math.sin(yaw)
    return mat([[c, s, 0, 1000 + 2 * frame], [s, -c, 0, 300 + 0.1 * frame], [0, 0, -1, 120]])


def write_png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0" * 4


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, __import__("io").BytesIO(data))


def make_raw(root: Path, extracted: bool = False, cam0_offset: float = 0.0) -> Path:
    """Dựng thư mục KITTI-360 giả; ``cam0_offset`` làm lệch cam0_to_world.txt (thử báo lỗi)."""
    calib = root / "calibration"
    calib.mkdir(parents=True)
    (calib / "calib_cam_to_pose.txt").write_text(
        "".join(f"{cam}: {flat12(m)}\n" for cam, m in CAM_TO_POSE.items())
    )
    (calib / "calib_cam_to_velo.txt").write_text(flat12(CAM0_TO_VELO) + "\n")
    rect = " ".join(f"{v:.12g}" for row in R_RECT_00[:3] for v in row[:3])
    (calib / "perspective.txt").write_text(
        "calib_time: 09-Jan-2013 15:40:47\n"
        f"R_rect_00: {rect}\n"
        "P_rect_00: 552.5 0 682.0 0 0 552.5 238.7 0 0 0 1 0\n"
        "R_rect_01: 1 0 0 0 1 0 0 0 1\n"
        "P_rect_01: 552.5 0 682.0 -328.3 0 552.5 238.7 0 0 0 1 0\n"
    )
    for cam in ("image_02", "image_03"):
        (calib / f"{cam}.yaml").write_text(
            "%YAML:1.0\nmodel_type: MEI\ncamera_name: " + cam + "\n"
            "image_width: 1400\nimage_height: 1400\nmirror_parameters:\n"
            f"   xi: {MEI_02['xi']}\ndistortion_parameters:\n   k1: {MEI_02['k1']}\n"
            f"   k2: {MEI_02['k2']}\n   p1: 4e-4\n   p2: 4e-4\nprojection_parameters:\n"
            f"   gamma1: {GAMMA_02[0]}\n   gamma2: {GAMMA_02[1]}\n   u0: 700\n   v0: 705\n"
        )
    poses = "".join(f"{f} {flat12(pose_of(f))}\n" for f in POSE_FRAMES)
    # cam0_to_world tính độc lập theo kitti360Scripts: pose · cam_to_pose[00] · R_rect_00⁻¹.
    cam0 = ""
    for f in POSE_FRAMES:
        m = matmul(matmul(pose_of(f), CAM_TO_POSE["image_00"]), rigid_inverse(R_RECT_00))
        m[0][3] += cam0_offset
        cam0 += f"{f} " + " ".join(f"{v:.12g}" for row in m for v in row) + "\n"
    with zipfile.ZipFile(root / "data_poses.zip", "w") as zipped:
        zipped.writestr(f"{SEQ}/poses.txt", poses)
        zipped.writestr(f"{SEQ}/cam0_to_world.txt", cam0)
    sizes = {"image_00": (1408, 376), "image_01": (1408, 376)}
    for cam in kitti360.CAMERAS:
        sub = "data_rect" if cam in sizes else "data_rgb"
        png = write_png(*sizes.get(cam, (1400, 1400)))
        if extracted:
            folder = root / "data_2d_raw" / SEQ / cam / sub
            folder.mkdir(parents=True)
            for f in IMAGE_FRAMES:
                (folder / f"{f:010d}.png").write_bytes(png)
        else:
            (root / "data_2d_raw").mkdir(exist_ok=True)
            with tarfile.open(root / "data_2d_raw" / f"{SEQ}_{cam}.tar", "w") as tar:
                for f in IMAGE_FRAMES:
                    add_bytes(tar, f"data_2d_raw/{SEQ}/{cam}/{sub}/{f:010d}.png", png)
    (root / "data_3d_raw").mkdir()
    with tarfile.open(root / "data_3d_raw" / f"{SEQ}.tar", "w") as tar:
        for f in IMAGE_FRAMES:
            add_bytes(tar, f"data_3d_raw/{SEQ}/velodyne_points/data/{f:010d}.bin", b"\1" * 16 * f)
    pkl = root / "ImageSets-rosa-6class"
    pkl.mkdir()
    (pkl / "kitti360_rosa_infos_val.pkl").write_bytes(b"not a pickle")
    return root


def read_matrix(path: Path) -> list[list[float]]:
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines()]


def close(a: list[list[float]], b: list[list[float]], tol: float = 1e-9) -> bool:
    return all(abs(x - y) <= tol for ra, rb in zip(a, b) for x, y in zip(ra, rb))


# ---------------------------------------------------------------- chọn khung


def test_select_frames_respects_gaps() -> None:
    frames = [0, 1, 2, 3, 8, 9, 10, 11, 12, 20]
    assert kitti360.select_frames(frames, max_gap=1) == [8, 9, 10, 11, 12]  # đoạn dài nhất
    assert kitti360.select_frames(frames, num_frames=3, max_gap=1) == [0, 1, 2]
    # Không đoạn nào đủ 6 khung: lấy đoạn dài nhất.
    assert kitti360.select_frames(frames, num_frames=6, max_gap=1) == [8, 9, 10, 11, 12]
    assert kitti360.select_frames(frames, start_frame=9, max_gap=1) == [9, 10, 11, 12]
    assert kitti360.select_frames(frames, max_gap=5) == [0, 1, 2, 3, 8, 9, 10, 11, 12]
    assert kitti360.select_frames(frames, max_gap=100) == frames
    with pytest.raises(IngestError, match="không có khung"):
        kitti360.select_frames(frames, start_frame=50)


# ---------------------------------------------------------------- nhận dạng và ingest


def test_detect_kitti360_without_reading_pickles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("không được đọc pickle")

    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setattr(pickle, "loads", refuse)
    raw = make_raw(tmp_path / "raw")
    report = inspect_source(raw)
    assert report.dataset == "kitti360"
    assert report.versions == {"": [SEQ]}
    assert report.cameras == list(kitti360.CAMERAS)
    assert report.has_lidar is True and report.has_cuboids is False
    assert report.ingest_supported
    assert any(".pkl" in note for note in report.notes)
    assert plan_ingest(report).sequence == SEQ


@pytest.mark.parametrize("extracted", [False, True])
def test_ingest_writes_kitti_layout(tmp_path: Path, extracted: bool) -> None:
    raw = make_raw(tmp_path / "raw", extracted=extracted)
    workdir = Workdir(tmp_path / "work")
    target = plan_ingest(inspect_source(raw))
    options = kitti360.Kitti360Options(max_gap=1)
    scene_dir = run_ingest(raw, workdir, target, kitti360_options=options)
    assert scene_dir == workdir.scene_dir(f"kitti360_{SEQ}")
    # Đoạn liền nhau dài nhất có đủ ảnh, LiDAR và pose: 8..12.
    assert (scene_dir / "frame_ids.txt").read_text().split() == ["8", "9", "10", "11", "12"]
    assert (scene_dir / "timestamps.txt").read_text().split() == [
        "800000",
        "900000",
        "1000000",
        "1100000",
        "1200000",
    ]
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    assert manifest.source.dataset == "kitti360"
    assert manifest.source.converter == "kitti360"
    assert [c.model for c in manifest.cameras] == [
        "opencv_pinhole",
        "opencv_pinhole",
        "opencv_fisheye",
        "opencv_fisheye",
    ]
    assert manifest.track == "fisheye"  # có camera mắt cá
    assert [list(c.resolution) for c in manifest.cameras][::2] == [[1408, 376], [1400, 1400]]
    assert manifest.lidars == ["velodyne"]
    assert manifest.available.lidar is True
    assert manifest.available.cuboids == "none"
    assert manifest.timeline.num_frames == 5

    # Ảnh và LiDAR chép nguyên byte; khung 0 của cảnh là khung gốc 8.
    assert (scene_dir / "lidar" / "000.bin").read_bytes() == b"\1" * 16 * 8
    assert (scene_dir / "images" / "003_3.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    pinhole = [float(v) for v in (scene_dir / "intrinsics" / "0.txt").read_text().split()]
    assert pinhole == [552.5, 552.5, 682.0, 238.7, 0, 0, 0, 0, 0]
    fisheye = [float(v) for v in (scene_dir / "intrinsics" / "2.txt").read_text().split()]
    assert len(fisheye) == 8
    assert fisheye[4:] == pytest.approx(NUMPY_COEFFS_02, abs=1e-9)
    assert fisheye[2:4] == [700.0, 705.0]

    # Extrinsics: camera → Velodyne = (cam_to_pose00 · velo→cam0)⁻¹ · cam_to_pose[k] · R_rect⁻¹.
    velo_to_pose = matmul(CAM_TO_POSE["image_00"], rigid_inverse(CAM0_TO_VELO))
    expected0 = matmul(
        matmul(rigid_inverse(velo_to_pose), CAM_TO_POSE["image_00"]), rigid_inverse(R_RECT_00)
    )
    assert close(read_matrix(scene_dir / "extrinsics" / "0.txt"), expected0)
    expected2 = matmul(rigid_inverse(velo_to_pose), CAM_TO_POSE["image_02"])
    assert close(read_matrix(scene_dir / "extrinsics" / "2.txt"), expected2)

    # Ego pose: khung đầu là gốc world; các khung sau tương đối với nó.
    ego0 = read_matrix(scene_dir / "ego_pose" / "000.txt")
    assert close(ego0, [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)])
    expected = matmul(
        rigid_inverse(matmul(pose_of(8), velo_to_pose)), matmul(pose_of(10), velo_to_pose)
    )
    assert close(read_matrix(scene_dir / "ego_pose" / "002.txt"), expected, tol=1e-8)

    report = files.load(scene_dir / "reports" / "validation.json", ValidationReport)
    assert report.summary.errors == 0
    checks = {check.check_id: check for check in report.checks}
    assert checks["pose.cam0_to_world"].severity == "info"
    assert checks["pose.cam0_to_world"].details["frames_compared"] == 5
    assert checks["pose.cam0_to_world"].details["max_translation_diff_m"] < 1e-6
    assert checks["camera.fisheye_fit"].details["fit_error_px"] < 0.01
    assert "camera.fisheye_fov" in checks
    assert "timestamps.nominal" in checks
    assert not (workdir.data_dir / ".ingest" / f"kitti360_{SEQ}").exists()

    with pytest.raises(IngestError, match="đã có"):
        run_ingest(raw, workdir, target)


def test_ingest_reports_wrong_pose_chain(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / "raw", cam0_offset=0.05)  # cam0_to_world lệch 5 cm
    workdir = Workdir(tmp_path / "work")
    scene_dir = kitti360.ingest_kitti360(raw, workdir, SEQ)
    report = files.load(scene_dir / "reports" / "validation.json", ValidationReport)
    assert report.summary.errors == 1
    check = next(c for c in report.checks if c.check_id == "pose.cam0_to_world")
    assert check.severity == "error"


def test_ingest_subset_of_cameras_and_frames(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / "raw")
    workdir = Workdir(tmp_path / "work")
    options = kitti360.Kitti360Options(
        start_frame=1, num_frames=2, cameras=("image_00", "image_03"), max_gap=1
    )
    scene_dir = kitti360.ingest_kitti360(
        raw, workdir, SEQ, scene_id="kitti360_short", options=options
    )
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    assert [c.camera_id for c in manifest.cameras] == ["image_00", "image_03"]
    assert manifest.track == "fisheye"
    assert (scene_dir / "frame_ids.txt").read_text().split() == ["1", "2"]
    assert sorted(p.name for p in (scene_dir / "intrinsics").iterdir()) == ["0.txt", "1.txt"]

    pinhole_only = kitti360.Kitti360Options(cameras=("image_00", "image_01"), max_gap=1)
    scene_dir = kitti360.ingest_kitti360(
        raw, workdir, SEQ, scene_id="kitti360_pinhole", options=pinhole_only
    )
    assert files.load(scene_dir / "manifest.yaml", Manifest).track == "pinhole"

    with pytest.raises(IngestError, match="image_09"):
        kitti360.ingest_kitti360(
            raw,
            workdir,
            SEQ,
            scene_id="kitti360_bad",
            options=kitti360.Kitti360Options(cameras=("image_09",)),
        )


def test_ingest_missing_poses(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / "raw")
    (raw / "data_poses.zip").unlink()
    with pytest.raises(IngestError, match="poses.txt"):
        kitti360.ingest_kitti360(raw, Workdir(tmp_path / "work"), SEQ)


def test_cli_ingest_split_and_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("D2S_WORKDIR", str(tmp_path / "work"))
    raw = make_raw(tmp_path / "raw")
    args = ["scene", "ingest", str(raw), "--max-gap", "5", "--num-frames", "6"]
    assert main([*args, "--scene", "kitti360_cli", "--quiet"]) == 0
    capsys.readouterr()
    manifest = Workdir(tmp_path / "work").load_scene("kitti360_cli").manifest
    assert manifest.timeline.num_frames == 6  # 0,1,2,3 rồi qua chỗ hở 3→8 (bước 5) tới 8, 9
    strategy = ["--strategy", "every_nth_frame:2", "--quiet"]
    assert main(["split", "create", "kitti360_cli", *strategy]) == 0
    capsys.readouterr()
    check = ["--scene", "kitti360_cli", "--split", "every2_allcams", "--json", "--quiet"]
    assert main(["approach", "check", "gsplat_3dgut", *check]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [s["stage_id"] for s in data["stages"]] == [
        "train.gsplat_3dgut",
        "eval.standard",
        "export.ply_world",
        "export.preview",
    ]
    # OmniRe cần box 3D, cảnh KITTI-360 không có: bị khoá (mã 3).
    assert main(["approach", "check", "omnire", *check]) == 3
