"""Converter PHENIKAA: chuyển một chuỗi PHENIKAA (Phenikaa-X Perception Data) sang dữ liệu chuẩn.

Một chuỗi PHENIKAA là một thư mục (ví dụ ``COMPLEX CITY``) gồm ``camera/<CAM>/<mốc>.jpg`` cho 10
camera (4 mắt cá ``CAM_F_*`` 1920x1080, 6 pinhole ``CAM_P_*`` 1920x1536), ``lidar/<mốc>.laz``,
``ego_pose/<mốc>.txt`` (ma trận 3x4 xe → world) và ``label/<mốc>.txt`` (box 3D từng khung, chưa có
ID theo dõi). Mốc thời gian là tên file ``<giây>-<nano giây>``, chung cho mọi cảm biến, 10 Hz.
Calib nằm ở ``calib/`` cạnh các chuỗi: ``Camera_Intrinsics.json`` và ``Sensor_Extrinsics.json``
(xe → camera, trục OpenCV).

Converter ghi theo **khung thư mục KITTI của DriveStudio** như KITTI-360 (CONTRACTS mục 2.1), xe
(``ego``) là hệ xe của PHENIKAA, cũng là hệ của điểm LiDAR trong file ``.laz``:

- ``images/<khung>_<chỉ số>.jpg``: camera mắt cá là hard link tới ảnh gốc (giữ nguyên ảnh,
  loader vẽ bằng mô hình Kannala–Brandt). Camera pinhole được **khử méo khi ingest** (5 hệ số
  Brown–Conrady, hoặc 8 hệ số rational của ``CAM_P_L``, ``CAM_P_R``) về camera pinhole lý tưởng,
  ghi JPEG chất lượng 95, vì trainer và renderer của tool không vẽ méo pinhole.
- ``intrinsics/<chỉ số>.txt``: pinhole 9 số ``fx fy cx cy 0 0 0 0 0`` (sau khử méo); mắt cá 8 số
  ``fx fy cx cy k1 k2 k3 k4`` (Kannala–Brandt, như ``cv2.fisheye``).
- ``extrinsics/<chỉ số>.txt``: camera → xe (nghịch đảo của ``Sensor_Extrinsics.json``).
- ``lidar/<khung>.bin``: ``float32`` x, y, z, cường độ (0–1) trong hệ xe.
- ``ego_pose/<khung>.txt``: xe → world, world là hệ xe ở khung đầu của cảnh.
- ``frame_ids.txt`` (số thứ tự khung trong chuỗi gốc), ``timestamps.txt`` (micro giây).

Chưa ghi box: nhãn PHENIKAA không có ID theo dõi nên chưa dựng được ``instances/`` (cần bước
tracking riêng); manifest ghi ``cuboids: none``. Box gốc vẫn ở thư mục dữ liệu gốc.

Giới hạn đã biết: ma trận camera mắt cá có skew khoảng 0,4–0,5 (``K[0, 1]``) mà mô hình mắt cá của
tool không có, nên bị bỏ (lệch dưới 1 pixel); chưa có mask thân xe.

Cần ``numpy``, ``opencv-python`` và ``laspy[lazrs]`` (môi trường ``drive2sim``), chỉ nạp khi
ingest.

File này chứa:
- ``DATASET``, ``CAMERAS``, ``LIDAR_ID``: tên dataset, camera và LiDAR ghi vào manifest.
- ``PhenikaaOptions``: lựa chọn khi ingest (camera).
- ``find_sequences``, ``find_calib``: tìm chuỗi và calib trong thư mục nguồn.
- ``read_calibration``: đọc và kiểm calib.
- ``fisheye_max_angle``: góc lệch trục lớn nhất của ống kính mắt cá.
- ``ingest_phenikaa``: hàm chính mà ``drive2sim/ingest/dispatch.py`` gọi.
"""

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from drive2sim.core import files, identifiers
from drive2sim.core.schema import ReportSummary, ValidationReport
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest.drivestudio import IngestError, default_scene_id
from drive2sim.ingest.manifest import build_manifest
from drive2sim.ingest.validate import validate_scene

if TYPE_CHECKING:
    import numpy as np

DATASET = "phenikaa"
CONVERTER = "phenikaa"
LIDAR_ID = "LIDAR_T"
DATA_LICENSE = "Phenikaa-X Perception Data (xem điều khoản của Phenikaa-X)"
# Thứ tự chỉ số camera trong cảnh: 4 pinhole tầm xa, 2 pinhole tầm trung, 4 mắt cá.
CAMERAS: tuple[str, ...] = (
    "CAM_P_F",
    "CAM_P_FL",
    "CAM_P_FR",
    "CAM_P_B",
    "CAM_P_L",
    "CAM_P_R",
    "CAM_F_F",
    "CAM_F_L",
    "CAM_F_R",
    "CAM_F_B",
)
CALIB_DIR = "calib"
INTRINSICS_FILE = "Camera_Intrinsics.json"
EXTRINSICS_FILE = "Sensor_Extrinsics.json"
JPEG_QUALITY = 95
# Ống kính mắt cá của PHENIKAA phủ 180 độ: tia lệch trục quá 90 độ không có trong ảnh.
FISHEYE_LENS_LIMIT = math.pi / 2
# Cường độ trong file .laz là số nguyên 0..255 (đo trên ba chuỗi mini); chia để về 0..1 như KITTI.
INTENSITY_SCALE = 255.0


@dataclass(frozen=True)
class PhenikaaOptions:
    """Lựa chọn khi ingest PHENIKAA.

    Attributes:
        cameras: Camera cần ghi, theo tên PHENIKAA; mặc định mọi camera có đủ ảnh.
        calib_dir: Thư mục calib; mặc định tìm ``calib/`` ở thư mục nguồn hoặc thư mục cha.
    """

    cameras: Optional[tuple[str, ...]] = None
    calib_dir: Optional[Path] = None


def _is_sequence(path: Path) -> bool:
    return (
        (path / "camera").is_dir() and (path / "ego_pose").is_dir() and (path / "lidar").is_dir()
    )


def find_sequences(raw_dir: Path) -> dict[str, Path]:
    """Các chuỗi trong thư mục nguồn: chính nó nếu là một chuỗi, hoặc các thư mục con là chuỗi.

    Returns:
        Tên chuỗi → thư mục chuỗi, sắp theo tên.
    """
    raw_dir = Path(raw_dir)
    if _is_sequence(raw_dir):
        return {raw_dir.name: raw_dir}
    found = {}
    if raw_dir.is_dir():
        for child in sorted(raw_dir.iterdir()):
            if child.is_dir() and _is_sequence(child):
                found[child.name] = child
    return found


def find_calib(sequence_dir: Path, override: Optional[Path] = None) -> Path:
    """Thư mục calib: ``override``, hoặc ``calib/`` trong chuỗi, cạnh chuỗi, rồi hai cấp trên.

    Raises:
        IngestError: Nếu không thấy đủ hai file calib.
    """
    candidates = [Path(override)] if override else []
    for base in (sequence_dir, *list(sequence_dir.parents)[:2]):
        candidates.append(base / CALIB_DIR)
    for folder in candidates:
        if (folder / INTRINSICS_FILE).is_file() and (folder / EXTRINSICS_FILE).is_file():
            return folder
    raise IngestError(
        f"không thấy {INTRINSICS_FILE} và {EXTRINSICS_FILE}: đặt thư mục {CALIB_DIR}/ cạnh các "
        "chuỗi PHENIKAA hoặc truyền --calib"
    )


def read_calibration(calib_dir: Path, cameras: list[str]) -> dict[str, dict[str, Any]]:
    """Đọc calib của các camera.

    Returns:
        Tên camera → ``{"model": "pinhole" | "fisheye", "K", "dist", "size": (w, h),
        "vehicle_to_camera"}`` (mảng numpy).

    Raises:
        IngestError: Nếu thiếu camera, mô hình lạ hoặc số hệ số méo sai.
    """
    import numpy as np

    intrinsics = json.loads((calib_dir / INTRINSICS_FILE).read_text(encoding="utf-8"))
    extrinsics = json.loads((calib_dir / EXTRINSICS_FILE).read_text(encoding="utf-8"))
    result = {}
    for camera in cameras:
        if camera not in intrinsics or camera not in extrinsics:
            raise IngestError(f"calib thiếu camera {camera}")
        item = intrinsics[camera]
        name = str(item["camera_model"]).lower()
        dist = np.asarray(item["distortion_coefficients"], dtype=np.float64)
        if "kannala" in name:
            model = "fisheye"
            if dist.size != 4:
                raise IngestError(f"{camera}: Kannala–Brandt cần 4 hệ số, có {dist.size}")
        elif "pinhole" in name:
            model = "pinhole"
            if dist.size not in (4, 5, 8, 12, 14):
                raise IngestError(f"{camera}: số hệ số méo {dist.size} không phải của OpenCV")
        else:
            raise IngestError(f"{camera}: mô hình camera lạ {item['camera_model']}")
        transform = np.asarray(extrinsics[camera], dtype=np.float64)
        if transform.shape != (4, 4):
            raise IngestError(f"{camera}: extrinsics cần ma trận 4x4")
        result[camera] = {
            "model": model,
            "K": np.asarray(item["camera_matrix"], dtype=np.float64),
            "dist": dist,
            "size": (int(item["image_width"]), int(item["image_height"])),
            "vehicle_to_camera": transform,
        }
    return result


def fisheye_max_angle(coeffs: list[float], limit: float = FISHEYE_LENS_LIMIT) -> float:
    """Góc lệch trục lớn nhất (rad) mà mô hình Kannala–Brandt còn đơn điệu, tối đa ``limit``.

    ``theta_d = theta·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)`` phải tăng theo ``theta`` thì mỗi pixel
    ứng với một tia; quá điểm gập, mô hình không còn đúng.
    """
    k1, k2, k3, k4 = coeffs
    step = 1e-3
    theta = step
    while theta < limit:
        t2 = theta * theta
        derivative = 1 + 3 * k1 * t2 + 5 * k2 * t2**2 + 7 * k3 * t2**3 + 9 * k4 * t2**4
        if derivative <= 0:
            return theta - step
        theta += step
    return limit


def _stamps(sequence_dir: Path) -> list[str]:
    """Mốc thời gian của chuỗi theo ``ego_pose/``, tăng dần."""
    stamps = [p.stem for p in (sequence_dir / "ego_pose").glob("*.txt")]
    return sorted(stamps, key=_microseconds)


def _microseconds(stamp: str) -> int:
    seconds, nanos = stamp.split("-")
    return int(seconds) * 1_000_000 + int(nanos) // 1000


def _matrix_text(matrix: "np.ndarray") -> str:
    return "".join(" ".join(f"{v:.18e}" for v in row) + "\n" for row in matrix)


def _write_lidar(source: Path, target: Path) -> int:
    """Đổi ``.laz`` thành ``.bin`` (``float32`` x, y, z, cường độ 0–1); trả số điểm."""
    import laspy
    import numpy as np

    las = laspy.read(source)
    points = np.empty((len(las.x), 4), dtype=np.float32)
    points[:, 0] = las.x
    points[:, 1] = las.y
    points[:, 2] = las.z
    points[:, 3] = np.asarray(las.intensity, dtype=np.float32) / INTENSITY_SCALE
    points = points[np.isfinite(points).all(axis=1)]
    points.tofile(target)
    return len(points)


def _undistort_maps(calib: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Bảng ánh xạ khử méo pinhole về camera lý tưởng cùng kích thước, không vùng đen (alpha 0)."""
    import cv2

    size = calib["size"]
    new_k, _ = cv2.getOptimalNewCameraMatrix(calib["K"], calib["dist"], size, 0, size)
    map_x, map_y = cv2.initUndistortRectifyMap(
        calib["K"], calib["dist"], None, new_k, size, cv2.CV_32FC1
    )
    return new_k, map_x, map_y


def _place_image(source: Path, target: Path) -> None:
    """Hard link ảnh gốc (cùng ổ đĩa), không được thì chép."""
    try:
        os.link(os.path.realpath(source), target)
    except OSError:
        shutil.copy2(source, target)


def ingest_phenikaa(
    raw_dir: Path,
    workdir: Workdir,
    sequence: str,
    scene_id: Optional[str] = None,
    options: Optional[PhenikaaOptions] = None,
) -> Path:
    """Ingest một chuỗi PHENIKAA vào ``data/<scene_id>/``.

    Các bước: tìm chuỗi và calib; chọn camera có ảnh ở mọi mốc; ghi vào thư mục tạm
    ``data/.ingest/<scene_id>/`` ảnh, LiDAR, pose, calib; chuyển sang ``data/<scene_id>/``; ghi
    ``manifest.yaml`` và ``reports/validation.json``.

    Args:
        raw_dir: Thư mục chứa các chuỗi (và ``calib/``), hoặc chính thư mục một chuỗi.
        workdir: Thư mục làm việc.
        sequence: Tên chuỗi, ví dụ ``COMPLEX CITY``.
        scene_id: Định danh cảnh; mặc định ``phenikaa_<chuỗi>`` (``phenikaa_complex-city``).
        options: Lựa chọn camera, thư mục calib.

    Returns:
        Thư mục cảnh ``data/<scene_id>/``.

    Raises:
        IngestError: Nếu thiếu chuỗi, calib, ảnh hoặc cảnh đã có.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise IngestError(f"ingest PHENIKAA cần numpy và OpenCV: {error}") from error
    try:
        import laspy  # noqa: F401
        import lazrs  # noqa: F401
    except ImportError as error:
        raise IngestError(
            'ingest PHENIKAA cần đọc LiDAR .laz: pip install "laspy[lazrs]" trong môi trường '
            "drive2sim"
        ) from error

    options = options or PhenikaaOptions()
    sequences = find_sequences(raw_dir)
    if sequence not in sequences:
        raise IngestError(f"không có chuỗi {sequence}; có: {', '.join(sequences) or '(không có)'}")
    sequence_dir = sequences[sequence]
    calib_dir = find_calib(sequence_dir, options.calib_dir)
    stamps = _stamps(sequence_dir)
    if not stamps:
        raise IngestError(f"{sequence}: ego_pose/ không có file nào")
    complete = [
        camera
        for camera in CAMERAS
        if all((sequence_dir / "camera" / camera / f"{s}.jpg").is_file() for s in stamps)
    ]
    cameras = list(options.cameras) if options.cameras else complete
    unknown = [camera for camera in cameras if camera not in CAMERAS]
    if unknown or not cameras:
        raise IngestError(
            f"camera không có trong PHENIKAA: {', '.join(unknown) or '(trống)'}; có: "
            f"{', '.join(CAMERAS)}"
        )
    partial = [camera for camera in cameras if camera not in complete]
    if partial:
        raise IngestError(
            f"{sequence}: camera thiếu ảnh ở một số mốc: {', '.join(partial)}; có đủ: "
            f"{', '.join(complete)}"
        )
    missing_lidar = [s for s in stamps if not (sequence_dir / "lidar" / f"{s}.laz").is_file()]
    if missing_lidar:
        raise IngestError(f"{sequence}: thiếu LiDAR ở {len(missing_lidar)} mốc")
    calibration = read_calibration(calib_dir, cameras)

    scene_id = scene_id or default_scene_id(DATASET, sequence)
    identifiers.check_scene_id(scene_id)
    scene_dir = workdir.scene_dir(scene_id)
    if scene_dir.exists():
        raise IngestError(
            f"cảnh {scene_id} đã có ở {workdir.relative(scene_dir)}; xoá cảnh đó hoặc đặt "
            "scene_id khác"
        )
    temp_dir = workdir.data_dir / ".ingest" / scene_id
    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        for name in ("images", "lidar", "ego_pose", "extrinsics", "intrinsics", "reports"):
            (temp_dir / name).mkdir(parents=True)
        models, max_angles, notes = [], [], {}
        maps = {}
        for index, camera in enumerate(cameras):
            calib = calibration[camera]
            k = calib["K"]
            if calib["model"] == "fisheye":
                values = [k[0, 0], k[1, 1], k[0, 2], k[1, 2], *calib["dist"].tolist()]
                models.append("opencv_fisheye")
                max_angles.append(fisheye_max_angle(calib["dist"].tolist()))
                notes[camera] = {"skew_dropped": float(k[0, 1])}
            else:
                new_k, map_x, map_y = _undistort_maps(calib)
                maps[camera] = (map_x, map_y)
                values = [new_k[0, 0], new_k[1, 1], new_k[0, 2], new_k[1, 2], 0, 0, 0, 0, 0]
                models.append("opencv_pinhole")
                max_angles.append(None)
                notes[camera] = {"undistorted_from": int(calib["dist"].size)}
            (temp_dir / "intrinsics" / f"{index}.txt").write_text(
                "".join(f"{v:.18e}\n" for v in values)
            )
            camera_to_vehicle = np.linalg.inv(calib["vehicle_to_camera"])
            (temp_dir / "extrinsics" / f"{index}.txt").write_text(_matrix_text(camera_to_vehicle))

        poses = []
        for stamp in stamps:
            pose = np.eye(4)
            pose[:3, :] = np.loadtxt(sequence_dir / "ego_pose" / f"{stamp}.txt").reshape(3, 4)
            poses.append(pose)
        # World của cảnh là hệ xe ở khung đầu, để toạ độ nhỏ (COMPLEX URBAN có pose toàn cục).
        world_from_global = np.linalg.inv(poses[0])
        points = 0
        for t, stamp in enumerate(stamps):
            (temp_dir / "ego_pose" / f"{t:03d}.txt").write_text(
                _matrix_text(world_from_global @ poses[t])
            )
            points += _write_lidar(
                sequence_dir / "lidar" / f"{stamp}.laz", temp_dir / "lidar" / f"{t:03d}.bin"
            )
            for index, camera in enumerate(cameras):
                source = sequence_dir / "camera" / camera / f"{stamp}.jpg"
                target = temp_dir / "images" / f"{t:03d}_{index}.jpg"
                if camera in maps:
                    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                    if image is None:
                        raise IngestError(f"không đọc được {source}")
                    image = cv2.remap(image, *maps[camera], interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                else:
                    _place_image(source, target)
        (temp_dir / "frame_ids.txt").write_text("".join(f"{t}\n" for t in range(len(stamps))))
        (temp_dir / "timestamps.txt").write_text("".join(f"{_microseconds(s)}\n" for s in stamps))
        (temp_dir / "reports" / "ingest.json").write_text(
            json.dumps(
                {
                    "source": str(sequence_dir.resolve()),
                    "calib": str(calib_dir.resolve()),
                    "stamps": [stamps[0], stamps[-1]],
                    "lidar_points": points,
                    "cameras": notes,
                    "cameras_skipped": [c for c in CAMERAS if c not in cameras],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    scene_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_dir), str(scene_dir))
    manifest = build_manifest(
        scene_dir,
        scene_id=scene_id,
        dataset=DATASET,
        sequence_id=sequence,
        camera_ids=cameras,
        converter=CONVERTER,
        lidar_ids=[LIDAR_ID],
        data_license=DATA_LICENSE,
        camera_models=models,
        camera_max_angles=max_angles,
    )
    files.dump(manifest, scene_dir / "manifest.yaml")
    report = validate_scene(scene_dir, manifest)
    report = ValidationReport(
        schema_version=report.schema_version,
        scene_id=report.scene_id,
        created_utc=report.created_utc,
        summary=ReportSummary(
            errors=sum(check.severity == "error" for check in report.checks),
            warnings=sum(check.severity == "warning" for check in report.checks),
        ),
        checks=report.checks,
    )
    files.dump(report, scene_dir / "reports" / "validation.json")
    return scene_dir
