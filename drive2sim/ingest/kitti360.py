"""Converter KITTI-360: chuyển một chuỗi KITTI-360 sang dữ liệu chuẩn.

Một chuỗi KITTI-360 có 2 camera pinhole, 2 camera mắt cá và LiDAR Velodyne.

DriveStudio không có script tiền xử lý cho KITTI-360, nên đây là converter tự viết (ARCHITECTURE
mục 3.1, 4.2). Theo ARCHITECTURE mục 2.4, nó ghi theo **khung thư mục KITTI của DriveStudio**
(``drive2sim/recon/preprocess/kitti/process.py``): xe (``ego``) trùng hệ LiDAR Velodyne, nên
LiDAR chép nguyên từng file ``.bin`` (``float32`` x, y, z, cường độ, hệ Velodyne), ``ego_pose/``
là pose Velodyne → world theo khung, ``extrinsics/<camera>.txt`` là camera → Velodyne (trục
OpenCV), cố định. Khác KITTI ở intrinsics của camera mắt cá (CONTRACTS mục 2.1).

Dữ liệu chỉ lấy từ file chính thức của KITTI-360 (quy ước của ``kitti360Scripts``):
- ``calibration/calib_cam_to_pose.txt``: camera (chưa rectify) → hệ pose (IMU/GPS);
- ``calibration/calib_cam_to_velo.txt``: camera 0 (chưa rectify) → Velodyne;
- ``calibration/perspective.txt``: ``R_rect_0k``, ``P_rect_0k`` của hai camera pinhole (ảnh trong
  ``data_rect/`` đã rectify, méo bằng 0);
- ``calibration/image_02.yaml``, ``image_03.yaml``: mô hình Mei của hai camera mắt cá;
- ``data_poses.zip`` (hoặc ``data_poses/``): ``<chuỗi>/poses.txt``, pose → world theo khung;
  ``cam0_to_world.txt`` chỉ dùng để kiểm tra chéo;
- ``data_2d_raw``, ``data_3d_raw``: file ``.tar`` đọc thẳng bằng ``tarfile`` hoặc thư mục đã giải
  nén; không bao giờ giải nén vào thư mục nguồn.

**Không bao giờ đọc file ``.pkl``** (ví dụ ``ImageSets-rosa-6class/``): đó là file pickle từ một
bản sao bên thứ ba, ``pickle.load`` chạy được mã tuỳ ý.

File này chứa:
- ``DATASET``, ``CAMERAS``, ``LIDAR_ID``: tên dataset, camera và LiDAR ghi vào manifest.
- ``Kitti360Options``: lựa chọn khi ingest (khung đầu, số khung, camera, khoảng hở tối đa).
- ``find_sequences``: các chuỗi có ảnh trong thư mục nguồn (cho ``source inspect``).
- ``read_calibration``, ``read_poses``: đọc calib và pose chính thức.
- ``rig_extrinsics``, ``camera_intrinsics``: dựng extrinsics (camera → Velodyne) và intrinsics.
- ``select_frames``: chọn đoạn khung liên tục theo pose có sẵn (như ``--max-gap`` của teammate).
- ``ingest_kitti360``: hàm chính mà ``drive2sim/ingest/dispatch.py`` gọi.

Chuỗi biến đổi đã kiểm trên dữ liệu thật (chiếu LiDAR lên ảnh, so ``cam0_to_world.txt``): xem
``docs/datasets/kitti360.md`` và ``drive2sim/scripts/check_kitti360_projection.py``.
Viết bằng Python thuần (không numpy) để chạy trong môi trường nhẹ của CLI.
"""

import re
import shutil
import tarfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from drive2sim.core import files, identifiers
from drive2sim.core.camera import (
    MEI_FIT_MAX_ANGLE,
    fit_mei_to_fisheye,
    mei_fit_error_px,
)
from drive2sim.core.schema import Check, ReportSummary, ValidationReport
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest.drivestudio import IngestError, default_scene_id
from drive2sim.ingest.manifest import build_manifest
from drive2sim.ingest.validate import validate_scene

DATASET = "kitti360"
CONVERTER = "kitti360"
DATA_LICENSE = "CC BY-NC-SA 3.0"
# Id camera giữ nguyên tên thư mục của KITTI-360 (CONTRACTS mục 10.1); thứ tự là chỉ số camera.
CAMERAS: tuple[str, ...] = ("image_00", "image_01", "image_02", "image_03")
PINHOLE_CAMERAS = frozenset({"image_00", "image_01"})
# Ảnh pinhole đã rectify nằm trong data_rect/, ảnh mắt cá trong data_rgb/.
IMAGE_SUBDIRS = {"image_00": "data_rect", "image_01": "data_rect"}
FISHEYE_SUBDIR = "data_rgb"
LIDAR_ID = "velodyne"
# Khoảng cách lớn nhất (theo số hiệu khung gốc) giữa hai khung liền nhau trong cảnh, như
# KITTI360_MAX_GAP của teammate; bản tải về của chuỗi 0003 cách nhau 5 khung.
DEFAULT_MAX_GAP = 5
# Camera KITTI-360 chạy 10 Hz; dùng khi nguồn không có timestamps.txt.
NOMINAL_FRAME_US = 100_000
SEQUENCE_RE = re.compile(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync")
_FRAME_RE = re.compile(r"(\d{10})\.(png|bin)$")

Matrix = list[list[float]]


@dataclass(frozen=True)
class Kitti360Options:
    """Lựa chọn khi ingest một chuỗi KITTI-360.

    Attributes:
        start_frame: Số hiệu khung gốc nhỏ nhất được xét (như ``KITTI360_START``).
        num_frames: Số khung tối đa; ``None`` là lấy hết đoạn liên tục dài nhất.
        cameras: Camera cần ghi, theo thứ tự chỉ số trong cảnh; mặc định cả bốn.
        max_gap: Hai khung liền nhau cách nhau quá số này (theo số hiệu gốc) thì cắt đoạn.
    """

    start_frame: int = 0
    num_frames: Optional[int] = None
    cameras: tuple[str, ...] = field(default=CAMERAS)
    max_gap: int = DEFAULT_MAX_GAP


@dataclass
class Kitti360Calibration:
    """Calib chính thức của KITTI-360, đọc từ thư mục ``calibration/``.

    Attributes:
        cam_to_pose: Camera (chưa rectify) → hệ pose, ma trận 4×4 theo camera.
        cam0_to_velo: Camera 0 (chưa rectify) → Velodyne, 4×4.
        rect: ``R_rect_0k`` dạng 4×4 cho ``image_00``, ``image_01``.
        projection: ``P_rect_0k`` dạng 3×4 cho ``image_00``, ``image_01``.
        mei: Tham số Mei của ``image_02``, ``image_03``: ``xi``, ``k1``, ``k2``, ``p1``, ``p2``,
            ``gamma1``, ``gamma2``, ``u0``, ``v0``, ``width``, ``height``.
    """

    cam_to_pose: dict[str, Matrix]
    cam0_to_velo: Matrix
    rect: dict[str, Matrix]
    projection: dict[str, Matrix]
    mei: dict[str, dict[str, float]]


# ---------------------------------------------------------------- ma trận 4×4 bằng Python thuần


def _identity() -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    """Nhân hai ma trận 4×4."""
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _inverse(m: Matrix) -> Matrix:
    """Nghịch đảo ma trận 4×4 bằng khử Gauss–Jordan có chọn trụ.

    Không dùng ``[Rᵀ, −Rᵀt]``: các ma trận calib của KITTI-360 không trực chuẩn tuyệt đối, và
    ``kitti360Scripts`` dùng ``np.linalg.inv``.
    """
    a = [list(row) + _identity()[i] for i, row in enumerate(m)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise IngestError("ma trận calib suy biến")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for r in range(4):
            if r != col and a[r][col] != 0.0:
                factor = a[r][col]
                a[r] = [v - factor * p for v, p in zip(a[r], a[col])]
    return [row[4:] for row in a]


def _from_values(values: Sequence[float]) -> Matrix:
    """Dựng 4×4 từ 12 số (ma trận 3×4 theo hàng) hoặc 9 số (3×3); hàng cuối ``0 0 0 1``."""
    m = _identity()
    if len(values) == 12:
        for i in range(3):
            m[i] = [float(v) for v in values[4 * i : 4 * i + 4]]
    elif len(values) == 9:
        for i in range(3):
            m[i][:3] = [float(v) for v in values[3 * i : 3 * i + 3]]
    else:
        raise IngestError(f"cần 9 hoặc 12 số cho một ma trận, có {len(values)}")
    return m


def _matrix_text(m: Matrix) -> str:
    """Viết ma trận 4×4 theo dạng ``np.savetxt`` (đọc lại được bằng ``np.loadtxt``)."""
    return "".join(" ".join(f"{v:.18e}" for v in row) + "\n" for row in m)


# ---------------------------------------------------------------- đọc file nguồn


def _read_records(text: str) -> dict[str, list[float]]:
    """Đọc file dạng ``khoá: số số ...`` (``perspective.txt``, ``calib_cam_to_pose.txt``)."""
    records = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep or not value.strip():
            continue
        try:
            records[key.strip()] = [float(v) for v in value.split()]
        except ValueError:
            continue  # dòng chữ như calib_time
    return records


def _read_mei(text: str, where: str) -> dict[str, float]:
    """Đọc tham số Mei trong ``image_02.yaml``, ``image_03.yaml`` (YAML của OpenCV).

    Không dùng ``yaml.safe_load`` vì dòng đầu ``%YAML:1.0`` không hợp chuẩn YAML 1.1 của PyYAML;
    chỉ đọc các khoá số cần dùng.
    """
    keys = ("xi", "k1", "k2", "p1", "p2", "gamma1", "gamma2", "u0", "v0")
    values = {}
    for key in (*keys, "image_width", "image_height"):
        match = re.search(rf"^\s*{key}\s*:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
        if not match:
            raise IngestError(f"{where} thiếu {key}")
        values[key] = float(match.group(1))
    values["width"] = values.pop("image_width")
    values["height"] = values.pop("image_height")
    return values


def read_calibration(raw_dir: Path) -> Kitti360Calibration:
    """Đọc calib chính thức trong ``<nguồn>/calibration/``.

    Args:
        raw_dir: Thư mục gốc KITTI-360.

    Returns:
        Calib đã đọc.

    Raises:
        IngestError: Nếu thiếu file hoặc file sai dạng.
    """
    calib_dir = raw_dir / "calibration"
    try:
        to_pose = _read_records((calib_dir / "calib_cam_to_pose.txt").read_text())
        to_velo = [float(v) for v in (calib_dir / "calib_cam_to_velo.txt").read_text().split()]
        perspective = _read_records((calib_dir / "perspective.txt").read_text())
        mei = {
            camera: _read_mei((calib_dir / f"{camera}.yaml").read_text(), f"{camera}.yaml")
            for camera in ("image_02", "image_03")
        }
    except OSError as error:
        raise IngestError(f"không đọc được calib trong {calib_dir}: {error}") from None
    rect, projection = {}, {}
    for camera in sorted(PINHOLE_CAMERAS):
        index = camera[-2:]
        try:
            rect[camera] = _from_values(perspective[f"R_rect_{index}"])
            p = perspective[f"P_rect_{index}"]
        except KeyError as error:
            raise IngestError(f"perspective.txt thiếu {error}") from None
        projection[camera] = [p[0:4], p[4:8], p[8:12]]
    missing = [camera for camera in CAMERAS if camera not in to_pose]
    if missing:
        raise IngestError(f"calib_cam_to_pose.txt thiếu {', '.join(missing)}")
    return Kitti360Calibration(
        cam_to_pose={camera: _from_values(to_pose[camera]) for camera in CAMERAS},
        cam0_to_velo=_from_values(to_velo),
        rect=rect,
        projection=projection,
        mei=mei,
    )


def _pose_text(raw_dir: Path, sequence: str, name: str) -> Optional[str]:
    """Đọc ``<chuỗi>/<name>`` trong ``data_poses.zip`` hoặc thư mục ``data_poses/``."""
    folder = raw_dir / "data_poses" / sequence / name
    if folder.is_file():
        return folder.read_text()
    archive = raw_dir / "data_poses.zip"
    if archive.is_file():
        with zipfile.ZipFile(archive) as zipped:
            for member in (f"{sequence}/{name}", f"data_poses/{sequence}/{name}"):
                if member in zipped.namelist():
                    return zipped.read(member).decode()
    return None


def _read_frame_matrices(text: str, count: int) -> dict[int, Matrix]:
    """Đọc file ``<số hiệu khung> <count số>`` mỗi dòng thành số hiệu → ma trận 4×4."""
    result = {}
    for line in text.splitlines():
        values = line.split()
        if len(values) == count + 1:
            result[int(values[0])] = _from_values([float(v) for v in values[1:]][:12])
    return result


def read_poses(raw_dir: Path, sequence: str) -> dict[int, Matrix]:
    """Đọc ``poses.txt`` của một chuỗi: số hiệu khung → pose (hệ IMU/GPS) → world.

    Mỗi dòng là số hiệu khung rồi 12 số của ma trận 3×4 (``kitti360Scripts``). Chỉ một phần các
    khung có pose; khung không có pose không dùng được.

    Raises:
        IngestError: Nếu không có ``poses.txt`` của chuỗi này.
    """
    text = _pose_text(raw_dir, sequence, "poses.txt")
    if text is None:
        raise IngestError(f"không thấy {sequence}/poses.txt trong data_poses.zip hay data_poses/")
    return _read_frame_matrices(text, 12)


def _read_cam0_to_world(raw_dir: Path, sequence: str) -> dict[int, Matrix]:
    """Đọc ``cam0_to_world.txt`` (camera 0 đã rectify → world, 16 số mỗi dòng) nếu có."""
    text = _pose_text(raw_dir, sequence, "cam0_to_world.txt")
    return _read_frame_matrices(text, 16) if text else {}


class _FrameSource:
    """Các file theo khung của một cảm biến, trong thư mục đã giải nén hoặc trong file ``.tar``.

    Đọc ``.tar`` bằng ``tarfile`` theo chế độ truy cập ngẫu nhiên (file ``.tar`` không nén), nên
    không giải nén gì ra đĩa.

    Attributes:
        label: Tên hiện trong thông báo lỗi.
        frames: Số hiệu khung → file trong thư mục, hoặc ``TarInfo`` trong file ``.tar``.
    """

    def __init__(self, label: str, directory: Optional[Path], tar_path: Optional[Path]) -> None:
        self.label = label
        self.frames: dict[int, Union[Path, tarfile.TarInfo]] = {}
        self.extra: dict[str, Union[Path, tarfile.TarInfo]] = {}
        self._tar: Optional[tarfile.TarFile] = None
        if directory is not None and directory.is_dir():
            for path in directory.iterdir():
                match = _FRAME_RE.search(path.name)
                if match:
                    self.frames[int(match.group(1))] = path
            timestamps = directory.parent / "timestamps.txt"
            if timestamps.is_file():
                self.extra["timestamps.txt"] = timestamps
        elif tar_path is not None and tar_path.is_file():
            self._tar = tarfile.open(tar_path, "r:")
            for member in self._tar.getmembers():
                match = _FRAME_RE.search(member.name)
                if member.isfile() and match:
                    self.frames[int(match.group(1))] = member
                elif member.isfile() and member.name.endswith("timestamps.txt"):
                    self.extra["timestamps.txt"] = member

    def read(self, item: Union[Path, tarfile.TarInfo]) -> bytes:
        """Đọc nội dung một file (trong thư mục hoặc trong ``.tar``)."""
        if isinstance(item, Path):
            return item.read_bytes()
        assert self._tar is not None
        stream = self._tar.extractfile(item)
        if stream is None:
            raise IngestError(f"{self.label}: không đọc được {item.name}")
        return stream.read()

    def copy(self, frame: int, target: Path) -> None:
        """Chép file của một khung tới ``target``, không giải mã ảnh hay điểm."""
        item = self.frames[frame]
        if isinstance(item, Path):
            shutil.copyfile(item, target)
        else:
            target.write_bytes(self.read(item))

    def close(self) -> None:
        """Đóng file ``.tar`` nếu đang mở."""
        if self._tar is not None:
            self._tar.close()


def _camera_source(raw_dir: Path, sequence: str, camera: str) -> _FrameSource:
    """Mở nguồn ảnh của một camera.

    Nguồn là thư mục ``data_2d_raw/<chuỗi>/<camera>/<data_rect|data_rgb>/`` hoặc file
    ``data_2d_raw/<chuỗi>_<camera>.tar``.
    """
    subdir = IMAGE_SUBDIRS.get(camera, FISHEYE_SUBDIR)
    return _FrameSource(
        camera,
        raw_dir / "data_2d_raw" / sequence / camera / subdir,
        raw_dir / "data_2d_raw" / f"{sequence}_{camera}.tar",
    )


def _lidar_source(raw_dir: Path, sequence: str) -> _FrameSource:
    """Mở nguồn LiDAR: ``data_3d_raw/<chuỗi>/velodyne_points/data/`` hoặc ``<chuỗi>.tar``."""
    return _FrameSource(
        LIDAR_ID,
        raw_dir / "data_3d_raw" / sequence / "velodyne_points" / "data",
        raw_dir / "data_3d_raw" / f"{sequence}.tar",
    )


def find_sequences(raw_dir: Path) -> list[str]:
    """Liệt kê các chuỗi có ảnh trong ``data_2d_raw/`` (thư mục hoặc file ``.tar``).

    Args:
        raw_dir: Thư mục gốc KITTI-360.

    Returns:
        Tên chuỗi, sắp theo tên, ví dụ ``["2013_05_28_drive_0003_sync"]``.
    """
    names = set()
    image_root = raw_dir / "data_2d_raw"
    if image_root.is_dir():
        for path in image_root.iterdir():
            match = SEQUENCE_RE.match(path.name)
            if match and (path.suffix == ".tar" or path.is_dir()):
                names.add(match.group(0))
    return sorted(names)


def has_lidar(raw_dir: Path, sequence: str) -> bool:
    """Cho biết nguồn có LiDAR Velodyne của chuỗi không (thư mục hoặc ``.tar``)."""
    return (raw_dir / "data_3d_raw" / f"{sequence}.tar").is_file() or (
        raw_dir / "data_3d_raw" / sequence / "velodyne_points" / "data"
    ).is_dir()


# ---------------------------------------------------------------- calib → dữ liệu chuẩn


def T_velo_pose(calib: Kitti360Calibration) -> Matrix:
    """Velodyne → hệ pose: ``cam_to_pose[image_00] · (cam0_to_velo)⁻¹`` (kitti360Scripts)."""
    return _matmul(calib.cam_to_pose["image_00"], _inverse(calib.cam0_to_velo))


def rig_extrinsics(calib: Kitti360Calibration) -> dict[str, Matrix]:
    """Dựng extrinsics camera → Velodyne (hệ ``ego`` của khung KITTI) cho mọi camera.

    Camera ``k`` chưa rectify → Velodyne là ``(T_velo_pose)⁻¹ · cam_to_pose[k]``. Hai camera
    pinhole dùng ảnh đã rectify: điểm trong hệ rectify là ``R_rect · điểm chưa rectify``, nên nhân
    thêm ``R_rect⁻¹`` bên phải (``kitti360Scripts``, ``CameraPerspective``). Trục camera theo
    OpenCV (x phải, y xuống, z tới trước), đúng như KITTI-360 và DriveStudio.

    Returns:
        ``camera_id`` → ma trận 4×4 camera → Velodyne.
    """
    pose_velo = _inverse(T_velo_pose(calib))
    result = {}
    for camera in CAMERAS:
        m = _matmul(pose_velo, calib.cam_to_pose[camera])
        if camera in calib.rect:
            m = _matmul(m, _inverse(calib.rect[camera]))
        result[camera] = m
    return result


def camera_intrinsics(calib: Kitti360Calibration, camera: str) -> tuple[str, list[float], dict]:
    """Dựng mô hình camera và nội dung ``intrinsics/<chỉ số>.txt`` cho một camera.

    - ``image_00``, ``image_01``: ``opencv_pinhole`` trên ảnh đã rectify; 9 số
      ``fx fy cx cy k1 k2 p1 p2 k3`` như khung Argoverse/KITTI, méo bằng 0. ``fx, fy, cx, cy``
      lấy từ ``P_rect_0k`` (cột thứ tư của ``P_rect_01`` là đường cơ sở stereo, đã nằm trong
      extrinsics nên bỏ).
    - ``image_02``, ``image_03``: ``opencv_fisheye``; 8 số ``fx fy cx cy k1 k2 k3 k4`` (thứ tự của
      ``cv2.fisheye``), khớp từ Mei bằng ``fit_mei_to_fisheye``.

    Args:
        calib: Calib đã đọc.
        camera: ``camera_id``.

    Returns:
        ``(mô hình, các số, thông tin khớp)``; thông tin khớp rỗng với camera pinhole, với camera
        mắt cá là tham số Mei gốc cộng sai số khớp (pixel) để ghi vào báo cáo kiểm tra.
    """
    if camera in PINHOLE_CAMERAS:
        p = calib.projection[camera]
        return "opencv_pinhole", [p[0][0], p[1][1], p[0][2], p[1][2], 0, 0, 0, 0, 0], {}
    mei = calib.mei[camera]
    fx, fy, coeffs = fit_mei_to_fisheye(
        mei["xi"], mei["k1"], mei["k2"], mei["gamma1"], mei["gamma2"]
    )
    error = mei_fit_error_px(
        mei["xi"], mei["k1"], mei["k2"], max(fx, fy) * (1 + mei["xi"]), coeffs
    )
    info = {**mei, "fit_max_angle": MEI_FIT_MAX_ANGLE, "fit_error_px": error}
    return "opencv_fisheye", [fx, fy, mei["u0"], mei["v0"], *coeffs], info


def select_frames(
    candidates: Sequence[int],
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    max_gap: int = DEFAULT_MAX_GAP,
) -> list[int]:
    """Chọn một đoạn khung liền nhau, như ``select_items`` của teammate.

    Khung xét là các số hiệu ``>= start_frame`` trong ``candidates`` (đã có ảnh mọi camera, LiDAR
    và pose). Hai khung liền nhau cách nhau quá ``max_gap`` thì cắt thành đoạn mới, để cảnh không
    nhảy cóc qua chỗ thiếu pose (chuỗi 0003 thiếu pose ở 142–152 và 162). Chọn đoạn đầu tiên có ít
    nhất ``num_frames`` khung; không đoạn nào đủ thì lấy đoạn dài nhất. ``num_frames`` là
    ``None`` thì lấy đoạn dài nhất.

    Args:
        candidates: Số hiệu khung gốc dùng được.
        start_frame: Số hiệu nhỏ nhất được xét.
        num_frames: Số khung tối đa; ``None`` là cả đoạn dài nhất.
        max_gap: Khoảng cách lớn nhất giữa hai khung liền nhau, theo số hiệu gốc.

    Returns:
        Số hiệu khung gốc đã chọn, tăng dần.

    Raises:
        IngestError: Nếu không có khung nào từ ``start_frame``.
    """
    pool = sorted(frame for frame in set(candidates) if frame >= start_frame)
    runs: list[list[int]] = []
    for frame in pool:
        if runs and frame - runs[-1][-1] <= max_gap:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    if not runs:
        raise IngestError(f"không có khung nào có đủ ảnh, LiDAR và pose từ khung {start_frame}")
    if num_frames is None:
        return max(runs, key=len)
    run = next((r for r in runs if len(r) >= num_frames), max(runs, key=len))
    return run[:num_frames]


def _timestamps_us(source: _FrameSource, frames: Sequence[int]) -> tuple[list[int], bool]:
    """Mốc thời gian (micro giây) của từng khung đã chọn.

    ``timestamps.txt`` của KITTI-360 có một dòng ``YYYY-MM-DD HH:MM:SS.nnnnnnnnn`` cho mỗi số hiệu
    khung (dòng thứ ``i`` là khung ``i``); giờ được coi là UTC. Nguồn không có file này (bản tải
    về chỉ có ảnh) thì dùng ``số hiệu × 100 ms`` (camera 10 Hz).

    Returns:
        ``(mốc thời gian, True nếu là mốc danh nghĩa)``.
    """
    item = source.extra.get("timestamps.txt")
    if item is not None:
        lines = source.read(item).decode().splitlines()
        if all(frame < len(lines) for frame in frames):
            stamps = []
            for frame in frames:
                date, _, fraction = lines[frame].strip().partition(".")
                base = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                micro = int((fraction + "000000")[:6])
                stamps.append(int(base.timestamp()) * 1_000_000 + micro)
            return stamps, False
    return [frame * NOMINAL_FRAME_US for frame in frames], True


def _check(check_id: str, severity: str, message: str, details: dict) -> Check:
    return Check(check_id=check_id, severity=severity, message=message, details=details)  # type: ignore[arg-type]


def _max_abs_difference(a: Matrix, b: Matrix) -> tuple[float, float]:
    """Sai khác lớn nhất giữa hai pose: phần xoay (phần tử 3×3) và phần tịnh tiến (mét)."""
    rotation = max(abs(a[i][j] - b[i][j]) for i in range(3) for j in range(3))
    translation = max(abs(a[i][3] - b[i][3]) for i in range(3))
    return rotation, translation


def ingest_kitti360(
    raw_dir: Path,
    workdir: Workdir,
    sequence: str,
    scene_id: Optional[str] = None,
    options: Optional[Kitti360Options] = None,
) -> Path:
    """Ingest một chuỗi KITTI-360 vào ``data/<scene_id>/``.

    Các bước: đọc calib và pose chính thức; liệt kê khung có ảnh của mọi camera đã chọn, có LiDAR
    và có pose; chọn một đoạn liền nhau (``select_frames``); ghi vào thư mục tạm
    ``data/.ingest/<scene_id>/``: ``images/<khung>_<chỉ số camera>.png`` (chép nguyên PNG),
    ``lidar/<khung>.bin`` (chép nguyên, hệ Velodyne), ``ego_pose/<khung>.txt`` (Velodyne → world,
    world là hệ Velodyne ở khung đầu của cảnh), ``extrinsics/<chỉ số>.txt``,
    ``intrinsics/<chỉ số>.txt``, ``frame_ids.txt`` (số hiệu khung gốc theo thứ tự khung),
    ``timestamps.txt`` (micro giây); rồi chuyển sang ``data/<scene_id>/`` và ghi
    ``manifest.yaml`` và ``reports/validation.json``.

    Báo cáo kiểm tra có thêm: so pose camera 0 của cảnh với ``cam0_to_world.txt`` chính thức,
    sai số khớp Mei → OpenCV fisheye, mốc thời gian danh nghĩa, chỗ hở giữa các khung.

    Args:
        raw_dir: Thư mục gốc KITTI-360 (có ``calibration/``, ``data_poses.zip``,
            ``data_2d_raw/``, ``data_3d_raw/``).
        workdir: Thư mục làm việc.
        sequence: Tên chuỗi, ví dụ ``2013_05_28_drive_0003_sync``.
        scene_id: Định danh cảnh; mặc định ``kitti360_<chuỗi>``.
        options: Lựa chọn khung và camera; mặc định ``Kitti360Options()``.

    Returns:
        Thư mục cảnh ``data/<scene_id>/``.

    Raises:
        IngestError: Nếu cảnh đã có, thiếu calib, pose, ảnh, hoặc không có khung nào dùng được.
    """
    options = options or Kitti360Options()
    unknown = [camera for camera in options.cameras if camera not in CAMERAS]
    if unknown or not options.cameras:
        raise IngestError(
            f"camera không có trong KITTI-360: {', '.join(unknown) or '(trống)'}; "
            f"có: {', '.join(CAMERAS)}"
        )
    scene_id = scene_id or default_scene_id(DATASET, sequence)
    identifiers.check_scene_id(scene_id)
    scene_dir = workdir.scene_dir(scene_id)
    if scene_dir.exists():
        raise IngestError(
            f"cảnh {scene_id} đã có ở {workdir.relative(scene_dir)}; xoá cảnh đó hoặc đặt "
            "scene_id khác"
        )
    calib = read_calibration(raw_dir)
    poses = read_poses(raw_dir, sequence)
    cameras = list(options.cameras)
    sources = {camera: _camera_source(raw_dir, sequence, camera) for camera in cameras}
    lidar = _lidar_source(raw_dir, sequence)
    temp_dir = workdir.data_dir / ".ingest" / scene_id
    try:
        for camera, source in sources.items():
            if not source.frames:
                raise IngestError(f"không thấy ảnh {camera} của {sequence} trong data_2d_raw/")
        if not lidar.frames:
            raise IngestError(f"không thấy LiDAR của {sequence} trong data_3d_raw/")
        candidates = set(poses) & set(lidar.frames)
        for source in sources.values():
            candidates &= set(source.frames)
        frames = select_frames(
            sorted(candidates), options.start_frame, options.num_frames, options.max_gap
        )

        shutil.rmtree(temp_dir, ignore_errors=True)
        for name in ("images", "lidar", "ego_pose", "extrinsics", "intrinsics", "reports"):
            (temp_dir / name).mkdir(parents=True)
        extrinsics = rig_extrinsics(calib)
        models, fits = [], {}
        for index, camera in enumerate(cameras):
            model, values, fit = camera_intrinsics(calib, camera)
            models.append(model)
            if fit:
                fits[camera] = fit
            (temp_dir / "intrinsics" / f"{index}.txt").write_text(
                "".join(f"{v:.18e}\n" for v in values)
            )
            (temp_dir / "extrinsics" / f"{index}.txt").write_text(_matrix_text(extrinsics[camera]))
        # T_velo_world của từng khung = pose → world · Velodyne → pose; world của cảnh là hệ
        # Velodyne ở khung đầu, để toạ độ nhỏ (CONTRACTS mục 1).
        velo_pose = T_velo_pose(calib)
        absolute = {frame: _matmul(poses[frame], velo_pose) for frame in frames}
        world_from_global = _inverse(absolute[frames[0]])
        ego_poses = [_matmul(world_from_global, absolute[frame]) for frame in frames]
        for t, frame in enumerate(frames):
            ego = ego_poses[t]
            (temp_dir / "ego_pose" / f"{t:03d}.txt").write_text(_matrix_text(ego))
            lidar.copy(frame, temp_dir / "lidar" / f"{t:03d}.bin")
            for index, camera in enumerate(cameras):
                sources[camera].copy(frame, temp_dir / "images" / f"{t:03d}_{index}.png")
        stamps, nominal = _timestamps_us(sources[cameras[0]], frames)
        (temp_dir / "frame_ids.txt").write_text("".join(f"{frame}\n" for frame in frames))
        (temp_dir / "timestamps.txt").write_text("".join(f"{stamp}\n" for stamp in stamps))
        extra_checks = _extra_checks(
            raw_dir, sequence, frames, ego_poses, world_from_global, extrinsics, fits, nominal
        )
    finally:
        for source in (*sources.values(), lidar):
            source.close()

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
    )
    files.dump(manifest, scene_dir / "manifest.yaml")
    report = validate_scene(scene_dir, manifest)
    checks = [*report.checks, *extra_checks]
    report = ValidationReport(
        schema_version=report.schema_version,
        scene_id=report.scene_id,
        created_utc=report.created_utc,
        summary=ReportSummary(
            errors=sum(check.severity == "error" for check in checks),
            warnings=sum(check.severity == "warning" for check in checks),
        ),
        checks=checks,
    )
    files.dump(report, scene_dir / "reports" / "validation.json")
    return scene_dir


def _extra_checks(
    raw_dir: Path,
    sequence: str,
    frames: list[int],
    ego_poses: list[Matrix],
    world_from_global: Matrix,
    extrinsics: dict[str, Matrix],
    fits: dict[str, dict],
    nominal_timestamps: bool,
) -> list[Check]:
    """Các mục kiểm tra riêng của KITTI-360, thêm vào ``reports/validation.json``.

    - ``pose.cam0_to_world``: pose camera 0 (đã rectify) dựng từ ``poses.txt`` và calib, so với
      ``cam0_to_world.txt`` chính thức ở mọi khung có cả hai; lệch quá 5 mm hoặc ``1e-4`` (phần tử
      ma trận xoay) là lỗi, vì khi đó chuỗi biến đổi sai. ``cam0_to_world.txt`` chỉ ghi 7 chữ số
      có nghĩa (toạ độ cỡ 3000 m làm tròn tới 1 mm), nên ngưỡng không thể nhỏ hơn cỡ đó.
    - ``camera.fisheye_fit``: sai số khớp Mei → OpenCV fisheye của từng camera mắt cá.
    - ``camera.fisheye_fov``: tia lệch trục hơn ``MEI_FIT_MAX_ANGLE`` không vẽ được; trainer phải
      che vùng đó.
    - ``timestamps.nominal``: nguồn không có ``timestamps.txt``.
    - ``frames.gaps``: các chỗ hai khung liền nhau cách nhau nhiều hơn khung gốc liền kề.
    """
    checks = []
    official = _read_cam0_to_world(raw_dir, sequence)
    common = [t for t, frame in enumerate(frames) if frame in official]
    if common:
        worst_rotation = worst_translation = 0.0
        for t in common:
            # Hai phía cùng là camera 0 (đã rectify) → world của cảnh: phía tool là
            # ego_pose · extrinsics; phía chính thức là cam0_to_world đưa về gốc của cảnh.
            ours = _matmul(ego_poses[t], extrinsics["image_00"])
            theirs = _matmul(world_from_global, official[frames[t]])
            rotation, translation = _max_abs_difference(ours, theirs)
            worst_rotation = max(worst_rotation, rotation)
            worst_translation = max(worst_translation, translation)
        bad = worst_rotation > 1e-4 or worst_translation > 5e-3
        message = (
            "pose camera 0 lệch so với cam0_to_world.txt"
            if bad
            else "pose camera 0 khớp cam0_to_world.txt chính thức"
        )
        checks.append(
            _check(
                "pose.cam0_to_world",
                "error" if bad else "info",
                message,
                {
                    "frames_compared": len(common),
                    "max_rotation_diff": worst_rotation,
                    "max_translation_diff_m": worst_translation,
                },
            )
        )
    for camera, fit in fits.items():
        checks.append(
            _check(
                "camera.fisheye_fit",
                "info",
                f"{camera}: Mei khớp sang OpenCV fisheye, sai số tối đa "
                f"{fit['fit_error_px']:.4f} px tới {MEI_FIT_MAX_ANGLE} rad; bỏ méo tiếp tuyến "
                "p1, p2",
                {key: fit[key] for key in sorted(fit)},
            )
        )
    if fits:
        checks.append(
            _check(
                "camera.fisheye_fov",
                "info",
                f"camera mắt cá nhìn rộng hơn 180 độ; tia lệch trục hơn {MEI_FIT_MAX_ANGLE} rad "
                "không vẽ được bằng 3DGUT của gsplat và phải được che khi train, đánh giá",
                {"max_angle": MEI_FIT_MAX_ANGLE, "cameras": sorted(fits)},
            )
        )
    if nominal_timestamps:
        checks.append(
            _check(
                "timestamps.nominal",
                "info",
                "nguồn không có timestamps.txt; mốc thời gian là số hiệu khung × 100 ms",
                {},
            )
        )
    gaps = [[a, b] for a, b in zip(frames, frames[1:]) if b - a > 1]
    if gaps:
        steps = sorted({b - a for a, b in gaps})
        checks.append(
            _check(
                "frames.gaps",
                "info",
                f"khung gốc không liền nhau (bước {', '.join(map(str, steps))}); "
                "xem frame_ids.txt",
                {"count": len(gaps), "first": gaps[:10]},
            )
        )
    return checks
