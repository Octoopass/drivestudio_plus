r"""Kiểm tra chuỗi hình học KITTI-360 của converter bằng cách chiếu LiDAR thật lên ảnh thật.

Với mỗi khung được chọn của một cảnh đã ingest (``drive2sim/ingest/kitti360.py``), điểm Velodyne
được chiếu vào cả bốn camera theo hai đường:

- **(a) chính thức**: calib và pose gốc trong thư mục nguồn KITTI-360, theo quy ước của
  ``kitti360Scripts`` (``helpers/project.py``). Velodyne → world là
  ``pose[f] · cam_to_pose[image_00] · calib_cam_to_velo⁻¹``; camera ``k`` → world là
  ``pose[f] · cam_to_pose[k]`` (nhân thêm ``R_rect⁻¹`` với camera pinhole). Pinhole chiếu bằng
  ``P_rect[:, :3]``; mắt cá chiếu bằng mô hình Mei đầy đủ (có méo tiếp tuyến ``p1``, ``p2``;
  ``kitti360Scripts`` bỏ hai hệ số này).
- **(b) dữ liệu chuẩn**: chỉ đọc thư mục cảnh: ``ego_pose/`` (Velodyne → world của cảnh),
  ``extrinsics/`` (camera → Velodyne), ``intrinsics/`` (pinhole hoặc OpenCV fisheye khớp từ Mei
  bằng ``fit_mei_to_fisheye``).

Hai đường phải cho cùng pixel: với pinhole lệch cỡ sai số làm tròn; với mắt cá lệch do phép khớp
đa thức và bỏ méo tiếp tuyến (chỉ so trong miền khớp ``MEI_FIT_MAX_ANGLE``).

Ngoài ra script đo **tỉ số cạnh** (không cần nhãn): gradient ngang của ảnh (Sobel) trung bình
tại các điểm LiDAR nằm ở mép trước của một bước nhảy độ sâu trên vòng quét, chia cho trung bình
tại mọi điểm chiếu vào ảnh. Hình học đúng thì viền vật trong LiDAR rơi vào viền vật trong ảnh, tỉ
số lớn hơn 1 rõ rệt; cùng phép đo với camera xoay lệch một góc nhỏ (``--perturb-deg``) cho tỉ số
thấp hơn. Với mắt cá chỉ xét điểm trong miền khớp (ngoài đó là mép ảnh có thân xe). Tỉ số của một
khung dao động theo cảnh (cây trong bóng râm, vạch kẻ đường sáng), nhất là ở mắt cá, nên script
in thêm tỉ số gộp mọi khung đã chọn của từng camera và thoát với mã 1 nếu tỉ số gộp của camera nào
không vượt ``--min-ratio``. Ảnh chồng điểm (màu theo độ sâu) ghi vào ``--out``.

Chạy (cần numpy, OpenCV; môi trường ``drive2sim``)::

    python -m drive2sim.scripts.check_kitti360_projection \
        --scene <workdir>/data/kitti360_2013_05_28_drive_0003_sync \
        --raw data/kitti360/raw --frame 0 15 --out <thư mục ảnh>

File này chứa:
- ``load_velodyne``, ``read_matrix``: đọc điểm ``.bin`` và ma trận ``np.savetxt``.
- ``transform``, ``yaw_rotation``: biến đổi điểm, phép xoay làm lệch camera.
- ``project_pinhole``, ``project_mei``, ``project_opencv_fisheye``: ba mô hình chiếu.
- ``depth_discontinuities``, ``edge_strength``, ``EdgeScore``, ``edge_score``: điểm ở mép bước
  nhảy độ sâu, độ mạnh cạnh ảnh và tỉ số cạnh.
- ``draw_overlay``: ảnh chồng điểm màu theo độ sâu.
- ``SceneData``, ``OfficialRig``: đọc thư mục cảnh và calib, pose chính thức.
- ``check_camera``: so (a) với (b) và đo tỉ số cạnh cho một camera ở một khung.
- ``main``: lệnh dòng lệnh.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from drive2sim.core import files
from drive2sim.core.camera import MEI_FIT_MAX_ANGLE
from drive2sim.core.schema import Manifest
from drive2sim.ingest.kitti360 import read_calibration, read_poses

DEFAULT_PERTURB_DEG = 1.0
DEFAULT_MIN_RATIO = 1.1
MAX_AZIMUTH_STEP_DEG = 0.5  # bước phương vị tối đa giữa hai điểm kề nhau trên một vòng quét
MIN_DEPTH_JUMP_M = 1.0
MIN_DEPTH_JUMP_REL = 0.2  # bước nhảy tối thiểu theo tỉ lệ độ sâu
MIN_DEPTH_M = 0.5


# ---------------------------------------------------------------- hình học cơ bản


def load_velodyne(path: Path) -> np.ndarray:
    """Đọc file ``.bin`` của Velodyne: ``float32`` x, y, z, cường độ; trả về ``(N, 3)`` float64."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)


def read_matrix(path: Path) -> np.ndarray:
    """Đọc ma trận 4×4 ghi bằng ``np.savetxt`` (``ego_pose/``, ``extrinsics/``)."""
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def as_array(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Ma trận dạng list của converter thành ``np.ndarray`` float64."""
    return np.asarray(matrix, dtype=np.float64)


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Áp phép biến đổi 4×4 lên điểm ``(N, 3)``."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def yaw_rotation(degrees: float) -> np.ndarray:
    """Phép xoay 4×4 quanh trục y của camera (trục OpenCV: y hướng xuống), tính theo độ."""
    a = np.deg2rad(degrees)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = np.cos(a), np.sin(a), -np.sin(a), np.cos(a)
    return m


def project_pinhole(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chiếu điểm trong hệ camera bằng ma trận nội ``k`` (3×3).

    Returns:
        ``(uv (N, 2), hợp lệ)``; hợp lệ là điểm nằm trước camera ít nhất ``MIN_DEPTH_M``.
    """
    valid = points[:, 2] > MIN_DEPTH_M
    z = np.where(valid, points[:, 2], 1.0)
    uvw = points @ k.T
    return uvw[:, :2] / z[:, None], valid


def project_mei(
    points: np.ndarray, mei: dict[str, float], tangential: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Chiếu điểm trong hệ camera bằng mô hình Mei (C. Mei, P. Rives, ICRA 2007).

    Điểm được đưa lên mặt cầu đơn vị ``(xs, ys, zs)``, rồi ``x = xs / (zs + ξ)``,
    ``y = ys / (zs + ξ)``; méo xuyên tâm ``1 + k1·r² + k2·r⁴`` với ``r² = x² + y²``; méo tiếp
    tuyến ``(2·p1·x·y + p2·(r² + 2x²), p1·(r² + 2y²) + 2·p2·x·y)``; cuối cùng
    ``u = γ1·x + u0``, ``v = γ2·y + v0``.

    Args:
        points: Điểm ``(N, 3)`` trong hệ camera (trục OpenCV).
        mei: ``xi``, ``k1``, ``k2``, ``p1``, ``p2``, ``gamma1``, ``gamma2``, ``u0``, ``v0``.
        tangential: ``False`` thì bỏ ``p1``, ``p2`` như ``kitti360Scripts``.

    Returns:
        ``(uv (N, 2), hợp lệ)``; hợp lệ là điểm cách tâm ít nhất ``MIN_DEPTH_M`` và chiếu được
        (``zs + ξ > 0``).
    """
    norm = np.linalg.norm(points, axis=1)
    safe = np.where(norm > 0, norm, 1.0)
    unit = points / safe[:, None]
    denom = unit[:, 2] + mei["xi"]
    valid = (norm > MIN_DEPTH_M) & (denom > 1e-9)
    denom = np.where(valid, denom, 1.0)
    x, y = unit[:, 0] / denom, unit[:, 1] / denom
    r2 = x * x + y * y
    radial = 1.0 + mei["k1"] * r2 + mei["k2"] * r2 * r2
    xd, yd = x * radial, y * radial
    if tangential:
        p1, p2 = mei["p1"], mei["p2"]
        xd = xd + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = yd + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    uv = np.stack([mei["gamma1"] * xd + mei["u0"], mei["gamma2"] * yd + mei["v0"]], axis=1)
    return uv, valid


def project_opencv_fisheye(
    points: np.ndarray, intrinsics: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chiếu điểm bằng OpenCV fisheye (``fx fy cx cy k1 k2 k3 k4``, thứ tự của ``cv2.fisheye``).

    ``θ`` là góc giữa tia và trục quang, tính bằng ``atan2`` nên đúng cả với tia lệch quá 90 độ
    (``cv2.fisheye`` dùng ``atan(r/z)``, chỉ đúng khi ``z > 0``);
    ``θ_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)``, ``u = fx·θ_d·x/ρ + cx``.

    Returns:
        ``(uv (N, 2), θ (N,), hợp lệ)``.
    """
    fx, fy, cx, cy = intrinsics[:4]
    coeffs = intrinsics[4:8]
    rho = np.hypot(points[:, 0], points[:, 1])
    theta = np.arctan2(rho, points[:, 2])
    t2 = theta * theta
    poly = np.ones_like(theta)
    power = np.ones_like(theta)
    for k in coeffs:
        power = power * t2
        poly = poly + k * power
    theta_d = theta * poly
    scale = np.where(rho > 1e-12, theta_d / np.where(rho > 1e-12, rho, 1.0), 0.0)
    uv = np.stack([fx * points[:, 0] * scale + cx, fy * points[:, 1] * scale + cy], axis=1)
    valid = np.linalg.norm(points, axis=1) > MIN_DEPTH_M
    return uv, theta, valid


def inside_image(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """Điểm có pixel làm tròn nằm trong ảnh ``width × height``."""
    px = np.round(uv)
    return (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)


# ---------------------------------------------------------------- tỉ số cạnh


def depth_discontinuities(
    points: np.ndarray,
    max_step_deg: float = MAX_AZIMUTH_STEP_DEG,
) -> np.ndarray:
    """Đánh dấu điểm LiDAR ở mép trước của một bước nhảy độ sâu, theo thứ tự quét.

    File ``.bin`` của Velodyne KITTI-360 ghi điểm theo từng tia laser, trong mỗi tia theo góc
    phương vị tăng dần, nên hai điểm liền nhau trong file là hai điểm kề nhau trên cùng một vòng
    quét (trừ chỗ đổi tia hoặc thiếu điểm phản xạ, nhận ra bằng bước phương vị lớn hơn
    ``max_step_deg``). Điểm ``i`` là mép trước nếu một điểm kề nó xa hơn ít nhất
    ``max(MIN_DEPTH_JUMP_M, MIN_DEPTH_JUMP_REL · khoảng cách)``: đó là viền vật phía trước, nơi ảnh
    thường có cạnh. Mép sau (điểm nền sát vật) không lấy, vì LiDAR và camera đặt lệch nhau nên nền
    sát vật dễ bị che trong ảnh.

    Args:
        points: Điểm ``(N, 3)`` trong hệ Velodyne, đúng thứ tự trong file.
        max_step_deg: Bước phương vị lớn nhất giữa hai điểm được coi là kề nhau.

    Returns:
        Mặt nạ ``(N,)``.
    """
    edge = np.zeros(len(points), bool)
    if len(points) < 2:
        return edge
    rng = np.linalg.norm(points, axis=1)
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    step = np.abs(np.diff(azimuth))
    step = np.minimum(step, 360.0 - step)
    near = rng[:-1]  # cặp (i, i + 1)
    far = rng[1:]
    adjacent = step <= max_step_deg
    threshold_left = np.maximum(MIN_DEPTH_JUMP_M, MIN_DEPTH_JUMP_REL * near)
    threshold_right = np.maximum(MIN_DEPTH_JUMP_M, MIN_DEPTH_JUMP_REL * far)
    edge[:-1] |= adjacent & (far - near > threshold_left)  # i gần hơn i + 1
    edge[1:] |= adjacent & (near - far > threshold_right)  # i + 1 gần hơn i
    return edge


def edge_strength(image: np.ndarray) -> np.ndarray:
    """Độ lớn gradient ngang ``|∂I/∂u|`` (Sobel) của ảnh xám, ``float32``.

    Velodyne quét theo phương ngang, nên bước nhảy độ sâu giữa hai điểm kề nhau trên một vòng quét
    là viền đứng của vật (cột, thân cây, thành xe); ảnh có cạnh ở đó thì gradient ngang lớn. Dùng
    gradient ngang thay cho độ lớn gradient để vạch kẻ đường, đường chân trời (cạnh nằm ngang)
    không làm tăng mẫu số.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    return np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))


@dataclass
class EdgeScore:
    """Tổng độ mạnh cạnh ảnh dưới điểm mép và dưới mọi điểm; cộng được qua nhiều khung.

    Attributes:
        edge_sum: Tổng độ mạnh cạnh tại điểm mép.
        edge_count: Số điểm mép.
        all_sum: Tổng độ mạnh cạnh tại mọi điểm.
        all_count: Số điểm.
    """

    edge_sum: float = 0.0
    edge_count: int = 0
    all_sum: float = 0.0
    all_count: int = 0

    @property
    def ratio(self) -> float:
        """Tỉ số cạnh: trung bình tại điểm mép / trung bình tại mọi điểm (``nan`` nếu không có)."""
        if self.edge_count == 0 or self.all_sum <= 0:
            return float("nan")
        return (self.edge_sum / self.edge_count) / (self.all_sum / self.all_count)

    def __add__(self, other: "EdgeScore") -> "EdgeScore":
        """Gộp hai kết quả (ví dụ của hai khung)."""
        return EdgeScore(
            self.edge_sum + other.edge_sum,
            self.edge_count + other.edge_count,
            self.all_sum + other.all_sum,
            self.all_count + other.all_count,
        )


def edge_score(strength: np.ndarray, uv: np.ndarray, is_edge: np.ndarray) -> EdgeScore:
    """Lấy độ mạnh cạnh ảnh dưới các điểm chiếu.

    Args:
        strength: Độ mạnh cạnh của ảnh (``edge_strength``).
        uv: Pixel ``(N, 2)`` của các điểm, đã lọc nằm trong ảnh.
        is_edge: Mặt nạ ``(N,)`` điểm ở mép bước nhảy độ sâu (``depth_discontinuities``).

    Returns:
        Các tổng; ``.ratio`` là tỉ số cạnh.
    """
    px = np.round(uv).astype(np.int64)
    values = strength[px[:, 1], px[:, 0]].astype(np.float64)
    return EdgeScore(
        float(values[is_edge].sum()), int(is_edge.sum()), float(values.sum()), int(values.size)
    )


def draw_overlay(image: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Vẽ điểm lên ảnh, màu theo độ sâu (thang log từ 2 m, đỏ, tới 80 m, xanh)."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if depth.size == 0:
        return canvas
    near, far = np.log(2.0), np.log(80.0)
    scaled = np.clip((far - np.log(np.maximum(depth, 1e-3))) / (far - near), 0.0, 1.0)
    colors = cv2.applyColorMap((scaled * 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)
    order = np.argsort(-depth)  # vẽ điểm gần sau cùng
    for (u, v), color in zip(np.round(uv[order]).astype(int), colors[order, 0]):
        cv2.circle(canvas, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)
    return canvas


# ---------------------------------------------------------------- dữ liệu


@dataclass
class SceneData:
    """Những gì converter ghi trong thư mục cảnh, đọc lại bằng numpy.

    Attributes:
        root: Thư mục cảnh.
        sequence: Tên chuỗi KITTI-360 (``manifest.yaml``).
        cameras: ``camera_id`` theo chỉ số camera trong cảnh.
        models: Mô hình camera theo chỉ số (``opencv_pinhole`` hoặc ``opencv_fisheye``).
        frame_ids: Số hiệu khung gốc theo khung của cảnh.
    """

    root: Path
    sequence: str
    cameras: list[str]
    models: list[str]
    frame_ids: list[int]

    @classmethod
    def load(cls, root: Path) -> "SceneData":
        """Đọc ``manifest.yaml`` và ``frame_ids.txt`` của một cảnh KITTI-360."""
        manifest = files.load(root / "manifest.yaml", Manifest)
        if manifest.source.dataset != "kitti360":
            raise SystemExit(f"{root} không phải cảnh KITTI-360 ({manifest.source.dataset})")
        frame_ids = [int(v) for v in (root / "frame_ids.txt").read_text().split()]
        return cls(
            root,
            manifest.source.sequence_id,
            [c.camera_id for c in manifest.cameras],
            [str(c.model) for c in manifest.cameras],
            frame_ids,
        )

    def ego_pose(self, t: int) -> np.ndarray:
        """Velodyne → world của cảnh ở khung ``t``."""
        return read_matrix(self.root / "ego_pose" / f"{t:03d}.txt")

    def extrinsics(self, index: int) -> np.ndarray:
        """Camera → Velodyne của camera ``index``."""
        return read_matrix(self.root / "extrinsics" / f"{index}.txt")

    def intrinsics(self, index: int) -> list[float]:
        """Các số trong ``intrinsics/<index>.txt``."""
        return [float(v) for v in (self.root / "intrinsics" / f"{index}.txt").read_text().split()]

    def lidar(self, t: int) -> np.ndarray:
        """Điểm Velodyne của khung ``t``."""
        return load_velodyne(self.root / "lidar" / f"{t:03d}.bin")

    def image(self, t: int, index: int) -> np.ndarray:
        """Ảnh BGR của camera ``index`` ở khung ``t``."""
        path = self.root / "images" / f"{t:03d}_{index}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"không đọc được {path}")
        return image


class OfficialRig:
    """Calib và pose chính thức của KITTI-360, ghép theo quy ước ``kitti360Scripts``."""

    def __init__(self, raw_dir: Path, sequence: str) -> None:
        calib = read_calibration(raw_dir)
        self.poses = {f: as_array(m) for f, m in read_poses(raw_dir, sequence).items()}
        self.cam_to_pose = {c: as_array(m) for c, m in calib.cam_to_pose.items()}
        self.rect = {c: as_array(m) for c, m in calib.rect.items()}
        self.k = {c: as_array(p)[:, :3] for c, p in calib.projection.items()}
        self.mei = calib.mei
        self.velo_to_pose = self.cam_to_pose["image_00"] @ np.linalg.inv(
            as_array(calib.cam0_to_velo)
        )

    def pose(self, frame: int) -> np.ndarray:
        """Hệ pose (IMU/GPS) → world ở khung gốc ``frame``."""
        if frame not in self.poses:
            raise SystemExit(f"poses.txt không có khung gốc {frame}")
        return self.poses[frame]

    def velo_to_world(self, frame: int) -> np.ndarray:
        """Velodyne → world chính thức ở khung gốc ``frame``."""
        return self.pose(frame) @ self.velo_to_pose

    def camera_to_world(self, camera: str, frame: int) -> np.ndarray:
        """Camera → world chính thức; camera pinhole tính theo hệ đã rectify."""
        m = self.pose(frame) @ self.cam_to_pose[camera]
        if camera in self.rect:
            m = m @ np.linalg.inv(self.rect[camera])
        return m


# ---------------------------------------------------------------- kiểm tra một camera


@dataclass
class CameraResult:
    """Kết quả của một camera ở một khung.

    Attributes:
        camera: ``camera_id``.
        diff_px: Lệch pixel giữa (a) và (b) ở từng điểm chiếu được theo cả hai đường.
        score: Tỉ số cạnh với hình học đúng.
        score_perturbed: Tỉ số cạnh khi xoay lệch camera.
    """

    camera: str
    diff_px: np.ndarray
    score: EdgeScore
    score_perturbed: EdgeScore


def _angle_from_axis(points: np.ndarray) -> np.ndarray:
    """Góc giữa tia tới điểm và trục quang (rad)."""
    return np.arctan2(np.hypot(points[:, 0], points[:, 1]), points[:, 2])


def check_camera(
    scene: SceneData,
    rig: OfficialRig,
    index: int,
    image_frame: int,
    lidar_frame: int,
    perturb_deg: float,
    out_dir: Optional[Path] = None,
) -> CameraResult:
    """So hai đường chiếu và đo tỉ số cạnh cho camera ``index``.

    LiDAR của khung ``lidar_frame`` được đưa lên world rồi chiếu vào ảnh của khung
    ``image_frame`` (hai khung khác nhau thì kiểm tra cả ``ego_pose`` và ``poses.txt``). Với camera
    mắt cá, phép so và tỉ số cạnh chỉ dùng điểm trong miền khớp ``MEI_FIT_MAX_ANGLE``: ngoài miền
    đó là mép ảnh có thân xe và vùng renderer bị che.

    Args:
        scene: Thư mục cảnh đã đọc.
        rig: Calib, pose chính thức.
        index: Chỉ số camera trong cảnh.
        image_frame: Khung (của cảnh) lấy ảnh.
        lidar_frame: Khung (của cảnh) lấy LiDAR.
        perturb_deg: Góc xoay lệch (độ) quanh trục y của camera để so tỉ số cạnh.
        out_dir: Thư mục ghi ảnh chồng điểm; ``None`` là không ghi.

    Returns:
        Kết quả của camera.
    """
    camera = scene.cameras[index]
    image = scene.image(image_frame, index)
    height, width = image.shape[:2]
    points = scene.lidar(lidar_frame)
    f_image, f_lidar = scene.frame_ids[image_frame], scene.frame_ids[lidar_frame]
    fisheye = camera not in rig.k

    # (a) calib và pose chính thức
    official = np.linalg.inv(rig.camera_to_world(camera, f_image)) @ rig.velo_to_world(f_lidar)
    # (b) dữ liệu chuẩn trong thư mục cảnh
    ours = np.linalg.inv(scene.ego_pose(image_frame) @ scene.extrinsics(index))
    ours = ours @ scene.ego_pose(lidar_frame)

    def project_official(extra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam = transform(extra @ official, points)
        if fisheye:
            uv, valid = project_mei(cam, rig.mei[camera])
        else:
            uv, valid = project_pinhole(cam, rig.k[camera])
        valid &= inside_image(uv, width, height)
        scored = valid & (_angle_from_axis(cam) <= MEI_FIT_MAX_ANGLE) if fisheye else valid
        return uv, valid, scored

    uv_a, ok_a, scored_a = project_official(np.eye(4))
    cam_b = transform(ours, points)
    intrinsics = scene.intrinsics(index)
    if scene.models[index] == "opencv_pinhole":
        fx, fy, cx, cy = intrinsics[:4]
        uv_b, ok_b = project_pinhole(cam_b, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
    else:
        uv_b, theta, ok_b = project_opencv_fisheye(cam_b, intrinsics)
        ok_b &= theta <= MEI_FIT_MAX_ANGLE
    both = ok_a & ok_b
    diff = np.linalg.norm(uv_a[both] - uv_b[both], axis=1)

    strength = edge_strength(image)
    is_edge = depth_discontinuities(points)
    score = edge_score(strength, uv_a[scored_a], is_edge[scored_a])
    uv_p, _, scored_p = project_official(yaw_rotation(perturb_deg))
    score_p = edge_score(strength, uv_p[scored_p], is_edge[scored_p])

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        depth = np.linalg.norm(transform(official, points), axis=1)
        overlay = draw_overlay(image, uv_a[ok_a], depth[ok_a])
        name = f"{image_frame:03d}_{camera}.png"
        if lidar_frame != image_frame:
            name = f"{image_frame:03d}_lidar{lidar_frame:03d}_{camera}.png"
        cv2.imwrite(str(out_dir / name), overlay)

    return CameraResult(camera, diff, score, score_p)


# ---------------------------------------------------------------- dòng lệnh


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m drive2sim.scripts.check_kitti360_projection",
        description="Chiếu LiDAR KITTI-360 lên ảnh để kiểm tra calib, pose và dữ liệu chuẩn.",
    )
    parser.add_argument("--scene", type=Path, required=True, help="thư mục cảnh đã ingest")
    parser.add_argument("--raw", type=Path, required=True, help="thư mục gốc KITTI-360")
    parser.add_argument(
        "--frame", type=int, nargs="+", default=[0], help="khung của cảnh (mặc định 0)"
    )
    parser.add_argument(
        "--lidar-frame",
        type=int,
        default=None,
        help="lấy LiDAR ở khung này thay cho khung ảnh (kiểm tra pose giữa hai khung)",
    )
    parser.add_argument("--cameras", nargs="+", default=None, help="camera_id (mặc định: mọi)")
    parser.add_argument(
        "--perturb-deg",
        type=float,
        default=DEFAULT_PERTURB_DEG,
        help=f"góc xoay lệch để so tỉ số cạnh (mặc định {DEFAULT_PERTURB_DEG})",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help=f"tỉ số cạnh gộp tối thiểu của mỗi camera (mặc định {DEFAULT_MIN_RATIO})",
    )
    parser.add_argument("--out", type=Path, default=None, help="thư mục ghi ảnh chồng điểm")
    return parser.parse_args(argv)


def _row(label: str, camera: str, diff: np.ndarray, score: EdgeScore, perturbed: EdgeScore) -> str:
    median = float(np.median(diff)) if diff.size else float("nan")
    largest = float(diff.max()) if diff.size else float("nan")
    return (
        f"{label:>5} {camera:<9} {diff.size:>7} {median:>11.2e} {largest:>12.2e} "
        f"{score.all_count:>7} {score.edge_count:>6} {score.ratio:>6.3f} {perturbed.ratio:>8.3f}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Chạy kiểm tra, in kết quả theo khung và tổng theo camera.

    Returns:
        0 nếu tỉ số cạnh gộp mọi khung của mọi camera vượt ``--min-ratio``, ngược lại 1.
    """
    args = _parse_args(argv)
    scene = SceneData.load(args.scene)
    rig = OfficialRig(args.raw, scene.sequence)
    wanted = args.cameras or scene.cameras
    unknown = sorted(set(wanted) - set(scene.cameras))
    if unknown:
        raise SystemExit(f"cảnh không có camera {', '.join(unknown)}")
    header = (
        f"{'khung':>5} {'camera':<9} {'so':>7} {'lệch tv px':>11} {'lệch max px':>12} "
        f"{'điểm':>7} {'mép':>6} {'tỉ số':>6} {f'lệch {args.perturb_deg:g}°':>8}"
    )
    print(header)
    totals: dict[str, list] = {}
    for frame in args.frame:
        lidar_frame = frame if args.lidar_frame is None else args.lidar_frame
        for index, camera in enumerate(scene.cameras):
            if camera not in wanted:
                continue
            r = check_camera(scene, rig, index, frame, lidar_frame, args.perturb_deg, args.out)
            print(_row(str(frame), camera, r.diff_px, r.score, r.score_perturbed))
            total = totals.setdefault(camera, [[], EdgeScore(), EdgeScore()])
            total[0].append(r.diff_px)
            total[1] += r.score
            total[2] += r.score_perturbed
    print(f"\ngộp {len(args.frame)} khung (ngưỡng tỉ số {args.min_ratio:g}):")
    print(header.replace("khung", "  gộp", 1))
    failed = []
    for camera, (diffs, score, perturbed) in totals.items():
        line = _row("tổng", camera, np.concatenate(diffs), score, perturbed)
        if not score.ratio > args.min_ratio:
            failed.append(camera)
            line += "  <- dưới ngưỡng"
        print(line)
    if args.out is not None:
        print(f"ảnh chồng điểm: {args.out}")
    if failed:
        print(f"tỉ số cạnh không vượt {args.min_ratio:g}: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
