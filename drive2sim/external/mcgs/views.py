"""Dựng các góc nhìn test để chấm bản đồ Gaussian của MCGS-SLAM bằng gsplat (cần numpy).

Bản đồ ``3dgs_final.ply`` của MCGS-SLAM nằm trong hệ riêng của SLAM, co 0,2 lần (``POSE_SCALE``),
còn ảnh và pose thật của cảnh nằm trong hệ ``world`` của cảnh. File này nối hai hệ đó và ghi
``views.npz`` theo đúng định dạng mà ``drive2sim/external/nurec/tools/ncore_eval_render.py`` đọc.

File này chứa:
- ``camera_to_world``: pose camera → world của cảnh KITTI, KITTI-360 theo ``ego_pose`` và
  ``extrinsics``.
- ``ScenePoses``: pose camera → world của mọi dataset pinhole của tool, qua loader DriveStudio của
  từng dataset (mỗi dataset lưu pose một kiểu).
- ``fit_alignment``: căn quỹ đạo keyframe của SLAM vào pose thật (Umeyama), kèm ATE.
- ``to_map_viewmat``: ma trận world → camera trong hệ bản đồ của MCGS-SLAM.
- ``read_intrinsics``, ``fisheye_theta``: ma trận K, hệ số méo và góc lệch trục của từng pixel.
- ``build_views``: ghi ``views.npz`` cho các cặp (khung, camera) test.

``drive2sim/stages/eval/mcgs.py`` gọi file này. Xem ARCHITECTURE mục 2.19.
"""

import importlib
from pathlib import Path
from typing import Optional

import numpy as np

from drive2sim.external.mcgs import PINHOLE_MODELS, POSE_SCALE
from drive2sim.external.mcgs.align import ate_rmse, match_by_time, read_tum

FISHEYE_MODELS = ("opencv_fisheye",)
# Dataset có ego_pose/<khung>.txt và extrinsics/<chỉ số>.txt (camera → ego, trục OpenCV): đọc trực
# tiếp, không cần nạp PyTorch.
DIRECT_POSE_DATASETS = ("kitti", "kitti360")
# dataset → (module trong recon/data/loaders, lớp camera). Mỗi lớp có ``get_camera2worlds`` đọc
# pose theo cách dataset đó lưu (nuScenes: extrinsics/<khung>_<camera>.txt, Waymo: camera pose).
# PhysicalAI (ftheta) không có: SLAM chỉ nhận camera pinhole.
LOADER_CAMERA_CLASSES = {
    "nuscenes": ("nuscenes", "NuScenesCameraData"),
    "waymo": ("waymo", "WaymoCameraData"),
    "argoverse2": ("argoverse", "ArgoVerseCameraData"),
    "pandaset": ("pandaset", "PandaCameraData"),
    "nuplan": ("nuplan", "NuPlanCameraData"),
}
SUPPORTED_DATASETS = (*DIRECT_POSE_DATASETS, *LOADER_CAMERA_CLASSES)


def read_matrix(path: Path) -> np.ndarray:
    """Đọc ma trận 4×4 từ file text của cảnh (``ego_pose``, ``extrinsics``)."""
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def camera_to_world(scene_root: Path, frame: int, camera_index: int) -> np.ndarray:
    """Pose camera → world của cảnh ở một khung: ``ego_pose[khung] @ extrinsics[camera]``.

    Giống cách loader KITTI tính (``OPENCV2DATASET`` là ma trận đơn vị với KITTI, KITTI-360), nhưng
    giữ world gốc của cảnh thay vì đặt gốc ở khung đầu của split; việc căn quỹ đạo bên dưới độc lập
    với lựa chọn đó.

    Args:
        scene_root: Thư mục ``data/<scene_id>/``.
        frame: Chỉ số khung của cảnh.
        camera_index: Vị trí của camera trong ``manifest.cameras`` (tên file ``<chỉ số>.txt``).
    """
    ego = read_matrix(scene_root / "ego_pose" / f"{frame:03d}.txt")
    cam_to_ego = read_matrix(scene_root / "extrinsics" / f"{camera_index}.txt")
    return ego @ cam_to_ego


class ScenePoses:
    """Pose camera → world của một cảnh, theo cách lưu của dataset gốc.

    KITTI, KITTI-360 tính trực tiếp (``camera_to_world``); các dataset còn lại gọi
    ``get_camera2worlds`` của lớp camera trong loader DriveStudio (nạp PyTorch khi cần), với world
    gốc đặt ở khung 0. Mọi camera của cùng một ``ScenePoses`` dùng chung một world, nên pose của
    các camera so được với nhau.

    Attributes:
        scene_root: Thư mục ``data/<scene_id>/``.
        dataset: ``source.dataset`` trong manifest, ví dụ ``waymo``.
        num_frames: Số khung của cảnh (``manifest.timeline.num_frames``).
    """

    def __init__(self, scene_root: Path, dataset: str, num_frames: int) -> None:
        """Kiểm tra dataset có hỗ trợ.

        Raises:
            ValueError: Nếu dataset không phải pinhole hoặc chưa có loader (ví dụ ``physicalai``).
        """
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"MCGS-SLAM chỉ nhận camera pinhole: dataset {dataset} chưa hỗ trợ "
                f"(có: {', '.join(SUPPORTED_DATASETS)})"
            )
        self.scene_root = scene_root
        self.dataset = dataset
        self.num_frames = num_frames
        self._cache: dict[int, np.ndarray] = {}

    def _loader_poses(self, camera_index: int) -> np.ndarray:
        module_name, class_name = LOADER_CAMERA_CLASSES[self.dataset]
        module = importlib.import_module(f"drive2sim.recon.data.loaders.{module_name}")
        camera_class = getattr(module, class_name)
        poses = camera_class.get_camera2worlds(
            str(self.scene_root), camera_index, 0, self.num_frames
        )
        return np.asarray(poses, dtype=np.float64)

    def c2w(self, frame: int, camera_index: int) -> np.ndarray:
        """Pose camera → world ``(4, 4)`` của một camera ở một khung của cảnh."""
        if self.dataset in DIRECT_POSE_DATASETS:
            return camera_to_world(self.scene_root, frame, camera_index)
        if camera_index not in self._cache:
            self._cache[camera_index] = self._loader_poses(camera_index)
        return self._cache[camera_index][frame]


def fit_alignment(
    traj_path: Path,
    poses_source: "ScenePoses",
    reference_index: int,
    frames_per_stamp: float = 1.0,
    with_scale: bool = True,
) -> dict:
    """Căn quỹ đạo keyframe của SLAM vào vị trí camera tham chiếu thật của cảnh.

    ``traj_mcgs.txt`` là pose camera 0 → hệ SLAM, chỉ ở keyframe, timestamp là số trong tên ảnh
    đưa vào SLAM (khung của cảnh nếu ``frames_per_stamp`` là 1). Tìm phép đồng dạng
    ``world = scale · R · slam + t`` khớp tốt nhất vị trí camera.

    Args:
        traj_path: ``traj_mcgs.txt``.
        poses_source: Pose thật của cảnh.
        reference_index: Chỉ số trong manifest của camera 0 của SLAM (thư mục ảnh đầu tiên).
        frames_per_stamp: Nhân timestamp để ra chỉ số khung của cảnh.
        with_scale: Ước lượng cả tỉ lệ; ``False`` giữ tỉ lệ 1 (khi tin vào tỉ lệ mét của stereo).

    Returns:
        ``{"scale", "R", "t", "ate_rmse_m", "keyframes"}``.

    Raises:
        ValueError: Nếu có dưới 3 keyframe khớp với một khung của cảnh.
    """
    stamps, poses = read_tum(traj_path)
    frames = np.rint(stamps * frames_per_stamp).astype(int)
    available = np.arange(poses_source.num_frames, dtype=float)
    pairs = match_by_time(frames.astype(float), available, tolerance=0.0)
    if len(pairs) < 3:
        raise ValueError(
            f"chỉ {len(pairs)} keyframe của SLAM khớp với khung của cảnh; kiểm tra timestamp "
            "(frames_per_stamp) và số khung của cảnh"
        )
    slam = np.array([poses[i, :3] for i, _ in pairs])
    truth = np.array([poses_source.c2w(int(j), reference_index)[:3, 3] for _, j in pairs])
    rmse, (scale, rotation, translation) = ate_rmse(slam, truth, with_scale)
    return {
        "scale": scale,
        "R": rotation,
        "t": translation,
        "ate_rmse_m": rmse,
        "keyframes": len(pairs),
    }


def to_map_viewmat(
    c2w_world: np.ndarray, alignment: dict, pose_scale: float = POSE_SCALE
) -> np.ndarray:
    """Ma trận world → camera trong hệ bản đồ Gaussian của MCGS-SLAM.

    Từ ``world = scale · R · slam + t``: vị trí trong hệ SLAM là ``Rᵀ (p − t) / scale``, hướng quay
    là ``Rᵀ R_cam``; hệ bản đồ là hệ SLAM co ``pose_scale`` lần (``Mcgs._scale_poses``).

    Args:
        c2w_world: Pose camera → world của cảnh ``(4, 4)``.
        alignment: Kết quả ``fit_alignment``.
        pose_scale: Hệ số co của bản đồ so với quỹ đạo.

    Returns:
        Ma trận ``(4, 4)`` world của cảnh → camera, trong hệ bản đồ (float32).
    """
    rotation, translation, scale = alignment["R"], alignment["t"], alignment["scale"]
    c2w = np.eye(4)
    c2w[:3, :3] = rotation.T @ c2w_world[:3, :3]
    c2w[:3, 3] = (pose_scale / scale) * rotation.T @ (c2w_world[:3, 3] - translation)
    return np.linalg.inv(c2w).astype(np.float32)


def read_intrinsics(
    scene_root: Path, camera_index: int, model: str
) -> tuple[np.ndarray, np.ndarray]:
    """Ma trận K và hệ số méo hướng tâm của một camera.

    ``intrinsics/<chỉ số>.txt`` theo mô hình camera (CONTRACTS mục 2.1): pinhole là 9 số
    ``fx fy cx cy k1 k2 p1 p2 k3``, fisheye là 8 số ``fx fy cx cy k1 k2 k3 k4``.

    Returns:
        ``(K (3, 3), radial (4,))``; ``radial`` là ``k1..k4`` của fisheye, 0 với pinhole.

    Raises:
        ValueError: Nếu mô hình không hỗ trợ hoặc file sai số phần tử.
    """
    values = np.loadtxt(scene_root / "intrinsics" / f"{camera_index}.txt", dtype=np.float64)
    values = np.atleast_1d(values)
    if model in FISHEYE_MODELS:
        if values.size != 8:
            raise ValueError(f"camera fisheye {camera_index}: cần 8 số, có {values.size}")
        radial = values[4:8]
    elif model in PINHOLE_MODELS:
        if values.size != 9:
            raise ValueError(f"camera pinhole {camera_index}: cần 9 số, có {values.size}")
        radial = np.zeros(4)
    else:
        raise ValueError(f"mô hình camera {model} chưa hỗ trợ khi chấm bản đồ MCGS-SLAM")
    fx, fy, cx, cy = values[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), radial


def fisheye_theta(
    intrinsics: np.ndarray, radial: np.ndarray, height: int, width: int, iterations: int = 12
) -> np.ndarray:
    """Góc lệch trục (rad) của tia qua từng pixel, mô hình fisheye của OpenCV.

    OpenCV: ``θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)`` với ``θ_d`` là bán kính chuẩn hoá của
    pixel; giải ``θ`` bằng phép lặp Newton.

    Args:
        intrinsics: Ma trận K ``(3, 3)``.
        radial: ``k1..k4``.
        height: Chiều cao ảnh, pixel.
        width: Chiều rộng ảnh, pixel.
        iterations: Số bước Newton.

    Returns:
        Mảng ``(height, width)``.
    """
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1]
    theta_d = np.hypot(x, y)
    k1, k2, k3, k4 = radial
    theta = theta_d.copy()
    for _ in range(iterations):
        t2 = theta * theta
        poly = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        slope = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        theta = theta - (poly - theta_d) / np.where(np.abs(slope) < 1e-9, 1e-9, slope)
    return theta


def build_views(
    scene_root: Path,
    poses_source: "ScenePoses",
    camera_ids: list[str],
    camera_models: dict[str, str],
    pairs: list[tuple[int, str]],
    alignment: dict,
    out: Path,
    max_angle: Optional[float] = None,
    ego_masks_dir: Optional[Path] = None,
    pose_scale: float = POSE_SCALE,
) -> list[dict]:
    """Ghi ``views.npz`` cho các cặp (khung, camera) test và trả thông tin từng góc nhìn.

    Định dạng như ``ncore_eval_export.py``: ``camera``, ``model``, ``frame``, ``viewmat``, ``K``,
    ``radial`` và các mảng đối tượng ``image`` (RGB uint8), ``mask`` (``True`` là pixel bỏ qua:
    thân xe từ asset của camera, và với fisheye là pixel lệch trục quá ``max_angle``).

    Args:
        scene_root: Thư mục cảnh.
        poses_source: Pose thật của cảnh (``ScenePoses``).
        camera_ids: ``camera_id`` theo thứ tự manifest (vị trí là chỉ số camera).
        camera_models: ``camera_id`` → mô hình camera trong manifest.
        pairs: Các cặp ``(khung, camera_id)`` test.
        alignment: Kết quả ``fit_alignment``.
        out: Đường dẫn ``views.npz``.
        max_angle: Góc lệch trục tối đa (rad) của camera fisheye; ``None`` là không giới hạn.
        ego_masks_dir: Thư mục ``<camera_id>.png`` (255 là vùng bỏ qua), ví dụ ảnh thân xe.
        pose_scale: Hệ số co của bản đồ.

    Returns:
        Danh sách theo thứ tự trong ``views.npz``: ``{"frame", "camera_id", "camera_index",
        "position"}``, với ``position`` là vị trí camera trong world của cảnh.

    Raises:
        FileNotFoundError: Nếu thiếu ảnh của một cặp.
    """
    import cv2

    records: dict[str, list] = {
        k: [] for k in ("camera", "model", "frame", "viewmat", "K", "radial", "image", "mask")
    }
    info = []
    for frame, camera_id in pairs:
        index = camera_ids.index(camera_id)
        model = camera_models[camera_id]
        found = sorted((scene_root / "images").glob(f"{frame:03d}_{index}.*"))
        if not found:
            raise FileNotFoundError(scene_root / "images" / f"{frame:03d}_{index}.*")
        path = found[0]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intrinsics, radial = read_intrinsics(scene_root, index, model)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        if ego_masks_dir is not None and (ego_masks_dir / f"{camera_id}.png").is_file():
            raw = cv2.imread(str(ego_masks_dir / f"{camera_id}.png"), cv2.IMREAD_GRAYSCALE)
            mask |= cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST) > 127
        if model in FISHEYE_MODELS and max_angle is not None:
            mask |= fisheye_theta(intrinsics, radial, height, width) > max_angle
        c2w = poses_source.c2w(frame, index)
        records["camera"].append(camera_id)
        records["model"].append("fisheye" if model in FISHEYE_MODELS else "pinhole")
        records["frame"].append(frame)
        records["viewmat"].append(to_map_viewmat(c2w, alignment, pose_scale))
        records["K"].append(intrinsics)
        records["radial"].append(radial.astype(np.float32))
        records["image"].append(image)
        records["mask"].append(mask)
        info.append(
            {
                "frame": frame,
                "camera_id": camera_id,
                "camera_index": index,
                "position": [float(v) for v in c2w[:3, 3]],
            }
        )
    arrays = {k: np.stack(records[k]) for k in ("viewmat", "K", "radial")}
    arrays["camera"], arrays["model"] = np.array(records["camera"]), np.array(records["model"])
    arrays["frame"] = np.array(records["frame"], dtype=np.int64)
    for key in ("image", "mask"):  # ảnh khác cỡ giữa các camera nên dùng mảng đối tượng
        holder = np.empty(len(records[key]), dtype=object)
        holder[:] = records[key]
        arrays[key] = holder
    np.savez(out, **arrays)
    return info
