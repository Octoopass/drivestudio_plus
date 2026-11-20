"""Ảnh toàn cảnh 360° và LiDAR chồng lên ảnh của một khung (CONTRACTS mục 2, 11.1).

Hai công cụ kiểm tra dữ liệu của Fisheye Fusion Lab, chuyển sang cảnh chuẩn của drive2sim:
- **Toàn cảnh**: ghép ảnh mọi camera của một khung thành ảnh equirectangular quanh xe. Mỗi pixel
  toàn cảnh là một tia trong hệ xe; tia đổi sang hệ camera bằng phần xoay của
  ``extrinsics/<camera>.txt`` (camera → xe, trục OpenCV), chiếu bằng mô hình ống kính của camera
  (``intrinsics/<camera>.txt`` theo ``cameras[].model``), lấy màu bằng ``cv2.remap``; chỗ hai
  camera chồng nhau trộn theo khoảng cách tới mép ảnh. Chỉ dùng phép xoay (bỏ khoảng cách giữa
  các camera), nên vật gần có thể lệch ở đường nối, như bản Fisheye Fusion Lab.
- **LiDAR chồng ảnh**: điểm ``lidar/<khung>.bin`` (``float32`` x, y, z, cường độ, hệ xe; dạng
  KITTI, KITTI-360) chiếu lên ảnh một camera, tô màu theo độ sâu (cùng thang với video độ sâu và
  ảnh chồng ở tab Kiểm tra, ``depth_colors``); điểm rơi lên thân xe tô xám (loader bỏ chúng). Calib
  đúng thì điểm nằm khít mép cột, xe, nhà. Nhanh (không cần PyTorch) nên xem được từng khung;
  ``tests/drive2sim/studio/test_frame_tools.py`` kiểm phép chiếu này trùng phép chiếu của loader
  (``recon/data/loaders/kitti360.py``), và trên cảnh KITTI-360 thật mọi pixel loader chiếu ra đều
  trùng. Dataset khác (calib theo khung như nuScenes): route trả ảnh chồng do loader tạo sẵn
  (``reports/lidar_overlay/``, ``recon/tools/lidar_overlay.py``) cho các khung có.

Cần extrinsics cố định theo camera (KITTI, KITTI-360, Waymo, Argoverse 2, nuPlan); dataset ghi
extrinsics theo khung hoặc LiDAR dạng khác bị từ chối kèm lý do. Cần numpy và OpenCV (môi
trường chính), nạp khi gọi.

File này chứa:
- ``FrameToolError``: không làm được với cảnh này.
- ``panorama``: ảnh toàn cảnh PNG và tỉ lệ phủ.
- ``lidar_overlay``: ảnh một camera có điểm LiDAR, PNG, và số điểm vẽ được.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from drive2sim.core.schema import Manifest

if TYPE_CHECKING:
    import numpy as np

LIDAR_DATASETS = {"kitti", "kitti360"}


class FrameToolError(ValueError):
    """Không làm được với cảnh này (thiếu calib cố định, LiDAR dạng khác, thiếu ảnh)."""


def _camera(scene_root: Path, manifest: Manifest, index: int) -> dict[str, Any]:
    import numpy as np

    camera = manifest.cameras[index]
    extrinsics = scene_root / "extrinsics" / f"{index}.txt"
    intrinsics = scene_root / "intrinsics" / f"{index}.txt"
    if not extrinsics.is_file() or not intrinsics.is_file():
        raise FrameToolError(
            "cảnh không có extrinsics/<camera>.txt cố định (dataset ghi calib theo khung): "
            "chưa hỗ trợ"
        )
    values = np.loadtxt(intrinsics).reshape(-1)
    return {
        "id": camera.camera_id,
        "model": camera.model,
        "size": tuple(camera.resolution),
        "max_angle": camera.max_angle,
        "T": np.loadtxt(extrinsics).reshape(4, 4),
        "K": np.array([[values[0], 0, values[2]], [0, values[1], values[3]], [0, 0, 1.0]]),
        "D": values[4:],
    }


def _image(scene_root: Path, frame: int, index: int) -> "np.ndarray":
    import cv2

    for suffix in (".png", ".jpg", ".jpeg"):
        path = scene_root / "images" / f"{frame:03d}_{index}{suffix}"
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    raise FrameToolError(f"không có ảnh khung {frame} camera {index}")


def _project(points: "np.ndarray", camera: dict[str, Any]) -> tuple["np.ndarray", "np.ndarray"]:
    """Chiếu điểm (hệ camera, ``[N, 3]``) ra pixel; trả ``(uv [N, 2], hợp lệ [N])``."""
    import cv2
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    norm = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(points[:, 2] / np.maximum(norm, 1e-12), -1, 1))
    zero = np.zeros(3)
    if camera["model"] == "opencv_fisheye":
        limit = camera["max_angle"] or np.pi / 2 * 0.999
        valid = (norm > 1e-9) & (theta < limit)
        uv = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), zero, zero, camera["K"], camera["D"][:4].reshape(4, 1)
        )[0].reshape(-1, 2)
    elif camera["model"] == "opencv_pinhole":
        valid = points[:, 2] > 1e-3
        safe = points.copy()
        safe[~valid] = [0, 0, 1]
        uv = cv2.projectPoints(safe, zero, zero, camera["K"], camera["D"][:5])[0].reshape(-1, 2)
    else:
        raise FrameToolError(f"mô hình camera {camera['model']} chưa hỗ trợ")
    width, height = camera["size"]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1)
    return uv, valid & inside & np.isfinite(uv).all(axis=1)


def _png(image: "np.ndarray") -> bytes:
    import cv2

    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise FrameToolError("không mã hoá được PNG")
    return data.tobytes()


def panorama(
    scene_root: Path, manifest: Manifest, frame: int, width: int = 2048
) -> tuple[bytes, float]:
    """Ảnh toàn cảnh equirectangular (rộng ``width``, cao ``width / 2``) của một khung.

    Kinh độ 0 là trục tới của xe (x), tăng sang trái; ảnh đọc từ trái qua phải là quay từ trái
    sang phải xe.

    Returns:
        ``(PNG, tỉ lệ pixel có ảnh)``.

    Raises:
        FrameToolError: Theo ``_camera``, ``_image``.
    """
    import cv2
    import numpy as np

    height = width // 2
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    lon = (0.5 - xs / width) * 2 * np.pi
    lat = (0.5 - ys / height) * np.pi
    rays = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    ).reshape(-1, 3)
    total = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    for index in range(len(manifest.cameras)):
        camera = _camera(scene_root, manifest, index)
        image = _image(scene_root, frame, index)
        rotation = camera["T"][:3, :3]  # camera → xe
        uv, valid = _project(rays @ rotation, camera)  # tia xe → camera: R^T · d
        map_x = uv[:, 0].reshape(height, width).astype(np.float32)
        map_y = uv[:, 1].reshape(height, width).astype(np.float32)
        valid = valid.reshape(height, width)
        map_x[~valid], map_y[~valid] = -1, -1
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        w, h = camera["size"]
        edge = np.minimum.reduce([map_x, w - 1 - map_x, map_y, h - 1 - map_y])
        weight = (valid * np.clip(edge / 20, 0.01, 1)).astype(np.float32)[..., None]
        total += warped.astype(np.float32) * weight
        weights += weight
    result = np.clip(total / np.maximum(weights, 1e-8), 0, 255).astype(np.uint8)
    return _png(result), float((weights[..., 0] > 0).mean())


def lidar_overlay(
    scene_root: Path, manifest: Manifest, frame: int, camera_id: str
) -> tuple[bytes, int]:
    """Ảnh của ``camera_id`` ở khung ``frame`` có điểm LiDAR tô theo độ sâu (gần đỏ, xa xanh).

    Returns:
        ``(PNG, số điểm vẽ được)``.

    Raises:
        FrameToolError: Dataset không có LiDAR dạng KITTI, camera lạ, thiếu file.
    """
    if manifest.source.dataset not in LIDAR_DATASETS:
        raise FrameToolError(
            f"LiDAR của {manifest.source.dataset} ghi dạng khác; hiện chỉ đọc dạng KITTI "
            "(lidar/<khung>.bin, hệ xe) của KITTI, KITTI-360"
        )
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # môi trường nhẹ: route dùng ảnh chồng do loader tạo
        raise FrameToolError(f"môi trường thiếu {error.name}") from None
    ids = [camera.camera_id for camera in manifest.cameras]
    if camera_id not in ids:
        raise FrameToolError(f"cảnh không có camera {camera_id}")
    index = ids.index(camera_id)
    path = scene_root / "lidar" / f"{frame:03d}.bin"
    if not path.is_file():
        raise FrameToolError(f"không có lidar/{frame:03d}.bin")
    camera = _camera(scene_root, manifest, index)
    image = _image(scene_root, frame, index)
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    in_camera = (points - camera["T"][:3, 3]) @ camera["T"][:3, :3]  # xe → camera
    uv, valid = _project(in_camera, camera)
    depth = np.linalg.norm(in_camera, axis=1)[valid]
    colors = depth_colors(depth)
    # Điểm rơi lên thân xe (mask ego của dataset) vẽ xám: loader bỏ chúng, model không học.
    pixels = uv[valid].astype(int)
    ego = _ego_mask(manifest, index, image.shape[1], image.shape[0])
    if ego is not None:
        colors[ego[pixels[:, 1], pixels[:, 0]]] = EGO_GREY
    for (x, y), color in zip(pixels, colors):
        cv2.circle(image, (int(x), int(y)), 2, tuple(int(c) for c in color), -1)
    return _png(image), int(valid.sum())


# Thang màu độ sâu chung với video độ sâu và ảnh chồng của tab Kiểm tra (``depth_visualizer`` của
# ``recon/utils/visualization.py``): turbo theo log, gần (DEPTH_NEAR_M) đỏ, xa (DEPTH_FAR_M) tím.
DEPTH_NEAR_M = 4.0
DEPTH_FAR_M = 120.0
EGO_GREY = (128, 128, 128)


def depth_colors(depth: "np.ndarray") -> "np.ndarray":
    """Màu BGR ``uint8`` ``[N, 3]`` của độ sâu (mét), cùng thang với ``depth_visualizer``."""
    import cv2
    import numpy as np

    near, far = np.log(DEPTH_NEAR_M), np.log(DEPTH_FAR_M)
    t = 1.0 - (np.log(np.maximum(depth, 1e-6)) - near) / (far - near)
    levels = (np.clip(t, 0, 1) * 255).round().astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(levels, cv2.COLORMAP_TURBO).reshape(-1, 3).copy()


def _ego_mask(manifest: Manifest, index: int, width: int, height: int) -> "np.ndarray | None":
    """Mask thân xe (``True`` là thân xe) cỡ ``width`` x ``height``; ``None`` nếu không có.

    Tìm như loader: theo ``camera_id`` (KITTI-360) rồi theo chỉ số camera (các dataset khác).
    """
    import cv2

    from drive2sim.paths import EGO_MASKS_DIR

    folder = EGO_MASKS_DIR / manifest.source.dataset
    for name in (manifest.cameras[index].camera_id, str(index)):
        path = folder / f"{name}.png"
        if path.is_file():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                return mask > 0
    return None
