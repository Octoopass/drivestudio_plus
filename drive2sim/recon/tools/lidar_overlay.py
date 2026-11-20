r"""Chiếu LiDAR lên ảnh của một cảnh: ảnh chồng để xem và số liệu để kiểm tra thô.

Calib (intrinsics, extrinsics) hay pose sai thì điểm LiDAR rơi lệch khỏi vật trong ảnh và model học
sai hình mà không báo lỗi. File này làm hai việc, trên vài khung cách đều:

- **Ảnh chồng** ``reports/lidar_overlay/<khung>_<camera>.jpg``: điểm LiDAR tô màu theo độ sâu vẽ
  lên ảnh camera, như video ``lidar_on_images`` của DriveStudio (``render.vis_lidar``) nhưng có
  ngay sau ingest, trước khi train. Người xem nhìn là biết điểm có nằm khít lên vật không. Không
  phải bước duyệt: ingest, train không chờ ai xem.
- **Số liệu** ``reports/lidar_check.json``: số điểm LiDAR rơi vào từng camera mỗi khung, mật độ
  trên ảnh và độ sâu trung vị. ``ingest/validate.py`` (``projection_checks``) đọc file này và
  cảnh báo lỗi lớn: camera gần như không có điểm (extrinsics sai hướng, sai vị trí, sai đơn vị), độ
  sâu vô lý. Calib hai camera bị đổi chỗ cho nhau thì không bắt được (vẫn thấy LiDAR), phải xem ảnh
  chồng. Lệch vài pixel cũng không bắt được: đã thử đo bằng gradient ảnh tại mép độ
  sâu (tỉ số cạnh, Levinson & Thrun 2013) và không tìm lại được độ lệch cố ý đưa vào
  (``docs/drive2sim/ARCHITECTURE.md`` mục 7.1).

Phép chiếu là của chính loader DriveStudio (``DrivingDataset`` tự chiếu LiDAR thành
``lidar_depth_maps`` theo đúng hình học khi train, kể cả ống kính mắt cá và vùng che), nên file này
không đọc file calib của từng dataset. Cần ảnh trên máy: cảnh ingest trên Kaggle được chiếu ngay
trong notebook ingest (``d2s scene ingest``), máy tải về ``lidar_check.json`` cùng file nhỏ.

Chạy trong môi trường ``drive2sim`` (cần PyTorch, OpenCV)::

    python -m drive2sim.recon.tools.lidar_overlay data/<scene_id>
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import OmegaConf

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.config import load_config
from drive2sim.core.schema import Manifest
from drive2sim.recon.data.driving_dataset import DrivingDataset
from drive2sim.recon.utils.visualization import depth_visualizer
from drive2sim.stages.scene_view import VIEW_SCENE, build_view
from drive2sim.stages.train.drivestudio import dataset_config_for

CHECK_FILE = "lidar_check.json"
OVERLAY_DIR = "lidar_overlay"
CHECK_VERSION = 1
NUM_FRAMES = 3
# LiDAR thưa (ít hơn ngần này điểm mỗi megapixel, như nuScenes 32 tia) thì vẽ điểm to ra
# ``DOT_RADIUS_PX`` pixel cho dễ nhìn; LiDAR dày (KITTI-360 64 tia trên ảnh mắt cá thu nhỏ) vẽ đúng
# một pixel để còn thấy ảnh bên dưới.
SPARSE_POINTS_PER_MEGAPIXEL = 40_000
DOT_RADIUS_PX = 1
JPEG_QUALITY = 85


def _dataset(view_root: Path, manifest: Manifest, start: int) -> DrivingDataset:
    """``DrivingDataset`` hai khung từ ``start``, có ảnh và LiDAR, không mask, box, SMPL.

    Hai khung vì loader chuẩn hoá mốc thời gian theo khoảng của đoạn; một khung thì chia cho 0 và
    phép so mốc thời gian ảnh, LiDAR của loader báo lệch.
    """
    name = dataset_config_for(manifest.source.dataset, len(manifest.cameras))
    cfg = OmegaConf.create(load_config(paths.CONFIGS_DIR / "datasets" / f"{name}.yaml")["data"])
    cfg.data_root = str(view_root)
    cfg.scene_idx = VIEW_SCENE
    cfg.start_timestep = start
    cfg.end_timestep = start + 1
    cfg.preload_device = "cpu"
    source = cfg.pixel_source
    source.cameras = list(range(len(manifest.cameras)))
    source.test_image_stride = 0
    source.undistort = False
    source.load_sky_mask = False
    source.load_dynamic_mask = False
    source.load_objects = False
    source.load_smpl = False
    cfg.lidar_source.load_lidar = True
    return DrivingDataset(data_cfg=cfg)


def overlay_image(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Ảnh (H, W, 3) uint8: điểm LiDAR tô màu độ sâu (thang của ``depth_visualizer``) trên ảnh."""
    hit = (depth > 0).astype(np.uint8)
    sparse = hit.sum() / (depth.size / 1e6) < SPARSE_POINTS_PER_MEGAPIXEL
    if sparse and DOT_RADIUS_PX > 0:
        kernel = np.ones((2 * DOT_RADIUS_PX + 1,) * 2, np.uint8)
        hit = cv2.dilate(hit, kernel)
        # độ sâu của điểm gần nhất trong vùng vẽ to (vật gần đè vật xa)
        near = np.where(depth > 0, depth, np.inf).astype(np.float32)
        depth = -cv2.dilate(-near, kernel)
        depth[~np.isfinite(depth)] = 0
    colours = depth_visualizer(depth, hit.astype(np.float32))
    mask = hit.astype(bool)[..., None]
    out = np.where(mask, colours, image)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def lidar_check(scene_dir: Path, num_frames: int = NUM_FRAMES) -> dict[str, Any]:
    """Ghi ảnh chồng vào ``reports/lidar_overlay/`` và trả số liệu của từng camera."""
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    total = manifest.timeline.num_frames
    frames = sorted({round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)})
    out_dir = scene_dir / "reports" / OVERLAY_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    stats: dict[str, dict[str, list]] = {}
    overlays = []
    with tempfile.TemporaryDirectory(prefix="d2s-lidar-") as tmp:
        view_root = Path(tmp)
        build_view(scene_dir, view_root, exclude=("sky_masks", "fine_dynamic_masks", "humanpose"))
        for frame in frames:
            start = max(0, min(frame, total - 2))
            index = frame - start
            dataset = _dataset(view_root, manifest, start)
            for cam in dataset.pixel_source.camera_data.values():
                name = str(cam.cam_name)
                if cam.lidar_depth_maps is None:
                    continue
                image = cam.images[index].detach().cpu().numpy()
                depth = cam.lidar_depth_maps[index].detach().cpu().numpy().astype(np.float32)
                values = depth[depth > 0]
                entry = stats.setdefault(name, {"points": [], "depths": [], "pixels": []})
                entry["points"].append(int(values.size))
                entry["depths"].append(values)
                entry["pixels"].append(int(depth.size))
                path = out_dir / f"{frame:03d}_{name}.jpg"
                cv2.imwrite(
                    str(path),
                    cv2.cvtColor(overlay_image(image, depth), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                overlays.append({"frame": frame, "camera_id": name, "file": path.name})
            del dataset
    cameras = []
    for camera in manifest.cameras:
        entry = stats.get(camera.camera_id)
        if entry is None:
            cameras.append({"camera_id": camera.camera_id, "frames": 0})
            continue
        depths = np.concatenate(entry["depths"]) if entry["depths"] else np.zeros(0)
        mean_points = float(np.mean(entry["points"]))
        cameras.append(
            {
                "camera_id": camera.camera_id,
                "frames": len(entry["points"]),
                "points_per_frame": round(mean_points, 1),
                "points_per_megapixel": round(mean_points / (np.mean(entry["pixels"]) / 1e6), 1),
                "median_depth_m": round(float(np.median(depths)), 3) if depths.size else None,
                "p95_depth_m": round(float(np.percentile(depths, 95)), 3) if depths.size else None,
            }
        )
    return {
        "version": CHECK_VERSION,
        "scene_id": manifest.scene_id,
        "frames": frames,
        "cameras": cameras,
        "overlays": overlays,
    }


def write_lidar_check(scene_dir: Path) -> Path:
    """Tính rồi ghi ``<scene_dir>/reports/lidar_check.json`` và ảnh chồng; trả đường dẫn file."""
    result = lidar_check(scene_dir)
    path = scene_dir / "reports" / CHECK_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> None:
    """Ghi ảnh chồng và ``reports/lidar_check.json`` cho thư mục cảnh trên dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scene_dir", type=Path)
    args = parser.parse_args()
    print(write_lidar_check(args.scene_dir.resolve()))


if __name__ == "__main__":
    main()
