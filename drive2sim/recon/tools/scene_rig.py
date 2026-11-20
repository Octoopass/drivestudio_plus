r"""Ghi ``reports/rig.json`` của một cảnh: bố trí camera trên xe và quỹ đạo xe, nhìn từ trên xuống.

Studio vẽ sơ đồ camera và quỹ đạo xe từ file này (SPEC mục 8A). Mỗi dataset lưu pose, calib theo
cách riêng (nuScenes: ``extrinsics/<khung>_<camera>.txt`` là camera → world; KITTI, KITTI-360:
``extrinsics/<camera>.txt`` là camera → xe cộng ``ego_pose/``...), nên file này **không** tự đọc
các file đó: nó dựng ``DrivingDataset`` của DriveStudio ở chế độ viewer (chỉ đọc pose, calib, không
đọc ảnh, LiDAR) và lấy pose camera, intrinsics mà loader của dataset đó đưa cho trainer. Sơ đồ vì
vậy đúng bằng những gì model được train với, và chạy được với cảnh ingest trên Kaggle (trên máy chỉ
có pose, calib).

Hệ toạ độ của file (``frame``):

- Trục lên: trung bình hướng "lên" của mọi camera ở mọi khung (``-Y`` của camera OpenCV), vì camera
  gắn trên xe gần như thẳng đứng; không dựa vào quy ước trục của dataset.
- Gốc: tâm các camera ở khung đầu. Trục tới: hướng xe chạy (``topdown_frame``), nên
  đúng cả khi cảnh không có camera nhìn thẳng phía trước; xe đứng yên cả đoạn thì lấy hướng nhìn
  của camera tham chiếu (``timeline.reference_sensor`` nếu là camera, không thì camera đầu tiên).
  ``forward_source`` ghi nguồn đã dùng.
- ``x`` tới, ``y`` sang trái, ``z`` lên, đơn vị mét.

Nội dung: ``cameras`` (vị trí so với tâm các camera trong hệ của xe ở khung đầu, góc lệch ngang
``yaw_deg`` và góc ngẩng ``pitch_deg`` của hướng nhìn, góc nhìn ngang ``hfov_deg``), ``trajectory``
(tâm các camera và hướng xe theo từng khung). Góc nhìn ngang đo trên tia của
chính loader: pinhole từ tiêu cự, mắt cá và F-theta là góc lớn nhất của pixel hợp lệ trên hàng giữa
ảnh (camera mắt cá KITTI-360: trong góc mô hình ống kính còn đúng).

Chạy trong môi trường ``drive2sim`` (cần PyTorch)::

    python -m drive2sim.recon.tools.scene_rig data/<scene_id>

``d2s scene ingest`` gọi sau khi ingest; ``d2s scene rig <scene_id>`` và Studio tạo bù cho cảnh
đã ingest trước đó.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.config import load_config
from drive2sim.core.schema import Manifest
from drive2sim.recon.data.driving_dataset import DrivingDataset
from drive2sim.recon.utils.geometry import camera_model_rays
from drive2sim.stages.scene_view import VIEW_SCENE, build_view
from drive2sim.stages.train.drivestudio import dataset_config_for

RIG_FILE = "rig.json"
RIG_VERSION = 1


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def hfov_deg(camera: Any) -> float:
    """Góc nhìn ngang (độ) của một ``CameraData`` theo tia của chính loader."""
    K = camera.intrinsics[0].detach().cpu()
    width, height = int(camera.WIDTH), int(camera.HEIGHT)
    model = getattr(camera, "camera_model", "pinhole")
    if model == "pinhole":
        return math.degrees(2 * math.atan(width / (2 * float(K[0, 0]))))
    if hasattr(camera, "camera_rays"):
        rays, valid = camera.camera_rays(K, height, width)
    else:
        rays, valid = camera_model_rays(
            height,
            width,
            K,
            model,
            getattr(camera, "radial_coeffs", None),
            getattr(camera, "ftheta_parameters", None),
        )
    row = height // 2
    theta = torch.atan2(rays[row, :, 0], rays[row, :, 2])  # lệch ngang, có dấu
    theta = theta[valid[row]]
    if theta.numel() == 0:
        return float("nan")
    return math.degrees(float(theta.max() - theta.min()))


# Đoạn giữa hai khung liền nhau phải dài hơn ngần này (m) mới dùng để đo hướng xe chạy.
MIN_STEP_M = 0.3


def topdown_frame(c2ws: dict[str, np.ndarray], reference: str) -> tuple[np.ndarray, str]:
    """Ma trận 4x4 đổi world của dataset sang hệ của file (x tới, y trái, z lên).

    Hướng tới là hướng xe chạy, đo bằng góc giữa đoạn đường đi được và hướng nhìn của camera
    ``reference`` ở mọi đoạn dài hơn ``MIN_STEP_M``, lấy trung bình (camera gắn cứng trên xe nên góc
    này không đổi; khúc cua chỉ làm lệch vài đoạn). Nhờ vậy đúng cả khi cảnh không có camera nhìn
    thẳng phía trước (ví dụ chỉ hai camera mắt cá nhìn ngang). Xe không đi đủ xa thì dùng hướng nhìn
    của camera ``reference``.

    Returns:
        ``(ma trận, nguồn của hướng tới: "motion" hoặc "camera")``.
    """
    ups = np.concatenate([-pose[:, :3, 1] for pose in c2ws.values()])
    up = _unit(ups.mean(0))
    num_frames = min(len(pose) for pose in c2ws.values())
    centres = np.mean([pose[:num_frames, :3, 3] for pose in c2ws.values()], axis=0)

    def horizontal_axes(frame: int) -> tuple[np.ndarray, np.ndarray]:
        look = c2ws[reference][frame, :3, 2]
        look = _unit(look - look.dot(up) * up)
        return look, np.cross(up, look)

    offsets = []
    for frame in range(num_frames - 1):
        step = centres[frame + 1] - centres[frame]
        step = step - step.dot(up) * up
        if np.linalg.norm(step) < MIN_STEP_M:
            continue
        look, side = horizontal_axes(frame)
        offsets.append(math.atan2(step.dot(side), step.dot(look)))
    look, side = horizontal_axes(0)
    if offsets:
        # trung bình trên vòng tròn (góc quanh ±180 độ không bị cộng sai)
        offset = math.atan2(np.mean(np.sin(offsets)), np.mean(np.cos(offsets)))
        forward, source = math.cos(offset) * look + math.sin(offset) * side, "motion"
    else:
        forward, source = look, "camera"
    left = np.cross(up, forward)
    rotation = np.stack([forward, left, up])  # hàng: trục mới tính trong world
    world_to_rig = np.eye(4)
    world_to_rig[:3, :3] = rotation
    world_to_rig[:3, 3] = -rotation @ centres[0]
    return world_to_rig, source


def _angles(direction: np.ndarray) -> tuple[float, float]:
    """``(yaw, pitch)`` (độ) của một hướng trong hệ x tới, y trái, z lên."""
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(direction[2])))))
    return yaw, pitch


def compute_rig(scene_dir: Path) -> dict[str, Any]:
    """Bố trí camera và quỹ đạo xe của cảnh, theo pose mà loader DriveStudio đọc ra."""
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    camera_ids = [camera.camera_id for camera in manifest.cameras]
    name = dataset_config_for(manifest.source.dataset, len(camera_ids))
    config = load_config(paths.CONFIGS_DIR / "datasets" / f"{name}.yaml")
    with tempfile.TemporaryDirectory(prefix="d2s-rig-") as tmp:
        view_root = Path(tmp)
        build_view(scene_dir, view_root, exclude=("sky_masks", "fine_dynamic_masks", "humanpose"))
        data = OmegaConf.create(config["data"])
        data.data_root = str(view_root)
        data.scene_idx = VIEW_SCENE
        data.start_timestep = 0
        data.end_timestep = -1
        data.preload_device = "cpu"
        data.viewer = True
        data.pixel_source.cameras = list(range(len(camera_ids)))
        data.pixel_source.test_image_stride = 0
        data.pixel_source.undistort = False
        dataset = DrivingDataset(data_cfg=data)
        cameras = {
            str(cam.cam_name): cam for cam in dataset.pixel_source.camera_data.values()
        }
        c2ws = {name: cam.cam_to_worlds.detach().cpu().numpy() for name, cam in cameras.items()}
        hfovs = {name: hfov_deg(cam) for name, cam in cameras.items()}

    reference = manifest.timeline.reference_sensor
    if reference not in c2ws:
        reference = next(iter(c2ws))
    world_to_rig, forward_source = topdown_frame(c2ws, reference)
    rotation, offset = world_to_rig[:3, :3], world_to_rig[:3, 3]
    num_frames = min(len(pose) for pose in c2ws.values())

    def to_rig(point: np.ndarray) -> np.ndarray:
        return rotation @ point + offset

    rig_cameras = []
    models = {camera.camera_id: camera.model for camera in manifest.cameras}
    for name, pose in c2ws.items():
        position = to_rig(pose[0, :3, 3])
        yaw, pitch = _angles(rotation @ pose[0, :3, 2])
        rig_cameras.append(
            {
                "camera_id": name,
                "model": models.get(name),
                "position_m": [round(float(v), 4) for v in position],
                "yaw_deg": round(yaw, 3),
                "pitch_deg": round(pitch, 3),
                "hfov_deg": None if math.isnan(hfovs[name]) else round(hfovs[name], 3),
            }
        )
    reference_yaw = next(c["yaw_deg"] for c in rig_cameras if c["camera_id"] == reference)
    trajectory = []
    for frame in range(num_frames):
        centre = to_rig(np.mean([pose[frame, :3, 3] for pose in c2ws.values()], axis=0))
        yaw, _ = _angles(rotation @ c2ws[reference][frame, :3, 2])
        yaw = (yaw - reference_yaw + 180) % 360 - 180  # hướng xe = hướng camera trừ góc gắn
        trajectory.append(
            {
                "frame": frame,
                "x": round(float(centre[0]), 4),
                "y": round(float(centre[1]), 4),
                "z": round(float(centre[2]), 4),
                "yaw_deg": round(yaw, 3),
            }
        )
    return {
        "version": RIG_VERSION,
        "scene_id": manifest.scene_id,
        "frame": "x forward, y left, z up (m); origin: camera centre at frame 0",
        "reference_camera": reference,
        "forward_source": forward_source,
        "cameras": rig_cameras,
        "trajectory": trajectory,
    }


def write_rig(scene_dir: Path) -> Path:
    """Tính rồi ghi ``<scene_dir>/reports/rig.json``; trả đường dẫn file."""
    rig = compute_rig(scene_dir)
    path = scene_dir / "reports" / RIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rig), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> None:
    """Ghi ``reports/rig.json`` cho thư mục cảnh trên dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scene_dir", type=Path)
    args = parser.parse_args()
    print(write_rig(args.scene_dir.resolve()))


if __name__ == "__main__":
    main()
