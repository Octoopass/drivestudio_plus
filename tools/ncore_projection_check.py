"""Overlay LiDAR points on every camera of an NCore V4 clip, using NCore's own camera models.

    <ncore-python> tools/ncore_projection_check.py clip.json --out check_dir [--frame 0]

Writes one PNG per camera. Points coloured by depth (near = red, far = blue). If extrinsics or
intrinsics are wrong, points visibly miss the edges of poles, cars and buildings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from upath import UPath

from ncore.data.v4 import (
    CameraSensorComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
)
from ncore.impl.sensors.camera import CameraModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=0, help="index of the lidar frame")
    a = p.parse_args()

    meta = json.loads(Path(a.json).read_text())
    base = UPath(a.json).parent
    r = SequenceComponentGroupsReader([base / s["path"] for s in meta["component_stores"]])
    poses = r.open_component_readers(PosesComponent.Reader)["default"]
    intr = r.open_component_readers(IntrinsicsComponent.Reader)["default"]
    lidars = r.open_component_readers(LidarSensorComponent.Reader)
    lidar_id, lidar = next(iter(lidars.items()))
    lt = int(lidar.frames_timestamps_us[a.frame, 1])
    d = lidar.get_frame_ray_bundle_data(lt, "direction")
    dist = lidar.get_frame_ray_bundle_return_data(lt, "distance_m", 0)
    pts_lidar = d * dist[:, None]
    T_lidar_rig = poses.get_static_pose(lidar_id, "rig")
    pts_rig = pts_lidar @ T_lidar_rig[:3, :3].T + T_lidar_rig[:3, 3]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cam_id, cam in r.open_component_readers(CameraSensorComponent.Reader).items():
        ts = cam.frames_timestamps_us
        ct = int(ts[np.argmin(np.abs(ts[:, 1].astype(np.int64) - lt)), 1])
        img = cam.get_frame_image(ct).convert("RGB")
        T_cam_rig = poses.get_static_pose(cam_id, "rig")
        T_rig_cam = np.linalg.inv(T_cam_rig)
        pc = pts_rig @ T_rig_cam[:3, :3].T + T_rig_cam[:3, 3]
        model = CameraModel.from_parameters(intr.get_camera_model_parameters(cam_id), device="cpu")
        proj = model.camera_rays_to_image_points(torch.from_numpy(pc.astype(np.float32)))
        uv = proj.image_points.numpy()
        ok = proj.valid_flag.numpy() & (pc[:, 2] > 0.5) & np.isfinite(uv).all(1)
        w, h = img.size
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        depth = np.linalg.norm(pc[ok], axis=1)
        draw = ImageDraw.Draw(img)
        t = np.clip((depth - 2) / 40, 0, 1)
        for (u, v), k in zip(uv[ok], t):
            draw.point((float(u), float(v)), fill=(int(255 * (1 - k)), 60, int(255 * k)))
        path = out / f"{cam_id}.png"
        img.save(path)
        print(f"{cam_id}: {ok.sum()} points projected -> {path}")


if __name__ == "__main__":
    main()
