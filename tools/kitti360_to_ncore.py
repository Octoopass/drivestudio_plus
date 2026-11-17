"""Convert a slice of a KITTI-360 sequence into an NCore V4 clip for Instant NuRec.

Run with the Python of an environment that has `nvidia-ncore` installed (the Instant NuRec
venv does):

    ~/nurec/instant-nurec/.venv/bin/python tools/kitti360_to_ncore.py \
        --data ./kitti360 --out ./ncore/kitti360_0003 --frames 150

Input is the same archive layout the 3DGUT trainer uses (calibration pickles, data_poses.zip,
data_2d_raw/<seq>_image_0X.tar, data_3d_raw/<seq>.tar).

Conventions (match NVIDIA's NCore clips):
- camera frames are OpenCV (x right, y down, z forward);
- the rig frame is FLU (x forward, y left, z up) with its origin at camera image_00;
- the world frame is the rig pose of the first converted frame.

Cameras:
- image_00 / image_01: rectified perspective images -> OpenCV pinhole, no distortion;
- image_02 / image_03: MEI fisheye, by --fisheye-model:
  rectify120 (default) resampled into NVIDIA's 120-degree F-theta cross camera, same pose: what
  Instant NuRec's multiview model was trained on;
  ftheta: the full fisheye as F-theta (fit < 0.6 px); opencv: as Kannala-Brandt (< 0.002 px);
  all with an 'ego' mask over the car body and the area outside the circle.

KITTI-360 has no per-frame exposure timestamps in this archive; frames are placed on a 10 Hz
clock from their frame id (gaps in the frame ids become gaps in time), global shutter.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from ncore.data.v4 import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.data import BBox3, CuboidTrackObservation, LabelSource
from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.data.types import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from upath import UPath

# KITTI-360 id, pickle key, NCore sensor id, model
CAMERAS = {
    "image_00": ("CAM_FRONT", "camera_front_left_00", "pinhole"),
    "image_01": ("CAM_FRONT_RIGHT", "camera_front_right_01", "pinhole"),
    "image_02": ("CAM_BACK_RIGHT", "camera_fisheye_left_02", "fisheye"),
    "image_03": ("CAM_BACK_LEFT", "camera_fisheye_right_03", "fisheye"),
}
LIDAR_ID = "lidar_velodyne"
# --target drivestudio: DriveStudio+'s PhysicalAI preprocessor only reads these sensor ids (index 0, the
# front-wide camera, is its reference clock) and only F-theta cameras. The names are labels; the geometry
# stays KITTI-360's. image_01 is left out: the preprocessor has no second front camera of the same kind.
DRIVESTUDIO_SENSOR_IDS = {
    "image_00": "camera_front_wide_120fov",
    "image_02": "camera_cross_left_120fov",
    "image_03": "camera_cross_right_120fov",
}
FRAME_DT_US = 100_000  # 10 Hz
FISHEYE_MAX_ANGLE = 1.45  # range of the MEI -> Kannala-Brandt fit
FTHETA_MAX_ANGLE = 1.6  # 92 deg: radius ~658 px, the edge of the KITTI-360 fisheye image circle
# OpenCV camera (x right, y down, z fwd) -> FLU rig (x fwd, y left, z up)
R_CAM_TO_FLU = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def log(msg: str) -> None:
    print(f"[kitti360->ncore] {msg}", flush=True)


# ------------------------------------------------------------------ KITTI-360 input
def load_items(data: Path, seq: str) -> list[dict]:
    items = []
    for split in ("val", "train"):
        path = data / f"ImageSets-rosa-6class/kitti360_rosa_infos_{split}.pkl"
        if not path.exists():
            sys.exit(f"missing {path}")
        with path.open("rb") as f:
            items += [x for x in pickle.load(f)["data_list"] if seq in x["lidar_path"]]
    return items


def load_poses(data: Path, seq: str) -> dict[int, np.ndarray]:
    with ZipFile(data / "data_poses.zip") as z, z.open(f"{seq}/cam0_to_world.txt") as f:
        poses = {}
        for line in f:
            v = np.array(line.split(), dtype=np.float64)
            if len(v) == 17:
                poses[int(v[0])] = v[1:].reshape(4, 4)
    return poses


def select_frames(items: list[dict], poses: dict, start: int, n: int, max_gap: int) -> list[dict]:
    pool = sorted((x for x in items if x["frame_id"] in poses and x["frame_id"] >= start), key=lambda x: x["frame_id"])
    runs, cur = [], []
    for x in pool:
        if cur and x["frame_id"] - cur[-1]["frame_id"] > max_gap:
            runs.append(cur)
            cur = []
        cur.append(x)
    if cur:
        runs.append(cur)
    if not runs:
        sys.exit(f"no frames with poses from frame {start}")
    run = next((r for r in runs if len(r) >= n), max(runs, key=len))
    if len(run) < n:
        log(f"WARNING: longest run has {len(run)} frames (< {n})")
    return run[:n]


def read_tar(path: Path, suffix: str, wanted: set[str]) -> dict[str, bytes]:
    out = {}
    with tarfile.open(path, "r:*") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(suffix) and (stem := Path(m.name).stem) in wanted:
                out[stem] = tar.extractfile(m).read()
                if len(out) == len(wanted):
                    break
    missing = wanted - out.keys()
    if missing:
        sys.exit(f"{path}: {len(missing)} frames missing, e.g. {sorted(missing)[:3]}")
    return out


# ------------------------------------------------------------------ object tracks
# Class order of the ROSA 6-class info files (metainfo["classes"]).
ROSA_CLASSES = ["car", "truck", "trailer", "motorcycle", "bicycle", "pedestrian"]


def cuboid_observations(frames: list[dict], t_us: dict[int, int], lidar_id: str = LIDAR_ID) -> list:
    """KITTI-360 ROSA boxes as NCore cuboid tracks.

    Each info item lists `instances` with `bbox_3d = [x, y, z, l, w, h, yaw]` in the Velodyne frame of that
    scan, z at the box's geometric centre (checked by projecting boxes onto image_00), yaw about +z, and an
    `instance_token` that stays the same across frames. The token becomes the track id; the scan time is
    the reference time, so the preprocessor places each box in the world with the LiDAR pose of its frame.
    Source AUTOLABEL, because that is the only source the DriveStudio+ PhysicalAI preprocessor keeps."""
    obs = []
    for item in frames:
        ts = int(t_us[item["frame_id"]])
        for inst in item.get("instances", []):
            x, y, z, l, w, h, yaw = (float(v) for v in inst["bbox_3d"])
            obs.append(CuboidTrackObservation(
                track_id=str(inst["instance_token"]), class_id=ROSA_CLASSES[int(inst["bbox_label_3d"])],
                timestamp_us=ts, reference_frame_id=lidar_id, reference_frame_timestamp_us=ts,
                bbox3=BBox3(centroid=(x, y, z), dim=(l, w, h), rot=(0.0, 0.0, yaw)),
                source=LabelSource.AUTOLABEL,
            ))
    return obs


# ------------------------------------------------------------------ camera models
def mei_to_kb(xi: float, k1: float, k2: float) -> np.ndarray:
    """Kannala-Brandt radial coefficients reproducing the MEI projection (same fit as the 3DGUT trainer)."""
    theta = np.linspace(0.02, FISHEYE_MAX_ANGLE, 256)
    u = np.sin(theta) / (np.cos(theta) + xi)
    u2 = u * u
    r = (1 + xi) * u * (1 + k1 * u2 + k2 * u2 * u2)
    X = np.column_stack([theta**2, theta**4, theta**6, theta**8])
    k, *_ = np.linalg.lstsq(X, r / theta - 1, rcond=None)
    return k.astype(np.float32)


def _poly_fit(x: np.ndarray, y: np.ndarray, deg: int = 5) -> np.ndarray:
    """y ~ sum c_i x^i with c_0 = 0, fitted on x / x.max() so the least squares stay well conditioned."""
    s = x.max()
    A = np.stack([(x / s) ** i for i in range(1, deg + 1)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.concatenate([[0.0], c / s ** np.arange(1, deg + 1)]).astype(np.float32)


def mei_to_ftheta(info: dict, w: int, h: int) -> FThetaCameraModelParameters:
    """NVIDIA F-theta model reproducing the MEI projection; Instant NuRec only accepts pinhole/F-theta.

    MEI pixel radius at angle theta: f * u (1 + k1 u^2 + k2 u^4), u = sin(theta) / (cos(theta) + xi).
    Radii are in fy pixels; linear_cde = [fx/fy, 0, 0] restores the x scale. Fit error at
    FTHETA_MAX_ANGLE: forward <= 0.32 px, backward <= 0.6 px on both KITTI-360 fisheyes."""
    K = np.array(info["camera_intrinsic"], dtype=np.float64)
    k1, k2, _, _, xi = info["distortion_coefficients"]
    theta = np.linspace(0.0, FTHETA_MAX_ANGLE, 4000)
    u = np.sin(theta) / (np.cos(theta) + xi)
    r = K[1, 1] * u * (1 + k1 * u**2 + k2 * u**4)
    if np.any(np.diff(r) <= 0):
        sys.exit("MEI projection is not monotonic up to FTHETA_MAX_ANGLE")
    return FThetaCameraModelParameters(
        resolution=np.array([w, h], dtype=np.uint64), shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        # F-theta principal points use the pixel-centre convention, same as the KITTI-360 K matrix
        principal_point=np.array([K[0, 2], K[1, 2]], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=_poly_fit(r, theta),
        angle_to_pixeldist_poly=_poly_fit(theta, r),
        max_angle=FTHETA_MAX_ANGLE,
        linear_cde=np.array([K[0, 0] / K[1, 1], 0.0, 0.0], dtype=np.float32),
    )


# NVIDIA's 120-degree cross cameras (from the public PhysicalAI-AV NCore demo clip): the camera geometry
# Instant NuRec's multiview model was trained on. KITTI-360 fisheyes look sideways like these.
NV_CROSS_120 = {
    "image_02": dict(principal_point=[961.3098754882812, 744.0254516601562], max_angle=1.352455637370406,
                     pixeldist_to_angle_poly=[0.0, 0.0010662720305845141, -1.5985817114483325e-09,
                                              8.953260008381925e-12, 2.3786245393589596e-14, -4.219298998532428e-18],
                     angle_to_pixeldist_poly=[0.0, 937.8035278320312, 1.5542490482330322, -6.7989678382873535,
                                              -19.430620193481445, 6.133647441864014]),
    "image_03": dict(principal_point=[964.480712890625, 743.8978881835938], max_angle=1.354026433697201,
                     pixeldist_to_angle_poly=[0.0, 0.001066190074197948, -5.264319202780143e-09,
                                              1.4043568391519656e-11, 2.1125440467588928e-14, -3.4421533004618145e-18],
                     angle_to_pixeldist_poly=[0.0, 937.8742065429688, 4.589958190917969, -10.717320442199707,
                                              -17.785009384155273, 5.831396579742432]),
}


def pinhole_to_ftheta(info: dict, w: int, h: int) -> FThetaCameraModelParameters:
    """F-theta model of a rectified KITTI-360 perspective camera (r = fy tan theta), for readers that only
    accept F-theta (DriveStudio+'s PhysicalAI loader). Fitted up to the image corner; the fit error is
    checked in tests/test_kitti360_ncore.py."""
    K = np.array(info["camera_intrinsic"], dtype=np.float64)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64) - K[:2, 2]
    corner_r = np.max(np.hypot(corners[:, 0] * K[1, 1] / K[0, 0], corners[:, 1]))
    max_angle = float(np.arctan(corner_r / K[1, 1]) + 0.02)
    theta = np.linspace(0.0, max_angle, 4000)
    r = K[1, 1] * np.tan(theta)
    return FThetaCameraModelParameters(
        resolution=np.array([w, h], dtype=np.uint64), shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=np.array([K[0, 2], K[1, 2]], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=_poly_fit(r, theta),
        angle_to_pixeldist_poly=_poly_fit(theta, r),
        max_angle=max_angle,
        linear_cde=np.array([K[0, 0] / K[1, 1], 0.0, 0.0], dtype=np.float32),
    )


def nv_cross_120(c: str) -> FThetaCameraModelParameters:
    t = NV_CROSS_120[c]
    return FThetaCameraModelParameters(
        resolution=np.array([1920, 1080], dtype=np.uint64), shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=np.array(t["principal_point"], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array(t["pixeldist_to_angle_poly"], dtype=np.float32),
        angle_to_pixeldist_poly=np.array(t["angle_to_pixeldist_poly"], dtype=np.float32),
        max_angle=t["max_angle"], linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )


class FisheyeRectifier:
    """Resample a KITTI-360 MEI fisheye into a virtual 120-degree F-theta camera with the same pose.

    Instant NuRec approximates every context camera by a pinhole from the ray angle at the horizontal
    image edges, which breaks beyond 90 degrees (the KITTI-360 fisheyes reach ~95). The virtual camera
    also mostly avoids the car body: it sees ~20 degrees below the optical axis, the body starts ~29."""

    def __init__(self, info: dict, target: FThetaCameraModelParameters, src_w: int, src_h: int):
        import torch
        from ncore.impl.sensors.camera import CameraModel

        self.target = target
        tw, th = (int(v) for v in target.resolution)
        model = CameraModel.from_parameters(target, device="cpu", dtype=torch.float64)
        ys, xs = torch.meshgrid(torch.arange(th, dtype=torch.float64), torch.arange(tw, dtype=torch.float64), indexing="ij")
        rays = model.image_points_to_camera_rays(torch.stack([xs, ys], -1).reshape(-1, 2) + 0.5)  # pixel centres
        rays = torch.nn.functional.normalize(rays, dim=-1).numpy()
        # exact MEI projection (same as the 3DGUT trainer): unit sphere -> shift by xi -> radial distortion
        K = np.array(info["camera_intrinsic"], dtype=np.float64)
        k1, k2, _, _, xi = info["distortion_coefficients"]
        denom = rays[:, 2] + xi
        ok = denom > 1e-6
        xn, yn = rays[:, 0] / np.where(ok, denom, 1), rays[:, 1] / np.where(ok, denom, 1)
        rho2 = xn * xn + yn * yn
        d = 1 + k1 * rho2 + k2 * rho2 * rho2
        u, v = K[0, 0] * xn * d + K[0, 2], K[1, 1] * yn * d + K[1, 2]  # OpenCV: pixel centres at integers
        ok &= (u >= 0) & (u <= src_w - 1) & (v >= 0) & (v <= src_h - 1)
        grid = np.stack([2 * u / (src_w - 1) - 1, 2 * v / (src_h - 1) - 1], -1)  # align_corners=True
        grid[~ok] = -2.0  # outside -> zeros, flagged by self.invalid
        self.grid = torch.from_numpy(grid.reshape(1, th, tw, 2).astype(np.float32))
        self.invalid = ~ok.reshape(th, tw)

    def _sample(self, arr: np.ndarray, mode: str) -> np.ndarray:
        import torch

        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
        out = torch.nn.functional.grid_sample(t, self.grid, mode=mode, padding_mode="zeros", align_corners=True)
        out = out[0].permute(1, 2, 0).numpy()
        return out[..., 0] if arr.ndim == 2 else out

    def image(self, img: Image.Image) -> Image.Image:
        return Image.fromarray(np.clip(self._sample(np.asarray(img), "bilinear") + 0.5, 0, 255).astype(np.uint8))

    def mask(self, mask: Image.Image) -> Image.Image:
        bad = (self._sample(np.asarray(mask), "nearest") > 127) | self.invalid
        return Image.fromarray(bad.astype(np.uint8) * 255, mode="L")


def camera_params(info: dict, model: str, w: int, h: int, fisheye_model: str = "ftheta", pinhole_ftheta: bool = False):
    K = np.array(info["camera_intrinsic"], dtype=np.float64)
    res = np.array([w, h], dtype=np.uint64)
    # NCore's OpenCV models put pixel centres at +0.5; KITTI-360 (OpenCV) K puts them at integers
    pp = np.array([K[0, 2] + 0.5, K[1, 2] + 0.5], dtype=np.float32)
    if model == "fisheye" and fisheye_model == "ftheta":
        return mei_to_ftheta(info, w, h)
    if model == "pinhole" and pinhole_ftheta:
        return pinhole_to_ftheta(info, w, h)
    if model == "pinhole":
        return OpenCVPinholeCameraModelParameters(
            resolution=res, shutter_type=ShutterType.GLOBAL, external_distortion_parameters=None,
            principal_point=pp, focal_length=np.array([K[0, 0], K[1, 1]], dtype=np.float32),
            radial_coeffs=np.zeros(6, np.float32), tangential_coeffs=np.zeros(2, np.float32),
            thin_prism_coeffs=np.zeros(4, np.float32),
        )
    k1, k2, _, _, xi = info["distortion_coefficients"]
    return OpenCVFisheyeCameraModelParameters(
        resolution=res, shutter_type=ShutterType.GLOBAL, external_distortion_parameters=None,
        principal_point=pp, focal_length=np.array([K[0, 0] / (1 + xi), K[1, 1] / (1 + xi)], dtype=np.float32),
        radial_coeffs=mei_to_kb(xi, k1, k2), max_angle=FISHEYE_MAX_ANGLE,
    )


def mask_frames(tar_path: Path, size: tuple[int, int], limit: int = 200) -> np.ndarray:
    """Up to `limit` evenly spaced grayscale frames of the whole sequence at `size`
    (half resolution: the mask thresholds were tuned at the 3DGUT trainer's 700x700 scale)."""
    with tarfile.open(tar_path, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".png")]
        step = max(1, len(members) // limit)
        return np.stack([
            np.asarray(Image.open(tar.extractfile(m)).convert("L").resize(size, Image.BOX), dtype=np.float32)
            for m in members[::step][:limit]
        ])


def fisheye_ego_mask(gray_frames: np.ndarray, persist: float = 0.5) -> Image.Image:
    """255 = ignore: the car body and the black area outside the fisheye circle.

    Same method as the 3DGUT trainer (tools/colab_cli_kitti360.py, static_masks): the body's colours
    change with the light, but its *edges* stay at the same pixels in every frame while scene edges
    move, so edges that persist in the temporal mean image mark the body."""
    from scipy import ndimage, signal

    def sobel(g):
        return np.hypot(ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0))

    mean_img = gray_frames.mean(0)
    mean_grad = np.mean([sobel(g) for g in gray_frames], axis=0)  # average edge strength per pixel
    persistence = sobel(mean_img) / (mean_grad + 1.0)  # ~1 where edges never move
    h, w = mean_grad.shape
    edges = (persistence > persist) & (mean_grad > 15) & (np.arange(h)[:, None] > 0.63 * h)
    closed = ndimage.binary_closing(edges, structure=np.ones((31, 31)))
    # body = everything below the highest persistent edge of each column
    top = np.where(closed.any(0), closed.argmax(0), h).astype(np.float64)
    top = signal.medfilt(top, 61)  # one-column noise must not cut into the scene
    hood = np.arange(h)[:, None] >= top[None, :]
    outside = (mean_img < 12) & (gray_frames.std(0) < 6)
    bad = ndimage.binary_dilation(hood | outside, structure=np.ones((9, 9)))
    return Image.fromarray(bad.astype(np.uint8) * 255, mode="L")


# ------------------------------------------------------------------ conversion
def convert(args) -> Path:
    data, seq = Path(args.data), args.sequence
    items = load_items(data, seq)
    poses = load_poses(data, seq)
    frames = select_frames(items, poses, args.start, args.frames, args.max_gap)
    fids = [x["frame_id"] for x in frames]
    log(f"{seq}: {len(fids)} frames {fids[0]}..{fids[-1]}")
    wanted = {f"{f:010d}" for f in fids}
    cams = [c for c in args.cameras.split(",") if c]
    for c in cams:
        if c not in CAMERAS:
            sys.exit(f"unknown camera {c}; choose from {list(CAMERAS)}")
    drivestudio = getattr(args, "target", "nurec") == "drivestudio"
    if drivestudio:
        dropped = [c for c in cams if c not in DRIVESTUDIO_SENSOR_IDS]
        cams = [c for c in cams if c in DRIVESTUDIO_SENSOR_IDS]
        if dropped:
            log(f"--target drivestudio: skipping {dropped} (no matching PhysicalAI sensor id)")
        if "image_00" not in cams:
            sys.exit("--target drivestudio needs image_00: it becomes the reference front-wide camera")
        args.fisheye_model = "ftheta"

    def sensor_id(c: str) -> str:
        return DRIVESTUDIO_SENSOR_IDS[c] if drivestudio else CAMERAS[c][1]

    # ---- timeline: 10 Hz from frame id; sequence = [0, t_last + 1)
    t_us = {f: (f - fids[0]) * FRAME_DT_US for f in fids}
    interval = HalfClosedInterval(0, t_us[fids[-1]] + 1)

    # ---- extrinsics (static, from the first frame's calibration)
    cal = frames[0]["images"]
    lidar_to_cam = {c: np.array(cal[CAMERAS[c][0]]["lidar2cam"], dtype=np.float64) for c in CAMERAS}
    cam0_to_rig = R_CAM_TO_FLU
    lidar_to_rig = cam0_to_rig @ lidar_to_cam["image_00"]
    cam_to_rig = {c: lidar_to_rig @ np.linalg.inv(lidar_to_cam[c]) for c in cams}

    # ---- rig trajectory, world = first rig pose
    rig_to_kworld = np.stack([poses[f] @ np.linalg.inv(cam0_to_rig) for f in fids])
    rig_to_world = np.linalg.inv(rig_to_kworld[0])[None] @ rig_to_kworld
    rig_to_world[:, 3, :] = [0, 0, 0, 1]
    log(f"trajectory length {np.linalg.norm(np.diff(rig_to_world[:, :3, 3], axis=0), axis=1).sum():.1f} m")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq_id = args.sequence_id or f"kitti360_{seq.split('_drive_')[1].split('_')[0]}_{fids[0]:06d}_{len(fids)}f"
    if drivestudio and not seq_id.startswith("pai_"):
        seq_id = "pai_" + seq_id  # the PhysicalAI preprocessor globs **/pai_*.json
    writer = SequenceComponentGroupsWriter(
        output_dir_path=UPath(out), store_base_name=seq_id, sequence_id=seq_id,
        sequence_timestamp_interval_us=interval,
        generic_meta_data={
            "source": "KITTI-360", "source_sequence": seq, "frame_ids": fids,
            "converter": "fisheye-fusion-lab/tools/kitti360_to_ncore.py", "converter_version": "0.2.0",
            "timestamps": "synthetic 10 Hz from frame id",
        },
        store_type="itar",
    )

    # ---- poses
    pw = writer.register_component_writer(PosesComponent.Writer, "default")
    for c in cams:
        pw.store_static_pose(sensor_id(c), "rig", cam_to_rig[c])
    if args.lidar:
        pw.store_static_pose(LIDAR_ID, "rig", lidar_to_rig)
    pw.store_static_pose("world", "world_global", np.eye(4))
    pw.store_dynamic_pose("rig", "world", rig_to_world, np.array([t_us[f] for f in fids], dtype=np.uint64))

    # ---- cameras
    iw = writer.register_component_writer(IntrinsicsComponent.Writer, "default")
    mw = writer.register_component_writer(MasksComponent.Writer, "default")
    for c in cams:
        key, _, model = CAMERAS[c]
        sid = sensor_id(c)
        raw = read_tar(data / f"data_2d_raw/{seq}_{c}.tar", ".png", wanted)
        first = Image.open(io.BytesIO(raw[f"{fids[0]:010d}"]))
        w, h = first.size
        info = frames[0]["images"][key]
        rect = None
        if model == "fisheye" and args.fisheye_model == "rectify120":
            rect = FisheyeRectifier(info, nv_cross_120(c), w, h)
            iw.store_camera_intrinsics(sid, rect.target)
        else:
            iw.store_camera_intrinsics(sid, camera_params(info, model, w, h, args.fisheye_model, pinhole_ftheta=drivestudio))
        cw = writer.register_component_writer(CameraSensorComponent.Writer, sid, group_name=sid)
        for f in fids:
            img = Image.open(io.BytesIO(raw[f"{f:010d}"])).convert("RGB")
            if rect is not None:
                img = rect.image(img)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=args.jpeg_quality)
            ts = np.array([t_us[f], t_us[f]], dtype=np.uint64)
            cw.store_frame(buf.getvalue(), "jpeg", ts, {}, {})
        if model == "fisheye":
            # The body is a property of the rig, not of the clip: estimate it from the whole sequence
            # (a short straight clip keeps lane markings at fixed pixels and they would look like body edges).
            grays = mask_frames(data / f"data_2d_raw/{seq}_{c}.tar", (w // 2, h // 2))
            mask = fisheye_ego_mask(grays).resize((w, h), Image.NEAREST)
            if rect is not None:
                mask = rect.mask(mask)
            log(f"{c}: ego mask covers {np.asarray(mask).mean() / 2.55:.0f}% of pixels")
        else:
            mask = Image.new("L", (w, h), 0)
        mw.store_camera_masks(sid, {"ego": mask})
        size = "x".join(str(int(v)) for v in rect.target.resolution) if rect is not None else f"{w}x{h}"
        log(f"{c} -> {sid} ({model if rect is None else 'fisheye -> F-theta 120deg'}, {size}, {len(fids)} frames)")
        del raw

    # ---- lidar
    if args.lidar:
        raw = read_tar(data / f"data_3d_raw/{seq}.tar", ".bin", wanted)
        lw = writer.register_component_writer(LidarSensorComponent.Writer, LIDAR_ID, group_name=LIDAR_ID)
        for f in fids:
            pts = np.frombuffer(raw[f"{f:010d}"], np.float32).reshape(-1, 4)
            dist = np.linalg.norm(pts[:, :3], axis=1)
            keep = dist > 1.0
            pts, dist = pts[keep], dist[keep]
            n = len(pts)
            lw.store_frame(
                direction=(pts[:, :3] / dist[:, None]).astype(np.float32),
                timestamp_us=np.full(n, t_us[f], dtype=np.uint64),
                model_element=None,
                distance_m=dist[None].astype(np.float32),
                intensity=np.clip(pts[:, 3], 0, 1)[None].astype(np.float32),
                frame_timestamps_us=np.array([t_us[f], t_us[f]], dtype=np.uint64),
                generic_data={}, generic_meta_data={},
            )
        log(f"lidar -> {LIDAR_ID} ({len(fids)} scans)")
        del raw

    # ---- cuboids: the NuRec loader needs the component (empty is fine); --target drivestudio fills it with the
    # ROSA boxes as tracks, which need the LiDAR pose to place them in the world
    cuboids = cuboid_observations(frames, t_us) if drivestudio and args.lidar and args.objects else []
    writer.register_component_writer(CuboidsComponent.Writer, "default").store_observations(cuboids)
    if cuboids:
        log(f"cuboids: {len(cuboids)} boxes in {len({o.track_id for o in cuboids})} tracks")

    store_paths = writer.finalize()
    meta = SequenceComponentGroupsReader(store_paths).get_sequence_meta()
    json_path = out / f"{seq_id}.json"
    json_path.write_text(json.dumps(meta.to_dict(), indent=2))
    log(f"wrote {json_path} ({len(store_paths)} stores)")
    return json_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="./kitti360", help="KITTI-360 archive root")
    p.add_argument("--out", required=True, help="output directory for the NCore clip")
    p.add_argument("--sequence", default="2013_05_28_drive_0003_sync")
    p.add_argument("--start", type=int, default=0, help="first frame id to consider")
    p.add_argument("--frames", type=int, default=150, help="number of frames (10 Hz; NuRec context needs >= 9 s)")
    p.add_argument("--max-gap", type=int, default=5, help="frame-id gap that ends a run of consecutive frames")
    p.add_argument("--cameras", default="image_00,image_01,image_02,image_03")
    p.add_argument("--no-lidar", dest="lidar", action="store_false")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--fisheye-model", choices=["rectify120", "ftheta", "opencv"], default="rectify120",
                   help="rectify120: resample to NVIDIA's 120-degree F-theta camera (Instant NuRec); "
                        "ftheta: full fisheye as F-theta; opencv: full fisheye as Kannala-Brandt (3DGUT/gsplat)")
    p.add_argument("--sequence-id", default="")
    p.add_argument("--no-objects", dest="objects", action="store_false",
                   help="--target drivestudio: do not write the ROSA 3D boxes as cuboid tracks")
    p.add_argument("--target", choices=["nurec", "drivestudio"], default="nurec",
                   help="nurec: Instant NuRec clip (default); drivestudio: clip for DriveStudio+'s PhysicalAI "
                        "preprocessor (F-theta cameras, PhysicalAI sensor ids, pai_ prefix; image_00/02/03 only). "
                        "Preprocess it with --frame_stride 1, KITTI-360 is 10 Hz")
    args = p.parse_args(argv)
    print(convert(args))


if __name__ == "__main__":
    main()
