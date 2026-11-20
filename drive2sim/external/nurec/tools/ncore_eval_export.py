"""(drive2sim: copied from Fisheye Fusion Lab tools/, plus --frame-indices.)

Step 1 of evaluating a 3DGS PLY against an NCore clip: export cameras + ground truth.

    <ncore-python> tools/ncore_eval_export.py clip.json --out views.npz [--every 4] [--cameras a,b]

Writes, for every selected camera frame: world->camera matrix (OpenCV camera, NCore world),
intrinsics (pinhole K or OpenCV-fisheye K + radial coeffs), image and ego mask. Step 2
(tools/ncore_eval_render.py) renders the PLY with gsplat and reports PSNR.
Cameras: OpenCV pinhole, OpenCV fisheye, and F-theta (refitted as Kannala-Brandt, which gsplat
renders; the fit error is printed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from upath import UPath

from ncore.data.v4 import (
    CameraSensorComponent,
    IntrinsicsComponent,
    MasksComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
)
from ncore.impl.data.types import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
)


def interp_pose(poses: np.ndarray, ts: np.ndarray, t: int) -> np.ndarray:
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    a = (t - ts[i - 1]) / max(1, ts[i] - ts[i - 1])
    if a <= 0:
        return poses[i - 1]
    if a >= 1:
        return poses[i]
    # translation-only interpolation is enough at camera timestamps that coincide with pose samples
    out = poses[i - 1].copy()
    out[:3, 3] = (1 - a) * poses[i - 1][:3, 3] + a * poses[i][:3, 3]
    return out


def ftheta_as_kb(params: FThetaCameraModelParameters, cam_id: str):
    """Kannala-Brandt (r = f (t + k1 t^3 + k2 t^5 + k3 t^7 + k4 t^9)) fitted to the F-theta forward
    polynomial. Returns K (pixel-corner convention, like NCore's OpenCV models) and radial coeffs."""
    c, d, e = (float(x) for x in params.linear_cde)
    if abs(d) > 1e-6 or abs(e) > 1e-6:
        raise ValueError(f"{cam_id}: skewed F-theta linear term not supported")
    t = np.linspace(1e-3, float(params.max_angle), 2000)
    r = np.polynomial.polynomial.polyval(t, params.angle_to_pixeldist_poly.astype(np.float64))
    # r / t = f + f k1 t^2 + ... : linear least squares in (f, f k1, ..., f k4)
    A = np.stack([t ** (2 * i) for i in range(5)], 1)
    sol, *_ = np.linalg.lstsq(A, r / t, rcond=None)
    f, radial = sol[0], (sol[1:] / sol[0]).astype(np.float32)
    err = np.abs(t * (A @ sol) - r).max()
    print(f"{cam_id}: F-theta -> Kannala-Brandt fit, max error {err:.3f} px")
    pp = params.principal_point.astype(np.float64) + 0.5  # F-theta: pixel-centre convention
    K = np.array([[f * c, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]], np.float32)
    return K, radial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=4, help="use every Nth frame of each camera")
    p.add_argument("--cameras", default="", help="comma-separated camera ids (default: all)")
    p.add_argument("--frame-indices", default="", help="comma-separated frame indices to export (drive2sim: the split's test frames); overrides --every")
    a = p.parse_args()

    meta = json.loads(Path(a.json).read_text())
    base = UPath(a.json).parent
    r = SequenceComponentGroupsReader([base / s["path"] for s in meta["component_stores"]])
    poses = r.open_component_readers(PosesComponent.Reader)["default"]
    intr = r.open_component_readers(IntrinsicsComponent.Reader)["default"]
    masks = r.open_component_readers(MasksComponent.Reader)["default"]
    rig_world, rig_ts = poses.get_dynamic_pose("rig", "world")
    wanted = [c for c in a.cameras.split(",") if c]

    rec = {k: [] for k in ("camera", "model", "viewmat", "K", "radial", "image", "mask", "frame")}
    for cam_id, cam in r.open_component_readers(CameraSensorComponent.Reader).items():
        if wanted and cam_id not in wanted:
            continue
        params = intr.get_camera_model_parameters(cam_id)
        if isinstance(params, OpenCVPinholeCameraModelParameters):
            model, radial = "pinhole", np.zeros(4, np.float32)
        elif isinstance(params, OpenCVFisheyeCameraModelParameters):
            model, radial = "fisheye", params.radial_coeffs.astype(np.float32)
        elif isinstance(params, FThetaCameraModelParameters):
            model = "fisheye"
            K, radial = ftheta_as_kb(params, cam_id)
        else:
            print(f"skip {cam_id}: {type(params).__name__} not supported")
            continue
        if not isinstance(params, FThetaCameraModelParameters):
            K = np.array([[params.focal_length[0], 0, params.principal_point[0]],
                          [0, params.focal_length[1], params.principal_point[1]], [0, 0, 1]], np.float32)
        T_cam_rig = poses.get_static_pose(cam_id, "rig")
        names = masks.get_camera_mask_names(cam_id)
        mask = np.asarray(masks.get_camera_mask_image(cam_id, "ego")) > 0 if "ego" in names else None
        stamps = cam.frames_timestamps_us[:, 1]
        picked = [int(i) for i in a.frame_indices.split(",") if i] if a.frame_indices else None
        stamps = stamps[[i for i in picked if 0 <= i < len(stamps)]] if picked is not None else stamps[:: a.every]
        for t_end in stamps:
            T_cam_world = interp_pose(rig_world, rig_ts, int(t_end)) @ T_cam_rig
            img = np.asarray(cam.get_frame_image(int(t_end)).convert("RGB"))
            rec["camera"].append(cam_id)
            rec["frame"].append(int(list(cam.frames_timestamps_us[:, 1]).index(t_end)))
            rec["model"].append(model)
            rec["viewmat"].append(np.linalg.inv(T_cam_world).astype(np.float32))
            rec["K"].append(K)
            rec["radial"].append(radial)
            rec["image"].append(img)
            rec["mask"].append(mask if mask is not None else np.zeros(img.shape[:2], bool))
        print(f"{cam_id}: {model}, {len(stamps)} views")

    out = {k: np.stack(rec[k]) for k in ("viewmat", "K", "radial")}
    out["camera"], out["model"] = np.array(rec["camera"]), np.array(rec["model"])
    out["frame"] = np.array(rec["frame"], dtype=np.int64)
    for k in ("image", "mask"):  # images differ in size between cameras
        arr = np.empty(len(rec[k]), dtype=object)
        arr[:] = rec[k]
        out[k] = arr
    np.savez(a.out, **out)
    print(f"wrote {a.out} ({len(rec['camera'])} views)")


if __name__ == "__main__":
    main()
