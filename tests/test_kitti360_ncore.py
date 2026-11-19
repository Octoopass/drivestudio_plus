"""Camera models written by tools/kitti360_to_ncore.py, checked with NCore's own camera code.

Needs `nvidia-ncore` and torch, i.e. the Instant NuRec venv:
    backends/instant-nurec/.venv/bin/python -m pytest tests/test_kitti360_ncore.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ncore")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import kitti360_to_ncore as k  # noqa: E402
from ncore.impl.sensors.camera import CameraModel  # noqa: E402

# KITTI-360 calibration of image_02 / image_03 (MEI): fx, fy, cx, cy, k1, k2, xi
CALIB = {
    "image_02": (1336.3221, 1335.7883, 716.94324, 705.76495, 0.016798235, 1.6548773, 2.2134047),
    "image_03": (1485.4388, 1484.9478, 698.8832, 698.14545, 0.049370397, 4.5068455, 2.553514),
}


def info(cam):
    fx, fy, cx, cy, k1, k2, xi = CALIB[cam]
    return {"camera_intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "distortion_coefficients": [k1, k2, 0, 0, xi]}


def mei_project(i, rays):
    K = np.array(i["camera_intrinsic"])
    k1, k2, _, _, xi = i["distortion_coefficients"]
    d = rays[:, 2] + xi
    xn, yn = rays[:, 0] / d, rays[:, 1] / d
    r2 = xn**2 + yn**2
    s = 1 + k1 * r2 + k2 * r2**2
    return np.stack([K[0, 0] * xn * s + K[0, 2], K[1, 1] * yn * s + K[1, 2]], 1)


def model(params):
    return CameraModel.from_parameters(params, device="cpu", dtype=torch.float64)


def image_points(m, rays):
    r = m.camera_rays_to_image_points(torch.from_numpy(rays))
    return (r.image_points if hasattr(r, "image_points") else r[0]).numpy()


def rays_up_to(theta_max, n=5000, seed=0):
    g = np.random.default_rng(seed)
    t, p = g.uniform(0, theta_max, n), g.uniform(0, 2 * np.pi, n)
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], 1)


@pytest.mark.parametrize("cam", CALIB)
@pytest.mark.parametrize("fisheye_model,tol", [("ftheta", 0.5), ("opencv", 0.01)])
def test_fisheye_models_reproduce_mei(cam, fisheye_model, tol):
    """Both NCore fisheye models land on the exact MEI pixel (OpenCV convention, +0.5 in NCore)."""
    rays = rays_up_to(1.42)
    got = image_points(model(k.camera_params(info(cam), "fisheye", 1400, 1400, fisheye_model)), rays)
    want = mei_project(info(cam), rays) + 0.5
    assert np.abs(got - want).max() < tol


def test_pinhole_principal_point_follows_ncore_convention():
    i = {"camera_intrinsic": [[552.55, 0, 682.05], [0, 552.55, 238.77], [0, 0, 1]]}
    rays = np.array([[0.0, 0.0, 1.0], [0.1, -0.05, 1.0]])
    got = image_points(model(k.camera_params(i, "pinhole", 1408, 376)), rays)
    want = np.array([[682.05, 238.77], [682.05 + 55.255, 238.77 - 27.6275]]) + 0.5
    assert np.abs(got - want).max() < 1e-3


@pytest.mark.parametrize("cam", CALIB)
def test_rectifier_samples_the_same_ray(cam):
    """A pixel of the virtual 120-degree camera samples the source pixel its ray projects to."""
    rect = k.FisheyeRectifier(info(cam), k.nv_cross_120(cam), 1400, 1400)
    tm = model(rect.target)
    th, tw = rect.invalid.shape
    assert (th, tw) == (1080, 1920) and not rect.invalid.any()  # fully inside the fisheye circle
    px = np.array([[0, 0], [tw - 1, 0], [tw // 2, th // 2], [tw - 1, th - 1], [137, 911]])
    rays = tm.image_points_to_camera_rays(torch.from_numpy(px + 0.5).double())
    rays = torch.nn.functional.normalize(rays, dim=-1).numpy()
    src = mei_project(info(cam), rays)
    g = rect.grid[0].numpy()[px[:, 1], px[:, 0]]
    sampled = np.stack([(g[:, 0] + 1) / 2 * 1399, (g[:, 1] + 1) / 2 * 1399], 1)
    assert np.abs(sampled - src).max() < 0.05


def test_rectifier_moves_image_and_mask_content():
    i = info("image_02")
    rect = k.FisheyeRectifier(i, k.nv_cross_120("image_02"), 1400, 1400)
    # a bright dot at the source pixel of the virtual camera's centre must show up there
    tm = model(rect.target)
    ray = torch.nn.functional.normalize(tm.image_points_to_camera_rays(torch.tensor([[960.5, 540.5]]).double()), dim=-1).numpy()
    u, v = np.rint(mei_project(i, ray)[0]).astype(int)
    img = np.zeros((1400, 1400, 3), np.uint8)
    img[v - 3 : v + 4, u - 3 : u + 4] = 255
    out = np.asarray(rect.image(k.Image.fromarray(img)))
    assert out.shape == (1080, 1920, 3) and out[540, 960].min() > 200 and out[100, 100].max() == 0
    mask = np.zeros((1400, 1400), np.uint8)
    mask[v - 3 : v + 4, u - 3 : u + 4] = 255
    m = np.asarray(rect.mask(k.Image.fromarray(mask)))
    assert m[540, 960] == 255 and m[100, 100] == 0


def test_eval_export_refits_ftheta_as_kannala_brandt():
    import ncore_eval_export as ev

    for cam in CALIB:
        params = k.nv_cross_120(cam)
        K, radial = ev.ftheta_as_kb(params, cam)
        rays = rays_up_to(1.25, seed=1)
        want = image_points(model(params), rays)
        kb = k.OpenCVFisheyeCameraModelParameters(
            resolution=params.resolution, shutter_type=params.shutter_type, external_distortion_parameters=None,
            principal_point=K[:2, 2], focal_length=np.array([K[0, 0], K[1, 1]], np.float32),
            radial_coeffs=radial, max_angle=float(params.max_angle))
        assert np.abs(image_points(model(kb), rays) - want).max() < 0.5


def test_pinhole_as_ftheta_matches_the_perspective_projection():
    """--target drivestudio writes image_00 as F-theta; it must land where the pinhole projects (+0.5 in NCore)."""
    i = {"camera_intrinsic": [[552.554261, 0, 682.049453], [0, 552.554261, 238.769549], [0, 0, 1]]}
    params = k.camera_params(i, "pinhole", 1408, 376, pinhole_ftheta=True)
    assert hasattr(params, "angle_to_pixeldist_poly")
    g = np.random.default_rng(2)
    uv = np.column_stack([g.uniform(0, 1408, 4000), g.uniform(0, 376, 4000)])  # every pixel of the image
    rays = np.column_stack([(uv[:, 0] - 682.049453) / 552.554261, (uv[:, 1] - 238.769549) / 552.554261, np.ones(len(uv))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    got = image_points(model(params), rays)
    assert np.abs(got - (uv + 0.5)).max() < 0.3
