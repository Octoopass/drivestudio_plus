r"""Kiểm tra hình học một cảnh KITTI-360 đã ingest: chiếu LiDAR lên ảnh và so với chuỗi chính thức.

Công cụ phát triển chạy một lần (ARCHITECTURE mục 3.1), không nằm trong luồng chính. Cần numpy và
OpenCV (môi trường GPU ``drive2sim``), nên không chạy trong ``.venv`` nhẹ. Cách làm theo
``tools/kitti360_projection_check.py`` của teammate (repo thaianh20021/fisheye-fusion-lab):

- **Chuỗi D (dữ liệu chuẩn)**: đọc ``lidar/``, ``extrinsics/``, ``intrinsics/``, ``ego_pose/`` của
  cảnh đã ingest, chiếu bằng mô hình mà trainer dùng (pinhole; OpenCV fisheye đã khớp, chỉ tia có
  ``theta < 1.45`` rad).
- **Chuỗi A (chính thức)**: tự dựng lại từ ``calibration/`` của nguồn bằng numpy, đúng quy ước
  ``kitti360Scripts`` (Velodyne → camera 0 bằng nghịch đảo ``calib_cam_to_velo``, camera 0 →
  camera k qua ``calib_cam_to_pose``, ``R_rect`` cho camera pinhole, Mei chính xác có cả méo tiếp
  tuyến cho camera mắt cá). In độ lệch pixel giữa D và A.
- **Chuỗi C (nhiều khung)**: LiDAR khung ``t`` đưa lên world bằng ``ego_pose`` rồi chiếu vào
  camera 0 ở khung ``t + offset``; so với cùng phép đó dùng ``cam0_to_world.txt`` chính thức.
- **Điểm cạnh** (``edge_ratio``): độ mạnh cạnh ảnh dưới các điểm LiDAR có bước nhảy độ sâu, chia
  cho độ mạnh cạnh dưới mọi điểm; calib đúng cho tỉ số lớn hơn khi làm lệch calib 1 độ.

Ví dụ::

    python -m drive2sim.scripts.check_kitti360_projection \\
        --scene <workdir>/data/kitti360_... --raw <thư mục KITTI-360> --frame 0 --out /tmp/check

In một JSON kết quả ra stdout và ghi ảnh chồng điểm vào ``--out``.
"""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from drive2sim.ingest.kitti360 import _read_mei, _read_records

FIT_MAX_ANGLE = 1.45


def _loadtxt(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def _to4(values: list) -> np.ndarray:
    m = np.eye(4)
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 9:
        m[:3, :3] = flat.reshape(3, 3)
    else:
        m[:3] = flat[:12].reshape(3, 4)
    return m


def project_pinhole(points: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple:
    """Chiếu điểm trong hệ camera bằng pinhole; trả ``(uv, hợp lệ)``."""
    z = points[:, 2]
    valid = z > 0.5
    zs = np.where(valid, z, 1.0)
    uv = np.column_stack([fx * points[:, 0] / zs + cx, fy * points[:, 1] / zs + cy])
    return uv, valid


def project_fisheye(points: np.ndarray, intr: np.ndarray, max_angle: float) -> tuple:
    """Chiếu bằng OpenCV fisheye (mô hình gsplat dùng); chỉ giữ tia có ``theta < max_angle``."""
    fx, fy, cx, cy, k1, k2, k3, k4 = intr
    norm_xy = np.linalg.norm(points[:, :2], axis=1)
    theta = np.arctan2(norm_xy, points[:, 2])
    t2 = theta * theta
    theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(norm_xy > 1e-9, theta_d / np.maximum(norm_xy, 1e-9), 1.0)
    uv = np.column_stack([fx * points[:, 0] * scale + cx, fy * points[:, 1] * scale + cy])
    valid = (np.linalg.norm(points, axis=1) > 0.5) & (theta < max_angle)
    return uv, valid, theta


def project_mei(points: np.ndarray, mei: dict) -> tuple:
    """Chiếu bằng mô hình Mei chính xác (có méo tiếp tuyến), như kitti360Scripts."""
    norm = np.linalg.norm(points, axis=1)
    denom = points[:, 2] + mei["xi"] * norm
    valid = (norm > 0.5) & (denom > 1e-6)
    d = np.where(valid, denom, 1.0)
    x, y = points[:, 0] / d, points[:, 1] / d
    r2 = x * x + y * y
    radial = 1 + mei["k1"] * r2 + mei["k2"] * r2 * r2
    xd = x * radial + 2 * mei["p1"] * x * y + mei["p2"] * (r2 + 2 * x * x)
    yd = y * radial + mei["p1"] * (r2 + 2 * y * y) + 2 * mei["p2"] * x * y
    uv = np.column_stack([mei["gamma1"] * xd + mei["u0"], mei["gamma2"] * yd + mei["v0"]])
    return uv, valid


def edge_ratio(gray: np.ndarray, uv: np.ndarray, valid: np.ndarray, depth: np.ndarray) -> float:
    """Tỉ số độ mạnh cạnh dưới điểm có bước nhảy độ sâu so với dưới mọi điểm (như teammate)."""
    h, w = gray.shape
    inside = valid & (uv[:, 0] >= 1) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 1)
    if inside.sum() < 200:
        return float("nan")
    edges = cv2.GaussianBlur(cv2.Canny(gray, 50, 150).astype(np.float32), (0, 0), 2.0)
    px = uv[inside].astype(int)
    d = depth[inside]
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(zbuf, (px[:, 1], px[:, 0]), d)
    kernel = np.ones((11, 11), np.uint8)
    near = cv2.erode(zbuf, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=np.inf)
    far = cv2.dilate(
        np.where(np.isinf(zbuf), -np.inf, zbuf).astype(np.float32),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=-np.inf,
    )
    jump = np.nan_to_num(far - near)[px[:, 1], px[:, 0]]
    disc = jump > 0.1 * np.clip(d, 1, None) + 0.5
    all_mean = edges[px[:, 1], px[:, 0]].mean()
    disc_mean = edges[px[disc, 1], px[disc, 0]].mean() if disc.any() else np.nan
    return float(disc_mean / max(all_mean, 1e-6))


def _yaw(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    m = np.eye(4)
    m[0, 0] = m[2, 2] = np.cos(a)
    m[0, 2], m[2, 0] = np.sin(a), -np.sin(a)
    return m


def _overlay(
    image: np.ndarray, uv: np.ndarray, valid: np.ndarray, depth: np.ndarray
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    hue = (np.clip(np.log(np.maximum(depth[inside], 1)) / np.log(60), 0, 1) * 120).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], -1)[None]
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    for (u, v), color in zip(uv[inside].astype(int), colors):
        cv2.circle(out, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)
    return out


def main() -> None:
    """Chạy kiểm tra và in kết quả JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", type=Path, required=True, help="thư mục cảnh đã ingest")
    parser.add_argument("--raw", type=Path, required=True, help="thư mục gốc KITTI-360")
    parser.add_argument("--frame", type=int, default=0, help="chỉ số khung trong cảnh")
    parser.add_argument("--offset", type=int, default=5, help="độ lệch khung cho chuỗi C")
    parser.add_argument("--out", type=Path, required=True, help="thư mục ghi ảnh chồng điểm")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scene, t = args.scene, args.frame
    manifest_text = (scene / "manifest.yaml").read_text()
    cameras = [
        line.split(":")[1].strip() for line in manifest_text.splitlines() if "camera_id" in line
    ]
    frame_ids = [int(v) for v in (scene / "frame_ids.txt").read_text().split()]
    sequence = next(
        line.split(":")[1].strip() for line in manifest_text.splitlines() if "sequence_id" in line
    )

    # Chuỗi A: calib chính thức, numpy.
    calib = args.raw / "calibration"
    to_pose = {
        k: _to4(v) for k, v in _read_records((calib / "calib_cam_to_pose.txt").read_text()).items()
    }
    cam0_from_velo = np.linalg.inv(_to4(np.loadtxt(calib / "calib_cam_to_velo.txt")))
    persp = _read_records((calib / "perspective.txt").read_text())

    def official_T(camera: str) -> np.ndarray:
        m = np.linalg.inv(to_pose[camera]) @ to_pose["image_00"] @ cam0_from_velo
        if camera in ("image_00", "image_01"):
            m = _to4(persp[f"R_rect_{camera[-2:]}"]) @ m
        return m

    points_velo = np.fromfile(scene / "lidar" / f"{t:03d}.bin", dtype=np.float32).reshape(-1, 4)
    points_velo = points_velo[:, :3].astype(np.float64)
    homog = np.column_stack([points_velo, np.ones(len(points_velo))])
    result: dict = {"frame_index": t, "source_frame": frame_ids[t], "cameras": {}}
    for index, camera in enumerate(cameras):
        image = cv2.imread(str(scene / "images" / f"{t:03d}_{index}.png"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        intr = _loadtxt(scene / "intrinsics" / f"{index}.txt")
        T_ours = np.linalg.inv(_loadtxt(scene / "extrinsics" / f"{index}.txt"))
        pc_ours = (homog @ T_ours.T)[:, :3]
        pc_off = (homog @ official_T(camera).T)[:, :3]
        entry: dict = {"extrinsic_max_abs_diff": float(np.abs(T_ours - official_T(camera)).max())}
        depth = np.linalg.norm(pc_ours, axis=1)
        if intr.size == 9:
            fx, fy, cx, cy = intr[:4]
            uv, valid = project_pinhole(pc_ours, fx, fy, cx, cy)
            P = np.array(persp[f"P_rect_{camera[-2:]}"]).reshape(3, 4)
            uv_a, valid_a = project_pinhole(pc_off, P[0, 0], P[1, 1], P[0, 2], P[1, 2])
            wrong = (homog @ (_yaw(1.0) @ T_ours).T)[:, :3]
            uv_w, valid_w = project_pinhole(wrong, fx, fy, cx, cy)
        else:
            uv, valid, theta = project_fisheye(pc_ours, intr, FIT_MAX_ANGLE)
            mei = _read_mei((calib / f"{camera}.yaml").read_text(), camera)
            uv_a, valid_a = project_mei(pc_off, mei)
            wrong = (homog @ (_yaw(1.0) @ T_ours).T)[:, :3]
            uv_w, valid_w, _ = project_fisheye(wrong, intr, FIT_MAX_ANGLE)
            entry["max_theta_rad_of_points"] = float(theta[valid].max()) if valid.any() else None
        h, w = gray.shape
        both = valid & valid_a & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0)
        both &= uv[:, 1] < h
        diff = np.linalg.norm(uv[both] - uv_a[both], axis=1)
        entry.update(
            points_in_image=int(both.sum()),
            pixel_diff_vs_official_median=float(np.median(diff)) if diff.size else None,
            pixel_diff_vs_official_p99=float(np.percentile(diff, 99)) if diff.size else None,
            pixel_diff_vs_official_max=float(diff.max()) if diff.size else None,
            edge_ratio=edge_ratio(gray, uv, valid, depth),
            edge_ratio_yaw_1deg=edge_ratio(gray, uv_w, valid_w, depth),
        )
        cv2.imwrite(str(args.out / f"overlay_{camera}.png"), _overlay(image, uv, valid, depth))
        result["cameras"][camera] = entry

    # Chuỗi C: khung t lên world, chiếu vào camera 0 ở khung t + offset.
    t2 = min(t + args.offset, len(frame_ids) - 1)
    ego_t, ego_t2 = (_loadtxt(scene / "ego_pose" / f"{i:03d}.txt") for i in (t, t2))
    E0 = _loadtxt(scene / "extrinsics" / "0.txt")
    world = homog @ ego_t.T
    cam_ours = (world @ np.linalg.inv(ego_t2 @ E0).T)[:, :3]
    with zipfile.ZipFile(args.raw / "data_poses.zip") as zipped:
        text = zipped.read(f"{sequence}/cam0_to_world.txt").decode()
    c2w = {int(r.split()[0]): _to4(r.split()[1:13]) for r in text.splitlines() if r.strip()}
    rect0_from_velo = official_T("image_00")
    cam_off = homog @ (np.linalg.inv(c2w[frame_ids[t2]]) @ c2w[frame_ids[t]] @ rect0_from_velo).T
    cam_off = cam_off[:, :3]
    fx, fy, cx, cy = _loadtxt(scene / "intrinsics" / "0.txt")[:4]
    uv_c, valid_c = project_pinhole(cam_ours, fx, fy, cx, cy)
    uv_o, valid_o = project_pinhole(cam_off, fx, fy, cx, cy)
    image = cv2.imread(str(scene / "images" / f"{t2:03d}_0.png"))
    h, w = image.shape[:2]
    both = valid_c & valid_o & (uv_c[:, 0] >= 0) & (uv_c[:, 0] < w) & (uv_c[:, 1] >= 0)
    both &= uv_c[:, 1] < h
    diff = np.linalg.norm(uv_c[both] - uv_o[both], axis=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth = np.linalg.norm(cam_ours, axis=1)
    result["multi_frame"] = {
        "from_frame": frame_ids[t],
        "to_frame": frame_ids[t2],
        "points_in_image": int(both.sum()),
        "pixel_diff_vs_cam0_to_world_median": float(np.median(diff)) if diff.size else None,
        "pixel_diff_vs_cam0_to_world_max": float(diff.max()) if diff.size else None,
        "edge_ratio": edge_ratio(gray, uv_c, valid_c, depth),
    }
    cv2.imwrite(
        str(args.out / "overlay_multiframe_image_00.png"), _overlay(image, uv_c, valid_c, depth)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
