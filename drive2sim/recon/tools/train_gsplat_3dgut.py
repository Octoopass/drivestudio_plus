"""Train một cảnh tĩnh bằng Gaussian 3DGUT của gsplat, cho camera pinhole và mắt cá cùng lúc.

Port từ trainer KITTI-360 của teammate (``tools/colab_cli_kitti360.py``, repo
thaianh20021/fisheye-fusion-lab), đổi phần dữ liệu: không tải gì, không đọc ``.pkl``; đọc thư mục
cảnh chuẩn của tool (khung KITTI của DriveStudio: ``images/``, ``lidar/``, ``ego_pose/``,
``extrinsics/``, ``intrinsics/``, ``manifest.yaml``) và file split của run (CONTRACTS mục 2.1,
3.1). Giữ nguyên thiết kế train của teammate:

- Gaussian khởi tạo từ điểm LiDAR (hệ ego = Velodyne) đưa lên world, màu lấy từ camera cùng khung
  thấy điểm đó; tối đa ``--max_init_gaussians`` điểm; kích thước đầu = khoảng cách trung bình tới
  3 điểm gần nhất.
- Tuỳ chọn vòm trời: Gaussian lớn trên mặt cầu 200 m quanh xe, màu lấy từ ảnh.
- Render bằng ``gsplat.rasterization(..., with_ut=True, with_eval3d=True, packed=False)`` (3DGUT),
  ``camera_model="fisheye"`` với ``radial_coeffs`` k1..k4 cho camera mắt cá.
- Densify bằng ``MCMCStrategy`` (3DGUT không có gradient ``means2d`` cho ``DefaultStrategy``).
- Tối ưu bằng ``SelectiveAdam`` (chỉ cập nhật Gaussian thấy được trong ảnh đang train, kiểu
  Fast3DGS) hoặc ``torch.optim.Adam``. Lưu ý: kernel CUDA ``SelectiveAdam`` của gsplat
  (``gsplat/cuda/csrc/AdamCUDA.cu``) **không hiệu chỉnh bias** của Adam, nên mấy trăm bước đầu
  bước cập nhật lớn hơn Adam chuẩn.
- Loss: L1 trên vùng được giám sát, tuỳ chọn ``0.8·L1 + 0.2·(1 − SSIM)`` (SSIM bằng conv2d của
  PyTorch hoặc ``fused-ssim``), cộng ``0.01·mean(opacity) + 0.01·mean(scale)``.
- Mask: luôn che vùng ảnh mắt cá có tia lệch trục hơn ``--fisheye_max_angle`` (gsplat UT không vẽ
  được tia gần và quá pi/2, đa thức chỉ khớp tới 1,45 rad); tuỳ chọn che nắp capô và vùng đen ngoài
  vòng tròn mắt cá (``static_masks`` của teammate). Đánh giá luôn dùng cả hai mask.
- ``scene_scale`` (nhân learning rate của vị trí) = 1,1 × bán kính quỹ đạo, kẹp trong [5, 20] m.

Khác teammate (có lý do):
- Khung train và test lấy từ split của tool, không tự chia mỗi khung thứ 10.
- Màu khởi tạo và mask capô chỉ dùng ảnh train (teammate dùng cả ảnh test, làm rò tập test).
- Mask vùng ngoài ``--fisheye_max_angle`` là mới: teammate để renderer vẽ đen ở đó rồi vẫn tính
  loss và PSNR toàn ảnh.
- Metric theo protocol của tool (SPEC mục 9): ảnh render làm tròn 8 bit; PSNR trên điểm ảnh hợp
  lệ; SSIM của ``skimage``; LPIPS ``torchmetrics`` backbone ``alex``.

Kết quả ghi vào ``--output_dir`` theo đúng tên file mà ``eval.standard``, ``export.ply_world`` và
``export.preview`` đọc (CONTRACTS mục 6.1, 6.3, 7): ``checkpoint_final.pth`` (node ``Background``
theo tên trường của DriveStudio), ``train_stats.json``, ``metrics/images_test_<thời điểm>.json``,
``cameras.json``; thêm ``compare_<camera>.png`` và ``masks/``. Chạy tiếp được từ
``checkpoint_resume.pth`` (``--resume_from``).

Chạy::

    python -m drive2sim.recon.tools.train_gsplat_3dgut --scene_dir data/<scene_id> \\
        --split_file data/<scene_id>/splits/<split_id>.json --output_dir <thư mục>
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from drive2sim.core.camera import fisheye_radius

FINAL_CHECKPOINT = "checkpoint_final.pth"
RESUME_CHECKPOINT = "checkpoint_resume.pth"
TRAIN_STATS = "train_stats.json"
CAMERAS_FILE = "cameras.json"
SH_C0 = 0.28209479177387814
SKY_RADIUS, SKY_SCALE, SKY_PER_FRAME = 200.0, 12.0, 400
# LPIPS của torchmetrics dựng AlexNet của torchvision với trọng số ImageNet. Trainer không tải gì:
# trọng số phải có sẵn trong cache của torch hub (docs/methods/gsplat_3dgut.md).
ALEXNET_WEIGHTS = "alexnet-owt-7be5be79.pth"


def parse_args(argv=None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh; stage ``train.gsplat_3dgut`` dựng đúng các tham số này."""
    p = argparse.ArgumentParser(description="Train cảnh tĩnh bằng 3DGUT của gsplat")
    p.add_argument("--scene_dir", type=Path, required=True)
    p.add_argument("--split_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--resume_from", type=Path, default=None)
    p.add_argument("--num_iters", type=int, default=10000)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--max_init_gaussians", type=int, default=120000)
    p.add_argument("--cap_max", type=int, default=1000000)
    p.add_argument("--sky", choices=["on", "off"], default="on")
    p.add_argument("--hood_mask", choices=["on", "off"], default="on")
    p.add_argument("--ssim", choices=["none", "torch", "fused"], default="fused")
    p.add_argument("--optimizer", choices=["selective_adam", "adam"], default="selective_adam")
    p.add_argument("--fisheye_max_angle", type=float, default=1.45)
    p.add_argument("--checkpoint_every", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--persist", type=float, default=0.5)
    return p.parse_args(argv)


# ------------------------------------------------------------------ đọc dữ liệu chuẩn
def load_views(args: argparse.Namespace) -> tuple[dict, list, list, list]:
    """Đọc manifest, split và mọi ảnh của các cặp (khung, camera) trong split.

    Pose camera → world của mỗi ảnh là ``ego_pose[khung] · extrinsics[camera]`` (khung KITTI của
    DriveStudio: ego trùng LiDAR). Ảnh được thu nhỏ ``--downscale`` lần bằng ``INTER_AREA``, nội
    tham số chia theo, như teammate.

    Returns:
        ``(manifest, views, frames, split)``; mỗi view là dict ``frame``, ``cam``, ``model``
        (``pinhole`` | ``fisheye``), ``image_np`` (RGB uint8), ``K_np``, ``radial_np``, ``T``
        (world → camera), ``c2w``, ``is_test``.
    """
    manifest = yaml.safe_load((args.scene_dir / "manifest.yaml").read_text())
    split = json.loads(args.split_file.read_text())
    camera_index = {c["camera_id"]: i for i, c in enumerate(manifest["cameras"])}
    camera_model = {c["camera_id"]: c["model"] for c in manifest["cameras"]}
    pairs = [(int(f), c, False) for f, c in split["train"]] + [
        (int(f), c, True) for f, c in split["test"]
    ]
    pairs.sort()
    ego = {}
    extrinsics = {
        cam: np.loadtxt(args.scene_dir / "extrinsics" / f"{camera_index[cam]}.txt")
        for cam in split["cameras"]
    }
    views = []
    for frame, cam, is_test in pairs:
        if frame not in ego:
            ego[frame] = np.loadtxt(args.scene_dir / "ego_pose" / f"{frame:03d}.txt")
        index = camera_index[cam]
        path = next((args.scene_dir / "images").glob(f"{frame:03d}_{index}.*"))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        h, w = bgr.shape[0] // args.downscale, bgr.shape[1] // args.downscale
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        intr = np.loadtxt(args.scene_dir / "intrinsics" / f"{index}.txt")
        k = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]], dtype=np.float64)
        k[:2] /= args.downscale
        model = camera_model[cam]
        if model == "opencv_fisheye":
            radial, gs_model = intr[4:8].astype(np.float32), "fisheye"
        elif model in ("opencv_pinhole", "ideal_pinhole"):
            if np.abs(intr[4:]).max() > 0:
                raise SystemExit(f"{cam}: trainer chỉ nhận pinhole không méo (ảnh đã rectify)")
            radial, gs_model = np.zeros(4, np.float32), "pinhole"
        else:
            raise SystemExit(f"{cam}: mô hình {model} chưa hỗ trợ")
        c2w = ego[frame] @ extrinsics[cam]
        views.append(
            dict(frame=frame, cam=cam, model=gs_model, h=h, w=w, image_np=rgb, K_np=k,
                 radial_np=radial, T=np.linalg.inv(c2w), c2w=c2w, is_test=is_test)
        )
    frames = sorted(ego)
    return manifest, views, frames, split


# ------------------------------------------------------------------ mô hình camera
def project_for_color(view: dict, pts_world: np.ndarray, max_angle: float) -> tuple:
    """Toạ độ pixel của điểm world trong một view, cùng mô hình camera mà renderer dùng."""
    pc = pts_world @ view["T"][:3, :3].T + view["T"][:3, 3]
    K = view["K_np"]
    if view["model"] == "pinhole":
        ok = pc[:, 2] > 0.5
        z = np.where(ok, pc[:, 2], 1.0)
        uv = np.column_stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]])
        return uv, ok
    norm_xy = np.linalg.norm(pc[:, :2], axis=1)
    theta = np.arctan2(norm_xy, pc[:, 2])
    ok = (np.linalg.norm(pc, axis=1) > 0.5) & (theta < max_angle)
    t2 = theta * theta
    r = view["radial_np"]
    theta_d = theta * (1 + r[0] * t2 + r[1] * t2**2 + r[2] * t2**3 + r[3] * t2**4)
    scale = np.where(norm_xy > 1e-9, theta_d / np.maximum(norm_xy, 1e-9), 1.0)
    uv = np.column_stack(
        [K[0, 0] * pc[:, 0] * scale + K[0, 2], K[1, 1] * pc[:, 1] * scale + K[1, 2]]
    )
    return uv, ok


def sample_colors(views_of_frame: list, pts_world: np.ndarray, max_angle: float) -> tuple:
    """Màu của mỗi điểm lấy từ camera đầu tiên cùng khung thấy nó (như teammate)."""
    colors = np.zeros((len(pts_world), 3), np.float32)
    seen = np.zeros(len(pts_world), bool)
    for view in views_of_frame:
        uv, ok = project_for_color(view, pts_world, max_angle)
        h, w = view["image_np"].shape[:2]
        inside = ok & ~seen & (uv[:, 0] >= 1) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 1)
        inside &= uv[:, 1] < h - 2
        if not inside.any():
            continue
        px = np.round(uv[inside]).astype(int)
        colors[inside] = view["image_np"][px[:, 1], px[:, 0]].astype(np.float32) / 255.0
        seen |= inside
    return colors, seen


def fov_mask(view: dict, max_angle: float) -> np.ndarray:
    """Mask 1 = vẽ được, 0 = tia lệch trục hơn ``max_angle`` (chỉ với camera mắt cá).

    Với OpenCV fisheye, bán kính chuẩn hoá của pixel là ``theta_d`` và tăng theo ``theta``, nên
    pixel có bán kính lớn hơn ``theta_d(max_angle)`` là tia lệch trục hơn ``max_angle``.
    """
    if view["model"] != "fisheye":
        return np.ones((view["h"], view["w"]), np.float32)
    K = view["K_np"]
    u = (np.arange(view["w"]) + 0.5 - K[0, 2]) / K[0, 0]
    v = (np.arange(view["h"]) + 0.5 - K[1, 2]) / K[1, 1]
    radius = np.sqrt(u[None, :] ** 2 + v[:, None] ** 2)
    limit = fisheye_radius(max_angle, [float(x) for x in view["radial_np"]])
    return (radius < limit).astype(np.float32)


def static_masks(views: list, persist: float, out_dir: Path) -> dict:
    """Mask nắp capô và vùng đen ngoài vòng tròn mắt cá, theo teammate (1 = giám sát, 0 = bỏ).

    Nắp capô và vùng đen cố định trong hệ camera. Màu capô đổi theo ánh sáng nhưng *cạnh* của nó
    nằm yên ở cùng pixel mọi khung, còn cạnh của cảnh thì chạy; nên độ "bền" của cạnh tìm ra capô.
    Chỉ dùng ảnh train.
    """
    from scipy.signal import medfilt

    masks = {}
    cams = sorted({v["cam"] for v in views if v["model"] == "fisheye"})
    for cam in cams:
        gray = np.stack(
            [cv2.cvtColor(v["image_np"], cv2.COLOR_RGB2GRAY) for v in views if v["cam"] == cam]
        ).astype(np.float32)

        def sobel(g: np.ndarray) -> np.ndarray:
            return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

        mean_grad = np.mean([sobel(g) for g in gray], axis=0)
        persistence = sobel(gray.mean(0)) / (mean_grad + 1.0)
        h, w = mean_grad.shape
        rows = np.arange(h)[:, None]
        edges = ((persistence > persist) & (mean_grad > 15) & (rows > 0.63 * h)).astype(np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        top = np.array(
            [np.flatnonzero(closed[:, x]).min() if closed[:, x].any() else h for x in range(w)],
            np.float32,
        )
        top = medfilt(top, 61)
        hood = (np.arange(h)[:, None] >= top[None, :]).astype(np.uint8)
        outside = ((gray.mean(0) < 12) & (gray.std(0) < 6)).astype(np.uint8)
        bad = cv2.dilate(np.maximum(hood, outside), np.ones((9, 9), np.uint8))
        masks[cam] = (1 - bad).astype(np.float32)
        cv2.imwrite(str(out_dir / f"hood_{cam}.png"), (masks[cam] * 255).astype(np.uint8))
        print(f"[MASK] {cam}: giám sát {masks[cam].mean() * 100:.0f}% điểm ảnh", flush=True)
    return masks


# ------------------------------------------------------------------ SSIM
def ssim_torch(a: torch.Tensor, b: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM bằng conv2d của PyTorch (ảnh HxWx3 trong [0, 1]), như teammate."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    g = torch.exp(-((torch.arange(win, device=x.device).float() - win // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(t, k, padding=win // 2, groups=3)

    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))).mean()


def make_ssim(kind: str):
    """Chọn hàm SSIM cho loss: ``fused`` (gói fused-ssim, CUDA) hoặc ``torch``."""
    if kind == "fused":
        from fused_ssim import fused_ssim

        print("[ACCEL] SSIM: fused-ssim", flush=True)
        return lambda a, b: fused_ssim(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
    print("[ACCEL] SSIM: torch conv2d", flush=True)
    return ssim_torch


# ------------------------------------------------------------------ ghi kết quả
def write_json(path: Path, data: object) -> None:
    """Ghi JSON qua file tạm rồi đổi tên, để file không bao giờ dở dang."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def background_state(params: torch.nn.ParameterDict) -> dict:
    """Đổi Gaussian sang tên trường node ``Background`` của DriveStudio.

    ``export_ply.py`` và ``export_preview.py`` đọc đúng các trường này. Màu RGB (sigmoid của
    ``colors``) đổi sang hệ số SH bậc 0: ``(rgb − 0.5) / C0``; không có SH bậc cao.
    """
    n = len(params["means"])
    rgb = params["colors"].detach().sigmoid().cpu()
    return {
        "_means": params["means"].detach().cpu(),
        "_scales": params["scales"].detach().cpu(),
        "_quats": params["quats"].detach().cpu(),
        "_features_dc": (rgb - 0.5) / SH_C0,
        "_features_rest": torch.zeros(n, 0, 3),
        "_opacities": params["opacities"].detach().cpu().reshape(-1, 1),
    }


def save_cameras(out_dir: Path, views: list) -> None:
    """Ghi ``cameras.json`` (CONTRACTS mục 7): pose camera → world (OpenCV) và K theo khung."""
    cameras = []
    for cam in sorted({v["cam"] for v in views}):
        own = sorted((v for v in views if v["cam"] == cam), key=lambda v: v["frame"])
        cameras.append(
            {
                "name": cam,
                "width": int(own[0]["w"]),
                "height": int(own[0]["h"]),
                "c2w": [[round(float(x), 6) for x in v["c2w"].reshape(-1)] for v in own],
                "K": [[round(float(x), 4) for x in v["K_np"].reshape(-1)] for v in own],
            }
        )
    payload = {"frame": "drivestudio_world", "convention": "opencv", "units": "metres",
               "cameras": cameras}
    (out_dir / CAMERAS_FILE).write_text(json.dumps(payload))


def alexnet_cached() -> bool:
    """Cho biết trọng số AlexNet của torchvision (LPIPS cần) đã có trong cache torch hub chưa."""
    return (Path(torch.hub.get_dir()) / "checkpoints" / ALEXNET_WEIGHTS).is_file()


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    """Train, đánh giá trên tập test, ghi kết quả; xem docstring của file."""
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    if not alexnet_cached():
        raise SystemExit(
            f"thiếu trọng số AlexNet ({ALEXNET_WEIGHTS}) trong {torch.hub.get_dir()}/checkpoints "
            "cho LPIPS; trainer không tự tải. Tải một lần bằng: python -c \"import torchvision; "
            "torchvision.models.alexnet(weights='IMAGENET1K_V1')\""
        )
    from gsplat import rasterization
    from gsplat.optimizers import SelectiveAdam
    from gsplat.strategy import MCMCStrategy
    from scipy.spatial import cKDTree

    dev = "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    import gsplat

    manifest, views, frames, split = load_views(args)
    train_views = [v for v in views if not v["is_test"]]
    test_views = [v for v in views if v["is_test"]]
    print(f"[DATA] {len(frames)} khung, {len(train_views)} ảnh train, {len(test_views)} ảnh test",
          flush=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(torch=torch.__version__, gsplat=gsplat.__version__,
                  gpu=torch.cuda.get_device_name(0), split_id=split["split_id"],
                  scene_id=manifest["scene_id"])
    write_json(out / "gsplat_config.json", config)
    save_cameras(out, views)

    hood = static_masks(train_views, args.persist, out / "masks")
    for v in views:
        fov = fov_mask(v, args.fisheye_max_angle)
        hood_mask = hood.get(v["cam"], np.ones_like(fov))
        v["eval_mask_np"] = fov * hood_mask
        v["mask_np"] = fov * hood_mask if args.hood_mask == "on" else fov
        if v["model"] == "fisheye":
            cv2.imwrite(str(out / "masks" / f"fov_{v['cam']}.png"), (fov * 255).astype(np.uint8))

    # ---- khởi tạo Gaussian: hình học từ LiDAR, màu từ camera cùng khung
    rng = np.random.default_rng(args.seed)
    first, last = split["frame_range"]
    lidar_frames = list(range(first, last + 1))
    by_frame = {}
    for v in train_views:
        by_frame.setdefault(v["frame"], []).append(v)
    ego = {t: np.loadtxt(args.scene_dir / "ego_pose" / f"{t:03d}.txt") for t in lidar_frames}
    per_frame = max(500, int(args.max_init_gaussians * 1.5 / len(lidar_frames)))
    pts_all, col_all = [], []
    for t in lidar_frames:
        if t not in by_frame:
            continue  # khung chỉ có ảnh test: không lấy màu từ ảnh test
        scan = np.fromfile(args.scene_dir / "lidar" / f"{t:03d}.bin", np.float32).reshape(-1, 4)
        scan = scan[:, :3].astype(np.float64)
        dist = np.linalg.norm(scan, axis=1)
        scan = scan[(dist > 2.5) & (dist < 80)]
        world = scan @ ego[t][:3, :3].T + ego[t][:3, 3]
        colors, seen = sample_colors(by_frame[t], world, args.fisheye_max_angle)
        world, colors = world[seen], colors[seen]
        pick = rng.choice(len(world), min(per_frame, len(world)), replace=False)
        pts_all.append(world[pick])
        col_all.append(colors[pick])
    points = np.concatenate(pts_all).astype(np.float32)
    colors0 = np.concatenate(col_all).astype(np.float32)
    if len(points) > args.max_init_gaussians:
        pick = rng.choice(len(points), args.max_init_gaussians, replace=False)
        points, colors0 = points[pick], colors0[pick]
    knn = cKDTree(points).query(points, k=4)[0][:, 1:].mean(1).clip(0.03, 1.0).astype(np.float32)
    max_scale = 40.0 if args.sky == "on" else 0.5
    if args.sky == "on":
        sky_pts, sky_col = [], []
        for t, frame_views in sorted(by_frame.items()):
            d = rng.normal(size=(SKY_PER_FRAME, 3))
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            d[:, 2] = np.abs(d[:, 2])
            d = d[d[:, 2] > np.sin(np.radians(12))]  # world z hướng lên: chỉ giữ nửa trên
            world = ego[t][:3, 3] + SKY_RADIUS * d
            colors, seen = sample_colors(frame_views, world, args.fisheye_max_angle)
            sky_pts.append(world[seen])
            sky_col.append(colors[seen])
        sky_pts = np.concatenate(sky_pts).astype(np.float32)
        sky_col = np.concatenate(sky_col).astype(np.float32)
        points = np.concatenate([points, sky_pts])
        colors0 = np.concatenate([colors0, sky_col])
        knn = np.concatenate([knn, np.full(len(sky_pts), SKY_SCALE, np.float32)])
        print(f"[INIT] +{len(sky_pts)} Gaussian trời trên vòm {SKY_RADIUS:.0f} m", flush=True)
    n = len(points)
    print(f"[INIT] {n} Gaussian, khoảng {np.ptp(points, axis=0).round(1)} m", flush=True)

    reference_cam = split["cameras"][0]
    centers = np.array([v["c2w"][:3, 3] for v in train_views if v["cam"] == reference_cam])
    # gsplat nhân learning rate vị trí và ngưỡng tách, tỉa với scene_scale; quỹ đạo lái xe dài
    # hàng trăm mét nên phải kẹp (teammate: 370 m cho lr vị trí 0,06).
    trajectory = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    scene_scale = float(np.clip(1.1 * trajectory, 5.0, 20.0))
    print(f"[INIT] scene_scale {scene_scale:.1f} m (bán kính quỹ đạo {trajectory:.0f} m)",
          flush=True)

    def logit(x: np.ndarray) -> np.ndarray:
        return np.log(x / (1 - x))

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(points, device=dev)),
        "quats": torch.nn.Parameter(
            torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)), dtype=torch.float32, device=dev)),
        "scales": torch.nn.Parameter(torch.tensor(np.log(knn[:, None].repeat(3, 1)), device=dev)),
        "opacities": torch.nn.Parameter(torch.full((n,), float(logit(0.3)), device=dev)),
        "colors": torch.nn.Parameter(torch.tensor(logit(colors0.clip(0.02, 0.98)), device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "quats": 1e-3, "scales": 5e-3, "opacities": 5e-2,
           "colors": 2.5e-2}
    optimizer_cls = SelectiveAdam if args.optimizer == "selective_adam" else torch.optim.Adam

    def make_optimizers(ps: torch.nn.ParameterDict) -> dict:
        return {k: optimizer_cls([{"params": [ps[k]], "lr": lr, "name": k}], eps=1e-15,
                                 betas=(0.9, 0.999)) for k, lr in lrs.items()}

    optimizers = make_optimizers(params)
    start_step, train_time = 0, 0.0
    if args.resume_from is not None and args.resume_from.is_file():
        ck = torch.load(args.resume_from, map_location=dev)
        params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.to(dev)) for k, v in ck["params"].items()})
        optimizers = make_optimizers(params)
        for k, opt in optimizers.items():
            opt.load_state_dict(ck["opt"][k])
        start_step, train_time = int(ck["step"]), float(ck.get("train_time_s", 0.0))
        print(f"[RESUME] bước {start_step}, {len(params['means'])} Gaussian", flush=True)
        del ck
    strategy = MCMCStrategy(cap_max=args.cap_max, refine_start_iter=500,
                            refine_stop_iter=int(args.num_iters * 0.8), refine_every=100,
                            min_opacity=0.005, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    for v in views:
        v["image"] = torch.from_numpy(v["image_np"]).to(dev)
        v["view"] = torch.tensor(v["T"], dtype=torch.float32, device=dev)[None]
        v["K"] = torch.tensor(v["K_np"], dtype=torch.float32, device=dev)[None]
        v["radial"] = torch.tensor(v["radial_np"], dtype=torch.float32, device=dev)[None]
        v["mask"] = torch.from_numpy(v["mask_np"]).to(dev)[..., None]
        v["eval_mask"] = torch.from_numpy(v["eval_mask_np"]).to(dev)[..., None]

    def render(v: dict) -> tuple:
        image, _, info = rasterization(
            params["means"], params["quats"], params["scales"].exp(),
            params["opacities"].sigmoid(), params["colors"].sigmoid(), v["view"], v["K"],
            v["w"], v["h"], packed=False, camera_model=v["model"],
            radial_coeffs=v["radial"] if v["model"] == "fisheye" else None,
            with_ut=True, with_eval3d=True)
        return image[0], info

    def gt(v: dict) -> torch.Tensor:
        return v["image"].float() / 255

    speedups = []
    if args.optimizer == "selective_adam":
        speedups.append("selective_adam")
    if args.ssim == "fused":
        speedups.append("fused_ssim")

    def save_stats(step: int) -> None:
        write_json(out / TRAIN_STATS, {
            "step": int(step),
            "num_iters": int(args.num_iters),
            "train_time_s": round(train_time, 3),
            "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 2**20),
            "num_gaussians": int(len(params["means"])),
            "gaussians_per_node": {"Background": int(len(params["means"]))},
            "speedups": speedups,
            "resolution": f"downscale={args.downscale}",
        })

    def save_resume(next_step: int) -> None:
        tmp = out / (RESUME_CHECKPOINT + ".tmp")
        torch.save({"step": next_step, "train_time_s": train_time,
                    "params": {k: v.detach().cpu() for k, v in params.items()},
                    "opt": {k: o.state_dict() for k, o in optimizers.items()}}, tmp)
        os.replace(tmp, out / RESUME_CHECKPOINT)
        save_stats(next_step)

    ssim_op = make_ssim(args.ssim) if args.ssim != "none" else None
    loop_start = time.perf_counter()
    for step in range(start_step, args.num_iters):
        v = random.choice(train_views)
        progress = step / max(1, args.num_iters - 1)
        optimizers["means"].param_groups[0]["lr"] = lrs["means"] * (0.01**progress)
        pred, info = render(v)
        target, m = gt(v), v["mask"]
        loss = ((pred - target).abs() * m).sum() / (m.sum() * 3).clamp_min(1)
        if ssim_op is not None:
            loss = 0.8 * loss + 0.2 * (1 - ssim_op(pred * m, target * m))
        loss = loss + 0.01 * params["opacities"].sigmoid().mean() + 0.01 * params["scales"].exp().mean()
        loss.backward()
        radii = info["radii"]
        visible = (radii[0] > 0).all(-1) if radii.ndim == 3 else radii[0] > 0
        for opt in optimizers.values():
            if args.optimizer == "selective_adam":
                opt.step(visible)
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info,
                                    lr=optimizers["means"].param_groups[0]["lr"])
        with torch.no_grad():
            params["scales"].clamp_(math.log(0.005), math.log(max_scale))
        done = step + 1
        if step == start_step or done % 500 == 0 or done == args.num_iters:
            torch.cuda.synchronize()
            train_time += time.perf_counter() - loop_start
            print(f"step {done}/{args.num_iters} loss={loss.item():.4f} "
                  f"gaussians={len(params['means'])} visible={int(visible.sum())} "
                  f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
            if args.checkpoint_every and done % args.checkpoint_every == 0 and done < args.num_iters:
                save_resume(done)
            loop_start = time.perf_counter()
    torch.cuda.synchronize()

    # ---- đánh giá theo protocol của tool (SPEC mục 9)
    from skimage.metrics import structural_similarity
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    per_camera: dict = {}
    render_seconds, timed = 0.0, 0
    with torch.no_grad():
        for i, v in enumerate(test_views):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred, _ = render(v)
            torch.cuda.synchronize()
            if i >= 3:  # bỏ 3 khung đầu khi đo FPS (CONTRACTS mục 6.3)
                render_seconds += time.perf_counter() - t0
                timed += 1
            pred8 = (pred.clamp(0, 1) * 255).round() / 255  # làm tròn 8 bit như ảnh gốc
            target, m = gt(v), v["eval_mask"]
            mse = ((pred8 - target) ** 2 * m).sum() / (m.sum() * 3).clamp_min(1)
            psnr = float(-10 * torch.log10(mse.clamp_min(1e-10)))
            a = (pred8 * m).cpu().numpy()
            b = (target * m).cpu().numpy()
            ssim = float(structural_similarity(a, b, data_range=1.0, channel_axis=2))
            lp = float(lpips((pred8 * m).permute(2, 0, 1)[None], (target * m).permute(2, 0, 1)[None]))
            per_camera.setdefault(v["cam"], []).append((psnr, ssim, lp))
    rows = [r for values in per_camera.values() for r in values]
    metrics = {}
    if rows:
        for j, name in enumerate(("psnr", "ssim", "lpips")):
            metrics[f"image_metrics/test/{name}"] = float(np.mean([r[j] for r in rows]))
            for cam, values in per_camera.items():
                metrics[f"image_metrics/test/per_camera/{cam}/{name}"] = float(
                    np.mean([r[j] for r in values]))
        if timed:
            metrics["image_metrics/test/render_fps"] = timed / render_seconds
        sizes = {(v["w"], v["h"]) for v in test_views}
        if len(sizes) == 1:
            w, h = sizes.pop()
            metrics["image_metrics/test/render_width"] = float(w)
            metrics["image_metrics/test/render_height"] = float(h)
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        write_json(out / "metrics" / f"images_test_{stamp}.json", metrics)
        print(f"[EVAL] {json.dumps(metrics)}", flush=True)
    else:
        print("[EVAL] split không có ảnh test", flush=True)

    for cam in sorted({v["cam"] for v in views}):
        v = next((x for x in test_views + train_views if x["cam"] == cam), None)
        with torch.no_grad():
            pred, _ = render(v)
        sheet = torch.cat([gt(v), pred.clamp(0, 1)], dim=1).cpu().numpy()
        cv2.imwrite(str(out / f"compare_{cam}.png"), (sheet[:, :, ::-1] * 255).astype(np.uint8))

    save_stats(args.num_iters)
    tmp = out / (FINAL_CHECKPOINT + ".tmp")
    torch.save({"step": args.num_iters, "models": {"Background": background_state(params)}}, tmp)
    os.replace(tmp, out / FINAL_CHECKPOINT)
    (out / RESUME_CHECKPOINT).unlink(missing_ok=True)
    print(f"[DONE] train {train_time:.0f}s, {len(params['means'])} Gaussian", flush=True)


if __name__ == "__main__":
    main()
