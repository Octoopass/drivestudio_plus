"""(drive2sim) Tối ưu tiếp PLY của Instant NuRec trên ảnh của clip NCore bằng gsplat.

    <python có gsplat> tools/ncore_refine.py scene_nurec.ply train_views.npz --out refined.ply
        [--iters 7000] [--strategy mcmc|none] [--cap-factor 1.5] [--max-side 1408]
        [--means-lr 1e-3] [--ssim-weight 0.2] [--seed 0]

``train_views.npz`` là file của ``ncore_eval_export.py`` (pose, thông số, ảnh, mask ego của các
khung train). Gaussian của NuRec là điểm xuất phát: vị trí, hướng, kích thước, độ đục và màu (SH
bậc 0) được tối ưu bằng L1 + D-SSIM. Camera mắt cá vẽ bằng 3DGUT (``with_ut``, ``with_eval3d``) như
lúc chấm. Lỗi chỉ tính ở pixel ngoài mask ego mà Gaussian đã phủ (alpha > ``--alpha-min``): bầu
trời của NuRec nằm ngoài PLY, nên không bắt Gaussian lấp trời.

``--strategy mcmc`` (thử nghiệm): ``MCMCStrategy`` của gsplat dời Gaussian gần trong suốt tới chỗ
cần và thêm Gaussian tới ``--cap-factor`` lần số ban đầu; không cần gradient 2D nên dùng được với
3DGUT. ``none`` (mặc định): giữ nguyên số Gaussian.

PLY ghi ra cùng cột với PLY gốc (``x y z``, ``rot_*``, ``scale_*``, ``opacity``, ``f_dc_*``...):
các stage đọc PLY NuRec dùng được ngay. Cột phụ (``nx``, ``road_mask``, ``sky_mask``) giữ của
Gaussian gốc khi số Gaussian không đổi, không thì đặt 0. In tiến độ dạng thanh tqdm và
``refine.json`` (thời gian, PSNR train đầu/cuối, VRAM) cạnh file ra.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

C0 = 0.28209479177387814
TYPES = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as f:
        header, props, n = [], [], 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            header.append(line)
            parts = line.split()
            if parts[0] == "element" and parts[1] == "vertex":
                n = int(parts[2])
            elif parts[0] == "property":
                props.append((parts[2], TYPES[parts[1]]))
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n).copy()
    return header, data


def write_ply(path: Path, header: list[str], data: np.ndarray) -> None:
    lines = [f"element vertex {len(data)}" if l.startswith("element vertex") else l for l in header]
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\nend_header\n").encode("ascii"))
        f.write(data.tobytes())


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM trung bình, ảnh [1, 3, H, W] trong [0, 1]."""
    try:
        from fused_ssim import fused_ssim

        return fused_ssim(a, b, padding="valid")
    except ImportError:
        pass
    k = torch.tensor(cv2.getGaussianKernel(11, 1.5), dtype=a.dtype, device=a.device)
    w = (k @ k.T)[None, None].repeat(3, 1, 1, 1)
    blur = lambda x: F.conv2d(x, w, groups=3)  # noqa: E731
    mu1, mu2 = blur(a), blur(b)
    s1, s2, s12 = blur(a * a) - mu1**2, blur(b * b) - mu2**2, blur(a * b) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))).mean()


def load_views(path: Path, max_side: int) -> list[dict]:
    with np.load(path, allow_pickle=True) as npz:
        v = {k: npz[k] for k in npz.files}
    views = []
    for i in range(len(v["camera"])):
        img = v["image"][i]
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        K = v["K"][i].astype(np.float32).copy()
        mask = v["mask"][i].astype(np.uint8)
        if s < 1:
            w, h = int(w * s), int(h * s)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            K[:2] *= s
        views.append({
            "camera": str(v["camera"][i]), "fisheye": str(v["model"][i]) == "fisheye",
            "viewmat": v["viewmat"][i].astype(np.float32), "K": K,
            "radial": v["radial"][i].astype(np.float32), "image": np.ascontiguousarray(img),
            "valid": torch.from_numpy(mask == 0), "w": w, "h": h,
        })  # fmt: skip
    return views


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("views")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=7000)
    p.add_argument("--strategy", choices=["mcmc", "none"], default="none")
    p.add_argument("--cap-factor", type=float, default=1.5)
    p.add_argument("--max-side", type=int, default=1408)
    p.add_argument("--means-lr", type=float, default=1e-3, help="lr vị trí (mét), giảm dần x0.01")
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--alpha-min", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    began = time.perf_counter()

    header, data = read_ply(Path(a.ply))
    col = lambda *names: torch.from_numpy(  # noqa: E731
        np.stack([data[k].astype(np.float32) for k in names], 1)
    ).to(dev)
    params = torch.nn.ParameterDict({
        "means": col("x", "y", "z"),
        "quats": col("rot_0", "rot_1", "rot_2", "rot_3"),
        "scales": col("scale_0", "scale_1", "scale_2"),
        "opacities": col("opacity")[:, 0],
        "sh0": col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :],
    })  # fmt: skip
    n0 = len(data)
    lrs = {"means": a.means_lr, "quats": 1e-3, "scales": 5e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optimizers = {
        k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
        for k, lr in lrs.items()
    }
    gamma = 0.01 ** (1 / max(1, a.iters))  # lr vị trí giảm tới 1% ở bước cuối
    strategy = state = None
    if a.strategy == "mcmc":
        from gsplat.strategy import MCMCStrategy

        strategy = MCMCStrategy(
            cap_max=int(n0 * a.cap_factor), refine_start_iter=500,
            refine_stop_iter=int(a.iters * 0.8), refine_every=100, min_opacity=0.005,
        )  # fmt: skip
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state()
    views = load_views(Path(a.views), a.max_side)
    print(f"{n0:,} Gaussian, {len(views)} view train, {a.iters} bước, strategy {a.strategy}",
          flush=True)  # fmt: skip

    def render(view: dict, sh0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        kw = dict(camera_model="fisheye", radial_coeffs=torch.from_numpy(view["radial"])[None].to(dev),
                  with_ut=True, with_eval3d=True) if view["fisheye"] else {}  # fmt: skip
        colors = (params["sh0"][:, 0] if sh0 is None else sh0) * C0 + 0.5
        rgb, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors,
            torch.from_numpy(view["viewmat"])[None].to(dev), torch.from_numpy(view["K"])[None].to(dev),
            view["w"], view["h"], render_mode="RGB", packed=False, **kw,
        )  # fmt: skip
        return rgb[0], alpha[0, ..., 0], info

    def train_psnr(sample: list[int]) -> float:
        values = []
        with torch.no_grad():
            for i in sample:
                rgb, alpha, _ = render(views[i])
                gt = torch.from_numpy(views[i]["image"]).to(dev).float() / 255
                valid = views[i]["valid"].to(dev) & (alpha > 0.9)
                if valid.any():
                    values.append(float(-10 * torch.log10(((rgb.clamp(0, 1) - gt) ** 2)[valid].mean())))
        return float(np.mean(values)) if values else float("nan")

    probe = rng.choice(len(views), size=min(40, len(views)), replace=False).tolist()
    psnr_before = train_psnr(probe)
    print(f"PSNR train (mẫu {len(probe)} view) trước: {psnr_before:.2f} dB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    order = rng.permutation(len(views))
    for step in range(a.iters):
        view = views[order[step % len(order)]]
        if step % len(order) == len(order) - 1:
            order = rng.permutation(len(views))
        rgb, alpha, info = render(view)
        gt = torch.from_numpy(view["image"]).to(dev).float() / 255
        keep = (view["valid"].to(dev) & (alpha.detach() > a.alpha_min))[..., None]
        pred = torch.where(keep, rgb, gt)  # pixel không tính lỗi lấy theo ảnh thật: gradient 0
        l1 = (pred - gt).abs().mean()
        loss = (1 - a.ssim_weight) * l1
        if a.ssim_weight > 0:
            loss = loss + a.ssim_weight * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        if strategy is not None:
            loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean()
            loss = loss + 0.01 * torch.exp(params["scales"]).mean()
            strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] *= gamma
        if strategy is not None:
            strategy.step_post_backward(
                params, optimizers, state, step, info, lr=optimizers["means"].param_groups[0]["lr"]
            )
        if step % 100 == 0 or step == a.iters - 1:
            done = step + 1
            bar = "#" * int(20 * done / a.iters)
            # dạng thanh tqdm để Studio (log_progress) đọc được tiến độ
            print(f"refine: {int(100 * done / a.iters):3d}%|{bar:<20}| {done}/{a.iters} "
                  f"[loss {float(loss):.4f}, {len(params['means']):,} Gaussian]", flush=True)  # fmt: skip

    psnr_after = train_psnr(probe)
    print(f"PSNR train (mẫu {len(probe)} view) sau: {psnr_after:.2f} dB", flush=True)
    with torch.no_grad():
        n = len(params["means"])
        out = data.copy() if n == n0 else np.zeros(n, dtype=data.dtype)
        get = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
        means, quats = get(params["means"]), get(F.normalize(params["quats"], dim=-1))
        scales, opac, sh0 = get(params["scales"]), get(params["opacities"]), get(params["sh0"][:, 0])
        for i, k in enumerate("xyz"):
            out[k] = means[:, i]
        for i in range(4):
            out[f"rot_{i}"] = quats[:, i]
        for i in range(3):
            out[f"scale_{i}"] = scales[:, i]
            out[f"f_dc_{i}"] = sh0[:, i]
        out["opacity"] = opac
        rgb8 = np.clip((sh0 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
        for i, k in enumerate(("red", "green", "blue")):
            if k in out.dtype.names:
                out[k] = rgb8[:, i]
        if "alpha" in out.dtype.names:
            out["alpha"] = np.clip(255 / (1 + np.exp(-opac)), 0, 255).astype(np.uint8)
    target = Path(a.out)
    write_ply(target, header, out)
    stats = {
        "iters": a.iters, "strategy": a.strategy, "views": len(views),
        "gaussians_before": n0, "gaussians_after": int(n),
        "train_psnr_before": round(psnr_before, 3), "train_psnr_after": round(psnr_after, 3),
        "refine_time_s": round(time.perf_counter() - began, 1),
        "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 2**20),
    }  # fmt: skip
    (target.parent / "refine.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {target} ({n:,} Gaussian) {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
