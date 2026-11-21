"""(drive2sim: copied from Fisheye Fusion Lab tools/, plus LPIPS and overall means.)

Step 2 of evaluating a 3DGS PLY against an NCore clip: render with gsplat (3DGUT) and score."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization


def load_ply(path: Path, device: str):
    with path.open("rb") as f:
        header, props, n = [], [], 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            header.append(line)
            parts = line.split()
            if parts[0] == "element" and parts[1] == "vertex":
                n = int(parts[2])
            elif parts[0] == "property":
                props.append((parts[2], {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}[parts[1]]))
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n)
    col = lambda *names: torch.from_numpy(np.stack([data[k].astype(np.float32) for k in names], 1)).to(device)
    means = col("x", "y", "z")
    quats = torch.nn.functional.normalize(col("rot_0", "rot_1", "rot_2", "rot_3"), dim=1)
    scales = torch.exp(col("scale_0", "scale_1", "scale_2"))
    opac = torch.sigmoid(col("opacity")[:, 0])
    colors = (col("f_dc_0", "f_dc_1", "f_dc_2") * 0.28209479 + 0.5).clamp(0, 1)
    return means, quats, scales, opac, colors


def ssim_masked(rgb: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    if not valid.any():
        return float("nan")
    mu1 = cv2.GaussianBlur(rgb, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gt, (11, 11), 1.5)
    s1 = cv2.GaussianBlur(rgb * rgb, (11, 11), 1.5) - mu1 * mu1
    s2 = cv2.GaussianBlur(gt * gt, (11, 11), 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(rgb * gt, (11, 11), 1.5) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    smap = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2))
    return float(smap.mean(axis=-1)[valid].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("views")
    p.add_argument("--out", required=True)
    p.add_argument("--max-side", type=int, default=1408, help="downscale views whose long side is larger")
    a = p.parse_args()
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    means, quats, scales, opac, colors = load_ply(Path(a.ply), dev)
    print(f"{len(means):,} Gaussians")
    with np.load(a.views, allow_pickle=True) as npz:  # nạp mỗi mảng một lần: đọc npz[k] mỗi lần là nạp lại cả mảng
        v = {k: npz[k] for k in npz.files}
    per_cam: dict[str, list[float]] = {}
    per_cam_ssim: dict[str, list[float]] = {}
    per_cam_lpips: dict[str, list[float]] = {}
    try:  # LPIPS like drive2sim eval.standard: torchmetrics, AlexNet backbone
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    except Exception as error:  # noqa: BLE001
        print(f"LPIPS unavailable: {error}")
        lpips_fn = None
    vid_frames: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i in range(len(v["camera"])):
        img = v["image"][i]
        h, w = img.shape[:2]
        s = min(1.0, a.max_side / max(h, w))
        if s < 1:
            w, h = int(w * s), int(h * s)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        K = v["K"][i].copy()
        K[:2] *= s
        mask = cv2.resize(v["mask"][i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        fisheye = v["model"][i] == "fisheye"
        kw = dict(camera_model="fisheye", radial_coeffs=torch.from_numpy(v["radial"][i])[None].to(dev),
                  with_ut=True, with_eval3d=True) if fisheye else {}
        with torch.no_grad():
            rgb, alpha, _ = rasterization(
                means, quats, scales, opac, colors,
                torch.from_numpy(v["viewmat"][i])[None].to(dev), torch.from_numpy(K)[None].to(dev),
                w, h, render_mode="RGB", packed=False, **kw,
            )
        rgb = rgb[0].clamp(0, 1).cpu().numpy()
        alpha = alpha[0, ..., 0].cpu().numpy()
        gt = img.astype(np.float32) / 255
        valid = (~mask) & (alpha > 0.9)
        mse = float(((rgb - gt) ** 2)[valid].mean()) if valid.any() else float("nan")
        psnr = -10 * np.log10(mse) if valid.any() else float("nan")
        ssim_val = ssim_masked(rgb, gt, valid)
        if lpips_fn is not None:
            # pixels outside the valid area take the ground truth, so only the reconstructed area counts
            filled = np.where(valid[..., None], rgb, gt)
            to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().to(dev)
            with torch.no_grad():
                lp = float(lpips_fn(to_t(filled), to_t(gt)))
        else:
            lp = float("nan")
        cam = str(v["camera"][i])
        per_cam.setdefault(cam, []).append(psnr)
        per_cam_ssim.setdefault(cam, []).append(ssim_val)
        per_cam_lpips.setdefault(cam, []).append(lp)
        vis = (np.concatenate([gt, rgb], 1) * 255).astype(np.uint8)
        if cam == str(v["camera"][0]):
            vid_frames.append(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if len(per_cam[cam]) <= 3:
            cv2.imwrite(str(out / f"{cam}_{len(per_cam[cam]) - 1}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"{cam} #{len(per_cam[cam]) - 1}: PSNR {psnr:.2f} dB SSIM {ssim_val:.3f} over {valid.mean() * 100:.0f}% of pixels")
    elapsed = max(time.perf_counter() - t0, 1e-6)
    if vid_frames:
        vp = out / "eval.mp4"
        h0, w0 = vid_frames[0].shape[:2]
        eh, ew = h0 - (h0 % 2), w0 - (w0 % 2)
        wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 6, (ew, eh))
        for fr in vid_frames:
            fr2 = cv2.resize(fr, (ew, eh)) if fr.shape[:2] != (eh, ew) else fr
            wr.write(fr2)
        wr.release()
        try:
            import imageio_ffmpeg
            tmp = vp.with_suffix(".h264.mp4")
            if subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(vp),
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(tmp)]).returncode == 0:
                tmp.replace(vp)
        except ImportError:
            pass
    metrics = {
        c: {"psnr_mean": float(np.nanmean(x)), "ssim_mean": float(np.nanmean(per_cam_ssim[c])),
            "lpips_mean": float(np.nanmean(per_cam_lpips[c])), "views": len(x)}
        for c, x in per_cam.items()
    }
    metrics["_summary"] = {
        "gaussians": int(len(means)),
        "psnr_mean": float(np.nanmean([y for x in per_cam.values() for y in x])),
        "ssim_mean": float(np.nanmean([y for x in per_cam_ssim.values() for y in x])),
        "lpips_mean": float(np.nanmean([y for x in per_cam_lpips.values() for y in x])),
        "render_resolution": f"{w}x{h}",
        "render_fps": float(len(v["camera"]) / elapsed),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 2**30),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
