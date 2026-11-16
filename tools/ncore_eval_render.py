"""Step 2 of evaluating a 3DGS PLY against an NCore clip: render with gsplat (3DGUT) and score.

    <gsplat-python> tools/ncore_eval_render.py scene.ply views.npz --out eval_dir

PSNR is computed over pixels outside the ego mask where the Gaussians are opaque (alpha > 0.9),
because Instant NuRec's sky is a separate cubemap that a plain PLY render does not include.
Writes side-by-side PNGs (ground truth | render) and metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2
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
    v = np.load(a.views, allow_pickle=True)
    per_cam: dict[str, list[float]] = {}
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
        cam = str(v["camera"][i])
        per_cam.setdefault(cam, []).append(psnr)
        if len(per_cam[cam]) <= 3:
            vis = (np.concatenate([gt, rgb], 1) * 255).astype(np.uint8)
            cv2.imwrite(str(out / f"{cam}_{len(per_cam[cam]) - 1}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"{cam} #{len(per_cam[cam]) - 1}: PSNR {psnr:.2f} dB over {valid.mean() * 100:.0f}% of pixels")
    metrics = {c: {"psnr_mean": float(np.nanmean(x)), "views": len(x)} for c, x in per_cam.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
