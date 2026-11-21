"""(drive2sim) Video đi dọc đoạn đường từ PLY của Instant NuRec, như render_flythrough của Fisheye Fusion Lab.

    <python môi trường chính> ncore_render_video.py <merged.ply> <views.npz> --out <video.mp4>
        [--cameras a,b] [--fps 10] [--height 360] [--no-gt]

``views.npz`` do ``ncore_eval_export.py --every 1`` ghi (mọi khung của clip NCore). Mỗi khung video
là một hàng ảnh dựng của các camera (theo thứ tự ``--cameras``), mỗi ảnh co về cao ``--height``;
mặc định có thêm hàng ảnh thật ở trên để so. Camera mắt cá vẽ bằng 3DGUT như
``ncore_eval_render.py``. Ghi H.264 yuv420p (trình duyệt phát được) khi có ``imageio_ffmpeg``.
In ``progress <đã vẽ>/<tổng>`` sau mỗi khung để stage báo tiến độ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ncore_eval_render import load_ply  # noqa: E402


def render(gaussians, viewmat, K, radial, model, w, h, dev):
    means, quats, scales, opac, colors = gaussians
    kw = dict(camera_model="fisheye", radial_coeffs=torch.from_numpy(radial)[None].to(dev),
              with_ut=True, with_eval3d=True) if model == "fisheye" else {}
    with torch.no_grad():
        rgb, _, _ = rasterization(
            means, quats, scales, opac, colors,
            torch.from_numpy(viewmat)[None].to(dev), torch.from_numpy(K)[None].to(dev),
            w, h, render_mode="RGB", packed=False, **kw,
        )
    return (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(2, round(w * height / h)), height), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("views")
    p.add_argument("--out", required=True)
    p.add_argument("--cameras", default="", help="thứ tự camera trong hàng; mặc định theo views")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--max-side", type=int, default=1408)
    p.add_argument("--no-gt", action="store_true", help="chỉ hàng ảnh dựng")
    a = p.parse_args()
    dev = "cuda"
    gaussians = load_ply(Path(a.ply), dev)
    with np.load(a.views, allow_pickle=True) as npz:  # nạp mỗi mảng một lần: đọc npz[k] mỗi lần là nạp lại cả mảng
        v = {k: npz[k] for k in npz.files}
    cameras = [c for c in a.cameras.split(",") if c] or list(dict.fromkeys(str(c) for c in v["camera"]))
    by_frame: dict[int, dict[str, int]] = {}
    for i, cam in enumerate(v["camera"]):
        by_frame.setdefault(int(v["frame"][i]), {})[str(cam)] = i
    frames = sorted(by_frame)
    widths: dict[str, int] = {}
    rows: list[np.ndarray] = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".mp4v.mp4")
    writer = None
    for n, frame in enumerate(frames):
        tiles_r, tiles_g = [], []
        for cam in cameras:
            i = by_frame[frame].get(cam)
            if i is None:  # camera thiếu ở khung này: ô đen cùng cỡ
                width = widths.get(cam, a.height)
                tiles_r.append(np.zeros((a.height, width, 3), np.uint8))
                tiles_g.append(np.zeros((a.height, width, 3), np.uint8))
                continue
            img = v["image"][i]
            h, w = img.shape[:2]
            s = min(1.0, a.max_side / max(h, w))
            K = v["K"][i].copy()
            if s < 1:
                w, h = int(w * s), int(h * s)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                K[:2] *= s
            rgb = render(gaussians, v["viewmat"][i], K, v["radial"][i], str(v["model"][i]), w, h, dev)
            tile_r = label(to_height(rgb, a.height), f"{cam} render")
            tile_g = label(to_height(img, a.height), f"{cam} ground truth")
            widths[cam] = tile_r.shape[1]
            tiles_r.append(tile_r)
            tiles_g.append(tile_g)
        row = np.concatenate(tiles_r, 1)
        if not a.no_gt:
            row = np.concatenate([np.concatenate(tiles_g, 1), row], 0)
        h0, w0 = row.shape[:2]
        if writer is None:
            size = (w0 - w0 % 2, h0 - h0 % 2)
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, size)
        if (row.shape[1], row.shape[0]) != size:
            row = cv2.resize(row, size)
        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f"progress {n + 1}/{len(frames)}", flush=True)
    if writer is None:
        sys.exit("views.npz không có khung nào")
    writer.release()
    try:
        import imageio_ffmpeg

        done = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                               "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                               "-movflags", "+faststart", str(out)]).returncode == 0
    except ImportError:
        done = False
    if done:
        raw.unlink()
    else:
        raw.replace(out)
    info = {"frames": len(frames), "cameras": cameras, "fps": a.fps, "duration_s": len(frames) / a.fps,
            "size": [int(size[0]), int(size[1])], "with_gt": not a.no_gt}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out}: {len(frames)} frames, {len(frames) / a.fps:.1f} s")


if __name__ == "__main__":
    main()
