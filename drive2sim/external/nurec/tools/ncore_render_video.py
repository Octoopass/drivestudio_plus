"""(drive2sim) Video đi dọc đoạn đường từ PLY của Instant NuRec, như render_flythrough của Fisheye Fusion Lab.

    <python môi trường chính> ncore_render_video.py <merged.ply> <views.npz> --out <video.mp4>
        [--cameras a,b] [--fps 10] [--speed 1] [--height 360] [--no-gt] [--interp]

``views.npz`` do ``ncore_eval_export.py --every 1`` ghi (mọi khung của clip NCore). Mỗi khung video
là một hàng ảnh dựng của các camera (theo thứ tự ``--cameras``), mỗi ảnh co về cao ``--height``;
mặc định có thêm hàng ảnh thật ở trên để so. Camera mắt cá vẽ bằng 3DGUT như
``ncore_eval_render.py``. Ghi H.264 yuv420p (trình duyệt phát được) khi có ``imageio_ffmpeg``.
``--speed`` > 0: video theo mốc thời gian thật của khung (``t_us`` của views.npz), mỗi khung giữ
trên màn hình đúng khoảng tới khung sau chia ``speed`` (1 là tốc độ gốc), video vẫn ``--fps``
khung/giây; ``--speed 0`` hoặc views không có ``t_us``: mỗi khung nguồn một khung video.
``--interp``: khung video nằm giữa hai khung nguồn được vẽ thật từ pose nội suy (quay theo slerp,
dời theo tuyến tính, trên pose camera→world) thay vì lặp lại khung trước, nên video mượt ở fps cao
dù nguồn thưa (KITTI-360 bộ ROSA: 0,5 s một ảnh). Hàng ảnh thật giữ ảnh của khung nguồn gần nhất
(giữa hai khung không có ảnh thật).
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


def _quat(R: np.ndarray) -> np.ndarray:
    """Ma trận quay 3x3 → quaternion (w, x, y, z) chuẩn hoá."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def interp_viewmat(va: np.ndarray, vb: np.ndarray, alpha: float) -> np.ndarray:
    """View matrix (world→camera) ở giữa ``va`` và ``vb``: nội suy trên pose camera→world."""
    if alpha <= 0:
        return va
    ca, cb = np.linalg.inv(va.astype(np.float64)), np.linalg.inv(vb.astype(np.float64))
    qa, qb = _quat(ca[:3, :3]), _quat(cb[:3, :3])
    dot = float(qa @ qb)
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + alpha * (qb - qa)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - alpha) * theta) * qa + np.sin(alpha * theta) * qb) / np.sin(theta)
    c = np.eye(4)
    c[:3, :3] = _rot(q / np.linalg.norm(q))
    c[:3, 3] = (1 - alpha) * ca[:3, 3] + alpha * cb[:3, 3]
    return np.linalg.inv(c).astype(va.dtype)


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
    p.add_argument("--speed", type=float, default=1, help="1: tốc độ gốc; 0: mỗi khung một khung video")
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--max-side", type=int, default=1408)
    p.add_argument("--no-gt", action="store_true", help="chỉ hàng ảnh dựng")
    p.add_argument("--interp", action="store_true",
                   help="vẽ khung giữa hai khung nguồn từ pose nội suy thay vì lặp khung")
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
    repeats = [1] * len(frames)
    if a.speed > 0 and "t_us" in v:
        stamp = [int(v["t_us"][next(iter(by_frame[f].values()))]) for f in frames]
        gaps = [(b - c) / 1e6 for c, b in zip(stamp, stamp[1:])]
        gaps.append(float(np.median(gaps)) if gaps else 1 / a.fps)  # khung cuối
        # làm tròn theo tổng dồn để thời lượng cả video không lệch dần
        done, clock = 0, 0.0
        for k, gap in enumerate(gaps):
            clock += max(gap, 0) / a.speed
            repeats[k] = max(1, round(clock * a.fps) - done)
            done += repeats[k]
    widths: dict[str, int] = {}
    rows: list[np.ndarray] = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".mp4v.mp4")
    writer = None
    size = None
    for n, frame in enumerate(frames):
        following = by_frame[frames[n + 1]] if n + 1 < len(frames) else {}
        # --interp: mỗi khung video vẽ lại từ pose nội suy; không thì vẽ một lần rồi lặp
        steps = repeats[n] if a.interp and following else 1
        tiles_g, gt_row = [], None
        for step in range(steps):
            alpha = step / steps
            tiles_r = []
            for cam in cameras:
                i = by_frame[frame].get(cam)
                if i is None:  # camera thiếu ở khung này: ô đen cùng cỡ
                    width = widths.get(cam, a.height)
                    tiles_r.append(np.zeros((a.height, width, 3), np.uint8))
                    if step == 0:
                        tiles_g.append(np.zeros((a.height, width, 3), np.uint8))
                    continue
                img = v["image"][i]
                h, w = img.shape[:2]
                s = min(1.0, a.max_side / max(h, w))
                K = v["K"][i].copy()
                if s < 1:
                    w, h = int(w * s), int(h * s)
                    K[:2] *= s
                viewmat = v["viewmat"][i]
                j = following.get(cam)
                if alpha > 0 and j is not None:
                    viewmat = interp_viewmat(viewmat, v["viewmat"][j], alpha)
                rgb = render(gaussians, viewmat, K, v["radial"][i], str(v["model"][i]), w, h, dev)
                tile_r = label(to_height(rgb, a.height), f"{cam} render")
                widths[cam] = tile_r.shape[1]
                tiles_r.append(tile_r)
                if step == 0:
                    shown = img if s >= 1 else cv2.resize(
                        img, (w, h), interpolation=cv2.INTER_AREA
                    )
                    tiles_g.append(label(to_height(shown, a.height), f"{cam} ground truth"))
            row = np.concatenate(tiles_r, 1)
            if not a.no_gt:
                if gt_row is None:
                    gt_row = np.concatenate(tiles_g, 1)
                row = np.concatenate([gt_row, row], 0)
            h0, w0 = row.shape[:2]
            if writer is None:
                size = (w0 - w0 % 2, h0 - h0 % 2)
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, size)
            if (row.shape[1], row.shape[0]) != size:
                row = cv2.resize(row, size)
            bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
            for _ in range(repeats[n] if steps == 1 else 1):
                writer.write(bgr)
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
    info = {"frames": len(frames), "video_frames": sum(repeats), "speed": a.speed if "t_us" in v else 0,
            "cameras": cameras, "fps": a.fps, "duration_s": sum(repeats) / a.fps,
            "size": [int(size[0]), int(size[1])], "with_gt": not a.no_gt,
            "interpolated": bool(a.interp)}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out}: {len(frames)} frames, {sum(repeats) / a.fps:.1f} s")


if __name__ == "__main__":
    main()
