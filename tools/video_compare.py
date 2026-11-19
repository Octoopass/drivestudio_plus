"""Compose two videos of the same drive (top/bottom or side-by-side) with labels.

    <python> tools/video_compare.py a.mp4 "label A" b.mp4 "label B" --out compare.mp4 [--layout vstack|hstack]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    h = max(28, out.shape[0] // 18)
    scale = h / 34
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(out, (0, 0), (tw + 24, h + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (12, h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def compose_videos(a_path: str | Path, label_a: str, b_path: str | Path, label_b: str, out_path: str | Path,
                   layout: str = "vstack") -> dict:
    if layout not in ("vstack", "hstack"):
        raise ValueError("layout must be vstack or hstack")
    ca, cb = cv2.VideoCapture(str(a_path)), cv2.VideoCapture(str(b_path))
    fps = ca.get(cv2.CAP_PROP_FPS) or 20
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, n, shape = None, 0, (0, 0)
    try:
        while True:
            ok_a, fa = ca.read()
            ok_b, fb = cb.read()
            if not (ok_a and ok_b):
                break
            if layout == "hstack":
                if fb.shape[0] != fa.shape[0]:
                    fb = cv2.resize(fb, (max(2, round(fb.shape[1] * fa.shape[0] / fb.shape[0])), fa.shape[0]), interpolation=cv2.INTER_AREA)
                frame = np.hstack([label(fa, label_a), label(fb, label_b)])
            else:
                if fb.shape[1] != fa.shape[1]:
                    fb = cv2.resize(fb, (fa.shape[1], max(2, round(fb.shape[0] * fa.shape[1] / fb.shape[1]))), interpolation=cv2.INTER_AREA)
                frame = np.vstack([label(fa, label_a), label(fb, label_b)])
            eh, ew = frame.shape[0] - (frame.shape[0] % 2), frame.shape[1] - (frame.shape[1] % 2)
            frame = frame[:eh, :ew]
            if writer is None:
                shape = (ew, eh)
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, shape)
            writer.write(frame)
            n += 1
    finally:
        ca.release()
        cb.release()
        if writer is not None:
            writer.release()
    if n == 0:
        raise ValueError("could not read frames from both videos")
    try:
        import imageio_ffmpeg

        tmp = out.with_suffix(".h264.mp4")
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(out),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(tmp)])
        if r.returncode == 0:
            tmp.replace(out)
    except ImportError:
        pass
    return {"frames": n, "width": shape[0], "height": shape[1], "path": str(out)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("label_a")
    p.add_argument("b")
    p.add_argument("label_b")
    p.add_argument("--out", required=True)
    p.add_argument("--layout", choices=("vstack", "hstack"), default="vstack")
    a = p.parse_args()
    info = compose_videos(a.a, a.label_a, a.b, a.label_b, a.out, a.layout)
    print(f"wrote {info['path']}: {info['frames']} frames, {info['width']}x{info['height']}")


if __name__ == "__main__":
    main()
