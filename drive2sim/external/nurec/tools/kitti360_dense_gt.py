"""(drive2sim) Ảnh thật KITTI-360 nằm giữa các khung của clip NCore, cho hàng ảnh thật của video.

    <python môi trường NuRec> kitti360_dense_gt.py --data <KITTI-360 gốc> --sequence <chuỗi> --clip <clip.json>
        --out <gt_dense.npz> [--height 360]

Clip của bộ ROSA thưa (``frame_ids`` cách nhau 5, tức 2 ảnh/giây) nhưng ``data_2d_raw`` có mọi
khung 10 Hz. Tool đọc các khung nằm giữa hai khung liên tiếp của clip từ archive gốc, xử lý như
``kitti360_to_ncore.py`` (camera mắt cá ``rectify120`` resample về camera F-theta 120°), co về cao
``--height`` và ghi JPEG vào ``gt_dense.npz``:
``clip_ids`` (frame id của từng khung clip), ``ids_<sensor>`` (frame id có ảnh), ``jpeg_<sensor>``
(byte JPEG nối liền) và ``off_<sensor>`` (vị trí bắt đầu, thêm phần tử cuối là tổng độ dài).
Khung thiếu trong archive thì bỏ qua (video giữ ảnh gần nhất).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti360_to_ncore import CAMERAS, FisheyeRectifier, load_items, nv_cross_120  # noqa: E402


def between(clip_ids: list[int]) -> list[int]:
    """Frame id nằm giữa hai khung liên tiếp của clip (không gồm chính các khung clip)."""
    return [f for a, b in zip(clip_ids, clip_ids[1:]) for f in range(a + 1, b)]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--clip", required=True, help="clip.json của prep.ncore")
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--jpeg-quality", type=int, default=90)
    a = p.parse_args(argv)
    data = Path(a.data)
    clip = json.loads(Path(a.clip).read_text(encoding="utf-8"))
    clip_ids = [int(f) for f in clip["frame_ids"]]
    wanted = between(clip_ids)
    out = {"clip_ids": np.array(clip_ids, dtype=np.int64)}
    items = {x["frame_id"]: x for x in load_items(data, a.sequence)}
    info_item = items.get(clip_ids[0]) or next(iter(items.values()))
    names = {f"{f:010d}": f for f in wanted}
    for cam, sensor in clip["sensors"].items():
        key, _, model = CAMERAS[cam]
        rect = None
        imgs: dict[int, bytes] = {}
        with tarfile.open(data / f"data_2d_raw/{a.sequence}_{cam}.tar", "r:*") as tar:
            for m in tar:
                stem = Path(m.name).stem
                if not (m.isfile() and m.name.endswith(".png") and stem in names):
                    continue
                img = Image.open(tar.extractfile(m)).convert("RGB")
                if model == "fisheye" and clip.get("fisheye_model", "rectify120") == "rectify120":
                    if rect is None:
                        rect = FisheyeRectifier(info_item["images"][key], nv_cross_120(cam), *img.size)
                    img = rect.image(img)
                w, h = img.size
                img = img.resize((max(2, round(w * a.height / h)), a.height), Image.BOX)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=a.jpeg_quality)
                imgs[names[stem]] = buf.getvalue()
        ids = sorted(imgs)
        blobs = [imgs[f] for f in ids]
        out[f"ids_{sensor}"] = np.array(ids, dtype=np.int64)
        out[f"jpeg_{sensor}"] = np.frombuffer(b"".join(blobs), dtype=np.uint8)
        out[f"off_{sensor}"] = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
        print(f"{cam}: {len(ids)}/{len(wanted)} khung giữa", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
