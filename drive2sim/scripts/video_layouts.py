r"""Tạo bù ``<bộ>_<bước>_layout.json`` cho video đánh giá của run train trước 20/11/2026.

Từ 20/11/2026 đánh giá cuối ghi file này cạnh video (``recon/rendering/videos.write_video_layout``)
để Studio cắt được từng camera ra khỏi khung ghép và vẽ chú thích. Run cũ không có; script dựng lại
nó từ ``cameras.json`` (tên, kích thước ảnh từng camera khi train) và ``config.yaml`` (dataset, để
chọn đúng hàm ghép ``get_layout``), cùng hàm ghép mà video đã dùng, nên khung khớp video.

Chạy trong môi trường ``drive2sim`` (cần numpy, OpenCV)::

    python -m drive2sim.scripts.video_layouts [<run_id> ...]   # bỏ trống: mọi run

File đã có thì giữ nguyên.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from drive2sim.engine.videos import _PREFIX, LAYOUT_SUFFIX, _split
from drive2sim.engine.workdir import Workdir
from drive2sim.recon.rendering.videos import write_video_layout
from drive2sim.recon.tools.eval import TEST_VIDEO_FPS
from drive2sim.recon.utils.visualization import get_layout


def backfill(train_dir: Path) -> list[Path]:
    """Ghi layout còn thiếu trong ``<train_dir>/videos``; trả các file đã ghi."""
    cameras_path, config_path = train_dir / "cameras.json", train_dir / "config.yaml"
    if not (cameras_path.is_file() and config_path.is_file()):
        return []
    cameras = json.loads(cameras_path.read_text())["cameras"]
    config = yaml.safe_load(config_path.read_text())
    layout = get_layout(config["data"]["dataset"])
    names = [c["name"] for c in cameras]
    frames = [np.zeros((c["height"], c["width"], 3), np.float32) for c in cameras]
    written = []
    prefixes = set()
    for video in sorted((train_dir / "videos").glob("*.mp4")):
        split = _split(video.stem)
        if split and _PREFIX.match(split[0]):
            prefixes.add(split[0])
    for prefix in sorted(prefixes):
        target = train_dir / "videos" / f"{prefix}{LAYOUT_SUFFIX}"
        if target.exists():
            continue
        fps = TEST_VIDEO_FPS if prefix.startswith("test_set") else config["render"]["fps"]
        write_video_layout(str(target.with_name(prefix + ".mp4")), layout, frames, names, fps)
        written.append(target)
    return written


def main() -> None:
    """Tạo bù layout cho các run được chọn (mặc định mọi run)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_ids", nargs="*")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    args = parser.parse_args()
    work = Workdir(args.workdir.resolve())
    runs = [work.find_run(r) for r in args.run_ids] or sorted(
        d for d in work.root.glob("runs/*/*/*") if (d / "d2s").is_dir()
    )
    for run in runs:
        for train_dir in sorted((run / "d2s" / "stages").glob("train.*")):
            for path in backfill(train_dir):
                print(path)


if __name__ == "__main__":
    main()
