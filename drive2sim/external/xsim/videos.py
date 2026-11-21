"""Xuất video RGB render và ground truth khi worker XSIM đánh giá.

write_videos ghép ngang các camera theo thứ tự split, ghi một khung cho mỗi
thời điểm có quan sát và chuyển các cặp test thật cho scorer. Studio đọc tên
video và layout theo CONTRACTS mục 6.3.
"""

import json
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, closing
from itertools import groupby
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

Pair = tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]


def write_videos(
    pairs: Iterable[Pair],
    folder: Path,
    step: int,
    cameras: dict[int, tuple[str, int, int]],
    train: set[tuple[int, str]],
    test: set[tuple[int, str]],
    timestamps: dict[int, tuple[int, int]],
) -> Iterator[Pair]:
    """Ghi video trong lúc duyệt các cặp đã sắp theo frame, chỉ yield ảnh test.

    Args:
        pairs: Chuỗi frame, sensor, RGB render, RGB thật và mask.
        folder: Thư mục tạm của lần đánh giá đang chạy.
        step: Số bước của checkpoint đang vẽ.
        cameras: Thứ tự ghép, tên và kích thước (rộng, cao) của mỗi camera.
        train: Các cặp frame và tên camera trong phần train.
        test: Các cặp frame và tên camera trong phần test.
        timestamps: Frame ingest sang frame nguồn và timestamp tuyệt đối (microsecond).

    Yields:
        Cặp test thật cho scorer, giữ nguyên RGB và mask trước khi ghép video.
    """
    if train & test or not train or not test:
        raise ValueError("Hai phần split phải khác nhau và đều có quan sát")
    boxes = {}
    width = 0
    for sensor, (name, w, h) in cameras.items():
        boxes[sensor] = dict(name=name, x=width, y=0, width=w, height=h)
        width += w
    height = max(h for _, _, h in cameras.values())
    width += width % 2
    height += height % 2
    groups = ("full_set", "test_set")
    channels = ("rgbs", "gt_rgbs")
    frames = {group: [] for group in groups}
    seen = set()
    expected = train | test
    previous = -1
    with ExitStack() as stack:
        writers = {}
        for group in groups:
            for channel in channels:
                path = folder / f"{group}_{step}_{channel}.mp4"
                writer = imageio_ffmpeg.write_frames(
                    str(path),
                    (width, height),
                    fps=10,
                    codec="libx264",
                    macro_block_size=1,
                    output_params=["-threads", "1", "-movflags", "+faststart"],
                )
                stack.enter_context(closing(writer))
                next(writer)
                writers[group, channel] = writer
        for frame, samples in groupby(pairs, key=lambda sample: sample[0]):
            if frame <= previous:
                raise ValueError("Thứ tự frame video XSIM không tăng")
            previous = frame
            tiles = {group: np.zeros((2, height, width, 3), np.uint8) for group in groups}
            present = {group: [] for group in groups}
            for pair in samples:
                _, sensor, rgb, gt, valid = pair
                name, w, h = cameras[sensor]
                key = (frame, name)
                if key not in expected or key in seen:
                    raise ValueError(f"Cặp video XSIM ngoài split hoặc trùng: {key}")
                seen.add(key)
                pixels = (
                    torch.stack([rgb, gt])
                    .mul(valid[..., None])
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .byte()
                    .cpu()
                    .numpy()
                )
                if pixels.shape != (2, h, w, 3):
                    raise ValueError(f"Sai kích thước camera trong video XSIM: {key}")
                x = boxes[sensor]["x"]
                for group in groups if key in test else ("full_set",):
                    tiles[group][:, :h, x : x + w] = pixels
                    present[group].append(name)
                if key in test:
                    yield pair
            for group in groups:
                if present[group]:
                    for image, channel in zip(tiles[group], channels):
                        writers[group, channel].send(image)
                    source, stamp = timestamps[frame]
                    frames[group].append(
                        dict(
                            frame=frame,
                            source_frame=source,
                            timestamp_us=stamp,
                            cameras=present[group],
                        )
                    )
    if seen != expected:
        raise ValueError("Video XSIM thiếu cặp ảnh trong split")
    for group in groups:
        layout = dict(
            width=width, height=height, fps=10, cameras=list(boxes.values()), frames=frames[group]
        )
        (folder / f"{group}_{step}_layout.json").write_text(json.dumps(layout), encoding="utf-8")
