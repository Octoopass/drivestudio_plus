"""Kiểm tra MP4 XSIM qua bộ đọc của Studio và giải mã nội dung video trên CPU."""

import json
from contextlib import closing
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ffmpeg = pytest.importorskip("imageio_ffmpeg")
np = pytest.importorskip("numpy")

from drive2sim.engine.videos import video_sets  # noqa: E402
from drive2sim.external.xsim.videos import write_videos  # noqa: E402


def test_sparse_split_writes_correct_channels_and_layout(tmp_path: Path) -> None:
    folder = tmp_path / "eval.xsim" / "videos"
    folder.mkdir(parents=True)
    gt = torch.ones(32, 32, 3)
    rgb = gt * 0.5
    mask = torch.ones(32, 32, dtype=torch.bool)
    rows = [(f, s, rgb, gt, mask) for f, s in ((0, 2), (0, 3), (1, 2))]
    outputs = list(
        write_videos(
            rows,
            folder,
            10,
            {2: ("left", 32, 32), 3: ("right", 32, 32)},
            {(0, "left"), (1, "left")},
            {(0, "right")},
            {0: (450, 123), 1: (451, 456)},
        )
    )
    assert [p[:2] for p in outputs] == [(0, 3)]
    sets = video_sets(tmp_path, str)
    assert [g["set"] for g in sets] == ["test_set", "full_set"]
    assert sets[0]["layout"]["frames"] == [
        {"frame": 0, "source_frame": 450, "timestamp_us": 123, "cameras": ["right"]}
    ]
    for group in sets:
        for channel, path in group["channels"].items():
            with closing(ffmpeg.read_frames(path, pix_fmt="rgb24")) as decoder:
                metadata = next(decoder)
                assert metadata["size"] == (64, 32)
                images = [np.frombuffer(buf, np.uint8).reshape(32, 64, 3) for buf in decoder]
            assert len(images) == (1 if group["set"] == "test_set" else 2)
            if group["set"] == "test_set":
                assert images[0][:, :32].max() < 5
                assert images[0][:, 40:].mean() == pytest.approx(
                    255 if channel == "gt_rgbs" else 128, abs=4
                )
    layout = json.loads((folder / "full_set_10_layout.json").read_text())
    assert layout["frames"][1]["cameras"] == ["left"]
