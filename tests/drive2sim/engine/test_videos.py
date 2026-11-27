"""Gom video đánh giá thành bộ cho trình phát so sánh của Studio (``engine/videos.py``)."""

import json
from pathlib import Path

from drive2sim.engine.videos import video_sets


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_videos_are_grouped_by_set_and_channel(tmp_path: Path) -> None:
    videos = tmp_path / "train.drivestudio" / "videos"
    for name in (
        "test_set_30000_gt_rgbs",
        "test_set_30000_rgbs",
        "test_set_30000_Background_rgbs",
        "test_set_30000_rgb_error_maps",
        "full_set_30000_rgbs",
        "full_set_30000_gt_rgbs",
    ):
        _touch(videos / f"{name}.mp4")
    _touch(videos / "novel_30000" / "front_center_interp.mp4")
    layout = {
        "width": 10,
        "height": 5,
        "fps": 2,
        "cameras": [{"name": "a", "x": 0, "y": 0, "width": 5, "height": 5}],
    }
    _touch(videos / "test_set_30000_layout.json", json.dumps(layout))

    sets = video_sets(tmp_path, lambda p: p.name)

    assert [(s["set"], s["step"]) for s in sets] == [("test_set", 30000), ("full_set", 30000)]
    test = sets[0]
    # Background_rgbs kết thúc bằng rgbs nhưng là kênh riêng; thứ tự theo CHANNELS.
    assert list(test["channels"]) == ["gt_rgbs", "rgbs", "rgb_error_maps", "Background_rgbs"]
    assert test["channels"]["rgbs"] == "test_set_30000_rgbs.mp4"
    assert test["layout"] == layout
    # run cũ không có layout: vẫn có bộ, layout rỗng
    assert sets[1]["layout"] is None


def test_broken_layout_is_ignored(tmp_path: Path) -> None:
    videos = tmp_path / "train.drivestudio" / "videos"
    _touch(videos / "test_set_10_rgbs.mp4")
    _touch(videos / "test_set_10_layout.json", "{oops")
    assert video_sets(tmp_path, str)[0]["layout"] is None
