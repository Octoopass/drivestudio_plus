"""Tracking box PHENIKAA: nối đúng hai vật đi gần nhau, lật yaw, nội suy khung hụt, bỏ vật ngắn."""

import math
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from drive2sim.ingest import phenikaa_tracking as tracking  # noqa: E402
from drive2sim.ingest.phenikaa_tracking import LabelBox  # noqa: E402


def frame_boxes(t: int) -> list[LabelBox]:
    """Hai xe máy song song cách 2 m chạy 0,8 m/khung; xe ô tô thiếu khung 5, lật yaw ở khung 3."""
    boxes = [
        LabelBox([0.8 * t, 0.0, 0.8], 0.0, [1.9, 0.7, 1.6], "Rider"),
        LabelBox([0.8 * t, 2.0, 0.8], 0.0, [1.9, 0.7, 1.6], "Rider"),
    ]
    if t != 5:
        yaw = math.pi if t == 3 else 0.0
        boxes.append(LabelBox([10.0 + 1.0 * t, -5.0, 0.8], yaw, [4.5, 2.0, 1.5 + 0.01 * t], "Car"))
    if t == 0:
        boxes.append(LabelBox([50.0, 50.0, 0.8], 0.0, [0.5, 0.5, 1.7], "Pedestrian"))
    return boxes[::-1] if t % 2 else boxes  # thứ tự dòng trong file đổi qua các khung


def test_tracks_keep_ids_fill_gaps_and_drop_short() -> None:
    boxes, stats = tracking.track([frame_boxes(t) for t in range(10)])
    by_id: dict[int, list] = {}
    for box in boxes:
        by_id.setdefault(box.key[1], []).append(box)
    assert stats["objects"] == 3 and stats["boxes_dropped_short_tracks"] == 1
    assert stats["boxes_interpolated"] == 1
    for items in by_id.values():
        frames = [b.frames[0] for b in items]
        assert frames == list(range(10))
        ys = {round(b.center[1], 3) for b in items}
        assert len(ys) == 1  # không đổi ID giữa hai xe máy song song
        sizes = {tuple(b.size) for b in items}
        assert len(sizes) == 1  # kích thước trung vị cho cả vật
    car = next(items for items in by_id.values() if items[0].class_name == "Car")
    assert all(abs(b.rotation[0][0] - 1.0) < 1e-9 for b in car)  # yaw lật được sửa
    assert car[5].center[0] == pytest.approx(15.0)  # khung hụt nội suy


def test_read_labels_moves_boxes_to_world(tmp_path: Path) -> None:
    path = tmp_path / "label.txt"
    path.write_text("1 0 0.5 4 2 1.5 0.0 Car\n1 1 1 1 1 1 0 Truck\nbad line\n")
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    pose = [[c, -s, 0, 10], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    (box,) = tracking.read_labels(path, pose)
    assert box.center == pytest.approx([10.0, 1.0, 0.5])
    assert box.yaw == pytest.approx(math.pi / 2)
