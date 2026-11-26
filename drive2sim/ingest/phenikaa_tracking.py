"""Nối box 3D của PHENIKAA qua các khung thành vật có ID (tracking), để OmniRe dựng xe, người.

Nhãn ``label/<mốc>.txt`` của Phenikaa-X là box 3D theo từng khung, **không có ID theo dõi**: mỗi
dòng ``x y z dài rộng cao yaw lớp`` trong hệ xe của khung đó, ``z`` là tâm hộp, ``yaw`` quanh +z,
lớp ``Car``, ``Rider`` (người trên xe máy, xe đạp, box bao cả xe), ``Pedestrian``. File này đưa box
sang world của cảnh rồi nối chúng thành vật bằng cách ghép theo khoảng cách:

1. Mỗi vật đang theo có vị trí dự đoán ở khung sau = vị trí cuối + vận tốc ước lượng (vận tốc
   không đổi, làm trơn theo hệ số ``VELOCITY_SMOOTHING``).
2. Ghép box mới với vật bằng thuật toán Hungarian (``scipy.optimize.linear_sum_assignment``) trên
   khoảng cách mặt đất (x, y), chỉ trong cùng lớp và trong ngưỡng ``GATES`` của lớp (Car 2,5 m,
   Rider 1,5 m, Pedestrian 1,0 m ở 10 Hz).
3. Box không ghép được mở vật mới; vật mất dấu quá ``MAX_AGE`` khung thì đóng.
4. Sau khi nối: khung bị hụt (tối đa ``MAX_AGE`` khung) được nội suy tuyến tính; yaw được lật
   ±180° cho liền mạch với khung trước (nhãn đôi khi lật hướng box); kích thước lấy trung vị của
   cả vật (box một vật không đổi cỡ); vật ngắn hơn ``MIN_LENGTH`` khung bị bỏ (thường là nhiễu).

Kết quả là các ``kitti360_boxes.Box`` có ``key = (-1, id)``, dùng lại ``build_instances`` và
``dynamic_mask_rects`` của KITTI-360 để ghi ``instances/`` và ``dynamic_masks/`` đúng định dạng
script KITTI của DriveStudio.

Rủi ro đã biết: hai vật cùng lớp đi sát nhau (xe máy chạy song song) có thể bị đổi ID cho nhau;
``tracking_stats`` báo số vật, độ dài và số lần nội suy để kiểm.

File này chứa ``CLASS_MAP``, ``GATES``, ``read_labels``, ``track``, ``tracking_stats``.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from drive2sim.ingest.kitti360_boxes import Box, Matrix

# Lớp Phenikaa-X → lớp của loader KITTI (``OBJECT_CLASS_NODE_MAPPING``): Rider là người cùng xe
# máy nên vào Cyclist (node biến dạng) như KITTI-360.
CLASS_MAP = {"Car": "Car", "Rider": "Cyclist", "Pedestrian": "Pedestrian"}
# Khoảng cách ghép lớn nhất (m) giữa vị trí dự đoán và box mới, theo lớp.
GATES = {"Car": 2.5, "Rider": 1.5, "Pedestrian": 1.0}
MAX_AGE = 3
MIN_LENGTH = 5
VELOCITY_SMOOTHING = 0.5


@dataclass
class LabelBox:
    """Một dòng nhãn đã đưa sang world của cảnh.

    Attributes:
        center: Tâm box trong world của cảnh.
        yaw: Góc quay quanh trục z của world (rad).
        size: ``[dài, rộng, cao]``.
        label: Lớp gốc của Phenikaa-X.
    """

    center: list[float]
    yaw: float
    size: list[float]
    label: str


@dataclass
class _Track:
    label: str
    frames: list[int] = field(default_factory=list)
    boxes: list[LabelBox] = field(default_factory=list)
    velocity: tuple[float, float] = (0.0, 0.0)

    def predict(self, frame: int) -> tuple[float, float]:
        gap = frame - self.frames[-1]
        x, y = self.boxes[-1].center[:2]
        return x + self.velocity[0] * gap, y + self.velocity[1] * gap

    def add(self, frame: int, box: LabelBox) -> None:
        if self.frames:
            gap = frame - self.frames[-1]
            last = self.boxes[-1].center
            measured = ((box.center[0] - last[0]) / gap, (box.center[1] - last[1]) / gap)
            a = VELOCITY_SMOOTHING if len(self.frames) > 1 else 1.0
            self.velocity = (
                a * measured[0] + (1 - a) * self.velocity[0],
                a * measured[1] + (1 - a) * self.velocity[1],
            )
        self.frames.append(frame)
        self.boxes.append(box)


def read_labels(path: Path, vehicle_to_world: Matrix) -> list[LabelBox]:
    """Đọc ``label/<mốc>.txt`` và đưa box từ hệ xe của khung sang world của cảnh.

    Args:
        path: File nhãn của một khung.
        vehicle_to_world: Pose xe → world của cảnh ở khung đó (4×4).

    Returns:
        Các box có lớp trong ``CLASS_MAP``; dòng hỏng bị bỏ.
    """
    boxes = []
    r = vehicle_to_world
    heading = math.atan2(r[1][0], r[0][0])
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 8 or parts[7] not in CLASS_MAP:
            continue
        x, y, z, length, width, height, yaw = (float(v) for v in parts[:7])
        center = [r[i][0] * x + r[i][1] * y + r[i][2] * z + r[i][3] for i in range(3)]
        boxes.append(LabelBox(center, yaw + heading, [length, width, height], parts[7]))
    return boxes


def _assign(tracks: list[_Track], boxes: list[LabelBox], frame: int) -> dict[int, int]:
    """Ghép box với vật đang theo (Hungarian, trong ngưỡng của lớp): ``{box: vật}``."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not tracks or not boxes:
        return {}
    big = 1e6
    cost = np.full((len(boxes), len(tracks)), big)
    for i, box in enumerate(boxes):
        for j, item in enumerate(tracks):
            if item.label != box.label:
                continue
            px, py = item.predict(frame)
            distance = math.hypot(box.center[0] - px, box.center[1] - py)
            if distance <= GATES[box.label]:
                cost[i, j] = distance
    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rows, cols) if cost[i, j] < big}


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _finish(item: _Track) -> tuple[list[int], list[LabelBox], int]:
    """Lật yaw cho liền mạch, nội suy khung hụt, kích thước trung vị; trả thêm số khung nội suy."""
    yaws = [item.boxes[0].yaw]
    for box in item.boxes[1:]:
        yaw = box.yaw
        if abs(_wrap(yaw - yaws[-1])) > math.pi / 2:
            yaw += math.pi  # box bị lật hướng so với khung trước
        yaws.append(yaws[-1] + _wrap(yaw - yaws[-1]))
    size = [sorted(b.size[k] for b in item.boxes)[len(item.boxes) // 2] for k in range(3)]
    frames, boxes, filled = [], [], 0
    for k, frame in enumerate(item.frames):
        if k:
            previous = item.frames[k - 1]
            for missing in range(previous + 1, frame):
                a = (missing - previous) / (frame - previous)
                center = [
                    (1 - a) * p + a * q
                    for p, q in zip(item.boxes[k - 1].center, item.boxes[k].center)
                ]
                frames.append(missing)
                yaw = (1 - a) * yaws[k - 1] + a * yaws[k]
                boxes.append(LabelBox(center, yaw, size, item.label))
                filled += 1
        frames.append(frame)
        boxes.append(LabelBox(item.boxes[k].center, yaws[k], size, item.label))
    return frames, boxes, filled


def track(labels: Sequence[list[LabelBox]]) -> tuple[list[Box], dict]:
    """Nối box qua các khung thành vật có ID.

    Args:
        labels: Box của từng khung (chỉ số khung trong cảnh), đã ở world của cảnh.

    Returns:
        ``(boxes, stats)``: mỗi box một khung, ``key = (-1, id vật)``; ``stats`` như
        ``tracking_stats``.
    """
    active: list[_Track] = []
    done: list[_Track] = []
    for frame, boxes in enumerate(labels):
        still = []
        for item in active:
            (done if frame - item.frames[-1] > MAX_AGE else still).append(item)
        active = still
        matched = _assign(active, list(boxes), frame)
        for i, box in enumerate(boxes):
            if i in matched:
                active[matched[i]].add(frame, box)
            else:
                item = _Track(box.label)
                item.add(frame, box)
                active.append(item)
    done.extend(active)
    done.sort(key=lambda item: item.frames[0])
    result: list[Box] = []
    kept, dropped, filled_total, lengths = 0, 0, 0, []
    for item in done:
        if len(item.frames) < MIN_LENGTH:
            dropped += len(item.frames)
            continue
        frames, boxes, filled = _finish(item)
        filled_total += filled
        lengths.append(len(frames))
        for frame, box in zip(frames, boxes):
            c, s = math.cos(box.yaw), math.sin(box.yaw)
            result.append(
                Box(
                    key=(-1, kept),
                    class_name=CLASS_MAP[item.label],
                    frames=(frame,),
                    rotation=[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    center=list(box.center),
                    size=list(box.size),
                )
            )
        kept += 1
    total = sum(len(boxes) for boxes in labels)
    return result, tracking_stats(done, kept, lengths, total, dropped, filled_total)


def tracking_stats(
    tracks: Sequence[_Track],
    kept: int,
    lengths: Sequence[int],
    total: int,
    dropped: int,
    filled: int,
) -> dict:
    """Số liệu để kiểm chất lượng tracking, ghi vào báo cáo của cảnh."""
    by_class: dict[str, int] = {}
    for item in tracks:
        if len(item.frames) >= MIN_LENGTH:
            by_class[CLASS_MAP[item.label]] = by_class.get(CLASS_MAP[item.label], 0) + 1
    ordered = sorted(lengths)
    return {
        "objects": kept,
        "objects_by_class": by_class,
        "label_boxes": total,
        "boxes_dropped_short_tracks": dropped,
        "boxes_interpolated": filled,
        "median_track_length": ordered[len(ordered) // 2] if ordered else 0,
        "gates_m": GATES,
        "max_age": MAX_AGE,
        "min_length": MIN_LENGTH,
    }
