"""Đọc box người trong ``instances/`` của dữ liệu chuẩn: khung nào của cảnh có người.

OmniRe dựng người đi bộ bằng ``SMPLNodes``, cần tư thế SMPL do ``prep.smpl`` (4D-Humans, tốn GPU)
hoặc ``prep.smpl_release`` tạo ra. Trainer chỉ nạp các khung trong ``frame_range`` của split
(loader DriveStudio cắt ``per_frame_instance_mask`` theo ``start_timestep``, ``end_timestep``), nên
split không có box người trong đoạn khung đó thì kết quả SMPL không được dùng. File này cho engine
biết điều đó trước khi chạy.

File này chứa:
- ``HUMAN_CLASSES``: tên lớp người (dựng bằng ``SMPLNodes``) của mọi dataset.
- ``human_frame_ranges``: đọc ``instances/`` của một cảnh, trả các đoạn khung có người; converter
  ghi kết quả vào ``available.human_frames`` của manifest (``build_manifest``), ``Workdir`` điền
  cho cảnh cũ có ``instances/`` trên máy.
- ``humans_in_range``, ``no_humans_reason``: split có người không; ``prep.smpl``,
  ``prep.smpl_release`` dùng trong ``unavailable_reason`` để ``smpl: auto`` chọn ``none``.

Xem CONTRACTS mục 2.2 (``available.human_frames``) và 4.1 (lựa chọn ``none``).
"""

import json
from pathlib import Path
from typing import Optional

from drive2sim.core.schema import Manifest, Split

# Lớp người theo OBJECT_CLASS_NODE_MAPPING của các loader DriveStudio
# (drive2sim/recon/data/loaders/*.py, các lớp ánh xạ tới ModelType.SMPLNodes); tên lớp của các
# dataset khác nhau nên gộp chung một tập. Test so tập này với mã loader.
HUMAN_CLASSES: frozenset[str] = frozenset(
    {
        "Pedestrian",  # KITTI, KITTI-360, Waymo, PandaSet, PhysicalAI
        "Person_sitting",  # KITTI, KITTI-360
        "Pedestrian with Object",  # PandaSet
        "pedestrian",  # nuPlan
        "PEDESTRIAN",  # Argoverse 2
        "OFFICIAL_SIGNALER",  # Argoverse 2
        "human.pedestrian.adult",  # nuScenes
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    }
)
INSTANCES_INFO = "instances_info.json"
FRAME_INSTANCES = "frame_instances.json"


def _ranges(frames: list[int]) -> list[tuple[int, int]]:
    """Gộp các chỉ số khung đã sắp thành các đoạn liền nhau ``(đầu, cuối)``, tính cả hai đầu."""
    ranges: list[tuple[int, int]] = []
    for frame in frames:
        if ranges and frame == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], frame)
        else:
            ranges.append((frame, frame))
    return ranges


def human_frame_ranges(scene_dir: Path) -> Optional[list[tuple[int, int]]]:
    """Đọc các đoạn khung có ít nhất một box người trong ``instances/`` của cảnh.

    Dùng đúng hai file mà loader DriveStudio đọc: ``frame_instances.json`` (khung → các vật có box
    ở khung đó) và lớp của từng vật trong ``instances_info.json``. Một khung có người khi một vật
    lớp ``HUMAN_CLASSES`` có box ở khung đó.

    Args:
        scene_dir: Thư mục cảnh ``data/<scene_id>/``.

    Returns:
        Các đoạn ``(đầu, cuối)`` theo chỉ số khung của cảnh, tính cả hai đầu, sắp tăng dần, ví dụ
        ``[(462, 495)]``; ``[]`` nếu không có người; ``None`` nếu thiếu một trong hai file hoặc
        không đọc được (không biết).
    """
    folder = Path(scene_dir) / "instances"
    try:
        info = json.loads((folder / INSTANCES_INFO).read_text(encoding="utf-8"))
        frame_instances = json.loads((folder / FRAME_INSTANCES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict) or not isinstance(frame_instances, dict):
        return None
    if any(
        not isinstance(value, dict) or not isinstance(value.get("class_name"), str)
        for value in info.values()
    ) or any(
        not str(frame).isdigit()
        or not isinstance(ids, list)
        or any(str(instance_id) not in info for instance_id in ids)
        for frame, ids in frame_instances.items()
    ):
        return None  # Nhãn hỏng không chứng minh được rằng cảnh không có người.
    humans = {
        str(key)
        for key, value in info.items()
        if isinstance(value, dict) and value.get("class_name") in HUMAN_CLASSES
    }
    frames = sorted(
        int(frame)
        for frame, ids in frame_instances.items()
        if str(frame).isdigit() and any(str(i) in humans for i in ids or [])
    )
    return _ranges(frames)


def humans_in_range(
    human_frames: list[tuple[int, int]], frame_range: tuple[int, int]
) -> list[tuple[int, int]]:
    """Phần của các đoạn có người nằm trong ``frame_range``.

    Args:
        human_frames: Các đoạn có người của cảnh, như ``available.human_frames``.
        frame_range: ``(đầu, cuối)`` của split, tính cả hai đầu.

    Returns:
        Các đoạn đã cắt theo ``frame_range``; rỗng nếu split không có người.
    """
    first, last = frame_range
    return [
        (max(start, first), min(end, last))
        for start, end in human_frames
        if start <= last and end >= first
    ]


def no_humans_reason(manifest: Manifest, split: Optional[Split]) -> Optional[str]:
    """Lý do bỏ qua bước dựng SMPL vì đoạn khung của split không có người.

    Chỉ trả lý do khi **biết chắc** không có người: manifest có ``available.human_frames`` và
    không đoạn nào chạm ``frame_range`` của split. Không biết (manifest cũ của cảnh chỉ có trên
    Kaggle, chưa chọn split) thì trả ``None`` để vẫn dựng SMPL: bỏ nhầm làm mất người trong cảnh,
    còn dựng thừa chỉ tốn GPU.

    Args:
        manifest: Manifest của cảnh.
        split: Split của run; ``None`` khi chưa chọn.

    Returns:
        Câu tiếng Việt, ví dụ ``"split every10_mixed4 (khung 0–199) không có người; cảnh có người
        ở khung 462–495"``; ``None`` nếu split có người hoặc không biết.
    """
    human_frames = manifest.available.human_frames
    if split is None or human_frames is None:
        return None
    first, last = split.frame_range
    if humans_in_range(human_frames, (first, last)):
        return None
    where = (
        "cảnh có người ở khung " + ", ".join(f"{a}–{b}" for a, b in human_frames[:5])
        if human_frames
        else "cảnh không có người"
    )
    return f"split {split.split_id} (khung {first}–{last}) không có người; {where}"
