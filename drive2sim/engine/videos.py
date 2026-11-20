"""Gom video đánh giá của một run thành bộ để Studio phát đồng bộ (SPEC mục 8C).

Đánh giá cuối của stage train (``recon/tools/eval.py``) ghi mỗi loại ảnh thành một video riêng,
mọi camera ghép chung một khung: ``videos/<bộ>_<bước>_<kênh>.mp4``, ví dụ
``test_set_30000_gt_rgbs.mp4`` (ảnh gốc), ``test_set_30000_rgbs.mp4`` (ảnh render),
``test_set_30000_rgb_error_maps.mp4`` (sai khác). Cạnh đó, run train sau 20/11/2026 có
``<bộ>_<bước>_layout.json`` (``recon/rendering/videos.write_video_layout``): kích thước khung
ghép, khung của từng camera, fps, thang màu sai khác và độ sâu. Video của run cũ không có file
này: Studio vẫn phát cả khung ghép, chỉ không cắt được từng camera.

File này chứa:
- ``CHANNELS``: kênh được nhận, theo thứ tự hiện trên Studio.
- ``video_sets``: các bộ video trong thư mục stage của một run.
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

# Kênh (đuôi tên file) theo thứ tự hiện. Khi so khớp (``video_sets``), đuôi dài được thử trước
# đuôi ngắn, vì ``Background_rgbs`` cũng kết thúc bằng ``rgbs``.
CHANNELS = (
    "gt_rgbs",
    "rgbs",
    "rgb_error_maps",
    "depths",
    "Background_rgbs",
    "RigidNodes_rgbs",
    "DeformableNodes_rgbs",
    "SMPLNodes_rgbs",
    "Dynamic_rgbs",
    "rgb_sky_blend",
    "rgb_sky",
    "lidar_on_images",
    "Background_depths",
    "RigidNodes_depths",
    "DeformableNodes_depths",
    "SMPLNodes_depths",
    "Dynamic_depths",
)
LAYOUT_SUFFIX = "_layout.json"
_PREFIX = re.compile(r"^(?P<set>test_set|full_set)_(?P<step>\d+)(?:_(?P<postfix>.+))?$")


def _split(stem: str) -> Optional[tuple[str, str]]:
    """``(tiền tố, kênh)`` của tên video, hoặc ``None`` nếu không phải video đánh giá."""
    for channel in sorted(CHANNELS, key=len, reverse=True):
        if stem.endswith("_" + channel):
            return stem[: -len(channel) - 1], channel
    return None


def video_sets(stages_dir: Path, url_of: Callable[[Path], str]) -> list[dict[str, Any]]:
    """Các bộ video đánh giá trong thư mục ``stages/`` của một run.

    Args:
        stages_dir: ``<run>/d2s/stages``.
        url_of: Đổi đường dẫn file thành url phát được.

    Returns:
        Mỗi bộ ``{"id", "stage_id", "set", "step", "postfix", "channels": {kênh: url},
        "layout": nội dung layout.json | None}``; ``set`` là ``test_set`` hoặc ``full_set``.
        Sắp theo stage, bộ test trước bộ full, bước lớn trước.
    """
    groups: dict[tuple[str, Path, str], dict[str, Any]] = {}
    for path in sorted(stages_dir.glob("*/**/*.mp4")):
        parts = path.relative_to(stages_dir).parts
        if "scene_view" in parts:
            continue
        split = _split(path.stem)
        if split is None:
            continue
        prefix, channel = split
        match = _PREFIX.match(prefix)
        if match is None:
            continue
        key = (parts[0], path.parent, prefix)
        folder = path.parent.relative_to(stages_dir / parts[0]).as_posix()
        group = groups.setdefault(
            key,
            {
                "id": f"{parts[0]}/{folder}/{prefix}",
                "stage_id": parts[0],
                "set": match["set"],
                "step": int(match["step"]),
                "postfix": match["postfix"],
                "channels": {},
                "layout": _read_layout(path.parent / f"{prefix}{LAYOUT_SUFFIX}"),
            },
        )
        group["channels"][channel] = url_of(path)
    for group in groups.values():
        group["channels"] = {
            channel: group["channels"][channel]
            for channel in CHANNELS
            if channel in group["channels"]
        }
    return sorted(
        groups.values(),
        key=lambda g: (g["stage_id"], g["set"] != "test_set", -g["step"], g["postfix"] or ""),
    )


def _read_layout(path: Path) -> Optional[dict[str, Any]]:
    """Nội dung ``layout.json``; ``None`` nếu không có hoặc hỏng (Studio phát cả khung ghép)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
        return None
    return data
