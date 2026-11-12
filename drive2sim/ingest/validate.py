"""Kiểm tra một thư mục cảnh sau ingest và ghi ``reports/validation.json``.

Theo cách NVIDIA kiểm tra dữ liệu (ARCHITECTURE mục 4.2): không phép kiểm tra nào đủ một mình,
nên đi từ rẻ tới kỹ. File này làm tầng rẻ: kiểm tra **cấu trúc** của thư mục cảnh so với
manifest. Tầng kỹ hơn, chiếu LiDAR lên ảnh, chưa có; báo cáo ghi rõ điều đó bằng một mục
``info`` để người đọc không tưởng dữ liệu đã được kiểm tra hình học.

File này chứa:
- ``validate_scene``: chạy mọi phép kiểm tra, trả về ``ValidationReport`` (CONTRACTS mục 3.2).

Các phép kiểm tra, mỗi lỗi gộp theo camera để báo cáo không dài theo số khung:
- ``images.missing`` (lỗi): thiếu ảnh của một camera ở một số khung.
- ``images.unreadable`` (lỗi): ảnh đầu của camera không đọc được kích thước.
- ``images.resolution`` (cảnh báo): ảnh có kích thước khác manifest.
- ``calib.intrinsics_missing`` (lỗi): thiếu ``intrinsics/<camera>.txt``.
- ``calib.extrinsics_missing`` (lỗi): thiếu cả ``extrinsics/<camera>.txt`` lẫn extrinsics theo
  từng khung ``extrinsics/<khung>_<camera>.txt``.
- ``pose.missing`` (lỗi): có ``ego_pose/`` nhưng thiếu pose ở một số khung.
- ``lidar.missing`` (cảnh báo): manifest có LiDAR nhưng thiếu file ở một số khung.
- ``objects.unreadable`` (lỗi): file ``instances/*.json`` không đọc được.
- ``projection.not_run`` (thông tin): chưa chiếu LiDAR lên ảnh.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from drive2sim.core.schema import SCHEMA_VERSIONS, Check, Manifest, ReportSummary, ValidationReport
from drive2sim.ingest.images import read_image_size
from drive2sim.ingest.manifest import frame_indices

_SHOWN_FRAMES = 10


def _check(
    check_id: str,
    severity: str,
    message: str,
    camera_id: Optional[str] = None,
    missing: Optional[list[int]] = None,
) -> Check:
    """Tạo một mục kiểm tra; với danh sách khung thiếu thì ghi số lượng và 10 khung đầu."""
    details: dict[str, object] = {}
    frame_index = None
    if missing:
        details = {"count": len(missing), "frames": missing[:_SHOWN_FRAMES]}
        frame_index = missing[0]
    return Check(
        check_id=check_id,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        camera_id=camera_id,
        frame_index=frame_index,
        details=details,
    )


def _missing(expected: range, present: set[int]) -> list[int]:
    """Các khung trong ``expected`` không có trong ``present``."""
    return [frame for frame in expected if frame not in present]


def _frames_in(directory: Path, pattern: str) -> set[int]:
    """Đọc chỉ số khung từ tên file ``<khung>.<đuôi>`` hoặc ``<khung>_...`` trong một thư mục."""
    frames = set()
    for path in directory.glob(pattern):
        head = path.stem.split("_", 1)[0]
        if head.isdigit():
            frames.add(int(head))
    return frames


def validate_scene(scene_dir: Path, manifest: Manifest) -> ValidationReport:
    """Kiểm tra cấu trúc của một thư mục cảnh so với manifest.

    Args:
        scene_dir: Thư mục cảnh ``data/<scene_id>/``.
        manifest: Manifest của cảnh.

    Returns:
        Báo cáo kiểm tra; ``summary.errors > 0`` nghĩa là cảnh không dùng được.
    """
    checks: list[Check] = []
    frames = range(manifest.timeline.num_frames)
    for index, camera in enumerate(manifest.cameras):
        camera_id = camera.camera_id
        missing = _missing(frames, set(frame_indices(scene_dir, index)))
        if missing:
            checks.append(
                _check("images.missing", "error", f"thiếu ảnh của {camera_id}", camera_id, missing)
            )
        first = sorted((scene_dir / "images").glob(f"{frames.start:03d}_{index}.*"))
        size = read_image_size(first[0]) if first else None
        if first and size is None:
            checks.append(
                _check(
                    "images.unreadable", "error", f"không đọc được ảnh của {camera_id}", camera_id
                )
            )
        elif size is not None and list(size) != list(camera.resolution):
            checks.append(
                _check(
                    "images.resolution",
                    "warning",
                    f"ảnh của {camera_id} có kích thước {size[0]}x{size[1]}, manifest ghi "
                    f"{camera.resolution[0]}x{camera.resolution[1]}",
                    camera_id,
                )
            )
        if not (scene_dir / "intrinsics" / f"{index}.txt").is_file():
            checks.append(
                _check(
                    "calib.intrinsics_missing",
                    "error",
                    f"thiếu intrinsics của {camera_id}",
                    camera_id,
                )
            )
        static = (scene_dir / "extrinsics" / f"{index}.txt").is_file()
        if not static:
            per_frame = _frames_in(scene_dir / "extrinsics", f"*_{index}.txt")
            missing = _missing(frames, per_frame)
            if missing:
                checks.append(
                    _check(
                        "calib.extrinsics_missing",
                        "error",
                        f"thiếu extrinsics của {camera_id}",
                        camera_id,
                        missing,
                    )
                )
    if (scene_dir / "ego_pose").is_dir():
        missing = _missing(frames, _frames_in(scene_dir / "ego_pose", "*.txt"))
        if missing:
            checks.append(_check("pose.missing", "error", "thiếu pose xe", missing=missing))
    if manifest.available.lidar:
        missing = _missing(frames, _frames_in(scene_dir / "lidar", "*.bin"))
        if missing:
            checks.append(_check("lidar.missing", "warning", "thiếu LiDAR", missing=missing))
    for path in sorted((scene_dir / "instances").glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            checks.append(
                _check("objects.unreadable", "error", f"không đọc được instances/{path.name}")
            )
    checks.append(
        _check(
            "projection.not_run",
            "info",
            "chưa chiếu LiDAR lên ảnh; calib và mốc thời gian chưa được kiểm tra hình học",
        )
    )
    return ValidationReport(
        schema_version=SCHEMA_VERSIONS["report"],
        scene_id=manifest.scene_id,
        created_utc=datetime.now(timezone.utc),
        summary=ReportSummary(
            errors=sum(check.severity == "error" for check in checks),
            warnings=sum(check.severity == "warning" for check in checks),
        ),
        checks=checks,
    )
