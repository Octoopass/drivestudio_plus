"""Xoá run, split, cảnh và dọn cache trong thư mục làm việc.

CLI (``d2s run delete``, ``d2s split delete``, ``d2s scene delete``, ``d2s cache list``,
``d2s cache gc``) và API của Studio cùng gọi các hàm ở đây. Mọi hàm xoá chỉ xoá bên trong
``D2S_WORKDIR`` và từ chối khi việc xoá làm hỏng một run đang chạy hoặc một run còn dùng thứ bị
xoá (CONTRACTS mục 10.4). Việc hỏi người dùng xác nhận do CLI hoặc giao diện lo; hàm ở đây chỉ làm
và trả về những gì đã xoá.

File này chứa:
- ``InUseError``: lỗi khi thứ cần xoá đang được dùng (mã lỗi ``conflict``).
- ``delete_run``, ``delete_split``, ``delete_scene``: xoá và trả về các đường dẫn đã xoá.
- ``CacheEntry``, ``cache_entries``: liệt kê kết quả trong ``cache/`` kèm dung lượng và run nào
  còn dùng.
- ``gc_cache``: xoá các kết quả cache không run nào dùng và các thư mục ``.partial`` bỏ dở.
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from drive2sim.engine.cache import OWNER_FILE, PARTIAL_SUFFIX, pid_alive
from drive2sim.engine.run import list_runs, run_is_active
from drive2sim.engine.workdir import Workdir


class InUseError(RuntimeError):
    """Thứ cần xoá đang được dùng: run đang chạy, hoặc split, cảnh còn run tham chiếu.

    Ứng với mã lỗi ``conflict`` (CONTRACTS mục 11.3).
    """


def _size_bytes(path: Path) -> int:
    """Tổng dung lượng các file trong một thư mục, không theo symlink."""
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            file_path = Path(root) / name
            if not file_path.is_symlink():
                total += file_path.stat().st_size
    return total


def delete_run(workdir: Workdir, run_id: str) -> list[str]:
    """Xoá thư mục của một run.

    Args:
        workdir: Thư mục làm việc.
        run_id: Run cần xoá.

    Returns:
        Đường dẫn đã xoá, tương đối với thư mục làm việc.

    Raises:
        WorkdirError: Nếu không có run này.
        InUseError: Nếu run đang chạy; huỷ run trước rồi xoá.
    """
    run_dir = workdir.find_run(run_id)
    if run_is_active(run_dir):
        raise InUseError(f"run {run_id} đang chạy; huỷ run trước rồi xoá")
    relative = workdir.relative(run_dir)
    shutil.rmtree(run_dir)
    return [relative]


def delete_split(workdir: Workdir, scene_id: str, split_id: str) -> list[str]:
    """Xoá một split của cảnh.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Cảnh.
        split_id: Split cần xoá.

    Returns:
        Đường dẫn đã xoá.

    Raises:
        WorkdirError: Nếu không có split này.
        InUseError: Nếu còn run dùng split này; xoá các run đó trước, vì không có split thì run
            không chạy tiếp được.
    """
    workdir.load_split(scene_id, split_id)
    users = [r.run_id for r in list_runs(workdir, scene_id) if r.split_id == split_id]
    if users:
        raise InUseError(
            f"split {split_id} còn được {len(users)} run dùng ({', '.join(users[:5])}); "
            "xoá các run đó trước"
        )
    path = workdir.split_path(scene_id, split_id)
    relative = workdir.relative(path)
    path.unlink()
    return [relative]


def delete_scene(workdir: Workdir, scene_id: str) -> list[str]:
    """Xoá một cảnh cùng mọi thứ của nó: dữ liệu chuẩn, split, cache và run.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Cảnh cần xoá.

    Returns:
        Các đường dẫn đã xoá.

    Raises:
        WorkdirError: Nếu cảnh chưa ingest.
        InUseError: Nếu một run của cảnh đang chạy.
    """
    workdir.load_scene(scene_id)
    for record in list_runs(workdir, scene_id):
        run_dir = workdir.run_dir(scene_id, record.approach.approach_id, record.run_id)
        if run_is_active(run_dir):
            raise InUseError(f"run {record.run_id} của cảnh đang chạy; huỷ run trước rồi xoá")
    removed = []
    for path in (
        workdir.runs_dir / scene_id,
        workdir.cache_dir / scene_id,
        workdir.scene_dir(scene_id),
    ):
        if path.exists():
            removed.append(workdir.relative(path))
            shutil.rmtree(path)
    return removed


@dataclass
class CacheEntry:
    """Một kết quả trong ``cache/``.

    Attributes:
        cache_key: Khoá, ví dụ ``nuscenes_scene-0061/prep.masks_segformer@0.1.0-4f1c9a0e``.
        scene_id: Cảnh.
        stage_id: Stage tạo ra kết quả.
        size_bytes: Dung lượng.
        partial: ``True`` nếu là thư mục ``.partial`` (đang ghi hoặc bị bỏ dở).
        stale: ``True`` nếu là ``.partial`` mà tiến trình ghi nó đã chết.
        used_by: Các run có ``cache_key`` này trong ``run.json``.
    """

    cache_key: str
    scene_id: str
    stage_id: str
    size_bytes: int
    partial: bool = False
    stale: bool = False
    used_by: list[str] = field(default_factory=list)


def cache_entries(workdir: Workdir, scene_id: Optional[str] = None) -> list[CacheEntry]:
    """Liệt kê các kết quả trong ``cache/``.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Chỉ lấy cache của cảnh này.

    Returns:
        Các kết quả, sắp theo khoá.
    """
    users: dict[str, list[str]] = {}
    for record in list_runs(workdir, scene_id):
        for stage in record.stages:
            if stage.cache_key:
                users.setdefault(stage.cache_key, []).append(record.run_id)
    entries = []
    for path in sorted(workdir.cache_dir.glob(f"{scene_id or '*'}/*")):
        if not path.is_dir():
            continue
        partial = path.name.endswith(PARTIAL_SUFFIX)
        name = path.name[: -len(PARTIAL_SUFFIX)] if partial else path.name
        key = f"{path.parent.name}/{name}"
        stale = False
        if partial:
            owner = path / OWNER_FILE
            pid = int(owner.read_text()) if owner.is_file() else None
            stale = pid is None or not pid_alive(pid)
        entries.append(
            CacheEntry(
                cache_key=key,
                scene_id=path.parent.name,
                stage_id=name.split("@", 1)[0],
                size_bytes=_size_bytes(path),
                partial=partial,
                stale=stale,
                used_by=[] if partial else users.get(key, []),
            )
        )
    return entries


def gc_cache(
    workdir: Workdir, scene_id: Optional[str] = None, dry_run: bool = False
) -> list[CacheEntry]:
    """Xoá các kết quả cache không run nào dùng và các thư mục ``.partial`` bỏ dở.

    Thư mục ``.partial`` của tiến trình còn sống (đang ghi) được giữ lại.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Chỉ dọn cache của cảnh này.
        dry_run: ``True`` thì chỉ trả về danh sách sẽ xoá, không xoá gì; CLI dùng để hỏi xác
            nhận trước.

    Returns:
        Các kết quả đã xoá (hoặc sẽ xoá khi ``dry_run``).
    """
    doomed = [
        entry
        for entry in cache_entries(workdir, scene_id)
        if (entry.partial and entry.stale) or (not entry.partial and not entry.used_by)
    ]
    if not dry_run:
        for entry in doomed:
            name = entry.cache_key.split("/", 1)[1] + (PARTIAL_SUFFIX if entry.partial else "")
            shutil.rmtree(workdir.cache_dir / entry.scene_id / name, ignore_errors=True)
    return doomed
