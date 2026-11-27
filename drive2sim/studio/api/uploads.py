"""Tải dữ liệu gốc từ máy người dùng lên qua trình duyệt (CONTRACTS mục 10.3, 11.1).

Studio có thể chạy trên máy GPU thuê hoặc server của lab, mở qua port forward (SPEC mục 1): khi
đó hộp thoại chọn thư mục (``/api/pick-folder``) mở trên server chứ không phải máy người dùng, và
dữ liệu gốc nằm trên máy người dùng. Trình duyệt gửi từng file theo khối (mặc định 8 MiB) vào
``uploads/<tên>/`` trong thư mục làm việc; file đang tải dở là ``<file>.part``, nên tải lại cùng
tên thì đi tiếp từ chỗ đã có. Tải xong, ô nạp cảnh xem và ingest thư mục đó như mọi thư mục khác.

File này chứa:
- ``safe_relative``: kiểm tra đường dẫn tương đối của một file trong lần tải.
- ``scenes_using``: cảnh nào đọc dữ liệu gốc từ một thư mục tải lên (``raw_source.json``).
- ``save_origin``, ``load_origin``: nhớ lần tải từ Drive của một thư mục (để tải tiếp sau khi
  huỷ hoặc khởi động lại Studio), trong ``uploads/.origins/<tên>.json``.
- ``list_uploads``, ``plan_upload``, ``write_chunk``, ``delete_upload``.
"""

import json
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from drive2sim.engine.workdir import Workdir

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$"
PART_SUFFIX = ".part"
CHUNK_MAX_BYTES = 64 * 1024 * 1024
RESERVE_BYTES = 5 * 1024**3  # chừa trên ổ đĩa sau khi tải xong
MAX_DEPTH = 12
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
ORIGINS_DIR = ".origins"
WORK_DIR = ".d2s_drive_tmp"  # thư mục tạm của drive.pull_folder trong thư mục đích


class UploadError(ValueError):
    """Yêu cầu tải lên không hợp lệ (tên, đường dẫn, kích thước, hết chỗ)."""


class UploadOffsetError(RuntimeError):
    """Khối gửi lên không nối tiếp phần đã có; trình duyệt lập kế hoạch lại rồi gửi tiếp."""


def safe_relative(path: str) -> Optional[str]:
    """Đường dẫn tương đối an toàn của một file, hoặc ``None`` nếu phải bỏ qua.

    Bỏ file rác của hệ điều hành; từ chối đường dẫn tuyệt đối, có ``..``, quá sâu, hoặc trùng
    đuôi ``.part`` của file tải dở.

    Raises:
        UploadError: Nếu đường dẫn không hợp lệ.
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or parts[-1] in SKIP_NAMES:
        return None
    if str(path).startswith(("/", "\\")) or ":" in parts[0] or ".." in parts:
        raise UploadError(f"đường dẫn không hợp lệ: {path}")
    if len(parts) > MAX_DEPTH or parts[-1].endswith(PART_SUFFIX):
        raise UploadError(f"đường dẫn không hợp lệ: {path}")
    return "/".join(parts)


def _root(workdir: Workdir, name: str) -> Path:
    if not re.match(NAME_PATTERN, name or ""):
        raise UploadError(f"tên lần tải không hợp lệ: {name}")
    return workdir.uploads_dir / name


def _origin_path(workdir: Workdir, name: str) -> Path:
    _root(workdir, name)
    return workdir.uploads_dir / ORIGINS_DIR / f"{name}.json"


def save_origin(workdir: Workdir, name: str, origin: dict) -> None:
    """Ghi nguồn của ``uploads/<tên>/`` (``{"kind": "drive", "session", "path", "include"}``)."""
    path = _origin_path(workdir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(origin, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_origin(workdir: Workdir, name: str) -> Optional[dict]:
    """Nguồn đã ghi của ``uploads/<tên>/``, hoặc ``None`` (tải từ trình duyệt, hoặc file hỏng)."""
    try:
        origin = json.loads(_origin_path(workdir, name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return origin if isinstance(origin, dict) else None


def scenes_using(workdir: Workdir) -> dict[str, list[str]]:
    """Cảnh nào đọc dữ liệu gốc từ thư mục tải lên nào.

    Ingest KITTI-360 ghi ``data/<cảnh>/raw_source.json`` trỏ về thư mục gốc, và ``prep.ncore``
    (Instant NuRec) đọc lại thư mục đó: xoá thư mục tải lên thì các cảnh này không chạy NuRec được.

    Returns:
        ``{tên lần tải: [scene_id]}``, chỉ gồm lần tải có cảnh dùng.
    """
    used: dict[str, list[str]] = {}
    if not workdir.data_dir.is_dir() or not workdir.uploads_dir.is_dir():
        return used
    base = workdir.uploads_dir.resolve()
    for source in sorted(workdir.data_dir.glob("*/raw_source.json")):
        try:
            raw = Path(json.loads(source.read_text(encoding="utf-8"))["raw_dir"]).resolve()
            name = raw.relative_to(base).parts[0]
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        used.setdefault(name, []).append(source.parent.name)
    return used


def list_uploads(workdir: Workdir) -> list[dict]:
    """Các thư mục đã tải lên.

    Returns:
        ``[{"name", "path" (tương đối), "path_abs", "files", "size_bytes", "partial",
        "partial_bytes", "modified", "used_by"}]``; ``partial`` là số file còn dở,
        ``partial_bytes`` số byte của chúng, ``modified`` lần ghi gần nhất (ISO, UTC),
        ``used_by`` các cảnh đọc dữ liệu gốc từ đây (``scenes_using``), ``origin`` là ``"drive"``
        khi tải từ Drive (tải tiếp được từ Studio), không thì ``"browser"``.
    """
    items = []
    if not workdir.uploads_dir.is_dir():
        return items
    used = scenes_using(workdir)
    for root in sorted(p for p in workdir.uploads_dir.iterdir() if p.is_dir()):
        if root.name.startswith("."):
            continue
        stats = [
            (f, f.stat())
            for f in root.rglob("*")
            if f.is_file() and WORK_DIR not in f.relative_to(root).parts
        ]
        origin = load_origin(workdir, root.name)
        parts = [st for f, st in stats if f.name.endswith(PART_SUFFIX)]
        latest = max([st.st_mtime for _, st in stats] or [root.stat().st_mtime])
        items.append(
            {
                "name": root.name,
                "path": workdir.relative(root),
                "path_abs": str(root),
                "files": len(stats) - len(parts),
                "size_bytes": sum(st.st_size for _, st in stats),
                "partial": len(parts),
                "partial_bytes": sum(st.st_size for st in parts),
                "modified": datetime.fromtimestamp(latest, timezone.utc).isoformat(),
                "used_by": used.get(root.name, []),
                "origin": "drive" if origin and origin.get("kind") == "drive" else "browser",
            }
        )
    return items


def plan_upload(workdir: Workdir, name: str, files: Iterable[dict]) -> dict:
    """Lập kế hoạch tải: file nào đã có, file nào tải tiếp từ đâu; kiểm tra chỗ trống.

    Args:
        workdir: Thư mục làm việc.
        name: Tên lần tải (thư mục ``uploads/<tên>/``).
        files: ``[{"path", "size"}]`` theo đường dẫn tương đối trong thư mục người dùng chọn.

    Returns:
        ``{"name", "path", "path_abs", "files": [{"path", "size", "offset", "done"}],
        "skipped": [path], "needed_bytes", "free_bytes"}``.

    Raises:
        UploadError: Tên hoặc đường dẫn sai, hoặc ổ đĩa không đủ chỗ.
    """
    root = _root(workdir, name)
    planned, skipped, seen = [], [], set()
    for item in files:
        rel = safe_relative(str(item.get("path", "")))
        size = int(item.get("size", -1))
        if rel is None:
            skipped.append(str(item.get("path", "")))
            continue
        if size < 0 or rel in seen:
            raise UploadError(f"file trùng hoặc kích thước sai: {rel}")
        seen.add(rel)
        target = root / rel
        part = target.with_name(target.name + PART_SUFFIX)
        if target.is_file() and target.stat().st_size == size:
            offset, done = size, True
        else:
            offset = part.stat().st_size if part.is_file() else 0
            offset, done = (offset if offset <= size else 0), False
        planned.append({"path": rel, "size": size, "offset": offset, "done": done})
    needed = sum(f["size"] - f["offset"] for f in planned)
    workdir.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workdir.root).free
    if needed + RESERVE_BYTES > free:
        raise UploadError(
            f"không đủ chỗ: cần {needed / 1024**3:.1f} GiB, còn trống {free / 1024**3:.1f} GiB "
            f"(chừa {RESERVE_BYTES // 1024**3} GiB)"
        )
    return {
        "name": name,
        "path": workdir.relative(root) if root.exists() else f"{workdir.uploads_dir.name}/{name}",
        "path_abs": str(root),
        "files": planned,
        "skipped": skipped,
        "needed_bytes": needed,
        "free_bytes": free,
    }


def write_chunk(
    workdir: Workdir, name: str, path: str, offset: int, total: int, data: bytes
) -> dict:
    """Ghi một khối vào ``<file>.part``; đủ ``total`` byte thì đổi tên thành file thật.

    Returns:
        ``{"path", "offset", "done"}``: số byte đã có sau khối này.

    Raises:
        UploadError: Tên, đường dẫn hoặc kích thước sai.
        UploadOffsetError: ``offset`` khác số byte đã có (khối trùng hoặc thiếu).
    """
    rel = safe_relative(path)
    if rel is None:
        raise UploadError(f"file này không được tải lên: {path}")
    if len(data) > CHUNK_MAX_BYTES or offset < 0 or total < 0 or offset + len(data) > total:
        raise UploadError("khối quá lớn hoặc vị trí sai")
    target = _root(workdir, name) / rel
    part = target.with_name(target.name + PART_SUFFIX)
    if target.is_file() and target.stat().st_size == total and offset >= total:
        return {"path": rel, "offset": total, "done": True}
    have = part.stat().st_size if part.is_file() else 0
    if have != offset:
        raise UploadOffsetError(f"{rel}: đã có {have} byte, khối bắt đầu ở {offset}")
    part.parent.mkdir(parents=True, exist_ok=True)
    with part.open("ab") as stream:
        stream.write(data)
    have += len(data)
    if have == total:
        part.replace(target)
    return {"path": rel, "offset": have, "done": have == total}


def delete_upload(workdir: Workdir, name: str) -> dict:
    """Xoá thư mục ``uploads/<tên>/``.

    Returns:
        ``{"removed": [đường dẫn tương đối]}``.
    """
    root = _root(workdir, name)
    _origin_path(workdir, name).unlink(missing_ok=True)
    if not root.is_dir():
        return {"removed": []}
    removed = workdir.relative(root)
    shutil.rmtree(root)
    return {"removed": [removed]}
