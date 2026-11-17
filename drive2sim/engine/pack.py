"""Đóng gói run, cảnh và cache để mang sang workdir khác (CONTRACTS mục 10.4).

``plan_pack`` kiểm kê mà không ghi dữ liệu. ``write_pack`` ghi tar không nén, giữ checkpoint,
thêm README và kiểm kê SHA-256. CLI gọi hai bước này; không gọi GPU hay dịch vụ từ xa.
View tạm do train và GPU viewer tạo được bỏ vì chứa symlink tuyệt đối và có thể dựng lại.
"""

import hashlib
import io
import json
import os
import posixpath
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine.registry import produces_artifact
from drive2sim.engine.run import D2S_DIR, RUN_FILE, run_is_active
from drive2sim.engine.workdir import Workdir

EXCLUDED_NAMES = frozenset({"kaggle.json", "access_token", ".env"})


@dataclass(frozen=True)
class PackEntry:
    """Một file hoặc symlink trong tar, kèm trạng thái nguồn để phát hiện thay đổi khi ghi.

    Attributes:
        source: File nguồn; mô hình SMPL có thể được tìm ở ngoài workdir.
        name: Tên tương đối trong tar, dùng dấu slash.
        size: Số byte nội dung; symlink dùng 0 vì chỉ ghi đích liên kết.
        mtime_ns: Thời điểm sửa file lúc kiểm kê, để kiểm tra lại khi ghi.
        link: Đích symlink tương đối trong tar; ``None`` với file thường.
    """

    source: Path
    name: str
    size: int
    mtime_ns: int
    link: Optional[str] = None


@dataclass
class PackPlan:
    """Danh sách file và dữ liệu phụ thuộc của các run đã chọn; không chứa nội dung file.

    Attributes:
        run_ids: Các run được chọn, đã sắp xếp và bỏ trùng.
        scene_ids: Các cảnh cần mang theo để mở lại những run này.
        cache_keys: Các cache được tham chiếu, tương đối với thư mục cache.
        entries: File nguồn, tên trong gói và trạng thái lúc kiểm kê.
    """

    run_ids: list[str]
    scene_ids: list[str]
    cache_keys: list[str]
    entries: list[PackEntry]

    def to_dict(self) -> dict:
        """Trả kiểm kê nhỏ cho CLI, gồm kích thước tar dự kiến và danh sách phụ thuộc."""
        # Tar có header 512 byte mỗi file, phần nội dung làm tròn và metadata PAX. Dự phòng
        # thêm 4 KiB mỗi mục cùng 1 MiB cho README, inventory và phần kết thúc.
        estimated = sum(((e.size + 511) // 512) * 512 + 4096 for e in self.entries) + (1 << 20)
        return {
            "run_ids": self.run_ids,
            "scene_ids": self.scene_ids,
            "cache_keys": self.cache_keys,
            "file_count": len(self.entries),
            "size_bytes": sum(e.size for e in self.entries),
            "estimated_tar_bytes": estimated,
        }


def _inside(path: Path, roots: list[Path]) -> bool:
    """Cho biết đường dẫn tuyệt đối có thuộc ít nhất một cây dữ liệu đã chọn không."""
    return any(path == root or root in path.parents for root in roots)


def _temporary(relative: Path) -> bool:
    """Nhận ra view có thể dựng lại, không bỏ nhầm ảnh hay checkpoint của run."""
    parts = relative.parts
    if "d2s" in parts:
        index = parts.index("d2s")
        if parts[index + 1 : index + 2] == ("viewer",):
            return True
        if parts[index + 1 : index + 2] == ("stages",):
            return len(parts) > index + 3 and parts[index + 3] in ("view", "scene_view")
    if parts and parts[0] == "cache":
        return len(parts) > 3 and parts[3] in ("view", "scene_view")
    return False


def plan_pack(workdir: Workdir, run_ids: list[str]) -> PackPlan:
    """Kiểm kê các run và đúng cảnh/cache mà chúng tham chiếu, không đọc checkpoint vào RAM.

    Args:
        workdir: Thư mục có run, cảnh và cache đầy đủ trên máy.
        run_ids: Định danh đầy đủ của các run cần bàn giao.

    Returns:
        Kế hoạch chứa đường dẫn tương đối, kích thước và trạng thái file nguồn.

    Raises:
        ValueError: Nếu không chọn run, run đang chạy, cache thiếu, hoặc symlink không thể
            mang sang máy khác bằng tập dữ liệu đã chọn.
        FileNotFoundError: Nếu thiếu cảnh, kết quả cache hoặc mô hình SMPL cần cho run.
    """
    if not run_ids:
        raise ValueError("cần chọn ít nhất một run")
    selected = sorted(set(run_ids))
    scenes: set[str] = set()
    caches: set[str] = set()
    roots = []
    needs_smpl = False
    for run_id in selected:
        folder = workdir.find_run(run_id)
        record = files.load(folder / D2S_DIR / RUN_FILE, RunRecord)
        if record.status == "running" or run_is_active(folder):
            raise ValueError(f"run {run_id} đang chạy; chờ xong rồi đóng gói")
        workdir.load_scene(record.scene_id)
        workdir.load_split(record.scene_id, record.split_id)
        if not (workdir.scene_dir(record.scene_id) / "images").is_dir():
            raise FileNotFoundError(
                f"cảnh {record.scene_id} chưa có ảnh trên máy; tải đầy đủ cảnh trước khi gói"
            )
        scenes.add(record.scene_id)
        needs_smpl |= produces_artifact((stage.stage_id for stage in record.stages), "smpl")
        roots.append(folder)
        for stage in record.stages:
            if stage.cache_key:
                cache = workdir.cache_dir / stage.cache_key
                if not cache.is_dir() or not (cache / "artifact.json").is_file():
                    raise FileNotFoundError(f"thiếu cache {stage.cache_key}; tải đầy đủ run trước")
                caches.add(stage.cache_key)
    roots += [workdir.scene_dir(scene) for scene in sorted(scenes)]
    roots += [workdir.cache_dir / key for key in sorted(caches)]
    roots = [p.absolute() for p in roots]
    entries = []
    seen: set[str] = set()
    for root in roots:
        if not _inside(root.resolve(), [workdir.root]):
            raise ValueError(f"dữ liệu nằm ngoài workdir: {root}")
        for base, directories, names in os.walk(root, followlinks=False):
            folder = Path(base)
            directories[:] = sorted(
                n for n in directories if not _temporary((folder / n).relative_to(workdir.root))
            )
            candidates = sorted(set(names) | {n for n in directories if (folder / n).is_symlink()})
            for name in candidates:
                source = folder / name
                relative = source.relative_to(workdir.root)
                archive_name = relative.as_posix()
                if name in EXCLUDED_NAMES or _temporary(relative) or archive_name in seen:
                    continue
                link = None
                stat = source.lstat()
                if source.is_symlink():
                    target = source.resolve()
                    target_name = (
                        target.relative_to(workdir.root) if _inside(target, roots) else None
                    )
                    if target_name is None or not target.exists() or _temporary(target_name):
                        raise ValueError(f"symlink trỏ ra ngoài dữ liệu được gói: {relative}")
                    link = posixpath.relpath(target_name.as_posix(), relative.parent.as_posix())
                elif not source.is_file():
                    raise ValueError(f"không phải file thường: {relative}")
                seen.add(archive_name)
                entries.append(
                    PackEntry(
                        source, archive_name, 0 if link else stat.st_size, stat.st_mtime_ns, link
                    )
                )
    if needs_smpl:
        # Gói kết quả phải tự đủ để render người. Chép file thật một lần vào workdir đích,
        # kể cả khi máy nguồn đang tìm mô hình qua biến môi trường hoặc symlink ngoài workdir.
        model = paths.find_smpl_model(workdir.root)
        if model is None:
            raise FileNotFoundError(
                f"gói có người SMPL nhưng thiếu {paths.SMPL_MODEL_NAME}; "
                "đặt vào smpl_models/ hoặc trỏ D2S_SMPL_MODEL tới file trước khi gói"
            )
        stat = model.stat()
        entries.append(
            PackEntry(
                model, f"smpl_models/{paths.SMPL_MODEL_NAME}", stat.st_size, stat.st_mtime_ns
            )
        )
    names = {entry.name for entry in entries}
    for entry in entries:
        if entry.link:
            target = posixpath.normpath(posixpath.join(posixpath.dirname(entry.name), entry.link))
            if target not in names and not any(n.startswith(target + "/") for n in names):
                raise ValueError(f"symlink không có đích trong gói: {entry.name}")
    return PackPlan(
        selected, sorted(scenes), sorted(caches), sorted(entries, key=lambda e: e.name)
    )


class _DigestReader:
    """Đọc file theo từng khối và tính SHA-256 trong cùng lượt ghi tar.

    Attributes:
        source: Luồng nhị phân do bên gọi mở và đóng.
        digest: Đối tượng SHA-256, cập nhật qua mỗi lần đọc.
    """

    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        """Đọc tối đa size byte và cộng vào mã băm."""
        data = self.source.read(size)
        self.digest.update(data)
        return data


def _add_text(archive: tarfile.TarFile, name: str, content: str) -> None:
    """Thêm metadata UTF-8 vào tar mà không tạo file rời."""
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def write_pack(plan: PackPlan, output: Path) -> dict:
    """Ghi tar theo kế hoạch, chỉ công bố file cuối khi mọi file nguồn còn nguyên.

    Args:
        plan: Kết quả ``plan_pack``.
        output: File .tar mới, không được trùng file có sẵn.

    Returns:
        Kiểm kê của gói và đường dẫn, kích thước file tar đã ghi.

    Raises:
        ValueError: Nếu sai đuôi, hết chỗ, hoặc nguồn đổi trong lúc ghi.
        FileExistsError: Nếu file đích đã có.
    """
    output = output.absolute()
    if output.suffix != ".tar":
        raise ValueError("gói không nén cần đuôi .tar")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"không ghi đè {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    required = plan.to_dict()["estimated_tar_bytes"]
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError(
            f"không đủ chỗ ghi gói: cần khoảng {required / 2**30:.2f} GiB; dùng ổ khác"
        )
    inventory = []
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=output.name + ".", suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for entry in plan.entries:
                    stat = entry.source.lstat()
                    if stat.st_mtime_ns != entry.mtime_ns or (
                        entry.link is None and stat.st_size != entry.size
                    ):
                        raise ValueError(f"nguồn đã đổi, kiểm kê lại: {entry.name}")
                    info = archive.gettarinfo(str(entry.source), arcname=entry.name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    if entry.link:
                        info.linkname = entry.link
                        archive.addfile(info)
                        inventory.append({"path": entry.name, "symlink": entry.link})
                    else:
                        # Không lưu hardlink theo inode: từng file vẫn độc lập trong gói.
                        info.type = tarfile.REGTYPE
                        info.size = entry.size
                        with entry.source.open("rb") as source:
                            reader = _DigestReader(source)
                            archive.addfile(info, reader)
                        inventory.append(
                            {
                                "path": entry.name,
                                "size_bytes": entry.size,
                                "sha256": reader.digest.hexdigest(),
                            }
                        )
                _add_text(
                    archive,
                    "README.md",
                    (paths.ASSETS_DIR / "pack/README.md").read_text(encoding="utf-8"),
                )
                _add_text(
                    archive,
                    "pack.json",
                    json.dumps(
                        {"format_version": 1, **plan.to_dict(), "files": inventory},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )
            stream.flush()
            os.fsync(stream.fileno())
        for entry in plan.entries:
            if entry.source.lstat().st_mtime_ns != entry.mtime_ns:
                raise ValueError(f"nguồn đã đổi khi đóng gói: {entry.name}")
        os.link(temporary, output)  # Tạo nguyên tử, thất bại nếu đích đã tồn tại.
        return {**plan.to_dict(), "output": str(output), "archive_bytes": output.stat().st_size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
