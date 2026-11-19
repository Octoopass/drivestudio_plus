"""Resumable browser uploads of KITTI-360 archives into the workspace dataset folder.

The browser sends each file in chunks (`PUT /api/upload/chunk`). The server appends to `<target>.part`
and renames it once the declared size is reached, so an upload resumes from the last full chunk after a
reload or a dropped connection. Only known KITTI-360 file names are accepted; each is placed at its
canonical location (e.g. `..._image_02.tar` -> `data_2d_raw/`) wherever it sat on the user's computer.
"""
from __future__ import annotations

import getpass
import re
import shutil
import threading
import zipfile
from pathlib import Path

from .settings import display_path

from .dataset_profiles import PROFILES, detect, safe_relative

SEQ = r"2013_05_28_drive_\d{4}_sync"
RULES = [
    (re.compile(r"data_poses\.zip"), lambda m: "data_poses.zip"),
    (re.compile(r"calibration\.zip"), lambda m: "calibration.zip"),
    (re.compile(r"kitti360_rosa_infos_(train|val)\.pkl"), lambda m: f"ImageSets-rosa-6class/{m.group(0)}"),
    (re.compile(rf"({SEQ})_image_0[0-3]\.tar"), lambda m: f"data_2d_raw/{m.group(0)}"),
    (re.compile(rf"({SEQ})\.tar"), lambda m: f"data_3d_raw/{m.group(0)}"),
    (re.compile(r"(calib_cam_to_pose|calib_cam_to_velo|calib_sick_to_velo|perspective)\.txt|image_0[23]\.yaml"),
     lambda m: f"calibration/{m.group(0)}"),
]
COMMON = ["data_poses.zip", "calibration.zip", "ImageSets-rosa-6class/kitti360_rosa_infos_train.pkl",
          "ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl"]
MAX_CHUNK = 64 * 2**20
MAX_FILE = 200 * 2**30
RESERVE = 5 * 2**30


def canonical(name: str) -> str | None:
    """KITTI-360 location for a file name (any folder on the user's side), or None if not a known file."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    for pattern, target in RULES:
        match = pattern.fullmatch(base)
        if match:
            return target(match)
    return None


def required(sequence: str) -> list[str]:
    """Files the converter / DriveStudio prepare step read for one sequence."""
    if not re.fullmatch(SEQ, sequence or ""):
        raise ValueError("sequence must look like 2013_05_28_drive_0003_sync")
    return [*COMMON, *(f"data_2d_raw/{sequence}_image_0{i}.tar" for i in range(4)), f"data_3d_raw/{sequence}.tar"]


IGNORED = {".DS_Store", "Thumbs.db", "desktop.ini"}


class UploadManager:
    """Targets are workspace-relative: `kitti360/<canonical KITTI-360 name>` or `data/<dataset>/raw/<relative path>`."""

    def __init__(self, workspace: Path, folder: str = "kitti360", ssh_user: str = "", ssh_host: str = ""):
        self.ssh_user, self.ssh_host = ssh_user, ssh_host
        self.workspace = Path(workspace).resolve()
        self.root = (self.workspace / folder).resolve()
        self.folder = folder
        self._lock = threading.Lock()

    def _target(self, target: str) -> Path:
        target = str(target or "")
        if target.startswith(self.folder + "/"):
            rel = target[len(self.folder) + 1:]
            if canonical(rel) != rel:
                raise ValueError(f"not a KITTI-360 file location: {target}")
            base = self.root
        else:
            match = re.fullmatch(r"data/([a-z0-9]+)/raw/(.+)", target)
            if not match or match.group(1) not in PROFILES or match.group(1) == "kitti360":
                raise ValueError(f"not a dataset location: {target}")
            rel = safe_relative(match.group(2))
            base = (self.workspace / PROFILES[match.group(1)]["raw"]).resolve()
        path = (base / rel).resolve()
        if not path.is_relative_to(base) or path.name.endswith(".part"):
            raise ValueError("target escapes the dataset folder")
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def status(self) -> dict:
        files, partial, sequences = {}, {}, set()
        if self.root.is_dir():
            for path in self.root.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = self._rel(path)
                if rel.endswith(".part") and canonical(rel[:-5]) == rel[:-5]:
                    partial[rel[:-5]] = path.stat().st_size
                elif canonical(rel) == rel:
                    files[rel] = path.stat().st_size
                    seq = re.match(SEQ, path.name)
                    if seq:
                        sequences.add(seq.group(0))
        disk = shutil.disk_usage(self.workspace)
        return {"folder": self.folder, "absolute": display_path(self.root), "workspace": display_path(self.workspace),  # "~/..." works in scp
                "user": self.ssh_user or getpass.getuser(), "host": self.ssh_host,
                "files": files, "partial": partial, "free_bytes": disk.free,
                "sequences": {s: required(s) for s in sorted(sequences)}}

    def plan(self, items: list[dict], dataset: str = "auto") -> dict:
        """Which dataset the picked files belong to, where each goes and how much the server already has."""
        if dataset != "auto" and dataset not in PROFILES:
            raise ValueError("unknown dataset")
        items = [i for i in items if str(i.get("name", "")).rsplit("/", 1)[-1] not in IGNORED]
        if dataset == "auto":
            dataset = detect(str(i.get("path") or i.get("name", "")) for i in items)
            if dataset is None and any(canonical(str(i.get("name", ""))) for i in items):
                dataset = "kitti360"
            if dataset is None:
                raise ValueError("Không nhận ra đây là bộ dữ liệu nào. Hãy chọn loại dataset trong ô 'Loại dữ liệu' rồi thêm lại file.")
        out, skipped, need = [], [], 0
        seen = set()
        for item in items:
            name, size = str(item.get("name", "")), item.get("size")
            if dataset == "kitti360":
                rel = canonical(name)
                target = f"{self.folder}/{rel}" if rel else None
            else:
                try:
                    target = f"{PROFILES[dataset]['raw']}/{safe_relative(item.get('path') or name)}"
                except ValueError:
                    target = None
            if target is None or type(size) is not int or size <= 0 or size > MAX_FILE or target in seen:
                skipped.append(name)
                continue
            seen.add(target)
            path = self._target(target)
            part = path.with_name(path.name + ".part")
            if path.is_file() and path.stat().st_size == size:
                offset, done = size, True
            else:
                offset = part.stat().st_size if part.is_file() and part.stat().st_size <= size else 0
                done = False
            need += size - offset
            out.append({"name": name, "path": item.get("path") or name, "target": target, "size": size,
                        "offset": offset, "done": done})
        free = shutil.disk_usage(self.workspace).free
        if need > max(0, free - RESERVE):
            raise ValueError(f"not enough disk space: need {need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB")
        return {"dataset": dataset, "label": PROFILES[dataset]["label"], "files": out, "skipped": skipped}
    def write(self, target: str, offset: int, total: int, data: bytes) -> dict:
        if type(offset) is not int or type(total) is not int or offset < 0 or not 0 < total <= MAX_FILE:
            raise ValueError("invalid offset or size")
        if len(data) > MAX_CHUNK or offset + len(data) > total or (not data and offset != total):
            raise ValueError("invalid chunk")
        path = self._target(target)
        part = path.with_name(path.name + ".part")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and path.stat().st_size == total:
                return {"target": target, "offset": total, "done": True}
            have = part.stat().st_size if part.is_file() else 0
            if have != offset:
                return {"target": target, "offset": have, "done": False, "mismatch": True}
            with open(part, "ab") as fh:
                fh.write(data)
            have += len(data)
            if have < total:
                return {"target": target, "offset": have, "done": False}
            part.replace(path)
        if target == f"{self.folder}/calibration.zip":
            self._extract_calibration(path)
        return {"target": target, "offset": total, "done": True}

    def cancel(self, target: str) -> dict:
        path = self._target(target)
        part = path.with_name(path.name + ".part")
        with self._lock:
            part.unlink(missing_ok=True)
        return {"target": target, "removed": True}

    def _extract_calibration(self, archive: Path):
        """Unpack calibration.zip next to it (tools read calibration/*.txt), refusing unsafe entries."""
        dest = self.root
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                out = (dest / info.filename).resolve()
                if not out.is_relative_to(dest):
                    raise ValueError(f"unsafe entry in calibration.zip: {info.filename}")
            zf.extractall(dest)
