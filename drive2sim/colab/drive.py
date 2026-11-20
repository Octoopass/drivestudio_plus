"""Lấy dữ liệu gốc từ Google Drive qua một phiên Colab (CONTRACTS mục 9.3, 11.1).

Dữ liệu gốc (KITTI-360, nuScenes...) hay nằm sẵn trên Drive của người dùng. Tool không gọi API
Drive trực tiếp (cần OAuth riêng): một phiên Colab (CPU là đủ) mount Drive bằng
``colab drivemount``, rồi Studio duyệt thư mục và tải file về qua ``colab exec`` và
``colab download`` như khi tải kết quả run. Lần mount đầu, Google hỏi quyền truy cập Drive: lệnh
in một đường dẫn, người dùng mở, đồng ý rồi bấm tiếp trên Studio (``MountFlow``).

File được đọc trên VM theo từng phần ``CHUNK_BYTES`` (MD5 mỗi phần) rồi nối vào
``<file>.part`` trên máy; đủ thì đổi tên. Tải lại cùng thư mục đích thì file đã đủ được bỏ qua và
file dở đi tiếp từ phần đã có.

File này chứa:
- ``safe_drive_path``: kiểm tra đường dẫn tương đối trong ``MyDrive``.
- ``MountFlow``: một lần ``colab drivemount`` đang chạy hoặc chờ người dùng đồng ý.
- ``browse``: liệt kê một thư mục trên Drive.
- ``pull_folder``: tải cả một thư mục (hoặc một file) trên Drive về máy.
"""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

from drive2sim.colab.accounts import _env
from drive2sim.colab.client import ColabClient, ColabError

MOUNT_POINT = "/content/drive"
MY_DRIVE = f"{MOUNT_POINT}/MyDrive"
VM_TMP = "/content/d2s_drive"
CHUNK_BYTES = 32 * 1024 * 1024
RESERVE_BYTES = 5 * 1024**3  # chừa trên ổ đĩa sau khi tải xong
MAX_FILES = 50000
MAX_LIST = 500
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
PART_SUFFIX = ".part"
MARKER = "D2S_JSON:"
# trang cấp quyền; trước đó lệnh còn in "Connecting to https://colab.research.google.com..."
URL = re.compile(r"https://accounts\.google\.com/\S+")
SESSION_PATTERN = r"^[A-Za-z0-9_-]{1,40}$"

Progress = Callable[[int, int, str], None]
# Quét cây thư mục để nhận dạng (inspect): đủ sâu cho thư mục con chứa dataset (3 cấp) cộng dấu
# hiệu của dataset (nuPlan sâu nhất: nuplan-v1.1/splits/<split>/*.db, 4 cấp); giới hạn số thư mục
# vì mỗi lần liệt kê trên Drive mount mất cỡ 0,1 tới 1 s.
INSPECT_DEPTH = 7
INSPECT_MAX_DIRS = 600
INSPECT_MAX_SUBDIRS = 200
INSPECT_MAX_FILES = 300
# File nhỏ mà bộ nhận dạng đọc nội dung (tên cảnh, cảm biến của nuScenes); file khác dựng rỗng.
INSPECT_CONTENT = ("scene.json", "sensor.json")
INSPECT_CONTENT_BYTES = 4 * 1024**2


def safe_drive_path(path: str) -> str:
    """Đường dẫn tương đối trong ``MyDrive`` đã chuẩn hoá (rỗng là gốc ``MyDrive``).

    Raises:
        ValueError: Nếu có ``..`` hoặc ký tự điều khiển.
    """
    parts = [p for p in str(path or "").replace("\\", "/").split("/") if p not in ("", ".")]
    if ".." in parts or any(ord(c) < 32 for c in "".join(parts)):
        raise ValueError(f"đường dẫn Drive không hợp lệ: {path}")
    return "/".join(parts)


def _remote(path: str) -> str:
    rel = safe_drive_path(path)
    return f"{MY_DRIVE}/{rel}" if rel else MY_DRIVE


def _json(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    raise ColabError(f"phiên Colab trả kết quả lạ: {output.strip()[-300:]}")


class MountFlow:
    """Một lần mount Drive vào phiên Colab (``colab drivemount -s <phiên>``).

    Lần đầu với một tài khoản, lệnh in đường dẫn trang cấp quyền Drive của Google rồi chờ Enter;
    ``start`` trả đường dẫn đó, người dùng đồng ý trong trình duyệt rồi gọi ``confirm``. Studio
    giữ một bản cho cả app; mỗi lúc một lần mount.
    """

    def __init__(
        self,
        popen: Callable[..., "subprocess.Popen[str]"] = subprocess.Popen,
        binary: str = "colab",
    ) -> None:
        self._popen = popen
        self._binary = binary
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._session = ""
        self._url: Optional[str] = None
        self._confirmed = False
        self._log = ""

    def view(self) -> dict:
        """Trạng thái: ``{"state", "session", "url", "log"}``.

        ``state`` là ``idle``, ``needs_consent`` (chờ người dùng đồng ý trên trang Google),
        ``running``, ``succeeded`` hoặc ``failed``.
        """
        with self._lock:
            if self._process is None:
                return {"state": "idle", "session": None, "url": None, "log": ""}
            code = self._process.poll()
            if code is None:
                state = "needs_consent" if self._url and not self._confirmed else "running"
            else:
                state = "succeeded" if code == 0 else "failed"
            return {
                "state": state,
                "session": self._session,
                "url": self._url,
                "log": self._log.strip()[-800:],
            }

    def _read(self, process: "subprocess.Popen[str]") -> None:
        for line in process.stdout or ():
            with self._lock:
                self._log = (self._log + line)[-6000:]
                found = URL.search(line)
                if found and not self._url:
                    self._url = found.group(0)

    def start(self, session: str, wait_s: float = 10) -> dict:
        """Bắt đầu mount (đang có lần mount chạy thì giữ nguyên); chờ đường dẫn tối đa ``wait_s``.

        Raises:
            ValueError: Nếu tên phiên sai dạng.
        """
        if not re.match(SESSION_PATTERN, session or ""):
            raise ValueError(f"tên phiên Colab không hợp lệ: {session}")
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                command = [self._binary, "drivemount", "-s", session, MOUNT_POINT]
                self._process = self._popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, env=_env(), start_new_session=True,
                )  # fmt: skip
                self._session, self._url, self._confirmed, self._log = session, None, False, ""
                threading.Thread(target=self._read, args=(self._process,), daemon=True).start()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            view = self.view()
            if view["state"] != "running":
                return view
            time.sleep(0.2)
        return self.view()

    def confirm(self, wait_s: float = 30) -> dict:
        """Người dùng đã đồng ý trên trang Google: cho lệnh đi tiếp, chờ tối đa ``wait_s``.

        Raises:
            ValueError: Nếu không có lần mount nào đang chờ.
        """
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise ValueError("không có lần kết nối Drive nào đang chờ; bắt đầu lại")
            try:
                assert process.stdin is not None
                process.stdin.write("\n")
                process.stdin.flush()
            except OSError:
                pass
            self._confirmed = True
        try:
            process.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            pass
        return self.view()

    def cancel(self) -> dict:
        """Dừng lần mount đang chạy."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
        return self.view()


BROWSE_CODE = r"""
import json, os
path = {path!r}
out = {{"mounted": os.path.isdir({my_drive!r}), "exists": os.path.isdir(path),
        "dirs": [], "files": [], "truncated": False}}
if out["exists"]:
    names = sorted(os.listdir(path), key=str.lower)
    out["truncated"] = len(names) > {limit}
    for name in names[:{limit}]:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            out["dirs"].append(name)
        else:
            try:
                out["files"].append([name, os.path.getsize(full)])
            except OSError:
                pass
print({marker!r} + json.dumps(out))
"""

TREE_CODE = r"""
import json, os
root = {path!r}
skip = {skip!r}
files = []
if os.path.isfile(root):
    files.append([os.path.basename(root), os.path.getsize(root)])
else:
    for here, dirs, names in os.walk(root):
        dirs.sort()
        for name in sorted(names):
            if name in skip:
                continue
            full = os.path.join(here, name)
            files.append([os.path.relpath(full, root), os.path.getsize(full)])
            if len(files) > {limit}:
                break
print({marker!r} + json.dumps({{"is_file": os.path.isfile(root),
    "exists": os.path.exists(root), "mounted": os.path.isdir({my_drive!r}), "files": files}}))
"""

CHUNK_CODE = r"""
import hashlib, json, os
os.makedirs({tmp!r}, exist_ok=True)
with open({source!r}, "rb") as stream:
    stream.seek({offset})
    data = stream.read({size})
with open({target!r}, "wb") as stream:
    stream.write(data)
print({marker!r} + json.dumps({{"bytes": len(data), "md5": hashlib.md5(data).hexdigest()}}))
"""


def browse(client: ColabClient, session: str, path: str = "") -> dict:
    """Liệt kê một thư mục trong ``MyDrive``.

    Returns:
        ``{"path", "mounted", "exists", "dirs": [tên], "files": [[tên, byte]], "truncated"}``;
        ``mounted`` sai thì phải mount Drive trước.

    Raises:
        ValueError: Đường dẫn sai.
        ColabError: Phiên lỗi hoặc đã mất.
    """
    code = BROWSE_CODE.format(path=_remote(path), my_drive=MY_DRIVE, limit=MAX_LIST, marker=MARKER)
    data = _json(client.exec_python(session, code, timeout_s=120))
    data["path"] = safe_drive_path(path)
    return data


SKELETON_CODE = r"""
import json, os
root = {path!r}
out = {{"mounted": os.path.isdir({my_drive!r}), "exists": os.path.isdir(root), "dirs": {{}},
        "truncated": False}}
queue = [("", 0)]
while queue and out["exists"]:
    rel, depth = queue.pop(0)
    if len(out["dirs"]) >= {max_dirs}:
        out["truncated"] = True
        break
    here = os.path.join(root, rel) if rel else root
    try:
        names = sorted(n for n in os.listdir(here) if not n.startswith("."))
    except OSError:
        continue
    subdirs, files, cut = [], [], False
    for name in names:
        full = os.path.join(here, name)
        if os.path.isdir(full):
            if len(subdirs) < {max_subdirs}:
                subdirs.append(name)
            else:
                cut = True
        elif len(files) < {max_files}:
            try:
                item = [name, os.path.getsize(full)]
            except OSError:
                continue
            if name in {content!r} and item[1] <= {content_bytes}:
                try:
                    with open(full, encoding="utf-8") as stream:
                        item.append(stream.read())
                except (OSError, ValueError):
                    pass
            files.append(item)
        else:
            cut = True
    out["dirs"][rel] = [subdirs, files]
    out["truncated"] = out["truncated"] or cut
    if depth < {depth}:
        queue.extend(((rel + "/" + n) if rel else n, depth + 1) for n in subdirs)
print({marker!r} + json.dumps(out))
"""


def _plain_name(name: str) -> bool:
    """Tên một mục trên Drive dựng lại được an toàn trong thư mục tạm."""
    return (
        name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and all(ord(c) >= 32 for c in name)
    )


def _build_skeleton(base: Path, dirs: dict) -> dict[str, int]:
    """Dựng lại cây thư mục Drive vào ``base``: thư mục thật, file rỗng (trừ ``INSPECT_CONTENT``).

    Returns:
        Kích thước trên Drive của từng file đã dựng, theo đường dẫn tương đối.
    """
    sizes: dict[str, int] = {}
    for rel, (subdirs, files) in sorted(dirs.items()):
        parts = [p for p in rel.split("/") if p]
        if not all(_plain_name(p) for p in parts):
            continue
        here = base.joinpath(*parts)
        here.mkdir(parents=True, exist_ok=True)
        for name in subdirs:
            if _plain_name(name):
                (here / name).mkdir(exist_ok=True)
        for item in files:
            name, size = item[0], int(item[1])
            if not _plain_name(name):
                continue
            if name == "sample_annotation.json":
                text = "[{}]" if size > 2 else "[]"  # chỉ cần biết có box hay không
            else:
                text = item[2] if len(item) > 2 else ""
            (here / name).write_text(text, encoding="utf-8")
            sizes["/".join([*parts, name])] = size
    return sizes


def inspect(client: ColabClient, session: str, path: str = "") -> dict:
    """Nhận dạng dataset ở một thư mục trong ``MyDrive``, như ``d2s source inspect`` trên máy.

    Quét cây thư mục trên Colab (chỉ tên và kích thước, ``SKELETON_CODE``), dựng lại trong thư
    mục tạm trên máy với file rỗng rồi chạy đúng ``inspect_source``, nên Drive và đường dẫn trên
    máy nhận dạng giống nhau, kể cả gợi ý thư mục con chứa dataset. Không đọc nội dung file nào
    ngoài ``INSPECT_CONTENT``.

    Returns:
        ``{"path", "mounted", "exists", "truncated", "report", "sequence_bytes", "pull_sets"}``:
        ``report`` là ``SourceReport.to_dict()`` với đường dẫn tương đối trong ``MyDrive`` (không
        có ``inventory``); ``sequence_bytes`` là dung lượng trên Drive của từng chuỗi;
        ``pull_sets`` là tiền tố file để chỉ tải một chuỗi (KITTI-360), dùng cho ``include`` của
        ``pull_folder``.

    Raises:
        ValueError: Đường dẫn sai.
        ColabError: Phiên lỗi hoặc đã mất.
    """
    from drive2sim.ingest import kitti360
    from drive2sim.ingest.detect import inspect_source

    rel_root = safe_drive_path(path)
    code = SKELETON_CODE.format(
        path=_remote(path), my_drive=MY_DRIVE, max_dirs=INSPECT_MAX_DIRS,
        max_subdirs=INSPECT_MAX_SUBDIRS, max_files=INSPECT_MAX_FILES,
        content=INSPECT_CONTENT, content_bytes=INSPECT_CONTENT_BYTES, depth=INSPECT_DEPTH,
        marker=MARKER,
    )  # fmt: skip
    tree = _json(client.exec_python(session, code, timeout_s=600))
    result: dict = {
        "path": rel_root,
        "mounted": tree["mounted"],
        "exists": tree["exists"],
        "truncated": tree["truncated"],
        "report": None,
        "sequence_bytes": {},
        "pull_sets": {},
    }
    if not (tree["mounted"] and tree["exists"]):
        return result
    shown = f"MyDrive/{rel_root}" if rel_root else "MyDrive"
    with tempfile.TemporaryDirectory(prefix="d2s_drive_") as tmp:
        base = Path(tmp) / "drive"
        base.mkdir()
        sizes = _build_skeleton(base, tree["dirs"])
        report = inspect_source(base).to_dict()
        candidates = []
        for candidate in report["candidates"]:
            found = Path(candidate["path"])
            if candidate["relation"] != "child" or base not in found.parents:
                continue  # thư mục cha của bản dựng là thư mục tạm trên máy, không phải Drive
            inner = found.relative_to(base).as_posix()
            candidates.append({**candidate, "path": f"{rel_root}/{inner}" if rel_root else inner})
        local = str(base)

        def drive_text(text: str) -> str:
            return text.replace(local, shown)

        report.update(
            path=rel_root,
            inventory=[],
            candidates=candidates,
            notes=[drive_text(n) for n in report["notes"]],
            errors=[drive_text(e) for e in report["errors"]],
        )
    result["report"] = report
    sequences = sorted({s for names in report["versions"].values() for s in names if s})
    for sequence in sequences:
        result["sequence_bytes"][sequence] = sum(
            size for rel, size in sizes.items() if sequence in rel
        )
    if report["dataset"] == kitti360.DATASET:
        result["pull_sets"] = {s: kitti360.sequence_prefixes(s) for s in sequences}
    return result


def _selected(rel: str, include: Sequence[str]) -> bool:
    """File ``rel`` có được chọn không: đường dẫn bắt đầu bằng một tiền tố trong ``include``."""
    return any(rel.startswith(prefix) for prefix in include)


def _chunk(
    client: ColabClient, session: str, source: str, offset: int, size: int, work: Path
) -> bytes:
    """Đọc ``size`` byte từ ``offset`` của file trên Drive, tải về và kiểm tra MD5."""
    remote_tmp = f"{VM_TMP}/chunk.bin"
    code = CHUNK_CODE.format(
        tmp=VM_TMP, source=source, offset=offset, size=size, target=remote_tmp, marker=MARKER
    )
    info = _json(client.exec_python(session, code, timeout_s=600))
    local_tmp = work / "chunk.bin"
    client.download(session, remote_tmp, local_tmp)
    data = local_tmp.read_bytes()
    local_tmp.unlink(missing_ok=True)
    if len(data) != info["bytes"] or hashlib.md5(data).hexdigest() != info["md5"]:
        raise ColabError(f"phần tải từ Drive sai MD5: {source} @ {offset}")
    return data


def pull_folder(
    client: ColabClient,
    session: str,
    path: str,
    target: Path,
    progress: Optional[Progress] = None,
    include: Sequence[str] = (),
) -> dict:
    """Tải một thư mục (hoặc một file) trong ``MyDrive`` về ``target``.

    Args:
        client: Client Colab.
        session: Phiên đã mount Drive.
        path: Đường dẫn tương đối trong ``MyDrive``.
        target: Thư mục đích trên máy (tạo nếu chưa có).
        progress: Gọi ``progress(byte đã có, tổng byte, file đang tải)`` sau mỗi phần.
        include: Chỉ tải file có đường dẫn (tương đối ``path``) bắt đầu bằng một trong các tiền
            tố này, ví dụ ``pull_sets`` của ``inspect`` cho một chuỗi KITTI-360, hoặc
            ``"<thư mục>/"``, ``"<file>"`` đã chọn; rỗng là tải hết.

    Returns:
        ``{"path_abs", "files", "bytes", "skipped"}``: ``skipped`` là số file đã đủ từ trước.

    Raises:
        ValueError: Đường dẫn sai, Drive chưa mount, không có file (hoặc không file nào khớp
            ``include``), hoặc ổ đĩa không đủ chỗ.
        ColabError: Phiên lỗi, đã mất, hoặc dữ liệu nhận về sai.
    """
    include = [safe_drive_path(p) + ("/" if p.endswith("/") else "") for p in include]
    source_root = _remote(path)
    code = TREE_CODE.format(
        path=source_root, skip=sorted(SKIP_NAMES), limit=MAX_FILES, my_drive=MY_DRIVE,
        marker=MARKER,
    )  # fmt: skip
    tree = _json(client.exec_python(session, code, timeout_s=600))
    if not tree["mounted"]:
        raise ValueError("Drive chưa mount vào phiên Colab này; kết nối Drive trước")
    if not tree["exists"] or not tree["files"]:
        raise ValueError(f"không có file nào ở MyDrive/{safe_drive_path(path)}")
    if len(tree["files"]) > MAX_FILES:
        raise ValueError(f"thư mục có hơn {MAX_FILES} file; chọn thư mục con")
    entries = [(rel, int(size)) for rel, size in tree["files"]]
    if include and not tree["is_file"]:
        entries = [(rel, size) for rel, size in entries if _selected(rel, include)]
        if not entries:
            raise ValueError("không file nào trong thư mục khớp phần đã chọn")
    for rel, _ in entries:
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ValueError(f"đường dẫn lạ trên Drive: {rel}")

    def have(rel: str, size: int) -> int:
        final = target / rel
        if final.is_file() and final.stat().st_size == size:
            return size
        part = final.with_name(final.name + PART_SUFFIX)
        done = part.stat().st_size if part.is_file() else 0
        return done if done <= size else 0

    total = sum(size for _, size in entries)
    done = sum(have(rel, size) for rel, size in entries)
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if total - done + RESERVE_BYTES > free:
        raise ValueError(
            f"không đủ chỗ: cần {(total - done) / 1024**3:.1f} GiB, còn trống "
            f"{free / 1024**3:.1f} GiB (chừa {RESERVE_BYTES // 1024**3} GiB)"
        )
    work = target / ".d2s_drive_tmp"
    work.mkdir(exist_ok=True)
    skipped = 0
    if progress:
        progress(done, total, "")
    try:
        for rel, size in entries:
            final = target / rel
            offset = have(rel, size)
            if final.is_file() and offset == size:
                skipped += 1
                continue
            part = final.with_name(final.name + PART_SUFFIX)
            part.parent.mkdir(parents=True, exist_ok=True)
            if offset == 0:
                part.write_bytes(b"")
            source = source_root if tree["is_file"] else f"{source_root}/{rel}"
            while offset < size:
                data = _chunk(
                    client, session, source, offset, min(CHUNK_BYTES, size - offset), work
                )
                if not data:
                    raise ColabError(f"file trên Drive ngắn hơn dự kiến: {rel}")
                with part.open("ab") as stream:
                    stream.write(data)
                offset += len(data)
                done += len(data)
                if progress:
                    progress(done, total, rel)
            part.replace(final)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"path_abs": str(target), "files": len(entries), "bytes": total, "skipped": skipped}
