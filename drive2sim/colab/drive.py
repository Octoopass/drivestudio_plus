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
import threading
import time
from collections.abc import Callable
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
) -> dict:
    """Tải một thư mục (hoặc một file) trong ``MyDrive`` về ``target``.

    Args:
        client: Client Colab.
        session: Phiên đã mount Drive.
        path: Đường dẫn tương đối trong ``MyDrive``.
        target: Thư mục đích trên máy (tạo nếu chưa có).
        progress: Gọi ``progress(byte đã có, tổng byte, file đang tải)`` sau mỗi phần.

    Returns:
        ``{"path_abs", "files", "bytes", "skipped"}``: ``skipped`` là số file đã đủ từ trước.

    Raises:
        ValueError: Đường dẫn sai, Drive chưa mount, không có file, hoặc ổ đĩa không đủ chỗ.
        ColabError: Phiên lỗi, đã mất, hoặc dữ liệu nhận về sai.
    """
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
