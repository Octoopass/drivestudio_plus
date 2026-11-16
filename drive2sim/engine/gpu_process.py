"""Ghi lại tiến trình GPU của trang Xem 3D, để train trên máy tắt nó trước khi chạy.

Trang Xem 3D của Studio bật một tiến trình giữ model trong VRAM (viewer GPU hoặc frame server,
khoảng 1 GB). Train trên cùng GPU khi tiến trình đó còn sống thì có thể hết VRAM: ngày 16/11/2026
run OmniRe 500 bước trên Quadro T2000 (4 GB) lỗi ``CUDA out of memory`` vì frame server còn giữ
984 MB, chạy lại khi GPU trống thì xong (đỉnh 2,9 GB).

Run có thể chạy trong Studio (hàng đợi) hoặc bằng lệnh ``d2s run start`` ở một tiến trình khác,
nên Studio ghi tiến trình đang bật vào ``<thư mục làm việc>/.d2s/gpu_process.json`` thay vì chỉ
giữ trong bộ nhớ; engine đọc file đó trước khi chạy stage. Chỉ tắt khi dòng lệnh của pid đúng là
module đã ghi (pid có thể đã được hệ điều hành cấp cho tiến trình khác), và chỉ trên hệ có
``/proc`` (Linux); nơi khác thì bỏ qua.

File này chứa ``record``, ``clear``, ``stop_recorded``.
"""

import json
import os
import signal
import time
from pathlib import Path
from typing import Optional

from drive2sim.engine.workdir import Workdir

STATE_DIR = ".d2s"
STATE_FILE = "gpu_process.json"
STOP_TIMEOUT_S = 10


def _path(workdir: Workdir) -> Path:
    return workdir.root / STATE_DIR / STATE_FILE


def record(workdir: Workdir, pid: int, module: str, run_id: str) -> None:
    """Ghi tiến trình GPU vừa bật (pid cũng là nhóm tiến trình: bật bằng ``start_new_session``)."""
    path = _path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "module": module, "run_id": run_id}), "utf-8")


def clear(workdir: Workdir, pid: Optional[int] = None) -> None:
    """Xoá bản ghi; có ``pid`` thì chỉ xoá khi bản ghi là của đúng tiến trình đó."""
    path = _path(workdir)
    try:
        if pid is not None and json.loads(path.read_text("utf-8")).get("pid") != pid:
            return
        path.unlink()
    except (OSError, ValueError):
        pass


def _command(pid: int) -> Optional[str]:
    """Dòng lệnh của tiến trình (Linux), ``None`` nếu không đọc được hoặc đã thoát.

    Tiến trình đã thoát mà cha chưa thu dọn (zombie) có ``cmdline`` rỗng: coi như đã dừng.
    """
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode(errors="replace") or None


def stop_recorded(workdir: Workdir) -> Optional[str]:
    """Tắt tiến trình GPU đã ghi nếu nó còn chạy đúng module đó.

    Returns:
        Mô tả tiến trình đã tắt, ví dụ ``"drive2sim.recon.tools.frame_server (run ...)"``, hoặc
        ``None`` nếu không có gì để tắt.
    """
    try:
        entry = json.loads(_path(workdir).read_text("utf-8"))
        pid, module = int(entry["pid"]), str(entry["module"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    command = _command(pid)
    if command is None or module not in command:
        clear(workdir, pid)
        return None
    try:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while _command(pid) is not None and time.monotonic() < deadline:
            time.sleep(0.2)
        if _command(pid) is not None:
            os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    clear(workdir, pid)
    return f"{module} (run {entry.get('run_id')})"
