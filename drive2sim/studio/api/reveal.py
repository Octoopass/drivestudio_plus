"""Mở một file hoặc thư mục của thư mục làm việc trong trình quản lý file của máy chạy Studio.

Studio chỉ nghe trên ``127.0.0.1`` nên máy chạy Studio cũng là máy người dùng đang ngồi (trừ khi
mở Studio qua SSH port forward: khi đó trình quản lý file mở trên máy từ xa, và máy không có giao
diện đồ hoạ thì báo lỗi để Studio hiện đường dẫn cho người dùng tự mở).

Chỉ mở được đường dẫn nằm trong thư mục làm việc; lệnh chạy không qua shell.

File này chứa ``reveal``, ``RevealError``.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path

from drive2sim.engine.workdir import Workdir


class RevealError(RuntimeError):
    """Không mở được trình quản lý file: đường dẫn không hợp lệ hoặc máy không có giao diện."""


def _command(target: Path) -> list[str]:
    """Câu lệnh mở ``target`` theo hệ điều hành; với file thì mở thư mục chứa nó."""
    system = platform.system()
    if system == "Darwin":
        return ["open", "-R", str(target)] if target.is_file() else ["open", str(target)]
    if system == "Windows":
        return ["explorer", f"/select,{target}"] if target.is_file() else ["explorer", str(target)]
    folder = target if target.is_dir() else target.parent
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RevealError("máy chạy Studio không có giao diện đồ hoạ; mở đường dẫn này bằng tay")
    opener = shutil.which("xdg-open") or shutil.which("gio")
    if opener is None:
        raise RevealError("không tìm thấy xdg-open để mở thư mục; mở đường dẫn này bằng tay")
    return [opener, "open", str(folder)] if opener.endswith("gio") else [opener, str(folder)]


def reveal(workdir: Workdir, relative: str) -> Path:
    """Mở file hoặc thư mục ``relative`` (tương đối với thư mục làm việc).

    Args:
        workdir: Thư mục làm việc.
        relative: Đường dẫn tương đối, ví dụ ``runs/<cảnh>/omnire/<run_id>/d2s``.

    Returns:
        Đường dẫn tuyệt đối đã mở.

    Raises:
        RevealError: Nếu đường dẫn nằm ngoài thư mục làm việc, không tồn tại, hoặc máy không mở
            được trình quản lý file.
    """
    root = workdir.root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise RevealError("đường dẫn nằm ngoài thư mục làm việc")
    if not target.exists():
        raise RevealError(f"không có {relative}")
    subprocess.Popen(  # noqa: S603  (danh sách tham số, không qua shell)
        _command(target),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return target
