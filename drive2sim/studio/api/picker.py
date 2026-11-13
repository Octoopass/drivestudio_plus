"""Mở hộp thoại chọn thư mục của hệ điều hành trên máy chạy Studio, trả về thư mục người dùng chọn.

Trình duyệt không cho trang web biết đường dẫn thật của thư mục được chọn, nên Studio nhờ máy chạy
Studio mở hộp thoại: Linux dùng ``zenity`` hoặc ``kdialog``, macOS dùng ``osascript``, Windows dùng
PowerShell. Studio chỉ nghe trên ``127.0.0.1`` nên hộp thoại hiện trên máy người dùng đang ngồi;
mở Studio qua SSH port forward thì hộp thoại hiện trên máy từ xa, và máy không có giao diện đồ hoạ
thì báo lỗi để người dùng gõ đường dẫn.

Lệnh chạy không qua shell. Hàm chờ tới khi người dùng chọn xong hoặc đóng hộp thoại.

File này chứa ``pick_folder``, ``PickerError``.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# Người dùng có thể mất vài phút để tìm thư mục; quá thời gian này thì coi như huỷ.
TIMEOUT_S = 600
TITLE = "Chọn thư mục dữ liệu gốc"


class PickerError(RuntimeError):
    """Không mở được hộp thoại chọn thư mục: máy không có giao diện hoặc thiếu công cụ."""


def _command(start: Optional[Path]) -> list[str]:
    """Câu lệnh mở hộp thoại chọn thư mục theo hệ điều hành, bắt đầu ở ``start`` nếu có."""
    system = platform.system()
    if system == "Darwin":
        script = f'POSIX path of (choose folder with prompt "{TITLE}"'
        if start:
            # Thoát dấu ngoặc kép để đường dẫn không thành mã AppleScript.
            quoted = str(start).replace("\\", "\\\\").replace('"', '\\"')
            script += f' default location POSIX file "{quoted}"'
        return ["osascript", "-e", script + ")"]
    if system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
            f"$d.Description = '{TITLE}';"
            # Nhân đôi dấu nháy đơn để đường dẫn không thành mã PowerShell.
            + (f"$d.SelectedPath = '{str(start).replace(chr(39), chr(39) * 2)}';" if start else "")
            + "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }"
        )
        return ["powershell", "-NoProfile", "-STA", "-Command", script]
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise PickerError("máy chạy Studio không có giao diện đồ hoạ; gõ đường dẫn thư mục vào ô")
    if shutil.which("zenity"):
        command = ["zenity", "--file-selection", "--directory", f"--title={TITLE}"]
        return command + [f"--filename={start}/"] if start else command
    if shutil.which("kdialog"):
        return ["kdialog", "--title", TITLE, "--getexistingdirectory", str(start or Path.home())]
    raise PickerError("không tìm thấy zenity hay kdialog để mở hộp thoại; gõ đường dẫn vào ô")


def pick_folder(start: Optional[str] = None) -> Optional[str]:
    """Mở hộp thoại chọn thư mục và chờ người dùng chọn.

    Args:
        start: Thư mục mở sẵn khi hộp thoại hiện (ví dụ đường dẫn đang có trong ô); bỏ qua nếu
            không phải thư mục có thật.

    Returns:
        Đường dẫn tuyệt đối của thư mục được chọn, hoặc ``None`` nếu người dùng đóng hộp thoại.

    Raises:
        PickerError: Nếu máy không mở được hộp thoại.
    """
    folder = Path(start).expanduser() if start else None
    if folder is not None and not folder.is_dir():
        folder = None
    try:
        result = subprocess.run(  # noqa: S603  (danh sách tham số, không qua shell)
            _command(folder),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as error:
        raise PickerError(f"không chạy được {error.filename}; gõ đường dẫn vào ô") from None
    except subprocess.TimeoutExpired:
        return None
    chosen = result.stdout.strip()
    # Mã khác 0 hoặc không in gì: người dùng bấm Huỷ hay đóng hộp thoại.
    if result.returncode != 0 or not chosen:
        return None
    return str(Path(chosen).resolve())
