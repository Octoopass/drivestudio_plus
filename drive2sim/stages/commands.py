"""Chạy một câu lệnh ngoài cho stage và ghi toàn bộ đầu ra vào file log.

Các stage bọc script DriveStudio (``train.drivestudio``, ``export.ply_world``...) chạy script đó
trong một tiến trình con, để engine không phải nạp PyTorch và lỗi CUDA không làm hỏng engine.
File này chứa ``run_logged``, hàm chung cho việc đó: chạy lệnh, nối stdout và stderr vào
``log.txt`` của stage, và báo lỗi kèm phần cuối log khi lệnh thất bại.
"""

import subprocess
from pathlib import Path

LOG_FILE = "log.txt"
TAIL_LINES = 20


def run_logged(command: list[str], log_path: Path) -> None:
    """Chạy một câu lệnh, nối stdout và stderr vào ``log_path``.

    Ghi nối thay vì ghi đè, nên khi chạy tiếp một run bị ngắt, log của các lần trước vẫn còn.
    Dòng đầu mỗi lần chạy là chính câu lệnh, để biết lệnh nào sinh ra log.

    Args:
        command: Câu lệnh dạng danh sách, ví dụ ``[sys.executable, "-m", "..."]``.
        log_path: File log của stage.

    Raises:
        RuntimeError: Nếu lệnh thoát với mã khác 0; thông báo kèm 20 dòng cuối của log, để
            người dùng thấy nguyên nhân mà không phải mở file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        returncode = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT).returncode
    if returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
        tail = "".join(lines[-TAIL_LINES:])
        raise RuntimeError(f"lệnh thoát với mã {returncode}; cuối {log_path.name}:\n{tail}")
