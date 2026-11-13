"""Chạy một câu lệnh ngoài cho stage và ghi toàn bộ đầu ra vào file log.

Các stage bọc script DriveStudio (``train.drivestudio``, ``export.ply_world``...) chạy script đó
trong một tiến trình con, để engine không phải nạp PyTorch và lỗi CUDA không làm hỏng engine.
File này chứa ``run_logged``, hàm chung cho việc đó: chạy lệnh, nối stdout và stderr vào
``log.txt`` của stage, dừng lệnh khi người dùng huỷ run, và báo lỗi kèm phần cuối log khi lệnh
thất bại.
"""

import os
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from drive2sim.engine.stage import StageCancelledError

LOG_FILE = "log.txt"
TAIL_LINES = 20
POLL_S = 1.0  # khoảng giữa hai lần kiểm tra yêu cầu huỷ
TERMINATE_GRACE_S = 10.0  # chờ tiến trình con tự dừng sau SIGTERM trước khi SIGKILL


def _stop_group(process: "subprocess.Popen[bytes]") -> None:
    """Dừng cả nhóm tiến trình của lệnh: SIGTERM, chờ, rồi SIGKILL nếu còn sống.

    Lệnh chạy trong một session riêng, nên các tiến trình con của nó (ví dụ worker của
    DataLoader) cũng bị dừng theo.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def run_logged(
    command: list[str],
    log_path: Path,
    cwd: Optional[Path] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    env: Optional[dict[str, str]] = None,
) -> None:
    """Chạy một câu lệnh, nối stdout và stderr vào ``log_path``.

    Ghi nối thay vì ghi đè, nên khi chạy tiếp một run bị ngắt, log của các lần trước vẫn còn.
    Dòng đầu mỗi lần chạy là chính câu lệnh, để biết lệnh nào sinh ra log. Trong lúc lệnh chạy,
    cứ mỗi giây hỏi ``should_stop``; trả ``True`` thì dừng cả nhóm tiến trình của lệnh.

    Args:
        command: Câu lệnh dạng danh sách, ví dụ ``[sys.executable, "-m", "..."]``.
        log_path: File log của stage.
        cwd: Thư mục chạy lệnh; mặc định là thư mục hiện tại.
        should_stop: Hàm cho biết người dùng đã yêu cầu huỷ chưa, thường là
            ``ctx.should_stop`` của stage.
        env: Biến môi trường thêm cho lệnh, cộng vào môi trường hiện tại.

    Raises:
        StageCancelledError: Nếu bị huỷ giữa chừng.
        RuntimeError: Nếu lệnh thoát với mã khác 0; thông báo kèm 20 dòng cuối của log, để
            người dùng thấy nguyên nhân mà không phải mở file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            start_new_session=True,
            env={**os.environ, **env} if env else None,
        )
        while True:
            try:
                returncode = process.wait(POLL_S)
                break
            except subprocess.TimeoutExpired:
                if should_stop is not None and should_stop():
                    _stop_group(process)
                    log_file.write("[d2s] đã dừng theo yêu cầu huỷ\n")
                    raise StageCancelledError("người dùng đã huỷ run") from None
            except BaseException:
                _stop_group(process)  # Ctrl-C hoặc lỗi khác: không để lại tiến trình con
                raise
    if returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
        tail = "".join(lines[-TAIL_LINES:])
        raise RuntimeError(f"lệnh thoát với mã {returncode}; cuối {log_path.name}:\n{tail}")
