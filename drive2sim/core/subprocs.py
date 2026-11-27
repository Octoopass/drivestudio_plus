"""Chạy lệnh ngoài (``kaggle``, ``colab``) mà Studio dừng được khi tắt.

``subprocess.run`` không cho luồng khác dừng lệnh đang chạy, nên một lượt tải output Kaggle
(có thể cả giờ) giữ Studio không thoát được khi bấm Ctrl+C. ``run`` ở đây làm như
``subprocess.run`` nhưng ghi lại tiến trình đang chạy; ``terminate_all`` (Studio gọi khi tắt)
dừng chúng, lệnh đó trả mã thoát khác 0 và việc nền kết thúc với lỗi, chạy lại sau là được.

``terminate_thread`` chỉ dừng lệnh do một luồng chạy (huỷ một việc nền mà không đụng việc khác) và
đánh dấu luồng đó đã huỷ: lệnh bị dừng và mọi lệnh sau của luồng ném ``CancelledError``, nên code
gọi có thử lại (``ColabClient.exec_python``) cũng dừng ngay; ``clear_thread`` bỏ dấu khi việc xong.

File này chứa ``run``, ``terminate_all``, ``terminate_thread``, ``clear_thread``,
``CancelledError``.
"""

import subprocess
import threading
from collections.abc import Sequence

_running: dict[subprocess.Popen, int] = {}  # tiến trình -> luồng đã chạy nó
_lock = threading.Lock()
_cancelled: set[int] = set()  # luồng đã bị huỷ (terminate_thread)


class CancelledError(RuntimeError):
    """Lệnh của một luồng đã bị huỷ (``terminate_thread``)."""


def run(command: Sequence[str], timeout: float) -> "subprocess.CompletedProcess[str]":
    """Như ``subprocess.run(command, capture_output=True, text=True, timeout=timeout)``.

    Raises:
        FileNotFoundError: Nếu không có lệnh.
        subprocess.TimeoutExpired: Nếu chạy quá ``timeout`` giây (tiến trình đã bị dừng).
        CancelledError: Nếu luồng gọi đã bị huỷ (trước hoặc trong lúc chạy).
    """
    thread = threading.get_ident()
    if thread in _cancelled:
        raise CancelledError("đã huỷ")
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    with _lock:
        _running[process] = threading.get_ident()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    finally:
        with _lock:
            _running.pop(process, None)
    if thread in _cancelled:
        raise CancelledError("đã huỷ")
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)


def terminate_all(grace_s: float = 5.0) -> None:
    """Dừng mọi lệnh đang chạy qua ``run`` (khi tắt Studio).

    Gửi SIGTERM; lệnh nào quá ``grace_s`` giây chưa dừng thì SIGKILL.
    """
    with _lock:
        processes = list(_running)
    _stop(processes, grace_s)


def terminate_thread(thread_id: int, grace_s: float = 5.0) -> None:
    """Dừng các lệnh đang chạy qua ``run`` do luồng ``thread_id`` gọi (huỷ một việc nền).

    Luồng bị đánh dấu huỷ tới khi gọi ``clear_thread``: lệnh sau của nó ném ``CancelledError``.
    """
    with _lock:
        _cancelled.add(thread_id)
        processes = [p for p, owner in _running.items() if owner == thread_id]
    _stop(processes, grace_s)


def clear_thread(thread_id: int) -> None:
    """Bỏ dấu huỷ của luồng (việc nền đã kết thúc, luồng được dùng lại cho việc khác)."""
    with _lock:
        _cancelled.discard(thread_id)


def _stop(processes: list[subprocess.Popen], grace_s: float) -> None:
    """SIGTERM, quá ``grace_s`` giây chưa dừng thì SIGKILL."""
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(grace_s)
        except subprocess.TimeoutExpired:
            process.kill()
