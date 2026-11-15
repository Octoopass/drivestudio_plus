"""Chạy lệnh ngoài (``kaggle``, ``colab``) mà Studio dừng được khi tắt.

``subprocess.run`` không cho luồng khác dừng lệnh đang chạy, nên một lượt tải output Kaggle
(có thể cả giờ) giữ Studio không thoát được khi bấm Ctrl+C. ``run`` ở đây làm như
``subprocess.run`` nhưng ghi lại tiến trình đang chạy; ``terminate_all`` (Studio gọi khi tắt)
dừng chúng, lệnh đó trả mã thoát khác 0 và việc nền kết thúc với lỗi, chạy lại sau là được.

File này chứa ``run``, ``terminate_all``.
"""

import subprocess
import threading
from collections.abc import Sequence

_running: set[subprocess.Popen] = set()
_lock = threading.Lock()


def run(command: Sequence[str], timeout: float) -> "subprocess.CompletedProcess[str]":
    """Như ``subprocess.run(command, capture_output=True, text=True, timeout=timeout)``.

    Raises:
        FileNotFoundError: Nếu không có lệnh.
        subprocess.TimeoutExpired: Nếu chạy quá ``timeout`` giây (tiến trình đã bị dừng).
    """
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    with _lock:
        _running.add(process)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    finally:
        with _lock:
            _running.discard(process)
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)


def terminate_all(grace_s: float = 5.0) -> None:
    """Dừng mọi lệnh đang chạy qua ``run`` (khi tắt Studio).

    Gửi SIGTERM; lệnh nào quá ``grace_s`` giây chưa dừng thì SIGKILL.
    """
    with _lock:
        processes = list(_running)
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(grace_s)
        except subprocess.TimeoutExpired:
            process.kill()
