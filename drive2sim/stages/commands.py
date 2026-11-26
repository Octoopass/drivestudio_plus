"""Chạy một câu lệnh ngoài cho stage và ghi toàn bộ đầu ra vào file log.

Các stage bọc script DriveStudio (``train.drivestudio``, ``export.ply_world``...) chạy script đó
trong một tiến trình con, để engine không phải nạp PyTorch và lỗi CUDA không làm hỏng engine.
File này chứa ``run_logged``, hàm chung cho việc đó: chạy lệnh, nối stdout và stderr vào
``log.txt`` của stage, dừng lệnh khi người dùng huỷ run, và báo lỗi kèm phần cuối log khi lệnh
thất bại. Lệnh lỗi vì thiếu gói Python (``ModuleNotFoundError``, ``ImportError``) thì báo bằng
câu hướng dẫn cài môi trường (``MissingModuleError``) thay vì traceback.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from drive2sim.engine.environment import MissingModuleError, missing_module_message
from drive2sim.engine.stage import StageCancelledError

LOG_FILE = "log.txt"
TAIL_LINES = 20
POLL_S = 1.0  # khoảng giữa hai lần kiểm tra yêu cầu huỷ
TERMINATE_GRACE_S = 10.0  # chờ tiến trình con tự dừng sau SIGTERM trước khi SIGKILL
# Dòng cuối của traceback Python khi nạp gói lỗi, ví dụ "ModuleNotFoundError: No module named 'x'".
_IMPORT_ERROR_RE = re.compile(r"^((?:ModuleNotFoundError|ImportError): .+)$")


def find_import_error(output: str) -> Optional[str]:
    """Tìm lỗi nạp gói Python trong phần cuối đầu ra của một lệnh.

    Chỉ xét ``TAIL_LINES`` dòng cuối và lấy dòng lỗi cuối cùng khớp, vì dòng cuối của traceback là
    lỗi làm lệnh dừng; thông báo ``ImportError`` in sớm hơn rồi được bỏ qua thì không tính.

    Args:
        output: Đầu ra của lệnh (stdout và stderr gộp).

    Returns:
        Dòng lỗi, ví dụ ``"ModuleNotFoundError: No module named 'numpy'"``; ``None`` nếu không
        có.
    """
    lines = output.strip().splitlines()
    match = _IMPORT_ERROR_RE.match(lines[-1].strip()) if lines else None
    return match.group(1) if match else None


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


def raise_open_file_limit() -> None:
    """Nâng giới hạn số file mở (mức soft) của tiến trình này lên mức hard, tối đa 1048576.

    Lệnh con thừa hưởng giới hạn này. DataLoader nhiều worker của PyTorch chuyển tensor qua file
    descriptor; với mức mặc định 1024, worker lỗi ``Too many open files`` và lệnh treo mãi
    (Instant NuRec trên clip dài). Không có module ``resource`` (Windows) thì bỏ qua.
    """
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 1048576 if hard == resource.RLIM_INFINITY else min(hard, 1048576)
    if soft != resource.RLIM_INFINITY and soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
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
        MissingModuleError: Nếu lệnh thoát với mã khác 0 và lỗi cuối của nó là thiếu gói Python;
            thông báo nói cách cài môi trường chính hoặc chạy trên Kaggle, Colab.
        RuntimeError: Nếu lệnh thoát với mã khác 0 vì lý do khác; thông báo kèm 20 dòng cuối của
            log, để người dùng thấy nguyên nhân mà không phải mở file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        start = log_file.tell()  # chỉ xét đầu ra của lần chạy này, không phải các lần trước
        raise_open_file_limit()
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
        with log_path.open("rb") as log_bytes:
            log_bytes.seek(start)
            output = log_bytes.read().decode("utf-8", errors="replace")
        detail = find_import_error(output)
        if detail is not None:
            python = command[0] if command else sys.executable
            raise MissingModuleError(missing_module_message(detail, python, str(log_path)), detail)
        tail = "".join(output.splitlines(True)[-TAIL_LINES:])
        raise RuntimeError(f"lệnh thoát với mã {returncode}; cuối {log_path.name}:\n{tail}")


CAMERAS_FILE = "cameras.json"


def copy_cameras(model_dir: Path, output_dir: Path) -> None:
    """Chép ``cameras.json`` (pose và thông số camera theo khung, ``train.py`` ghi) sang kết quả.

    Để file xuất (``scene.ply``, gói xem trước) mang theo góc nhìn của các camera gốc: viewer mở
    cảnh ở đúng góc của video. Run train trước khi có file này thì bỏ qua.
    """
    source = model_dir / CAMERAS_FILE
    if source.is_file():
        shutil.copy2(source, output_dir / CAMERAS_FILE)
