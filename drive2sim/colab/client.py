"""Gọi Google Colab qua lệnh ``colab`` (gói ``google-colab-cli``).

Như Kaggle, tool không import thư viện Colab mà chạy lệnh ``colab`` trong tiến trình con, nên máy
không dùng Colab không cần cài gì, và test thay được toàn bộ Colab bằng một client giả.

Cần: ``uv tool install google-colab-cli --with keyrings.alt`` (hoặc ``pip install
google-colab-cli``) và đăng nhập Google một lần (lệnh ``colab`` mở trang đồng ý ở lần gọi đầu).
Máy không có keyring của hệ điều hành (server không màn hình) đặt thêm
``PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring``. Cách dùng các lệnh đã đọc từ
``colab --help`` và mã nguồn của google-colab-cli 0.7.5 (chưa chạy job thật qua file này).

Những điều đã gặp khi lái Colab bằng lệnh này (theo kinh nghiệm của teammate với ``fflab``):
- ``colab exec`` giữ kết nối lâu (vài phút) hay bị rớt; job dài phải chạy tách khỏi kết nối trên
  VM và đọc log theo từng đoạn (``drive2sim/colab/executor.py``).
- ``colab exec --env`` giữ biến trong kernel cho mọi lần gọi sau; tool không dùng cờ này, mọi
  tham số của job ghi thẳng vào lệnh chạy job.
- API upload báo lỗi (HTTP 500) khi thư mục đích chưa có; phải tạo thư mục trước.
- Upload và download đi qua API nội dung của Jupyter: cả file nằm trong một request, nên file lớn
  được cắt thành từng phần (``CHUNK_BYTES`` ở executor).

File này chứa:
- ``ColabError``: lỗi khi gọi Colab (chưa cài, chưa đăng nhập, lệnh lỗi).
- ``ColabSessionLostError``: phiên Colab đã mất (VM bị thu hồi hoặc đã tắt).
- ``ColabClient``: phiên bản lệnh, danh sách phiên, mức dùng compute unit, tạo và tắt phiên, chạy
  một đoạn Python trên VM (có thử lại), upload và download một file.
"""

import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

Runner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]

# GPU mà ``colab new --gpu`` nhận; giá trị lạ bị CLI âm thầm đổi thành A100, nên kiểm tra trước.
GPUS = ("T4", "L4", "G4", "A100", "H100")
INSTALL_HINT = "chưa có lệnh colab; cài bằng: uv tool install google-colab-cli --with keyrings.alt"
EXEC_RETRY_S = 3.0


class ColabError(RuntimeError):
    """Lỗi khi gọi Colab: chưa cài lệnh ``colab``, chưa đăng nhập, hoặc lệnh báo lỗi."""


class ColabSessionLostError(ColabError):
    """Phiên Colab không còn (VM bị Colab thu hồi, hết giờ, hoặc đã bị tắt)."""


def _run_cli(command: Sequence[str], timeout_s: float) -> "subprocess.CompletedProcess[str]":
    """Chạy một lệnh ``colab``; không ném lỗi khi lệnh thoát khác 0 (``ColabClient`` xử lý).

    Quá ``timeout_s`` thì trả mã thoát ``124`` (như lệnh ``timeout``) thay vì ném lỗi.

    Raises:
        ColabError: Nếu chưa cài lệnh ``colab``.
    """
    try:
        return subprocess.run(list(command), capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        raise ColabError(INSTALL_HINT) from None
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(list(command), 124, "", f"quá {int(timeout_s)} giây")


def _tail(result: "subprocess.CompletedProcess[str]", lines: int = 3) -> str:
    """Vài dòng cuối của stderr (hoặc stdout) của một lệnh, gộp thành một dòng."""
    text = (result.stderr or "").strip() or (result.stdout or "").strip()
    return " ".join(text.splitlines()[-lines:])


class ColabClient:
    """Các thao tác với Colab mà executor cần.

    Args:
        runner: Hàm chạy một lệnh (danh sách tham số, giới hạn giây) và trả ``CompletedProcess``;
            mặc định ``_run_cli``. Test truyền hàm giả hoặc lớp con thay các phương thức.
        binary: Lệnh ``colab``; mặc định ``colab`` trong ``PATH``.
    """

    def __init__(self, runner: Optional[Runner] = None, binary: str = "colab") -> None:
        self._runner = runner or _run_cli
        self._binary = binary

    def _run(self, *args: str, timeout_s: float = 120) -> "subprocess.CompletedProcess[str]":
        """Chạy ``colab <args>``, không kiểm tra mã thoát."""
        return self._runner([self._binary, *args], timeout_s)

    @staticmethod
    def _raise_if_lost(session: str, result: "subprocess.CompletedProcess[str]") -> None:
        """Ném ``ColabSessionLostError`` nếu output cho thấy phiên đã mất.

        google-colab-cli in ``Session '<tên>' appears to be lost (404/401)`` khi VM bị thu hồi
        lúc đang gọi, và ``Session '<tên>' not found.`` khi phiên không còn trên server.
        """
        text = f"{result.stdout}\n{result.stderr}"
        if "appears to be lost" in text or f"Session '{session}' not found" in text:
            raise ColabSessionLostError(
                f"phiên Colab {session} đã mất (VM bị thu hồi, hết giờ hoặc đã tắt)"
            )

    def version(self) -> Optional[str]:
        """Phiên bản của lệnh ``colab``, ví dụ ``0.7.5``; ``None`` nếu chưa cài hoặc lệnh lỗi."""
        try:
            result = self._run("version", timeout_s=30)
        except ColabError:
            return None
        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", result.stdout or "")
        return match.group(1) if result.returncode == 0 and match else None

    def sessions(self) -> list[dict[str, str]]:
        """Các phiên đang có trên server (``colab sessions``); cần đăng nhập được.

        Mỗi dòng CLI in dạng ``[tên] endpoint | Hardware: T4 | Shape: ... | Variant: GPU``; phiên
        không có bản ghi trên máy này có tên ``?``.

        Returns:
            Mỗi phiên một mục ``{"name", "hardware"}``, ``hardware`` là ``CPU``, ``T4``...

        Raises:
            ColabError: Nếu chưa cài lệnh hoặc chưa đăng nhập được.
        """
        result = self._run("sessions", timeout_s=90)
        if result.returncode != 0:
            raise ColabError(f"colab sessions lỗi: {_tail(result)}")
        found = []
        for line in (result.stdout or "").splitlines():
            match = re.match(r"^\[([^\]]+)\]", line)
            if match:
                hardware = re.search(r"Hardware: (\S+)", line)
                found.append(
                    {"name": match.group(1), "hardware": hardware.group(1) if hardware else ""}
                )
        return found

    def usage(self) -> dict[str, Optional[float]]:
        """Số compute unit còn lại, mức tiêu thụ mỗi giờ và số phiên đang có (``colab usage``).

        Returns:
            ``{"balance", "rate_per_h", "assignments"}``; trường nào không đọc được là ``None``.

        Raises:
            ColabError: Nếu lệnh lỗi.
        """
        result = self._run("usage", timeout_s=60)
        if result.returncode != 0:
            raise ColabError(f"colab usage lỗi: {_tail(result)}")
        text = result.stdout or ""

        def number(label: str) -> Optional[float]:
            match = re.search(rf"{label}: ([\d.]+)", text)
            return float(match.group(1)) if match else None

        return {
            "balance": number("Current balance"),
            "rate_per_h": number("Usage rate"),
            "assignments": number("Active assignments"),
        }

    def create_session(self, session: str, gpu: str) -> None:
        """Tạo phiên Colab có GPU (``colab new -s <tên> --gpu <gpu>``); mất tới vài phút.

        Args:
            session: Tên phiên.
            gpu: Một trong ``GPUS``.

        Raises:
            ColabError: Nếu ``gpu`` lạ, tài khoản không được cấp GPU đó, hoặc Colab hết chỗ.
        """
        if gpu not in GPUS:
            raise ColabError(f"GPU {gpu} không có trên Colab; chọn một trong {', '.join(GPUS)}")
        result = self._run("new", "-s", session, "--gpu", gpu, timeout_s=900)
        if result.returncode != 0 or "READY" not in f"{result.stdout}{result.stderr}":
            raise ColabError(f"không tạo được phiên Colab {session} ({gpu}): {_tail(result)}")

    def stop(self, session: str) -> None:
        """Tắt phiên (``colab stop``), trả VM cho Colab; phiên không còn thì không làm gì.

        Raises:
            ColabError: Nếu lệnh lỗi.
        """
        result = self._run("stop", "-s", session, timeout_s=120)
        if result.returncode != 0:
            raise ColabError(f"colab stop lỗi: {_tail(result)}")

    def exec_python(
        self,
        session: str,
        code: str,
        timeout_s: float = 60,
        attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> str:
        """Chạy một đoạn Python trong kernel của phiên và trả stdout (``colab exec -f``).

        Kết nối tới VM thỉnh thoảng rớt, nên lệnh lỗi được thử lại tối đa ``attempts`` lần, nghỉ
        tăng dần giữa các lần. Đoạn code phải chạy lại được an toàn (executor viết như vậy). Phiên
        đã mất thì không thử lại.

        Args:
            session: Tên phiên.
            code: Mã Python.
            timeout_s: Giới hạn thời gian chạy đoạn code trên VM, giây.
            attempts: Số lần thử.
            sleep: Hàm nghỉ; test truyền hàm không nghỉ.

        Returns:
            stdout của đoạn code.

        Raises:
            ColabSessionLostError: Nếu phiên đã mất.
            ColabError: Nếu mọi lần thử đều lỗi.
        """
        handle, script = tempfile.mkstemp(prefix="d2s-colab-", suffix=".py")
        last = ""
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(code)
            for attempt in range(attempts):
                result = self._run(
                    "exec", "-s", session, "--timeout", str(int(timeout_s)), "-f", script,
                    timeout_s=timeout_s + 60,
                )  # fmt: skip
                self._raise_if_lost(session, result)
                if result.returncode == 0 and "Connection was lost" not in result.stdout:
                    return result.stdout
                last = _tail(result)
                if attempt + 1 < attempts:
                    sleep(EXEC_RETRY_S * (attempt + 1))
        finally:
            Path(script).unlink(missing_ok=True)
        raise ColabError(f"colab exec lỗi sau {attempts} lần thử: {last}")

    def upload(self, session: str, local: Path, remote: str) -> None:
        """Upload một file lên VM (``colab upload``); thư mục đích phải có sẵn.

        Args:
            session: Tên phiên.
            local: File trên máy.
            remote: Đường dẫn tuyệt đối trên VM, ví dụ ``/content/d2s/upload/x.part000``.

        Raises:
            ColabSessionLostError: Nếu phiên đã mất.
            ColabError: Nếu lệnh lỗi.
        """
        result = self._run("upload", "-s", session, str(local), remote, timeout_s=1800)
        self._raise_if_lost(session, result)
        if result.returncode != 0 or "Upload failed" in result.stdout:
            raise ColabError(f"upload {local.name} lên Colab lỗi: {_tail(result)}")

    def download(self, session: str, remote: str, local: Path) -> None:
        """Tải một file từ VM về (``colab download``).

        Args:
            session: Tên phiên.
            remote: Đường dẫn tuyệt đối trên VM.
            local: File đích trên máy (thư mục cha được tạo nếu chưa có).

        Raises:
            ColabSessionLostError: Nếu phiên đã mất.
            ColabError: Nếu lệnh lỗi.
        """
        local.parent.mkdir(parents=True, exist_ok=True)
        result = self._run("download", "-s", session, remote, str(local), timeout_s=1800)
        self._raise_if_lost(session, result)
        if result.returncode != 0 or "Download failed" in result.stdout:
            raise ColabError(f"tải {remote} từ Colab lỗi: {_tail(result)}")
