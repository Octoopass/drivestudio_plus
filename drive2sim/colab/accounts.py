"""Tài khoản Google và phiên bản của lệnh ``colab``, cho Studio và ``d2s colab`` (CONTRACTS 9.3).

Lệnh ``colab`` lưu một token Google trong keyring; Drive và phiên Colab đều theo tài khoản đó.
File này cho phép xem tài khoản đang dùng, chuyển giữa các tài khoản đã đăng nhập, thêm tài khoản
ngay trên Studio (mở trang đăng nhập Google, dán mã xác thực) và cập nhật google-colab-cli. Việc
đọc, ghi token chạy trong ``account_helper.py`` bằng Python của google-colab-cli (đường dẫn lấy từ
dòng ``#!`` của lệnh ``colab``), vì cần mã đăng nhập của gói đó.

File này chứa:
- ``cli_python``: trình thông dịch của google-colab-cli.
- ``list_accounts``, ``use_account``, ``forget_account``: danh sách và chuyển tài khoản.
- ``LoginFlow``: một lần đăng nhập đang chờ mã xác thực (Studio giữ một bản).
- ``cli_versions``, ``update_cli``: phiên bản đang cài, bản mới nhất, cập nhật.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

from drive2sim.colab.client import INSTALL_HINT, ColabClient, ColabError

HELPER = Path(__file__).with_name("account_helper.py")
EMAIL_PATTERN = r"^[^@\s]{1,64}@[^@\s]{1,190}$"
CODE_PATTERN = r"^[A-Za-z0-9/_.~-]{10,300}$"
LOGIN_URL = re.compile(r"https://accounts\.google\.com/\S+")
MARKER = "D2S_JSON:"
CLI_PACKAGE = "google-colab-cli"

Runner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]


def _env() -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHON_KEYRING_BACKEND", "keyrings.alt.file.PlaintextKeyring")
    return env


def _run(command: Sequence[str], timeout_s: float) -> "subprocess.CompletedProcess[str]":
    try:
        return subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, env=_env(),
            stdin=subprocess.DEVNULL, check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(list(command), 124, "", f"quá {int(timeout_s)} giây")


def cli_python(binary: str = "colab") -> Path:
    """Trình thông dịch Python của google-colab-cli, đọc từ dòng ``#!`` của lệnh ``colab``.

    Raises:
        ColabError: Nếu chưa cài lệnh ``colab`` hoặc không đọc được trình thông dịch.
    """
    found = shutil.which(binary) or str(Path.home() / ".local" / "bin" / binary)
    try:
        first = Path(found).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError) as error:
        raise ColabError(INSTALL_HINT) from error
    python = Path(first[2:].strip()) if first.startswith("#!") else None
    if python is None or not python.is_file():
        raise ColabError(f"không tìm thấy Python của google-colab-cli (dòng đầu của {found})")
    return python


def _parse(result: "subprocess.CompletedProcess[str]") -> dict:
    for line in reversed((result.stdout or "").splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    detail = ((result.stderr or "") + (result.stdout or "")).strip()[-400:]
    raise ColabError(f"đọc tài khoản Google lỗi: {detail or 'không có kết quả'}")


def _helper(*args: str, runner: Optional[Runner] = None, timeout_s: float = 60) -> dict:
    command = [str(cli_python()), str(HELPER), *args]
    return _parse((runner or _run)(command, timeout_s))


def _check_email(email: str) -> str:
    if not re.match(EMAIL_PATTERN, email or ""):
        raise ValueError(f"email không hợp lệ: {email}")
    return email


def list_accounts(runner: Optional[Runner] = None) -> dict:
    """Các tài khoản đã đăng nhập và tài khoản đang dùng.

    Returns:
        ``{"accounts": [email], "active": email hoặc None}``.
    """
    return _helper("list", runner=runner)


def use_account(email: str, runner: Optional[Runner] = None) -> dict:
    """Chuyển lệnh ``colab`` sang một tài khoản đã đăng nhập; trả danh sách sau khi chuyển."""
    _helper("use", _check_email(email), runner=runner)
    return list_accounts(runner)


def forget_account(email: str, runner: Optional[Runner] = None) -> dict:
    """Bỏ một tài khoản khỏi danh sách (token đang dùng giữ nguyên); trả danh sách mới."""
    _helper("forget", _check_email(email), runner=runner)
    return list_accounts(runner)


class LoginFlow:
    """Một lần thêm tài khoản: chạy ``account_helper.py login`` và chờ mã xác thực.

    ``start`` trả đường dẫn trang đăng nhập Google; người dùng đăng nhập, Google hiện một mã;
    ``submit_code`` chuyển mã cho tiến trình và chờ nó lưu token. Mỗi lúc một lần đăng nhập.
    """

    def __init__(self, popen: Callable[..., "subprocess.Popen[str]"] = subprocess.Popen) -> None:
        self._popen = popen
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._url: Optional[str] = None
        self._email: Optional[str] = None
        self._log = ""
        self._sent = False

    def view(self) -> dict:
        """Trạng thái: ``{"state", "url", "email", "error"}``.

        ``state`` là ``idle``, ``waiting_code`` (đang chờ mã), ``checking`` (đã gửi mã),
        ``succeeded`` hoặc ``failed``.
        """
        with self._lock:
            if self._process is None:
                return {"state": "idle", "url": None, "email": None, "error": None}
            code = self._process.poll()
            if code is None:
                state = "checking" if self._sent else "waiting_code"
            else:
                state = "succeeded" if code == 0 and self._email else "failed"
            error = self._log.strip()[-400:] if state == "failed" else None
            return {"state": state, "url": self._url, "email": self._email, "error": error}

    def _read(self, process: "subprocess.Popen[str]") -> None:
        for line in process.stdout or ():
            with self._lock:
                self._log = (self._log + line)[-4000:]
                found = LOGIN_URL.search(line)
                if found and not self._url:
                    self._url = found.group(0)
                if line.startswith(MARKER):
                    try:
                        self._email = json.loads(line[len(MARKER) :]).get("email")
                    except ValueError:
                        pass

    def start(self, wait_s: float = 15) -> dict:
        """Bắt đầu đăng nhập (lần đang chờ mã thì giữ nguyên); chờ đường dẫn tối đa ``wait_s``."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                command = [str(cli_python()), "-u", str(HELPER), "login"]
                self._process = self._popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, env=_env(), start_new_session=True,
                )  # fmt: skip
                self._url, self._email, self._log, self._sent = None, None, "", False
                threading.Thread(target=self._read, args=(self._process,), daemon=True).start()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            view = self.view()
            if view["url"] or view["state"] != "waiting_code":
                return view
            time.sleep(0.2)
        return self.view()

    def submit_code(self, code: str, wait_s: float = 60) -> dict:
        """Gửi mã xác thực Google đã dán và chờ tiến trình lưu token.

        Raises:
            ValueError: Mã sai dạng, hoặc không có lần đăng nhập nào đang chờ mã.
        """
        code = (code or "").strip()
        if not re.match(CODE_PATTERN, code):
            raise ValueError("mã xác thực không hợp lệ; dán đúng mã Google hiển thị")
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or self._sent:
                raise ValueError("không có lần đăng nhập nào đang chờ mã; bắt đầu lại")
            assert process.stdin is not None
            process.stdin.write(code + "\n")
            process.stdin.flush()
            self._sent = True
        try:
            process.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.2)  # cho luồng đọc kịp lấy dòng kết quả cuối
        return self.view()

    def cancel(self) -> dict:
        """Huỷ lần đăng nhập đang chờ."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
            self._process = None
        return self.view()


def cli_versions(client: Optional[ColabClient] = None) -> dict:
    """Phiên bản google-colab-cli đang cài và bản mới nhất (``colab update`` chỉ kiểm tra).

    Returns:
        ``{"version", "latest", "up_to_date"}``; ``latest`` là ``None`` khi không hỏi được.
    """
    client = client or ColabClient()
    version = client.version()
    latest = client.latest_version() if version else None
    up_to_date = bool(version and latest == version)
    return {"version": version, "latest": latest, "up_to_date": up_to_date}


def update_cli(runner: Optional[Runner] = None, client: Optional[ColabClient] = None) -> dict:
    """Cập nhật google-colab-cli: ``uv tool upgrade``, không có uv thì ``colab update --install``.

    Returns:
        Kết quả ``cli_versions`` sau khi cập nhật.

    Raises:
        ColabError: Nếu lệnh cập nhật lỗi.
    """
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if Path(uv).is_file():
        command = [uv, "tool", "upgrade", CLI_PACKAGE]
    else:
        command = ["colab", "update", "--install"]
    result = (runner or _run)(command, 600)
    if result.returncode != 0:
        detail = ((result.stderr or "") + (result.stdout or "")).strip()[-400:]
        raise ColabError(f"cập nhật google-colab-cli lỗi: {detail}")
    return cli_versions(client)
