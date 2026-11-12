"""Gọi Kaggle qua lệnh ``kaggle`` (gói ``kaggle`` trên PyPI).

Tool không import thư viện Kaggle mà chạy lệnh ``kaggle`` trong tiến trình con, nên máy không
dùng Kaggle không cần cài gì thêm, và test thay được toàn bộ Kaggle bằng một hàm giả.

Cần: ``pip install kaggle`` và khoá API (trang Settings của Kaggle, mục API): token mới
(``~/.kaggle/access_token``) hoặc ``~/.kaggle/kaggle.json`` kiểu cũ, hoặc biến môi trường
``KAGGLE_USERNAME``, ``KAGGLE_KEY``. Cờ của các lệnh đã kiểm với Kaggle CLI 2.2.5.

File này chứa:
- ``KaggleError``: lỗi khi gọi Kaggle (chưa cài, chưa đăng nhập, lệnh lỗi).
- ``KaggleClient``: các thao tác tool cần: tên người dùng, trạng thái và tạo dataset, đẩy
  notebook, trạng thái notebook, tải output của notebook.
"""

import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


class KaggleError(RuntimeError):
    """Lỗi khi gọi Kaggle: chưa cài lệnh ``kaggle``, chưa có khoá API, hoặc lệnh báo lỗi."""


def _run_cli(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    """Chạy một lệnh ``kaggle``; không ném lỗi khi lệnh thoát khác 0 (``KaggleClient`` xử lý)."""
    try:
        return subprocess.run(list(command), capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        raise KaggleError("chưa có lệnh kaggle; cài bằng: pip install kaggle") from None


class KaggleClient:
    """Các thao tác với Kaggle mà executor cần.

    Args:
        runner: Hàm chạy một lệnh và trả ``CompletedProcess``; mặc định ``_run_cli``. Test
            truyền hàm giả.
        config_dir: Thư mục chứa ``kaggle.json``; mặc định ``~/.kaggle``.
    """

    def __init__(self, runner: Optional[Runner] = None, config_dir: Optional[Path] = None) -> None:
        self._runner = runner or _run_cli
        self._config_dir = config_dir or Path(
            os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")
        )

    def _call(self, *args: str) -> str:
        """Chạy ``kaggle <args>``.

        Returns:
            stdout của lệnh.

        Raises:
            KaggleError: Nếu lệnh thoát khác 0; thông báo kèm stderr.
        """
        result = self._runner(["kaggle", *args])
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip().splitlines()
            raise KaggleError(f"kaggle {args[0]} {args[1]} lỗi: {' '.join(message[-3:])}")
        return result.stdout

    def username(self) -> str:
        """Tên người dùng Kaggle.

        Thứ tự: biến ``KAGGLE_USERNAME``; ``kaggle.json`` kiểu cũ; dòng ``- username: ...`` của
        ``kaggle config view`` (cách duy nhất với token mới ``access_token``).

        Raises:
            KaggleError: Nếu chưa có khoá API.
        """
        if os.environ.get("KAGGLE_USERNAME"):
            return os.environ["KAGGLE_USERNAME"]
        path = self._config_dir / "kaggle.json"
        try:
            return json.loads(path.read_text("utf-8"))["username"]
        except (OSError, ValueError, KeyError):
            pass
        result = self._runner(["kaggle", "config", "view"])
        match = re.search(r"^- username: (\S+)", result.stdout or "", re.MULTILINE)
        if result.returncode == 0 and match and match.group(1) != "None":
            return match.group(1)
        raise KaggleError(
            "chưa có khoá API Kaggle; tạo token ở trang Settings của Kaggle, mục API, rồi làm "
            "theo hướng dẫn ở đó"
        )

    def dataset_exists(self, slug: str) -> bool:
        """Kiểm tra dataset ``<người dùng>/<tên>`` đã có trên Kaggle chưa.

        Args:
            slug: Định danh đầy đủ, ví dụ ``alice/d2s-1a2b3c4d``.

        Returns:
            ``True`` nếu có (kể cả đang xử lý).
        """
        result = self._runner(["kaggle", "datasets", "status", slug])
        return result.returncode == 0 and bool(result.stdout.strip())

    def create_dataset(self, folder: Path) -> None:
        """Tạo dataset private từ một thư mục có ``dataset-metadata.json``.

        Thư mục con được nén zip khi upload và Kaggle giải nén lại khi gắn vào notebook.

        Args:
            folder: Thư mục cần upload.
        """
        self._call("datasets", "create", "-p", str(folder), "--dir-mode", "zip")

    def push_kernel(self, folder: Path) -> None:
        """Đẩy notebook trong thư mục có ``kernel-metadata.json``; Kaggle chạy nó ngay.

        Args:
            folder: Thư mục chứa notebook và ``kernel-metadata.json``.
        """
        self._call("kernels", "push", "-p", str(folder))

    def kernel_status(self, slug: str) -> str:
        """Trạng thái của lần chạy mới nhất của notebook.

        Args:
            slug: Định danh đầy đủ của notebook.

        Kaggle in ``<notebook> has status "<trạng thái>"``; trạng thái có thể kèm tên enum
        (``KernelWorkerStatus.RUNNING``), nên chỉ lấy phần cuối và viết thường.

        Returns:
            Ví dụ ``queued``, ``running``, ``complete``, ``error``, ``cancel_acknowledged``; dòng
            gốc nếu không đọc ra được.
        """
        text = self._call("kernels", "status", slug).strip()
        match = re.search(r'has status "(?:\w+\.)?(\w+)"', text)
        return match.group(1).lower() if match else text

    def kernel_output(self, slug: str, destination: Path) -> None:
        """Tải output của notebook về ``destination``.

        Args:
            slug: Định danh đầy đủ của notebook.
            destination: Thư mục nhận file.
        """
        destination.mkdir(parents=True, exist_ok=True)
        self._call("kernels", "output", slug, "-p", str(destination))
