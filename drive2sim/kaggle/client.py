"""Gọi Kaggle qua lệnh ``kaggle`` (gói ``kaggle`` trên PyPI).

Tool không import thư viện Kaggle mà chạy lệnh ``kaggle`` trong tiến trình con, nên máy không
dùng Kaggle không cần cài gì thêm, và test thay được toàn bộ Kaggle bằng một hàm giả.

Cần: ``pip install kaggle`` và khoá API (trang Settings của Kaggle, mục API). Kaggle CLI 2.2.5
(``kagglesdk/kaggle_env.py``, ``KaggleApi.authenticate``) tìm khoá theo thứ tự:

1. Token mới: biến ``KAGGLE_API_TOKEN`` (giá trị hoặc đường dẫn file), rồi
   ``~/.kaggle/access_token`` (luôn ở ``~/.kaggle``, không theo ``KAGGLE_CONFIG_DIR``). CLI chỉ
   lấy token đầu tiên tìm thấy; token đó sai hoặc đã thu hồi thì bị bỏ qua không báo và CLI xuống
   bước 2 (đã thử: ``KAGGLE_API_TOKEN`` sai thì file ``access_token`` cũng không được đọc).
2. Khoá kiểu cũ: biến ``KAGGLE_USERNAME`` và ``KAGGLE_KEY``, hoặc ``kaggle.json`` trong thư mục
   cấu hình (``KAGGLE_CONFIG_DIR``; không đặt thì ``~/.kaggle``, trên Linux khi chưa có thư mục đó
   thì ``$XDG_CONFIG_HOME/kaggle`` hoặc ``~/.config/kaggle``).
3. Đăng nhập OAuth (``kaggle auth login``): ``~/.kaggle/credentials.json``.

Cờ của các lệnh đã kiểm với Kaggle CLI 2.2.5.

File này chứa:
- ``KaggleError``: lỗi khi gọi Kaggle (chưa cài, chưa đăng nhập, lệnh lỗi).
- ``KaggleClient``: các thao tác tool cần: tên người dùng, trạng thái và tạo dataset, đẩy
  notebook, trạng thái notebook, tải output của notebook; phiên bản lệnh, nguồn khoá API và hạn
  mức GPU cho màn hình thiết lập Kaggle.
"""

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


# Giá trị ``auth_method`` mà ``kaggle config view`` in ra, đổi sang tên nguồn khoá của tool.
AUTH_METHODS = {"ACCESS_TOKEN": "access_token", "LEGACY_API_KEY": "kaggle.json", "OAUTH": "oauth"}


def default_config_dir() -> Path:
    """Thư mục cấu hình (chứa ``kaggle.json``) theo đúng cách Kaggle CLI chọn."""
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        return Path(os.environ["KAGGLE_CONFIG_DIR"])
    folder = Path.home() / ".kaggle"
    if sys.platform.startswith("linux") and not folder.exists():
        return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "kaggle"
    return folder


def default_token_dir() -> Path:
    """Thư mục Kaggle CLI đọc ``access_token`` và ``credentials.json``: luôn ``~/.kaggle``."""
    return Path.home() / ".kaggle"


class KaggleError(RuntimeError):
    """Lỗi khi gọi Kaggle: chưa cài lệnh ``kaggle``, chưa có khoá API, hoặc lệnh báo lỗi."""


# Một lệnh kaggle chạy lâu nhất bao lâu. Tải output của một run (vài GB, hàng nghìn file) qua mạng
# chậm mất hơn một giờ (đã gặp: 2,4 GB ở khoảng 0,7 MB/s), nên để 6 giờ.
CLI_TIMEOUT_S = 6 * 3600


def _run_cli(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    """Chạy một lệnh ``kaggle``; không ném lỗi khi lệnh thoát khác 0 (``KaggleClient`` xử lý)."""
    try:
        return subprocess.run(list(command), capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    except FileNotFoundError:
        raise KaggleError("chưa có lệnh kaggle; cài bằng: pip install kaggle") from None
    except subprocess.TimeoutExpired:
        raise KaggleError(
            f"lệnh {' '.join(command[:3])} chạy quá {CLI_TIMEOUT_S // 3600} giờ; chạy lại lệnh"
        ) from None


class KaggleClient:
    """Các thao tác với Kaggle mà executor cần.

    Args:
        runner: Hàm chạy một lệnh và trả ``CompletedProcess``; mặc định ``_run_cli``. Test
            truyền hàm giả.
        config_dir: Thư mục chứa ``kaggle.json``; mặc định như Kaggle CLI (``default_config_dir``).
        token_dir: Thư mục chứa ``access_token`` và ``credentials.json``; mặc định ``~/.kaggle``
            (Kaggle CLI cố định chỗ này).
    """

    def __init__(
        self,
        runner: Optional[Runner] = None,
        config_dir: Optional[Path] = None,
        token_dir: Optional[Path] = None,
    ) -> None:
        self._runner = runner or _run_cli
        self._config_dir = config_dir or default_config_dir()
        self._token_dir = token_dir or default_token_dir()

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

    def login(self) -> tuple[Optional[str], Optional[str]]:
        """Tài khoản và nguồn khoá Kaggle CLI thật sự dùng (``kaggle config view``).

        Khác ``credential_source`` (chỉ xem file, biến môi trường nào đang có): CLI kiểm token với
        Kaggle và bỏ qua token sai, nên chỉ lệnh này biết đang đăng nhập bằng gì.

        Returns:
            ``(tên người dùng, nguồn khoá)``; nguồn là ``access_token``, ``kaggle.json``, ``oauth``
            (``AUTH_METHODS``). ``(None, None)`` nếu chưa đăng nhập được.
        """
        try:
            result = self._runner(["kaggle", "config", "view"])
        except KaggleError:
            return None, None
        text = (result.stdout or "") if result.returncode == 0 else ""
        user = re.search(r"^- username: (\S+)", text, re.MULTILINE)
        method = re.search(r"^- auth_method: (\S+)", text, re.MULTILINE)
        username = user.group(1) if user and user.group(1) != "None" else None
        if username is None:
            return None, None
        return username, AUTH_METHODS.get(method.group(1)) if method else None

    def username(self) -> str:
        """Tên người dùng Kaggle mà lệnh ``kaggle`` đang đăng nhập (``login``).

        Raises:
            KaggleError: Nếu chưa đăng nhập được.
        """
        username, _ = self.login()
        if username is None:
            raise KaggleError(
                "chưa có khoá API Kaggle dùng được; tạo token ở trang Settings của Kaggle, "
                "mục API, rồi làm theo hướng dẫn ở đó"
            )
        return username

    def dataset_exists(self, slug: str) -> bool:
        """Kiểm tra dataset ``<người dùng>/<tên>`` đã có trên Kaggle chưa.

        Args:
            slug: Định danh đầy đủ, ví dụ ``alice/d2s-1a2b3c4d``.

        Returns:
            ``True`` nếu có (kể cả đang xử lý).
        """
        result = self._runner(["kaggle", "datasets", "status", slug])
        return result.returncode == 0 and bool(result.stdout.strip())

    def dataset_status(self, slug: str) -> str:
        """Trạng thái xử lý của dataset, viết thường, ví dụ ``ready``, ``pending``, ``error``.

        Args:
            slug: Định danh đầy đủ, ví dụ ``alice/d2s-1a2b3c4d``.

        Returns:
            Dòng trạng thái CLI in ra, viết thường.

        Raises:
            KaggleError: Nếu lệnh lỗi (ví dụ dataset không tồn tại).
        """
        return self._call("datasets", "status", slug).strip().lower()

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
        output = self._call("kernels", "push", "-p", str(folder))
        # Kaggle CLI 2.2.5 có lúc thoát 0 mà không tạo notebook (in lỗi ra stdout); chỉ tin khi nó
        # báo đã đẩy.
        if "successfully pushed" not in output.lower():
            said = " ".join(output.strip().splitlines()[-3:]) or "(không in gì)"
            raise KaggleError(f"kaggle kernels push không tạo được notebook: {said}")

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

    @property
    def config_dir(self) -> Path:
        """Thư mục chứa ``kaggle.json`` của Kaggle CLI (``default_config_dir``)."""
        return self._config_dir

    def version(self) -> Optional[str]:
        """Phiên bản của lệnh ``kaggle``, ví dụ ``2.2.5``; ``None`` nếu chưa cài hoặc lệnh lỗi."""
        try:
            result = self._runner(["kaggle", "--version"])
        except KaggleError:
            return None
        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", result.stdout or "")
        return match.group(1) if result.returncode == 0 and match else None

    @property
    def token_dir(self) -> Path:
        """Thư mục chứa ``access_token`` và ``credentials.json`` (``~/.kaggle``)."""
        return self._token_dir

    def credential_source(self) -> Optional[str]:
        """Nguồn khoá API đứng đầu theo thứ tự Kaggle CLI đọc (đầu file); ``None`` nếu chưa có.

        Chỉ xem file và biến môi trường nào đang có, không kiểm khoá còn dùng được không; nguồn
        CLI thật sự dùng là ``login``.

        Returns:
            ``env`` (biến môi trường), ``access_token`` (token mới), ``kaggle.json`` (kiểu cũ),
            ``oauth`` (``kaggle auth login``), hoặc ``None``.
        """
        if os.environ.get("KAGGLE_API_TOKEN"):
            return "env"
        if any(
            (self._token_dir / name).is_file() for name in ("access_token", "access_token.txt")
        ):
            return "access_token"
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return "env"
        if (self._config_dir / "kaggle.json").is_file():
            return "kaggle.json"
        if (self._token_dir / "credentials.json").is_file():
            return "oauth"
        return None

    def quota(self) -> list[dict[str, object]]:
        """Hạn mức GPU, TPU của tài khoản trong tuần (``kaggle quota``); cần đăng nhập được.

        Returns:
            Mỗi tài nguyên một mục ``{"resource", "used_h", "remaining_h", "total_h",
            "refresh_at"}``; số giờ là số thực.

        Raises:
            KaggleError: Nếu chưa đăng nhập được hoặc không đọc được kết quả.
        """
        text = self._call("quota", "--format", "json")
        try:
            rows = json.loads(text)
        except ValueError:
            raise KaggleError(
                f"kaggle quota trả về không đọc được: {text.strip()[:120]}"
            ) from None

        def hours(value: object) -> Optional[float]:
            match = re.match(r"^\s*([\d.]+)\s*h?\s*$", str(value))
            return float(match.group(1)) if match else None

        return [
            {
                "resource": str(row.get("resource", "")),
                "used_h": hours(row.get("used")),
                "remaining_h": hours(row.get("remaining")),
                "total_h": hours(row.get("total")),
                "refresh_at": row.get("refreshAt"),
            }
            for row in rows
            if isinstance(row, dict)
        ]

    def kernel_output(
        self, slug: str, destination: Path, file_pattern: Optional[str] = None
    ) -> None:
        """Tải output của notebook về ``destination``.

        Args:
            slug: Định danh đầy đủ của notebook.
            destination: Thư mục nhận file.
            file_pattern: Biểu thức chính quy; chỉ tải file có đường dẫn khớp (``--file-pattern``
                của Kaggle CLI), ví dụ chỉ các file nhỏ của một cảnh.
        """
        destination.mkdir(parents=True, exist_ok=True)
        extra = ["--file-pattern", file_pattern] if file_pattern else []
        self._call("kernels", "output", slug, "-p", str(destination), *extra)
