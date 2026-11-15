"""Thiết lập Kaggle từ Studio: kiểm tra mọi thứ cần để chạy run trên Kaggle, và lưu khoá API.

Kiểm tra gồm: lệnh ``kaggle`` đã cài chưa, khoá API lấy từ đâu, đăng nhập được không (gọi
``kaggle quota``, chỉ đọc), còn bao nhiêu giờ GPU trong tuần, và commit hiện tại đã push chưa
(notebook ``git clone`` code từ remote).

Khoá được ghi vào đúng chỗ Kaggle CLI đọc (``drive2sim/kaggle/client.py``), quyền ``600``, để
lệnh ``kaggle`` và mọi công cụ khác dùng chung: token mới (``KGAT_...``) vào
``~/.kaggle/access_token``, khoá kiểu cũ kèm tên người dùng vào ``kaggle.json`` của thư mục cấu
hình. Khoá không bao giờ được trả lại hay ghi vào log; đã có khoá thì chỉ ghi đè khi người dùng
đồng ý.

File này chứa:
- ``check``: kiểm tra và trả về trạng thái để Studio hiển thị.
- ``save_token``: kiểm tra dạng khoá rồi ghi vào thư mục cấu hình của Kaggle.
- ``CredentialExistsError``: đã có khoá mà chưa đồng ý ghi đè.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from drive2sim.kaggle.client import KaggleClient, KaggleError
from drive2sim.kaggle.executor import _git, code_source

NEW_TOKEN_RE = re.compile(r"^KGAT_[A-Za-z0-9_-]{8,256}$")
LEGACY_KEY_RE = re.compile(r"^[0-9a-f]{32}$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class CredentialExistsError(KaggleError):
    """Đã có khoá API trong thư mục cấu hình mà người dùng chưa đồng ý ghi đè."""


def check(client: Optional[KaggleClient] = None) -> dict:
    """Kiểm tra Kaggle đã sẵn sàng để chạy run chưa.

    Mỗi bước độc lập: bước trước lỗi thì các bước sau vẫn chạy được nếu có thể, để người dùng
    thấy hết những gì còn thiếu một lần.

    Args:
        client: Client Kaggle; mặc định dùng lệnh ``kaggle`` thật.

    Returns:
        ``{"ready", "cli": {"installed", "version"}, "credentials": {"source", "active", "warning",
        "config_dir"},
        "account": {"ok", "username", "error"}, "gpu": {...} | None, "code": {"ok", "commit",
        "error", "dirty"}}``; ``dirty`` là có thay đổi chưa commit (notebook sẽ không có).
        ``ready`` đúng khi đã cài lệnh, đăng nhập được, còn giờ GPU và code đã push.
        ``source`` là nguồn khoá đứng đầu theo thứ tự Kaggle CLI đọc, ``active`` là nguồn CLI thật
        sự dùng; khác nhau (token sai hoặc đã thu hồi bị CLI bỏ qua) thì ``warning`` nói rõ, vì khi
        đó run đẩy bằng một tài khoản khác với tài khoản người dùng nghĩ.
    """
    client = client or KaggleClient()
    version = client.version()
    source = client.credential_source()
    account: dict[str, object] = {"ok": False, "username": None, "error": None}
    active, warning = None, None
    gpu = None
    if version is None:
        account["error"] = "chưa cài lệnh kaggle; cài bằng: pip install kaggle"
    elif source is None:
        account["error"] = "chưa có khoá API"
    else:
        try:
            account["username"], active = client.login()
            if account["username"] is None:
                raise KaggleError("khoá API không dùng được (sai hoặc đã thu hồi)")
            expected = source
            if source == "env":
                expected = "access_token" if os.environ.get("KAGGLE_API_TOKEN") else "kaggle.json"
            if active is not None and active != expected:
                warning = (
                    f"khoá ở nguồn {source} không dùng được (sai hoặc đã thu hồi), nên Kaggle CLI "
                    f"bỏ qua nó và đang đăng nhập bằng {active}, tài khoản {account['username']}"
                )
            quota = client.quota()
            account["ok"] = True
            gpu = next((row for row in quota if row["resource"].upper() == "GPU"), None)
        except KaggleError as error:
            account["error"] = str(error)
    try:
        commit, _ = code_source()
        code = {"ok": True, "commit": commit[:7], "error": None}
    except KaggleError as error:
        code = {"ok": False, "commit": None, "error": str(error)}
    # Notebook clone đúng commit này: thay đổi chưa commit trên máy sẽ không có trên Kaggle.
    code["dirty"] = bool(_git("status", "--porcelain", "--untracked-files=no"))
    has_gpu = gpu is not None and (gpu.get("remaining_h") or 0) > 0
    return {
        "ready": bool(account["ok"] and has_gpu and code["ok"]),
        "cli": {"installed": version is not None, "version": version},
        "credentials": {
            "source": source,
            "active": active,
            "warning": warning,
            "config_dir": str(client.config_dir),
        },
        "account": account,
        "gpu": gpu,
        "code": code,
    }


def _write_private(path: Path, text: str) -> None:
    """Ghi file chỉ chủ sở hữu đọc được (quyền ``600``), thay file cũ trong một bước."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def save_token(
    token: str,
    username: Optional[str] = None,
    replace: bool = False,
    config_dir: Optional[Path] = None,
    token_dir: Optional[Path] = None,
) -> dict:
    """Kiểm tra dạng khoá API rồi ghi vào đúng chỗ Kaggle CLI đọc.

    Args:
        token: Token mới (``KGAT_...``) hoặc khoá kiểu cũ (32 ký tự hex).
        username: Tên người dùng Kaggle; chỉ cần với khoá kiểu cũ.
        replace: Cho phép ghi đè file khoá đã có.
        config_dir: Thư mục ghi ``kaggle.json`` (khoá kiểu cũ); mặc định như ``KaggleClient``.
        token_dir: Thư mục ghi ``access_token`` (token mới); mặc định ``~/.kaggle``.

    Returns:
        ``{"kind": "access_token" | "kaggle.json", "path": <file đã ghi>, "note": str | None}``;
        không chứa khoá. ``note`` báo khi có nguồn khoá khác mà Kaggle CLI đọc trước file vừa ghi.

    Raises:
        ValueError: Nếu khoá hoặc tên người dùng sai dạng (thông báo không chứa khoá).
        CredentialExistsError: Nếu file khoá đã có mà ``replace`` là ``False``.
    """
    token = token.strip()
    if NEW_TOKEN_RE.match(token):
        folder = token_dir or KaggleClient().token_dir
        path, text, kind = folder / "access_token", token + "\n", "access_token"
    elif LEGACY_KEY_RE.match(token):
        if not username or not USERNAME_RE.match(username.strip()):
            raise ValueError("khoá kiểu cũ (32 ký tự) cần kèm tên người dùng Kaggle")
        text = json.dumps({"username": username.strip(), "key": token}) + "\n"
        folder = config_dir or KaggleClient().config_dir
        path, kind = folder / "kaggle.json", "kaggle.json"
    else:
        raise ValueError(
            "khoá không đúng dạng: token mới bắt đầu bằng KGAT_, khoá cũ gồm 32 ký tự 0-9 a-f"
        )
    if path.exists() and not replace:
        raise CredentialExistsError(f"đã có khoá ở {path}; chọn thay khoá cũ để ghi đè")
    _write_private(path, text)
    # Thứ tự Kaggle CLI đọc (client.py, đầu file): biến KAGGLE_API_TOKEN đứng trước mọi file, token
    # mới đứng trước khoá kiểu cũ.
    ahead = []
    if os.environ.get("KAGGLE_API_TOKEN"):
        ahead.append("biến KAGGLE_API_TOKEN")
    if (
        kind == "kaggle.json"
        and (KaggleClient(token_dir=token_dir).token_dir / "access_token").is_file()
    ):
        ahead.append("token mới ~/.kaggle/access_token")
    note = None
    if ahead:
        note = (
            f"Kaggle CLI dùng {' và '.join(ahead)} trước khoá vừa lưu; xoá đi nếu muốn dùng "
            "khoá này"
        )
    return {"kind": kind, "path": str(path), "note": note}
