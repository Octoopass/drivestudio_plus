"""Tài khoản Google của lệnh ``colab``; chạy bằng Python của google-colab-cli (CONTRACTS 9.3).

File này **không** import ``drive2sim``: ``accounts.py`` chạy nó bằng trình thông dịch trong môi
trường của google-colab-cli (``uv tool``), vì cần mã đăng nhập của gói đó. Phần dùng tới là hàm nội
bộ của google-colab-cli 0.8.0 (``auth._run_remote_flow``, ``_get_token_from_keyring``,
``_save_token_to_keyring``); bản khác chưa kiểm tra.

Lệnh (kết quả in một dòng ``D2S_JSON:<json>``):
- ``whoami``: email của token đang dùng, ``null`` nếu chưa đăng nhập.
- ``list``: các tài khoản đã lưu (chỉ email) và tài khoản đang dùng.
- ``login``: in đường dẫn đăng nhập Google, đọc mã xác thực từ stdin, lưu token vào keyring của
  lệnh ``colab`` và vào danh sách tài khoản.
- ``use <email>``: chuyển sang một tài khoản đã lưu.
- ``forget <email>``: bỏ một tài khoản khỏi danh sách (token đang dùng giữ nguyên).

Danh sách tài khoản ở ``~/.config/drive2sim/colab_accounts.json`` (quyền ``600``).
"""

import json
import os
import sys
from importlib import resources
from pathlib import Path

os.environ.setdefault("PYTHON_KEYRING_BACKEND", "keyrings.alt.file.PlaintextKeyring")

from colab_cli import auth  # noqa: E402
from google.auth.transport.requests import AuthorizedSession, Request  # noqa: E402
from google.oauth2.credentials import Credentials  # noqa: E402

STORE = Path.home() / ".config" / "drive2sim" / "colab_accounts.json"
CLIENT_CONFIG = Path.home() / ".colab-cli-oauth-config.json"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def _out(data: dict) -> None:
    print("D2S_JSON:" + json.dumps(data), flush=True)


def _load() -> dict:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(STORE)


def _email_of(token_json: str) -> tuple:
    """Email của một token (làm mới token khi cần) và token sau khi làm mới."""
    creds = Credentials.from_authorized_user_info(json.loads(token_json), auth.PUBLIC_SCOPES)
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
    info = AuthorizedSession(creds).get(USERINFO_URL, timeout=20).json()
    if "email" not in info:
        raise SystemExit("không đọc được email của tài khoản: " + json.dumps(info)[:200])
    return info["email"], creds.to_json()


def _client_config() -> dict:
    if CLIENT_CONFIG.is_file():
        return json.loads(CLIENT_CONFIG.read_text(encoding="utf-8"))
    return json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())


def _remember(email: str, token_json: str) -> None:
    store = _load()
    store[email] = token_json
    _save(store)


def main() -> None:
    """Chạy một lệnh theo ``sys.argv``."""
    command = sys.argv[1] if len(sys.argv) > 1 else "whoami"
    if command in ("whoami", "list"):
        token = auth._get_token_from_keyring()
        active = None
        if token:
            try:
                active, token = _email_of(token)
                _remember(active, token)  # tài khoản đang dùng luôn có trong danh sách
            except Exception:  # noqa: BLE001 - token hết hạn nghĩa là chưa đăng nhập
                active = None
        if command == "whoami":
            _out({"email": active})
        else:
            _out({"accounts": sorted(_load()), "active": active})
    elif command == "login":
        creds = auth._run_remote_flow(_client_config())
        email, token = _email_of(creds.to_json())
        auth._save_token_to_keyring(token)
        _remember(email, token)
        _out({"email": email})
    elif command == "use":
        token = _load().get(sys.argv[2])
        if not token:
            raise SystemExit(f"chưa lưu tài khoản {sys.argv[2]}")
        email, token = _email_of(token)
        auth._save_token_to_keyring(token)
        _remember(email, token)
        _out({"email": email})
    elif command == "forget":
        store = _load()
        store.pop(sys.argv[2], None)
        _save(store)
        _out({"accounts": sorted(store)})
    else:
        raise SystemExit(f"lệnh không hợp lệ: {command}")


if __name__ == "__main__":
    main()
