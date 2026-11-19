"""Google account behind the `colab` CLI (and so behind the Drive it mounts). Runs with the CLI's own python.

    whoami            -> FFLAB_JSON {"email": ...} for the token in the CLI keyring
    login             -> prints the Google sign-in URL, reads the authorization code from stdin,
                         stores the new token in the CLI keyring and in the saved-accounts file
    use EMAIL         -> make a saved account the active one
    list              -> saved accounts (emails only) and the active one
    forget EMAIL      -> drop a saved account (the active token is left alone)

Saved accounts live in ~/.config/fflab/colab_accounts.json (mode 600), next to the CLI's own keyring copy.
"""
from __future__ import annotations

import json
import os
import sys
from importlib import resources
from pathlib import Path

os.environ.setdefault("PYTHON_KEYRING_BACKEND", "keyrings.alt.file.PlaintextKeyring")

from colab_cli import auth  # noqa: E402
from google.auth.transport.requests import AuthorizedSession, Request  # noqa: E402
from google.oauth2.credentials import Credentials  # noqa: E402

STORE = Path.home() / ".config" / "fflab" / "colab_accounts.json"
CLIENT = Path.home() / ".colab-cli-oauth-config.json"


def out(data):
    print("FFLAB_JSON:" + json.dumps(data), flush=True)


def load_store() -> dict:
    try:
        return json.loads(STORE.read_text())
    except (OSError, ValueError):
        return {}


def save_store(data: dict):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.chmod(tmp, 0o600)
    tmp.replace(STORE)


def email_of(token_json: str) -> tuple[str, str]:
    """Email for a token (refreshing it when needed) and the possibly refreshed token JSON."""
    creds = Credentials.from_authorized_user_info(json.loads(token_json), auth.PUBLIC_SCOPES)
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
    info = AuthorizedSession(creds).get("https://openidconnect.googleapis.com/v1/userinfo", timeout=20).json()
    if "email" not in info:
        raise SystemExit("Không đọc được email của tài khoản: " + json.dumps(info)[:200])
    return info["email"], creds.to_json()


def client_config() -> dict:
    if CLIENT.is_file():
        return json.loads(CLIENT.read_text())
    return json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())


def active_token() -> str | None:
    return auth._get_token_from_keyring()


def remember(email: str, token_json: str):
    store = load_store()
    store[email] = token_json
    save_store(store)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "whoami"
    if cmd == "whoami":
        token = active_token()
        if not token:
            out({"email": None})
            return
        email, token = email_of(token)
        remember(email, token)          # keeps the current account in the switch list
        out({"email": email})
    elif cmd == "list":
        token = active_token()
        active = None
        if token:
            try:
                active = email_of(token)[0]
            except Exception:  # noqa: BLE001 - an expired token just means "not signed in"
                active = None
        out({"accounts": sorted(load_store()), "active": active})
    elif cmd == "login":
        creds = auth._run_remote_flow(client_config())
        token = creds.to_json()
        email, token = email_of(token)
        auth._save_token_to_keyring(token)
        remember(email, token)
        out({"email": email})
    elif cmd == "use":
        email = sys.argv[2]
        token = load_store().get(email)
        if not token:
            raise SystemExit(f"Chưa lưu tài khoản {email}")
        email2, token = email_of(token)
        auth._save_token_to_keyring(token)
        remember(email2, token)
        out({"email": email2})
    elif cmd == "forget":
        store = load_store()
        store.pop(sys.argv[2], None)
        save_store(store)
        out({"accounts": sorted(store)})
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
