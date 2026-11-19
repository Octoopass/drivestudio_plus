"""Colab CLI version check and Google account switching (fake CLI + fake account helper)."""
import sys

import pytest

import fisheye_fusion.colab as colab_mod
from fisheye_fusion.colab import ColabHelper

FAKE_COLAB = r'''#!__PY__
import sys
cmd = sys.argv[1]
if cmd == "version":
    print("Version: 0.8.0")
elif cmd == "update":
    print("[colab] Colab CLI is up to date (version: 0.8.0, latest: 0.8.0).")
'''

FAKE_ACCOUNT = r'''
import json, os, sys
store = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.json")
data = json.load(open(store)) if os.path.exists(store) else {"accounts": ["a@x.com"], "active": "a@x.com"}
cmd = sys.argv[1]
def out(d): print("FFLAB_JSON:" + json.dumps(d), flush=True)
if cmd == "list":
    out(data)
elif cmd == "use":
    if sys.argv[2] not in data["accounts"]:
        sys.exit("Chưa lưu tài khoản")
    data["active"] = sys.argv[2]; json.dump(data, open(store, "w")); out({"email": sys.argv[2]})
elif cmd == "forget":
    data["accounts"].remove(sys.argv[2]); json.dump(data, open(store, "w")); out(data)
elif cmd == "login":
    print("To authorize colab-cli, visit this URL in any browser:\n\n  https://accounts.google.com/o/oauth2/auth?client_id=x\n", flush=True)
    code = sys.stdin.readline().strip()
    if code != "4/0AbCdEfGhIjKlMn":
        sys.exit("invalid_grant")
    data["accounts"].append("b@y.com"); data["active"] = "b@y.com"; json.dump(data, open(store, "w"))
    out({"email": "b@y.com"})
'''


@pytest.fixture
def helper(tmp_path, monkeypatch):
    fake = tmp_path / "colab"
    fake.write_text(FAKE_COLAB.replace("__PY__", sys.executable))
    fake.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "colab_account.py").write_text(FAKE_ACCOUNT)
    monkeypatch.setattr(colab_mod, "TOOLS", tools)
    h = ColabHelper(lambda: str(fake))
    yield h
    h.shutdown()


def test_cli_version(helper):
    info = helper.cli_info()
    assert info["version"] == "0.8.0" and info["latest"] == "0.8.0" and info["up_to_date"]


def test_login_flow_adds_and_activates_account(helper):
    assert helper.accounts()["active"] == "a@x.com"
    view = helper.login_start()
    assert view["state"] == "waiting_code" and view["url"].startswith("https://accounts.google.com/")
    with pytest.raises(ValueError):
        helper.login_code("short")
    data = helper.login_code("4/0AbCdEfGhIjKlMn")
    assert data["active"] == "b@y.com" and data["accounts"] == ["a@x.com", "b@y.com"]
    assert data["login"]["state"] == "succeeded"
    assert helper.use_account("a@x.com")["active"] == "a@x.com"
    assert helper.forget_account("b@y.com")["accounts"] == ["a@x.com"]
    with pytest.raises(ValueError):
        helper.use_account("not an email")


def test_wrong_code_reports_failure(helper):
    helper.login_start()
    data = helper.login_code("4/0WrongCodeXYZ")
    assert data["login"]["state"] == "failed" and data["active"] == "a@x.com"
