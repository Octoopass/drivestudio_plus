"""Test tài khoản Google và phiên bản của lệnh ``colab`` (``drive2sim/colab/accounts.py``).

``account_helper.py`` thật cần google-colab-cli, nên test thay nó bằng một script giả cùng giao
thức (dòng ``D2S_JSON:``) chạy bằng Python của máy test; không gọi Google hay lệnh ``colab`` thật.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.colab import accounts
from drive2sim.colab.client import ColabClient, ColabError
from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.app import create_app

FAKE_HELPER = r"""
import json, os, sys
store = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.json")
first = {"accounts": ["a@x.com"], "active": "a@x.com"}
data = json.load(open(store)) if os.path.exists(store) else first
def out(d): print("D2S_JSON:" + json.dumps(d), flush=True)
def save(): json.dump(data, open(store, "w"))
cmd = sys.argv[1]
if cmd == "list":
    out(data)
elif cmd == "use":
    if sys.argv[2] not in data["accounts"]:
        sys.exit("chưa lưu tài khoản")
    data["active"] = sys.argv[2]; save(); out({"email": sys.argv[2]})
elif cmd == "forget":
    data["accounts"].remove(sys.argv[2]); save(); out({"accounts": data["accounts"]})
elif cmd == "login":
    print("Visit:\n  https://accounts.google.com/o/oauth2/auth?client_id=x\n", flush=True)
    if sys.stdin.readline().strip() != "4/0AbCdEfGhIjKlMn":
        sys.exit("invalid_grant")
    data["accounts"].append("b@y.com"); data["active"] = "b@y.com"
    save(); out({"email": "b@y.com"})
"""


@pytest.fixture()
def helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "helper.py"
    script.write_text(FAKE_HELPER, encoding="utf-8")
    monkeypatch.setattr(accounts, "HELPER", script)
    monkeypatch.setattr(accounts, "cli_python", lambda binary="colab": Path(sys.executable))
    return script


def test_list_use_forget(helper: Path) -> None:
    assert accounts.list_accounts() == {"accounts": ["a@x.com"], "active": "a@x.com"}
    with pytest.raises(ValueError):
        accounts.use_account("không phải email")
    with pytest.raises(ColabError, match="chưa lưu"):
        accounts.use_account("z@z.com")


def test_login_flow_adds_account(helper: Path) -> None:
    flow = accounts.LoginFlow()
    assert flow.view()["state"] == "idle"
    view = flow.start()
    assert view["state"] == "waiting_code"
    assert view["url"].startswith("https://accounts.google.com/")
    with pytest.raises(ValueError):
        flow.submit_code("ngắn")
    done = flow.submit_code("4/0AbCdEfGhIjKlMn")
    assert done["state"] == "succeeded" and done["email"] == "b@y.com"
    assert accounts.list_accounts()["active"] == "b@y.com"
    assert accounts.forget_account("b@y.com")["accounts"] == ["a@x.com"]
    with pytest.raises(ValueError, match="không có lần đăng nhập"):
        flow.submit_code("4/0AbCdEfGhIjKlMn")


def test_wrong_code_fails(helper: Path) -> None:
    flow = accounts.LoginFlow()
    flow.start()
    failed = flow.submit_code("4/0WrongCodeXYZ")
    assert failed["state"] == "failed" and "invalid_grant" in failed["error"]
    assert flow.cancel()["state"] == "idle"


class Done:
    """Kết quả giả của một lệnh."""

    def __init__(self, code: int, out: str = "") -> None:
        self.returncode, self.stdout, self.stderr = code, out, ""


def test_cli_versions_and_update(tmp_path: Path) -> None:
    def colab(command: Sequence[str], timeout_s: float) -> Done:
        if command[1] == "version":
            return Done(0, "Version: 0.8.0\n")
        return Done(0, "[colab] Colab CLI is up to date (version: 0.8.0, latest: 0.9.1).\n")

    client = ColabClient(runner=colab)  # type: ignore[arg-type]
    info = accounts.cli_versions(client)
    assert info == {"version": "0.8.0", "latest": "0.9.1", "up_to_date": False}
    calls: list[list[str]] = []

    def upgrade(command: Sequence[str], timeout_s: float) -> Done:
        calls.append(list(command))
        return Done(0)

    assert accounts.update_cli(runner=upgrade, client=client)["version"] == "0.8.0"  # type: ignore[arg-type]
    assert calls and ("upgrade" in calls[0] or "update" in calls[0])
    with pytest.raises(ColabError, match="cập nhật"):
        accounts.update_cli(runner=lambda c, t: Done(1, "no network"), client=client)  # type: ignore[arg-type]


def test_routes_are_not_taken_for_run_ids(helper: Path, tmp_path: Path) -> None:
    """``/api/colab/accounts`` và ``/login`` không bị route ``/api/colab/{run_id}`` bắt mất."""
    app = create_app(Workdir(tmp_path / "work"), registry=StageRegistry(), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/colab/accounts").json()["active"] == "a@x.com"
        assert client.get("/api/colab/login").json()["state"] == "idle"
        started = client.post("/api/colab/login").json()
        assert started["state"] == "waiting_code" and started["url"]
        bad = client.post("/api/colab/login/code", json={"code": "x"})
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_request"
        done = client.post("/api/colab/login/code", json={"code": "4/0AbCdEfGhIjKlMn"}).json()
        assert done["state"] == "succeeded"
        assert client.post("/api/colab/accounts/a@x.com/use").json()["active"] == "a@x.com"
        missing = client.post("/api/colab/accounts/z@z.com/use")
        assert missing.status_code == 502 and missing.json()["error"]["code"] == "colab_failed"
        assert client.delete("/api/colab/accounts/b@y.com").json()["accounts"] == ["a@x.com"]


def test_cli_python_reads_shebang(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "colab"
    binary.write_text(f"#!{sys.executable}\nprint('x')\n", encoding="utf-8")
    assert accounts.cli_python(str(binary)) == Path(sys.executable)
    binary.write_text("#!/không/có/python\n", encoding="utf-8")
    with pytest.raises(ColabError):
        accounts.cli_python(str(binary))
