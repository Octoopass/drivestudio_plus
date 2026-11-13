"""Kiểm tra ``drive2sim/kaggle/setup.py``: kiểm tra trạng thái Kaggle và lưu khoá API.

Lệnh ``kaggle`` được thay bằng hàm giả (không gọi mạng); thư mục cấu hình là thư mục tạm, nên
test không đọc hay ghi khoá thật của máy.
"""

import json
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from drive2sim.kaggle import setup
from drive2sim.kaggle.client import KaggleClient

QUOTA = json.dumps(
    [
        {"resource": "GPU", "used": "7.61h", "remaining": "22.39h", "total": "30.00h",
         "refreshAt": "2026-11-14T00:00:00+00:00"},
        {"resource": "TPU", "used": "0.00h", "remaining": "20.00h", "total": "20.00h",
         "refreshAt": "2026-11-14T00:00:00+00:00"},
    ]
)  # fmt: skip


def fake_cli(responses: dict[str, tuple[int, str]]):  # noqa: ANN201
    """Lệnh ``kaggle`` giả: trả theo lệnh con đầu tiên, ví dụ ``quota``."""

    def run(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
        key = command[1]
        code, out = responses.get(key, (1, ""))
        return subprocess.CompletedProcess(list(command), code, out, "" if code == 0 else out)

    return run


def test_check_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "access_token").write_text("KGAT_test\n")
    client = KaggleClient(
        fake_cli(
            {
                "--version": (0, "Kaggle CLI 2.2.5"),
                "config": (0, "- username: demo\n"),
                "quota": (0, QUOTA),
            }
        ),
        config_dir=tmp_path,
    )
    monkeypatch.setattr(setup, "code_source", lambda: ("a" * 40, "https://example.invalid/r.git"))
    result = setup.check(client)
    assert result["cli"] == {"installed": True, "version": "2.2.5"}
    assert result["credentials"]["source"] == "access_token"
    assert result["account"] == {"ok": True, "username": "demo", "error": None}
    assert result["gpu"]["remaining_h"] == 22.39 and result["gpu"]["total_h"] == 30.0
    assert result["code"]["commit"] == "aaaaaaa" and result["ready"] is True


def test_check_reports_every_missing_piece(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_push() -> tuple[str, str]:
        raise setup.KaggleError("commit abc chưa có trên remote")

    monkeypatch.setattr(setup, "code_source", no_push)
    installed = KaggleClient(fake_cli({"--version": (0, "Kaggle CLI 2.2.5")}), config_dir=tmp_path)
    result = setup.check(installed)
    assert result["ready"] is False and result["credentials"]["source"] is None
    assert result["account"]["error"] == "chưa có khoá API"
    assert result["code"]["ok"] is False
    assert result["code"]["error"] == "commit abc chưa có trên remote"
    missing = KaggleClient(fake_cli({}), config_dir=tmp_path)
    assert setup.check(missing)["cli"]["installed"] is False


def test_save_new_token(tmp_path: Path) -> None:
    saved = setup.save_token("  KGAT_abcdef123456  ", config_dir=tmp_path / "kaggle")
    path = Path(saved["path"])
    assert saved["kind"] == "access_token" and path.read_text() == "KGAT_abcdef123456\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(setup.CredentialExistsError):
        setup.save_token("KGAT_other1234567", config_dir=tmp_path / "kaggle")
    setup.save_token("KGAT_other1234567", replace=True, config_dir=tmp_path / "kaggle")
    assert path.read_text() == "KGAT_other1234567\n"


def test_save_legacy_key_and_reject_bad_input(tmp_path: Path) -> None:
    key = "0123456789abcdef0123456789abcdef"
    with pytest.raises(ValueError, match="tên người dùng"):
        setup.save_token(key, config_dir=tmp_path)
    saved = setup.save_token(key, username="demo", config_dir=tmp_path)
    assert json.loads(Path(saved["path"]).read_text()) == {"username": "demo", "key": key}
    with pytest.raises(ValueError) as error:
        setup.save_token("not-a-token-secret-value", config_dir=tmp_path)
    assert "secret" not in str(error.value)  # thông báo lỗi không chứa khoá
