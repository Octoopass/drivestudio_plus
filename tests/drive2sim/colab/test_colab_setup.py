"""Kiểm tra ``drive2sim/colab/setup.py``: trạng thái Colab cho màn hình thiết lập.

Lệnh ``colab`` được thay bằng hàm giả (không gọi mạng), git bằng hàm trả commit cố định.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from drive2sim import paths
from drive2sim.colab import setup
from drive2sim.colab.client import ColabClient, ColabError


class Done:
    """Kết quả giả của một lệnh ``colab``."""

    def __init__(self, code: int, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = code, out, err


def fake_cli(answers: dict[str, Done]):  # noqa: ANN201
    """Lệnh ``colab`` giả: trả theo lệnh con, lệnh khác báo lỗi."""

    def runner(command: Sequence[str], timeout_s: float) -> Done:
        return answers.get(command[1], Done(1, "", "lỗi"))

    return runner


def test_check_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "code_source", lambda: ("a" * 40, "https://example.invalid/r"))
    monkeypatch.setattr(paths, "find_smpl_model", lambda root=None: tmp_path / "SMPL.pkl")
    client = ColabClient(
        runner=fake_cli(
            {
                "version": Done(0, "Version: 0.7.5\n"),
                "sessions": Done(0, "[d2s] e | Hardware: T4 | Shape: Standard | Variant: GPU\n"),
                "usage": Done(0, "Current balance: 3.00 compute units\nUsage rate: 0.00/hr\n"),
            }
        )  # type: ignore[arg-type]
    )
    status = setup.check(client)
    assert status["ready"] and status["cli"] == {"installed": True, "version": "0.7.5"}
    assert status["gpu"]["sessions"] == [{"name": "d2s", "hardware": "T4"}]
    assert status["gpu"]["balance"] == 3.0 and "T4" in status["gpu"]["choices"]
    assert status["smpl"]["found"] and status["code"]["commit"] == "aaaaaaa"


def test_check_reports_every_missing_piece(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_push() -> tuple[str, str]:
        raise ColabError("commit abc chưa có trên remote origin")

    monkeypatch.setattr(setup, "code_source", no_push)
    monkeypatch.setattr(paths, "find_smpl_model", lambda root=None: None)
    monkeypatch.delenv("PYTHON_KEYRING_BACKEND", raising=False)
    client = ColabClient(
        runner=fake_cli(
            {
                "version": Done(0, "Version: 0.7.5\n"),
                "sessions": Done(1, "", "keyring.errors.NoKeyringError: No recommended backend"),
            }
        )  # type: ignore[arg-type]
    )
    status = setup.check(client)
    assert not status["ready"] and not status["account"]["ok"]
    assert "keyrings.alt" in status["account"]["hint"]
    assert not status["code"]["ok"] and "chưa có trên remote" in status["code"]["error"]
    assert status["smpl"] == {"found": False, "path": None}
    missing = setup.check(ColabClient(runner=fake_cli({})))  # type: ignore[arg-type]
    assert not missing["cli"]["installed"] and "uv tool install" in missing["account"]["error"]
