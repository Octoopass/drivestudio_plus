"""Kiểm tra ``drive2sim/studio/api/picker.py``: câu lệnh mở hộp thoại và cách đọc kết quả.

``subprocess.run`` được thay bằng hàm giả, nên test không mở hộp thoại nào.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pytest

from drive2sim.studio.api import picker


def fake_run(
    stdout: str, returncode: int = 0, calls: Optional[list] = None
) -> Callable[..., "subprocess.CompletedProcess[str]"]:
    """Hàm thay ``subprocess.run``: ghi lại câu lệnh, trả kết quả cho trước."""

    def run(command: list[str], **kwargs: object) -> "subprocess.CompletedProcess[str]":
        if calls is not None:
            calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    return run


def test_pick_folder_with_zenity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(picker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(picker.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(picker.subprocess, "run", fake_run(f"{tmp_path}\n", calls=calls))
    assert picker.pick_folder(str(tmp_path)) == str(tmp_path.resolve())
    assert calls[0][:3] == ["zenity", "--file-selection", "--directory"]
    assert calls[0][-1] == f"--filename={tmp_path}/"
    # Đường dẫn bắt đầu không có thật thì bỏ qua.
    calls.clear()
    picker.pick_folder(str(tmp_path / "missing"))
    assert not any(arg.startswith("--filename") for arg in calls[0])


def test_pick_folder_cancel_and_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(picker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(picker.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(picker.subprocess, "run", fake_run("", returncode=1))
    assert picker.pick_folder() is None  # bấm Huỷ
    monkeypatch.delenv("DISPLAY")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(picker.PickerError, match="giao diện đồ hoạ"):
        picker.pick_folder()


def test_pick_folder_escapes_start_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / 'a"b'
    folder.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(picker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(picker.subprocess, "run", fake_run("", calls=calls))
    picker.pick_folder(str(folder))
    assert 'a\\"b' in calls[0][2]
