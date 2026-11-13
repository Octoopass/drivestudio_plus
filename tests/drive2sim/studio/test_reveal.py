"""Kiểm tra ``drive2sim/studio/api/reveal.py``: chỉ mở đường dẫn trong thư mục làm việc.

Lệnh mở trình quản lý file được thay bằng hàm ghi lại câu lệnh, nên test không mở cửa sổ nào.
"""

from pathlib import Path

import pytest

from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api import reveal


def test_reveal_opens_folder_inside_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        reveal.subprocess, "Popen", lambda command, **kwargs: calls.append(command)
    )
    monkeypatch.setattr(reveal.platform, "system", lambda: "Linux")
    monkeypatch.setattr(reveal.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("DISPLAY", ":0")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "a.txt").write_text("x")
    work = Workdir(tmp_path)
    assert reveal.reveal(work, "runs/a.txt") == (tmp_path / "runs" / "a.txt").resolve()
    assert calls == [
        ["/usr/bin/xdg-open", str((tmp_path / "runs").resolve())]
    ]  # file: mở thư mục chứa


def test_reveal_rejects_bad_paths_and_headless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reveal.subprocess, "Popen", lambda *a, **k: pytest.fail("không được mở"))
    work = Workdir(tmp_path)
    with pytest.raises(reveal.RevealError, match="ngoài thư mục làm việc"):
        reveal.reveal(work, "../")
    with pytest.raises(reveal.RevealError, match="không có"):
        reveal.reveal(work, "runs/missing")
    monkeypatch.setattr(reveal.platform, "system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(reveal.RevealError, match="giao diện đồ hoạ"):
        reveal.reveal(work, ".")
