"""Fixture dùng chung cho mọi test của drive2sim."""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from drive2sim.colab import client as colab_client
from drive2sim.kaggle import client


def _blocked_cli(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    """Thay lệnh ``kaggle`` thật: luôn báo lỗi, không gọi mạng."""
    return subprocess.CompletedProcess(list(command), 1, "", "test không được gọi Kaggle thật")


@pytest.fixture(autouse=True)
def no_real_kaggle(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chặn mọi lần gọi Kaggle thật trong test.

    Máy phát triển thường đã có khoá API, nên một test lỡ đi tới executor Kaggle sẽ upload dataset
    và chạy notebook GPU thật trên tài khoản người chạy test. Test cần Kaggle thì truyền runner giả
    vào ``KaggleClient`` hoặc tự patch ``_run_cli``.
    """
    monkeypatch.setattr(client, "_run_cli", _blocked_cli)
    config: Path = tmp_path_factory.mktemp("kaggle_config")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)


def _blocked_colab(command: Sequence[str], timeout_s: float) -> "subprocess.CompletedProcess[str]":
    """Thay lệnh ``colab`` thật: luôn báo lỗi, không gọi mạng."""
    return subprocess.CompletedProcess(list(command), 1, "", "test không được gọi Colab thật")


@pytest.fixture(autouse=True)
def no_real_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chặn mọi lần gọi Colab thật trong test.

    Máy phát triển thường đã đăng nhập ``colab``, nên một test lỡ đi tới executor Colab sẽ tạo
    VM GPU thật (tốn compute unit) trên tài khoản người chạy test. Test cần Colab thì truyền
    client giả hoặc runner giả vào ``ColabClient``.
    """
    monkeypatch.setattr(colab_client, "_run_cli", _blocked_colab)


@pytest.fixture(autouse=True)
def no_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coi máy chạy test như máy không có giao diện đồ hoạ.

    Trên máy phát triển, test lỡ gọi ``/api/reveal`` hay ``/api/pick-folder`` sẽ mở trình quản lý
    file hoặc hộp thoại thật và chờ người bấm. Test cần giao diện thì tự đặt ``DISPLAY`` và patch
    lệnh chạy.
    """
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
