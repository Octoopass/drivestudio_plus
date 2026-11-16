"""Tắt tiến trình GPU của trang Xem 3D trước khi train trên máy (engine/gpu_process.py)."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from drive2sim.engine import gpu_process
from drive2sim.engine.workdir import Workdir

pytestmark = pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="cần /proc")

MARKER = "d2s_fake_frame_server"


def spawn() -> subprocess.Popen:
    """Tiến trình giả, dòng lệnh chứa MARKER, nhóm tiến trình riêng như GpuViewers bật."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep(60)  # {MARKER}"],
        start_new_session=True,
    )


def test_stop_recorded_kills_matching_process(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    process = spawn()
    try:
        gpu_process.record(workdir, process.pid, MARKER, "20261116-000000-omnire-abcd")
        stopped = gpu_process.stop_recorded(workdir)
        assert stopped is not None and MARKER in stopped
        assert process.wait(timeout=5) is not None
        assert gpu_process.stop_recorded(workdir) is None  # bản ghi đã xoá
    finally:
        process.kill()


def test_stop_recorded_leaves_other_process_alone(tmp_path: Path) -> None:
    """PID đã được cấp cho tiến trình khác (dòng lệnh không có module đã ghi): không tắt."""
    workdir = Workdir(tmp_path)
    process = spawn()
    try:
        gpu_process.record(workdir, process.pid, "drive2sim.recon.tools.viewer", "r")
        assert gpu_process.stop_recorded(workdir) is None
        time.sleep(0.2)
        assert process.poll() is None
    finally:
        process.kill()


def test_nothing_recorded(tmp_path: Path) -> None:
    assert gpu_process.stop_recorded(Workdir(tmp_path)) is None


def test_clear_only_own_pid(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    gpu_process.record(workdir, 123, MARKER, "r")
    gpu_process.clear(workdir, 456)  # tiến trình khác: giữ bản ghi
    assert (tmp_path / gpu_process.STATE_DIR / gpu_process.STATE_FILE).is_file()
    gpu_process.clear(workdir, 123)
    assert not (tmp_path / gpu_process.STATE_DIR / gpu_process.STATE_FILE).exists()
