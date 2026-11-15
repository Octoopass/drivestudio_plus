"""Kiểm tra ``drive2sim/core/subprocs.py``: lệnh đang chạy dừng được khi tắt Studio."""

import sys
import threading
import time

from drive2sim.core import subprocs


def test_run_returns_output() -> None:
    result = subprocs.run([sys.executable, "-c", "print('ok')"], timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == "ok"


def test_terminate_all_stops_running_command() -> None:
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            subprocs.run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=120)
        )
    )
    start = time.monotonic()
    worker.start()
    while not subprocs._running and time.monotonic() - start < 10:
        time.sleep(0.01)
    subprocs.terminate_all()
    worker.join(10)
    assert not worker.is_alive()
    assert results[0].returncode != 0
    assert time.monotonic() - start < 10
