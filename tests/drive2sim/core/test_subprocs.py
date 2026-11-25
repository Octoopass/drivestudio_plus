"""Kiểm tra ``drive2sim/core/subprocs.py``: lệnh đang chạy dừng được khi tắt Studio."""

import sys
import threading
import time

import pytest

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


def test_terminate_thread_stops_only_that_threads_command() -> None:
    results: dict[str, object] = {}

    def sleeper(key: str) -> None:
        try:
            results[key] = subprocs.run(
                [sys.executable, "-c", "import time; time.sleep(60)"], timeout=120
            )
        except subprocs.CancelledError as error:
            results[key] = error

    victim = threading.Thread(target=sleeper, args=("victim",))
    other = threading.Thread(target=sleeper, args=("other",))
    start = time.monotonic()
    victim.start()
    other.start()
    while len(subprocs._running) < 2 and time.monotonic() - start < 10:
        time.sleep(0.01)
    subprocs.terminate_thread(victim.ident)
    victim.join(10)
    assert not victim.is_alive() and isinstance(results["victim"], subprocs.CancelledError)
    assert other.is_alive()
    subprocs.terminate_all()
    other.join(10)
    assert not other.is_alive()


def test_cancelled_thread_refuses_new_commands_until_cleared() -> None:
    me = threading.get_ident()
    subprocs.terminate_thread(me)
    try:
        with pytest.raises(subprocs.CancelledError):
            subprocs.run([sys.executable, "-c", "print(1)"], timeout=30)
    finally:
        subprocs.clear_thread(me)
    assert subprocs.run([sys.executable, "-c", "print(1)"], timeout=30).returncode == 0
