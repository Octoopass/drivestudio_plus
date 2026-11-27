"""Kiểm tra ``BackgroundJobs`` huỷ và chạy lại việc tải (``drive2sim/studio/api/jobs.py``)."""

import threading
from pathlib import Path

import pytest

from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.jobs import BackgroundJobs


def test_cancel_queued_job_and_retry(tmp_path: Path) -> None:
    jobs = BackgroundJobs(Workdir(tmp_path))
    gate = threading.Event()
    first = jobs.submit("ingest", "chặn hàng đợi", lambda: None if gate.wait(10) else None)
    queued = jobs.submit("drive_pull", "chờ", lambda: "xong", target="t")
    assert queued["cancellable"] and not first["cancellable"]
    with pytest.raises(ValueError):
        jobs.cancel(first["job_id"])
    with pytest.raises(KeyError):
        jobs.cancel("nope")
    assert jobs.active_for("t")[0]["job_id"] == queued["job_id"]
    assert jobs.cancel(queued["job_id"])["status"] == "cancelled"
    assert jobs.active_for("t") == []
    gate.set()
    jobs.wait_idle()
    assert jobs.get(queued["job_id"])["status"] == "cancelled"  # không chạy
    again = jobs.retry(queued["job_id"])
    jobs.wait_idle()
    assert jobs.get(again["job_id"])["status"] == "succeeded"
    with pytest.raises(ValueError):
        jobs.retry(again["job_id"])  # chưa huỷ, chưa lỗi
    jobs.shutdown()


def test_cancel_running_job_stops_at_next_report(tmp_path: Path) -> None:
    jobs = BackgroundJobs(Workdir(tmp_path))
    started, gate = threading.Event(), threading.Event()

    def work() -> str:
        jobs.report(1, 3)
        started.set()
        gate.wait(10)
        jobs.report(2, 3)  # đã huỷ: ném JobCancelled
        return "không tới đây"

    job = jobs.submit("drive_pull", "chạy", work, target="t")
    assert started.wait(10)
    assert jobs.cancel(job["job_id"])["status"] == "running"
    gate.set()
    assert jobs.wait_stopped([job["job_id"]], timeout=10)
    done = jobs.get(job["job_id"])
    assert done["status"] == "cancelled" and done["error"] is None and done["result"] is None
    jobs.shutdown()
