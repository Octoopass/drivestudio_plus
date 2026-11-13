"""Kiểm tra ``drive2sim/studio/api/kaggle_watch.py`` với Kaggle và hàng việc giả.

Chỉ run đã đẩy mà chưa tải kết quả của lần đẩy mới nhất được hỏi; notebook xong thì xếp đúng một
việc tải; notebook còn chạy thì chờ; không có run nào thì không gọi Kaggle.
"""

import json
from pathlib import Path
from typing import Optional

import pytest

from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.kaggle_watch import KaggleWatcher, watched_runs


class FakeJobs:
    """Hàng việc giả: chỉ ghi lại việc được xếp."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, str]] = []

    def submit(self, kind: str, label: str, work: object) -> dict:
        """Ghi lại loại và nhãn của việc."""
        self.submitted.append((kind, label))
        return {}


class FakeClient:
    """Client Kaggle giả: trả trạng thái đặt sẵn cho từng notebook."""

    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self.calls: list[str] = []

    def kernel_status(self, slug: str) -> str:
        """Trả trạng thái đặt sẵn, ghi lại notebook đã hỏi."""
        self.calls.append(slug)
        return self.statuses[slug]


def add_run(work: Workdir, run_id: str, pushed: str, pulled: Optional[str]) -> None:
    d2s = work.runs_dir / "scene" / "omnire" / run_id / "d2s"
    d2s.mkdir(parents=True)
    state = {
        "username": "alice",
        "kernel": f"alice/{run_id}",
        "attempt": 1,
        "dataset": "alice/scene",
        "commit": "abc",
        "pushed_utc": pushed,
        "pulled_utc": pulled,
    }
    (d2s / "kaggle.json").write_text(json.dumps(state))


def test_watch_queues_pull_once_when_finished(tmp_path: Path) -> None:
    work = Workdir(tmp_path)
    add_run(work, "run-a", "2026-11-13T10:00:00+00:00", None)
    add_run(work, "run-b", "2026-11-13T10:00:00+00:00", None)
    add_run(work, "run-c", "2026-11-13T09:00:00+00:00", "2026-11-13T09:30:00+00:00")  # đã tải
    assert [d.name for d, _ in watched_runs(work)] == ["run-a", "run-b"]
    client = FakeClient({"alice/run-a": "complete", "alice/run-b": "running"})
    jobs = FakeJobs()
    watcher = KaggleWatcher(work, jobs, client_factory=lambda: client)  # type: ignore[arg-type]
    assert watcher.check_once() == ["run-a"]
    assert jobs.submitted == [("kaggle_pull", "tự tải run-a từ Kaggle")]
    client.calls.clear()
    assert watcher.check_once() == []  # run-a đã xếp việc: không hỏi lại
    assert client.calls == ["alice/run-b"]


def test_watch_does_not_call_kaggle_without_runs(tmp_path: Path) -> None:
    def no_client() -> None:
        pytest.fail("không có run nào thì không được gọi Kaggle")

    watcher = KaggleWatcher(Workdir(tmp_path), FakeJobs(), client_factory=no_client)  # type: ignore[arg-type]
    assert watcher.check_once() == []


def test_cancelled_notebook_is_pulled_and_errors_do_not_block(tmp_path: Path) -> None:
    from drive2sim.kaggle.client import KaggleError

    work = Workdir(tmp_path)
    add_run(work, "run-a", "2026-11-13T10:00:00+00:00", None)
    add_run(work, "run-b", "2026-11-13T10:00:00+00:00", None)

    class Mixed(FakeClient):
        def kernel_status(self, slug: str) -> str:
            """run-a đã bị xoá trên Kaggle; run-b bị huỷ xong."""
            if slug == "alice/run-a":
                raise KaggleError("404 Not Found")
            return "cancel_acknowledged"

    jobs = FakeJobs()
    watcher = KaggleWatcher(work, jobs, client_factory=lambda: Mixed({}))  # type: ignore[arg-type]
    assert watcher.check_once() == ["run-b"]
    # mọi lần hỏi đều lỗi thì báo lên để vòng lặp giãn khoảng chờ
    only_a = KaggleWatcher(Workdir(tmp_path), FakeJobs(), client_factory=lambda: Mixed({}))  # type: ignore[arg-type]
    only_a._submitted.add("alice/run-b")
    with pytest.raises(KaggleError):
        only_a.check_once()
