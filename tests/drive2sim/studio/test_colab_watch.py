"""Kiểm tra ``drive2sim/studio/api/colab_watch.py`` với Colab giả và hàng việc giả.

Chỉ run đã đẩy mà chưa tải về và phiên chưa mất được hỏi; job xong thì xếp đúng một việc tải;
job còn chạy thì chờ; phiên mất thì thôi theo dõi; không có run nào thì không gọi Colab.
"""

import json
from pathlib import Path
from typing import Optional

import pytest

from drive2sim.colab import executor
from drive2sim.colab.client import ColabError, ColabSessionLostError
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.colab_watch import ColabWatcher

RUN = "20261113-010203-omnire-{}"


class FakeJobs:
    """Hàng việc giả: chỉ ghi lại việc được xếp."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, str]] = []

    def submit(self, kind: str, label: str, work: object) -> dict:
        """Ghi lại loại và nhãn của việc."""
        self.submitted.append((kind, label))
        return {}


class FakeClient:
    """Colab giả: trả kết quả đọc trạng thái đặt sẵn cho từng job."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.calls = 0

    def exec_python(self, session: str, code: str, timeout_s: float = 60) -> str:
        """Trả ``@@STATE`` của job có tên trong đoạn mã; lỗi nếu đặt sẵn là ngoại lệ."""
        self.calls += 1
        for job, answer in self.answers.items():
            if f'"job": "{job}"' in code:
                if isinstance(answer, Exception):
                    raise answer
                return f"@@STATE {json.dumps(answer)}\n@@L dòng log\n"
        raise AssertionError("job lạ")


def add_run(work: Workdir, run_id: str, pulled: Optional[str] = None) -> None:
    d2s = work.runs_dir / "scene" / "omnire" / run_id / "d2s"
    d2s.mkdir(parents=True)
    state = executor.ColabState(
        "d2s", "T4", 1, f"{run_id}-1", "abc", "drive2sim", "2026-11-13T10:00:00+00:00", pulled
    )
    (d2s / "colab.json").write_text(json.dumps(state.to_dict()))


def running(alive: bool = True, exit_code: Optional[int] = None) -> dict:
    return {"known": True, "alive": alive, "exit_code": exit_code, "offset": 10}


def test_watch_queues_pull_once_and_stops_on_lost(tmp_path: Path) -> None:
    work = Workdir(tmp_path)
    for run_id in (
        "20261113-010203-omnire-aaaa",
        "20261113-010203-omnire-bbbb",
        "20261113-010203-omnire-cccc",
        "20261113-010203-omnire-dddd",
    ):
        add_run(work, run_id)
    add_run(work, "20261113-010203-omnire-eeee", pulled="2026-11-13T11:00:00+00:00")
    client = FakeClient(
        {
            "20261113-010203-omnire-aaaa-1": running(alive=False, exit_code=0),
            "20261113-010203-omnire-bbbb-1": running(),
            "20261113-010203-omnire-cccc-1": ColabSessionLostError("mất"),
            "20261113-010203-omnire-dddd-1": running(alive=False),
        }
    )
    jobs = FakeJobs()
    watcher = ColabWatcher(work, jobs, client_factory=lambda: client)  # type: ignore[arg-type]
    assert sorted(watcher.check_once()) == [RUN.format("aaaa"), RUN.format("dddd")]
    assert sorted(kind for kind, _ in jobs.submitted) == ["colab_pull", "colab_pull"]
    calls = client.calls
    assert watcher.check_once() == []  # aaaa, dddd đã xếp; cccc đã mất; chỉ hỏi bbbb
    assert client.calls == calls + 1
    run_c = work.find_run("20261113-010203-omnire-cccc")
    assert executor.load_state(run_c).lost_utc is not None


def test_watch_without_runs_does_not_call_colab(tmp_path: Path) -> None:
    def no_client() -> None:
        raise AssertionError("không được gọi Colab")

    watcher = ColabWatcher(Workdir(tmp_path), FakeJobs(), client_factory=no_client)  # type: ignore[arg-type]
    assert watcher.check_once() == []


def test_watch_raises_when_every_poll_fails(tmp_path: Path) -> None:
    work = Workdir(tmp_path)
    add_run(work, "20261113-010203-omnire-aaaa")
    client = FakeClient({"20261113-010203-omnire-aaaa-1": ColabError("mất mạng")})
    watcher = ColabWatcher(work, FakeJobs(), client_factory=lambda: client)  # type: ignore[arg-type]
    with pytest.raises(ColabError, match="mất mạng"):
        watcher.check_once()
