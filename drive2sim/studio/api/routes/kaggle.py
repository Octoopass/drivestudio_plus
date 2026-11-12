"""Route Kaggle: ``/api/kaggle/{run_id}/...`` (như ``d2s kaggle``; CONTRACTS mục 11.1).

Đẩy và tải có thể mất nhiều phút (upload cảnh, tải output), nên chạy như việc nền
(``/api/jobs``); hỏi trạng thái thì trả lời ngay.

File này chứa:
- ``GET /api/kaggle/{run_id}``: ``kaggle.json`` của run cộng trạng thái notebook.
- ``POST /api/kaggle/{run_id}/push``: đẩy (lần sau chạy tiếp); trả ``202`` kèm việc.
- ``POST /api/kaggle/{run_id}/pull``: tải kết quả về; trả ``202`` kèm việc.
"""

from fastapi import APIRouter, Request

from drive2sim.kaggle import executor
from drive2sim.studio.api.state import StudioState, state_of

router = APIRouter(prefix="/api/kaggle", tags=["kaggle"])


def submit_push(state: StudioState, run_id: str) -> dict:
    """Xếp việc đẩy một run lên Kaggle; ``POST /api/runs`` với ``executor: kaggle`` cũng gọi.

    Returns:
        Việc vừa xếp hàng.
    """

    def work() -> str:
        executor.push_run(state.workdir, run_id)
        return run_id

    return state.jobs.submit("kaggle_push", f"đẩy {run_id} lên Kaggle", work)


@router.get("/{run_id}")
def kaggle_status(run_id: str, request: Request) -> dict:
    """Trạng thái notebook của run.

    Returns:
        ``username``, ``kernel``, ``attempt``, ``dataset``, ``commit``, ``pushed_utc``,
        ``pulled_utc``, ``status``.
    """
    return executor.kernel_state(state_of(request).workdir, run_id)


@router.post("/{run_id}/push", status_code=202)
def push(run_id: str, request: Request) -> dict:
    """Đẩy run lên Kaggle ở nền."""
    state = state_of(request)
    state.workdir.find_run(run_id)
    return submit_push(state, run_id)


@router.post("/{run_id}/pull", status_code=202)
def pull(run_id: str, request: Request) -> dict:
    """Tải kết quả của run trên Kaggle về ở nền."""
    state = state_of(request)
    state.workdir.find_run(run_id)

    def work() -> str:
        executor.pull_run(state.workdir, run_id)
        return run_id

    return state.jobs.submit("kaggle_pull", f"tải {run_id} từ Kaggle", work)
