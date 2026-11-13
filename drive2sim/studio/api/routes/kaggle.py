"""Route Kaggle: ``/api/kaggle/{run_id}/...`` (như ``d2s kaggle``; CONTRACTS mục 11.1).

Đẩy và tải có thể mất nhiều phút (upload cảnh, tải output), nên chạy như việc nền
(``/api/jobs``); hỏi trạng thái thì trả lời ngay.

File này chứa:
- ``GET /api/kaggle/setup/status``: kiểm tra lệnh ``kaggle``, khoá API, tài khoản, giờ GPU còn lại
  và code đã push chưa (``drive2sim/kaggle/setup.py``).
- ``POST /api/kaggle/setup/token``: lưu khoá API vào thư mục cấu hình của Kaggle.
- ``GET /api/kaggle/{run_id}``: ``kaggle.json`` của run cộng trạng thái notebook.
- ``POST /api/kaggle/{run_id}/push``: đẩy (lần sau chạy tiếp); trả ``202`` kèm việc.
- ``POST /api/kaggle/{run_id}/pull``: tải kết quả về; trả ``202`` kèm việc.
"""

from collections.abc import Callable
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle import executor, setup
from drive2sim.kaggle.client import KaggleError
from drive2sim.studio.api.jobs import RunQueue
from drive2sim.studio.api.state import StudioState, state_of

router = APIRouter(prefix="/api/kaggle", tags=["kaggle"])


class TokenBody(BaseModel):
    """Khoá API người dùng nhập trên Studio.

    Attributes:
        token: Token mới (``KGAT_...``) hoặc khoá kiểu cũ.
        username: Tên người dùng Kaggle; chỉ cần với khoá kiểu cũ.
        replace: Cho phép ghi đè khoá đã có.
    """

    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1, max_length=512, repr=False)
    username: Optional[str] = None
    replace: bool = False


@router.get("/setup/status")
def setup_status() -> dict:
    """Kiểm tra Kaggle đã sẵn sàng chạy run chưa; gọi ``kaggle quota`` (chỉ đọc).

    Returns:
        Kết quả ``setup.check`` (``ready``, ``cli``, ``credentials``, ``account``, ``gpu``,
        ``code``).
    """
    return setup.check()


@router.post("/setup/token")
def setup_token(body: TokenBody) -> dict:
    """Lưu khoá API vào thư mục cấu hình của Kaggle CLI (quyền ``600``); không trả lại khoá.

    Returns:
        ``{"kind", "path"}``. Khoá sai dạng trả ``400``; đã có khoá mà ``replace`` là ``false``
        trả ``409``.
    """
    return setup.save_token(body.token, body.username, body.replace)


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
    return state.jobs.submit(
        "kaggle_pull", f"tải {run_id} từ Kaggle", pull_work(state.workdir, state.queue, run_id)
    )


def pull_work(workdir: Workdir, queue: RunQueue, run_id: str) -> Callable[[], str]:
    """Việc tải kết quả Kaggle về, có giữ chỗ run trong lúc ghép.

    Không giữ chỗ thì tải về chạy cùng lúc với xuất lại hay chạy tiếp trên cùng thư mục run, hai
    bên xoá và ghi đè run.json, thư mục stage của nhau.
    """

    def work() -> str:
        run_dir = workdir.find_run(run_id)
        if not queue.claim(run_dir):
            raise KaggleError(f"run {run_id} đang chạy hoặc xếp hàng trên máy; chờ xong rồi tải")
        try:
            executor.pull_run(workdir, run_id)
        finally:
            queue.release(run_dir)
        return run_id

    return work
