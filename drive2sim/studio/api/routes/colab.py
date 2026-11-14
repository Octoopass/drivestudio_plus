"""Route Colab: ``/api/colab/...`` (như ``d2s colab``; CONTRACTS mục 9.3 và 11.1).

Đẩy và tải có thể mất nhiều phút (tạo phiên, upload cảnh, tải output), nên chạy như việc nền
(``/api/jobs``); hỏi trạng thái và tắt phiên thì trả lời ngay (một lệnh ``colab`` vài giây).

File này chứa:
- ``GET /api/colab/setup/status``: kiểm tra lệnh ``colab``, đăng nhập, compute unit, file SMPL và
  code đã push chưa (``drive2sim/colab/setup.py``).
- ``GET /api/colab/{run_id}``: ``colab.json`` của run cộng trạng thái job và log cuối.
- ``POST /api/colab/{run_id}/push``: đẩy (lần sau chạy tiếp); thân tuỳ chọn ``gpu``, ``session``;
  trả ``202`` kèm việc.
- ``POST /api/colab/{run_id}/pull``: tải kết quả về; trả ``202`` kèm việc.
- ``POST /api/colab/{run_id}/stop``: tắt phiên Colab của run.
"""

from collections.abc import Callable
from typing import Literal, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.colab import executor, setup
from drive2sim.colab.client import ColabError
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.jobs import RunQueue
from drive2sim.studio.api.state import StudioState, state_of

router = APIRouter(prefix="/api/colab", tags=["colab"])


class PushBody(BaseModel):
    """Lựa chọn khi đẩy run lên Colab; bỏ trống thì dùng mặc định của ``d2s colab push``.

    Attributes:
        gpu: GPU khi phải tạo phiên mới.
        session: Tên phiên Colab.
    """

    model_config = ConfigDict(extra="forbid")

    gpu: Literal["T4", "L4", "G4", "A100", "H100"] = executor.DEFAULT_GPU
    session: str = Field(default=executor.DEFAULT_SESSION, pattern=r"^[A-Za-z0-9_-]{1,40}$")


@router.get("/setup/status")
def setup_status(request: Request) -> dict:
    """Kiểm tra Colab đã sẵn sàng chạy run chưa; gọi ``colab sessions`` và ``colab usage``.

    Returns:
        Kết quả ``setup.check`` (``ready``, ``cli``, ``account``, ``gpu``, ``smpl``, ``code``).
    """
    return setup.check(workdir_root=state_of(request).workdir.root)


def submit_push(state: StudioState, run_id: str, body: Optional[PushBody] = None) -> dict:
    """Xếp việc đẩy một run lên Colab; ``POST /api/runs`` với ``executor: colab`` cũng gọi.

    Returns:
        Việc vừa xếp hàng.
    """
    options = body or PushBody()

    def work() -> str:
        executor.push_run(state.workdir, run_id, gpu=options.gpu, session=options.session)
        return run_id

    return state.jobs.submit("colab_push", f"đẩy {run_id} lên Colab", work)


@router.get("/{run_id}")
def colab_status(run_id: str, request: Request) -> dict:
    """Trạng thái job Colab của run.

    Returns:
        Nội dung ``colab.json`` (``session``, ``gpu``, ``attempt``, ``job``, ``commit``,
        ``branch``, ``pushed_utc``, ``pulled_utc``, ``lost_utc``, ``stopped_utc``) cộng
        ``status``, ``exit_code``, ``log_tail``, ``status_error``.
    """
    return executor.job_state(state_of(request).workdir, run_id)


@router.post("/{run_id}/push", status_code=202)
def push(run_id: str, request: Request, body: Optional[PushBody] = None) -> dict:
    """Đẩy run lên Colab ở nền."""
    state = state_of(request)
    state.workdir.find_run(run_id)
    return submit_push(state, run_id, body)


@router.post("/{run_id}/pull", status_code=202)
def pull(run_id: str, request: Request) -> dict:
    """Tải kết quả của run trên Colab về ở nền."""
    state = state_of(request)
    state.workdir.find_run(run_id)
    return state.jobs.submit(
        "colab_pull", f"tải {run_id} từ Colab", pull_work(state.workdir, state.queue, run_id)
    )


@router.post("/{run_id}/stop")
def stop(run_id: str, request: Request) -> dict:
    """Tắt phiên Colab của run; mọi thứ trên VM chưa tải về sẽ mất.

    Returns:
        ``colab.json`` sau khi ghi ``stopped_utc``.
    """
    return executor.stop_session(state_of(request).workdir, run_id)


def pull_work(workdir: Workdir, queue: RunQueue, run_id: str) -> Callable[[], str]:
    """Việc tải kết quả Colab về, có giữ chỗ run trong lúc ghép (như Kaggle).

    Không giữ chỗ thì tải về chạy cùng lúc với xuất lại hay chạy tiếp trên cùng thư mục run, hai
    bên xoá và ghi đè run.json, thư mục stage của nhau.
    """

    def work() -> str:
        run_dir = workdir.find_run(run_id)
        if not queue.claim(run_dir):
            raise ColabError(f"run {run_id} đang chạy hoặc xếp hàng trên máy; chờ xong rồi tải")
        try:
            executor.pull_run(workdir, run_id)
        finally:
            queue.release(run_dir)
        return run_id

    return work
