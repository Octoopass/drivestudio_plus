"""Theo dõi các run đang chạy trên Kaggle và tự tải kết quả về khi notebook xong.

Studio chạy một luồng nền, cứ ``interval_s`` giây (mặc định 5 phút) hỏi trạng thái notebook của
các run đã đẩy lên Kaggle mà chưa tải kết quả của lần đẩy mới nhất. Notebook xong (thành công,
lỗi hay bị huỷ) thì xếp một việc ``kaggle_pull`` như khi người dùng bấm "Tải kết quả về"; việc
hiện trong ngăn Hoạt động.

Kaggle không công bố giới hạn tần suất gọi API, nên luồng này hỏi thưa và giãn dần khi lỗi: mỗi
lần hỏi lỗi (mạng, chưa có khoá API, 429) thì khoảng chờ nhân đôi, tối đa ``max_interval_s``
(1 giờ); hỏi được thì về lại ``interval_s``. Không có run nào cần theo dõi thì không gọi Kaggle.

Vòng hỏi nền và cách giãn khoảng chờ dùng chung với Colab (``remote_watch.py``).

File này chứa ``KaggleWatcher``, ``watched_runs``.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Optional

from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle import executor
from drive2sim.kaggle.client import KaggleClient, KaggleError
from drive2sim.studio.api.jobs import BackgroundJobs, RunQueue
from drive2sim.studio.api.remote_watch import RemoteWatcher

LOGGER = logging.getLogger(__name__)
DEFAULT_INTERVAL_S = 300
MAX_INTERVAL_S = 3600
# Trạng thái notebook đã dừng hẳn. KaggleClient.kernel_status trả tên enum KernelWorkerStatus viết
# thường, giữ dấu gạch dưới: CANCEL_ACKNOWLEDGED → "cancel_acknowledged". "cancel_requested" chưa
# dừng hẳn (output có thể chưa ghi xong), nên chờ tới "cancel_acknowledged".
FINISHED = {"complete", "error", "cancel_acknowledged"}


def watched_runs(workdir: Workdir) -> list[tuple[Path, executor.KaggleState]]:
    """Các run đã đẩy lên Kaggle mà chưa tải kết quả của lần đẩy mới nhất.

    Args:
        workdir: Thư mục làm việc.

    Returns:
        ``(thư mục run, kaggle.json)`` theo thứ tự đường dẫn.
    """
    found = []
    for state_file in sorted(workdir.runs_dir.glob(f"*/*/*/d2s/{executor.STATE_FILE}")):
        run_dir = state_file.parents[1]
        try:
            state = executor.load_state(run_dir)
        except (OSError, ValueError, TypeError):
            continue
        # Thời điểm ghi theo ISO 8601 UTC cùng một định dạng, nên so chuỗi là so thời gian.
        if state and not (state.pulled_utc and state.pulled_utc >= state.pushed_utc):
            found.append((run_dir, state))
    return found


class KaggleWatcher(RemoteWatcher):
    """Luồng nền hỏi trạng thái notebook và tự xếp việc tải kết quả về.

    Args:
        workdir: Thư mục làm việc.
        jobs: Hàng việc nền của Studio, nơi xếp việc ``kaggle_pull``.
        interval_s: Khoảng giữa hai lần hỏi khi mọi thứ bình thường.
        max_interval_s: Khoảng chờ tối đa khi hỏi lỗi liên tiếp.
        client_factory: Tạo ``KaggleClient``; test truyền client giả.
        queue: Hàng đợi run của Studio, để giữ chỗ run trong lúc tải về.
    """

    THREAD_NAME: ClassVar[str] = "kaggle-watch"
    SERVICE: ClassVar[str] = "Kaggle"

    def __init__(
        self,
        workdir: Workdir,
        jobs: BackgroundJobs,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_interval_s: float = MAX_INTERVAL_S,
        client_factory: Callable[[], KaggleClient] = KaggleClient,
        queue: Optional["RunQueue"] = None,
    ) -> None:
        super().__init__(interval_s, max_interval_s)
        self.workdir = workdir
        self.jobs = jobs
        self.client_factory = client_factory
        self.queue = queue
        self._submitted: set[str] = set()  # notebook đã xếp việc tải, không xếp lại

    def check_once(self) -> list[str]:
        """Hỏi trạng thái mọi run cần theo dõi một lượt, xếp việc tải cho notebook đã xong.

        Returns:
            ``run_id`` vừa được xếp việc tải.

        Raises:
            KaggleError: Nếu mọi lần hỏi trạng thái đều lỗi (để vòng lặp giãn khoảng chờ); lỗi
                của một vài notebook chỉ ghi log, các run khác vẫn được xét.
        """
        runs = [(d, s) for d, s in watched_runs(self.workdir) if s.kernel not in self._submitted]
        if not runs:
            return []
        client = self.client_factory()
        queued = []
        failures: list[KaggleError] = []
        for run_dir, state in runs:
            # Một notebook lỗi khi hỏi (ví dụ đã bị xoá trên Kaggle) không được chặn các run khác.
            try:
                status = client.kernel_status(state.kernel)
            except KaggleError as error:
                failures.append(error)
                continue
            if status not in FINISHED:
                continue
            run_id = run_dir.name
            self._submitted.add(state.kernel)
            self.jobs.submit("kaggle_pull", f"tự tải {run_id} từ Kaggle", self._pull(run_id))
            queued.append(run_id)
        # Mọi lần hỏi đều lỗi (mất mạng, khoá API hết hạn, 429): báo lên để giãn khoảng chờ.
        if failures and len(failures) == len(runs):
            raise failures[-1]
        if failures:
            LOGGER.warning("không hỏi được %d notebook Kaggle: %s", len(failures), failures[-1])
        return queued

    def _pull(self, run_id: str) -> Callable[[], str]:
        if self.queue is not None:
            from drive2sim.studio.api.routes.kaggle import pull_work

            return pull_work(self.workdir, self.queue, run_id)

        def work() -> str:
            executor.pull_run(self.workdir, run_id)
            return run_id

        return work
