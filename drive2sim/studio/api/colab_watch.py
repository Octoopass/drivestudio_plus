"""Theo dõi các run đang chạy trên Colab và tự tải kết quả về khi job xong (như ``kaggle_watch``).

Studio chạy một luồng nền, cứ ``interval_s`` giây (mặc định 5 phút) hỏi trạng thái job của các run
đã đẩy lên Colab mà chưa tải kết quả của lần đẩy mới nhất và phiên chưa mất
(``executor.watched_runs``). Job xong (thành công, lỗi, hoặc tiến trình mất) thì xếp một việc
``colab_pull`` như khi người dùng bấm "Tải kết quả về". Phiên đã mất thì ``job_state`` ghi
``lost_utc`` và run đó thôi được theo dõi. Mỗi lần hỏi là một lệnh ``colab exec`` ngắn; không có
run nào cần theo dõi thì không gọi Colab.

File này chứa ``ColabWatcher``.
"""

import logging
from collections.abc import Callable
from typing import ClassVar, Optional

from drive2sim.colab import executor
from drive2sim.colab.client import ColabClient, ColabError
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.jobs import BackgroundJobs, RunQueue
from drive2sim.studio.api.remote_watch import RemoteWatcher

LOGGER = logging.getLogger(__name__)
DEFAULT_INTERVAL_S = 300
MAX_INTERVAL_S = 3600
# Trạng thái job đã dừng hẳn (executor.job_state): output trên VM không đổi nữa, tải về được.
FINISHED = {"succeeded", "failed", "died"}


class ColabWatcher(RemoteWatcher):
    """Luồng nền hỏi trạng thái job Colab và tự xếp việc tải kết quả về.

    Args:
        workdir: Thư mục làm việc.
        jobs: Hàng việc nền của Studio, nơi xếp việc ``colab_pull``.
        interval_s: Khoảng giữa hai lần hỏi khi mọi thứ bình thường.
        max_interval_s: Khoảng chờ tối đa khi hỏi lỗi liên tiếp.
        client_factory: Tạo ``ColabClient``; test truyền client giả.
        queue: Hàng đợi run của Studio, để giữ chỗ run trong lúc tải về.
    """

    THREAD_NAME: ClassVar[str] = "colab-watch"
    SERVICE: ClassVar[str] = "Colab"

    def __init__(
        self,
        workdir: Workdir,
        jobs: BackgroundJobs,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_interval_s: float = MAX_INTERVAL_S,
        client_factory: Callable[[], ColabClient] = ColabClient,
        queue: Optional[RunQueue] = None,
    ) -> None:
        super().__init__(interval_s, max_interval_s)
        self.workdir = workdir
        self.jobs = jobs
        self.client_factory = client_factory
        self.queue = queue
        self._submitted: set[str] = set()  # job đã xếp việc tải, không xếp lại

    def check_once(self) -> list[str]:
        """Hỏi trạng thái mọi run cần theo dõi một lượt, xếp việc tải cho job đã xong.

        Returns:
            ``run_id`` vừa được xếp việc tải.

        Raises:
            ColabError: Nếu mọi lần hỏi đều lỗi (để vòng lặp giãn khoảng chờ).
        """
        runs = [
            (run_dir, state)
            for run_dir, state in executor.watched_runs(self.workdir)
            if state.job not in self._submitted
        ]
        if not runs:
            return []
        client = self.client_factory()
        queued, failures = [], []
        for run_dir, state in runs:
            result = executor.job_state(self.workdir, run_dir.name, client=client)
            if result["status"] is None:
                failures.append(ColabError(result["status_error"] or "không rõ lỗi"))
                continue
            if result["status"] not in FINISHED:
                continue
            self._submitted.add(state.job)
            self.jobs.submit(
                "colab_pull", f"tự tải {run_dir.name} từ Colab", self._pull(run_dir.name)
            )
            queued.append(run_dir.name)
        if failures and len(failures) == len(runs):
            raise failures[-1]
        if failures:
            LOGGER.warning("không hỏi được %d job Colab: %s", len(failures), failures[-1])
        return queued

    def _pull(self, run_id: str) -> Callable[[], str]:
        if self.queue is not None:
            from drive2sim.studio.api.routes.colab import pull_work

            return pull_work(self.workdir, self.queue, run_id)

        def work() -> str:
            executor.pull_run(self.workdir, run_id)
            return run_id

        return work
