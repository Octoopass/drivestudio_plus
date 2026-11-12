"""Chạy run nền cho Studio, mỗi lần một run.

Khi người dùng bấm chạy trên Studio, API tạo run rồi trả lời ngay (``202``), còn việc chạy các
stage diễn ra ở đây, trong một luồng nền. Chỉ một luồng, nên các run xếp hàng chạy lần lượt: tool
chạy cho một người trên một máy, và hai run train cùng lúc sẽ tranh nhau GPU (SPEC mục 2).

Tiến trình Studio không nạp model: các stage bọc DriveStudio chạy script trong tiến trình con
(ARCHITECTURE mục 2.9), nên luồng nền chỉ điều phối và ghi ``run.json``.

File này chứa ``RunQueue``: nhận run cần chạy, chạy bằng ``execute_run``, chuyển sự kiện tới
``EventBroker``.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from drive2sim.engine.registry import REGISTRY, StageRegistry
from drive2sim.engine.run import execute_run
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.events import EventBroker

logger = logging.getLogger(__name__)


class RunQueue:
    """Hàng đợi chạy run trong một luồng nền.

    Attributes:
        workdir: Thư mục làm việc.
        broker: Nơi nhận sự kiện tiến độ.
        registry: Bảng stage.
    """

    def __init__(
        self, workdir: Workdir, broker: EventBroker, registry: StageRegistry = REGISTRY
    ) -> None:
        self.workdir = workdir
        self.broker = broker
        self.registry = registry
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d2s-run")

    def submit(self, run_dir: Path) -> "Future[None]":
        """Xếp một run vào hàng đợi.

        Args:
            run_dir: Thư mục run đã tạo bằng ``create_run``, hoặc run cần chạy tiếp.

        Returns:
            ``Future`` xong khi run kết thúc; test dùng để chờ.
        """
        return self._executor.submit(self._run, Path(run_dir))

    def _run(self, run_dir: Path) -> None:
        """Chạy một run; lỗi ngoài dự kiến được ghi log và phát thành sự kiện ``log``.

        Lỗi của stage đã được engine ghi vào ``run.json``; ở đây chỉ bắt lỗi xảy ra trước khi
        engine chạy được stage nào (ví dụ ``run.json`` hỏng), để luồng nền không chết.
        """
        try:
            execute_run(run_dir, self.workdir, self.registry, sink=self.broker.publish)
        except Exception as error:
            logger.exception("run %s failed outside a stage", run_dir.name)
            self.broker.publish(
                {
                    "event": "log",
                    "run_id": run_dir.name,
                    "stage_id": None,
                    "level": "error",
                    "message": f"lỗi khi chạy run: {type(error).__name__}: {error}",
                }
            )
            self.broker.publish({"event": "done", "run_id": run_dir.name, "status": "failed"})

    def wait_idle(self, timeout: float = 60.0) -> None:
        """Chờ mọi run đã xếp hàng chạy xong.

        Hàng đợi chỉ có một luồng và chạy theo thứ tự, nên một việc rỗng xếp sau cùng xong thì
        mọi run trước nó đã xong.

        Args:
            timeout: Thời gian chờ tối đa, giây.

        Raises:
            TimeoutError: Nếu chờ quá ``timeout``.
        """
        self._executor.submit(lambda: None).result(timeout)

    def shutdown(self) -> None:
        """Dừng nhận run mới và chờ run đang chạy xong; gọi khi tắt Studio."""
        self._executor.shutdown(wait=True)
