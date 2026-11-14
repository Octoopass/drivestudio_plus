"""Chạy run nền cho Studio, mỗi lần một run.

Khi người dùng bấm chạy trên Studio, API tạo run rồi trả lời ngay (``202``), còn việc chạy các
stage diễn ra ở đây, trong một luồng nền. Chỉ một luồng, nên các run xếp hàng chạy lần lượt: tool
chạy cho một người trên một máy, và hai run train cùng lúc sẽ tranh nhau GPU (SPEC mục 2).

Tiến trình Studio không nạp model: các stage bọc DriveStudio chạy script trong tiến trình con
(ARCHITECTURE mục 2.9), nên luồng nền chỉ điều phối và ghi ``run.json``.

File này chứa:
- ``RunQueue``: nhận run cần chạy, chạy bằng ``execute_run``, chuyển sự kiện tới ``EventBroker``.
- ``BackgroundJobs``: chạy việc nền dài (ingest, Kaggle) ở luồng riêng, giữ trạng thái từng việc.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from drive2sim.engine.registry import REGISTRY, StageRegistry
from drive2sim.engine.run import RunActiveError, execute_run
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
        # Run đang xếp hàng, đang chạy, hoặc đang được việc khác giữ (tải Kaggle về). run_is_active
        # chỉ đúng khi engine đã ghi file pid, nên run chờ trong hàng đợi phải giữ chỗ ở đây: không
        # thì bấm "chạy tiếp" hai lần, hay xuất lại trong lúc đang tải Kaggle về, sẽ chạy trùng.
        self._busy: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, run_dir: Path) -> bool:
        """Giữ chỗ cho một run; ``False`` nếu run đã được giữ (xếp hàng, chạy, đang tải về)."""
        key = str(Path(run_dir).resolve())
        with self._lock:
            if key in self._busy:
                return False
            self._busy.add(key)
            return True

    def release(self, run_dir: Path) -> None:
        """Trả chỗ của một run đã giữ bằng ``claim``."""
        with self._lock:
            self._busy.discard(str(Path(run_dir).resolve()))

    def is_busy(self, run_dir: Path) -> bool:
        """Run có đang được giữ chỗ không."""
        with self._lock:
            return str(Path(run_dir).resolve()) in self._busy

    def submit(self, run_dir: Path, claimed: bool = False) -> "Future[None]":
        """Xếp một run vào hàng đợi.

        Args:
            run_dir: Thư mục run đã tạo bằng ``create_run``, hoặc run cần chạy tiếp.
            claimed: Người gọi đã ``claim`` run này (để sửa run.json trước khi xếp hàng).

        Returns:
            ``Future`` xong khi run kết thúc; test dùng để chờ.

        Raises:
            RunActiveError: Nếu run đã xếp hàng, đang chạy hoặc đang được tải về.
        """
        if not claimed and not self.claim(run_dir):
            raise RunActiveError(f"run {Path(run_dir).name} đang xếp hàng hoặc đang được xử lý")
        return self._executor.submit(self._run, Path(run_dir))

    def _run(self, run_dir: Path) -> None:
        """Chạy một run; lỗi ngoài dự kiến được ghi log và phát thành sự kiện ``log``.

        Lỗi của stage đã được engine ghi vào ``run.json``; ở đây chỉ bắt lỗi xảy ra trước khi
        engine chạy được stage nào (ví dụ ``run.json`` hỏng), để luồng nền không chết.
        """
        try:
            execute_run(run_dir, self.workdir, self.registry, sink=self.broker.publish)
        except RunActiveError as error:
            # Run được một tiến trình khác (lệnh d2s) chạy: báo, không ghi đè trạng thái của nó.
            self.broker.publish(
                {
                    "event": "log",
                    "run_id": run_dir.name,
                    "stage_id": None,
                    "level": "error",
                    "message": str(error),
                }
            )
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
        finally:
            self.release(run_dir)

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
        """Dừng nhận run mới, bỏ các run còn chờ, và chờ run đang chạy xong; gọi khi tắt Studio.

        Run bị bỏ vẫn ở ``pending`` trong ``run.json``, bấm chạy tiếp khi mở lại Studio.
        """
        self._executor.shutdown(wait=True, cancel_futures=True)


class BackgroundJobs:
    """Chạy việc nền dài (ingest, đẩy và tải Kaggle) và giữ trạng thái từng việc trong bộ nhớ.

    Các việc này có thể chạy nhiều phút, nên route trả lời ngay kèm ``job_id`` và giao diện hỏi
    lại ``GET /api/jobs/{job_id}``. Một luồng riêng, tách khỏi hàng đợi run, để không phải chờ
    run train xong. Trạng thái mất khi tắt Studio; kết quả thật vẫn còn trên đĩa.

    Attributes:
        workdir: Thư mục làm việc.
    """

    def __init__(self, workdir: Workdir) -> None:
        self.workdir = workdir
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d2s-ingest")

    def submit(self, kind: str, label: str, work: Callable[[], Optional[str]]) -> dict:
        """Xếp một việc vào hàng đợi.

        Args:
            kind: Loại việc: ``ingest``, ``kaggle_push``, ``kaggle_pull``, ``colab_push``,
                ``colab_pull``.
            label: Mô tả ngắn hiện trên giao diện, ví dụ ``nuscenes scene-0061``.
            work: Hàm làm việc, trả về ``scene_id`` hoặc ``run_id`` liên quan (để giao diện mở);
                lỗi gì thì việc thành ``failed`` kèm thông báo lỗi.

        Returns:
            Trạng thái ban đầu của việc (``queued``).
        """
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "kind": kind,
            "label": label,
            "status": "queued",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "finished_at": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, work)
        return dict(job)

    def _run(self, job_id: str, work: Callable[[], Optional[str]]) -> None:
        """Chạy một việc và ghi kết quả vào trạng thái."""
        self._update(job_id, status="running")
        try:
            result = work()
        except Exception as error:
            logger.exception("job %s failed", job_id)
            self._update(job_id, status="failed", error=str(error), finished_at=time.time())
        else:
            self._update(job_id, status="succeeded", result=result, finished_at=time.time())

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def get(self, job_id: str) -> Optional[dict]:
        """Trạng thái một việc, hoặc ``None`` nếu không có."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list[dict]:
        """Mọi việc, mới nhất trước."""
        with self._lock:
            return sorted((dict(j) for j in self._jobs.values()), key=lambda j: -j["created_at"])

    def wait_idle(self, timeout: float = 60.0) -> None:
        """Chờ mọi việc đã xếp hàng xong; test dùng."""
        self._executor.submit(lambda: None).result(timeout)

    def shutdown(self) -> None:
        """Bỏ các việc còn chờ và chờ việc đang chạy xong; gọi khi tắt Studio."""
        self._executor.shutdown(wait=True, cancel_futures=True)
