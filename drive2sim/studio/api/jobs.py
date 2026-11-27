"""Chạy run nền cho Studio, mỗi lần một run.

Khi người dùng bấm chạy trên Studio, API tạo run rồi trả lời ngay (``202``), còn việc chạy các
stage diễn ra ở đây, trong một luồng nền. Chỉ một luồng, nên các run xếp hàng chạy lần lượt: tool
chạy cho một người trên một máy, và hai run train cùng lúc sẽ tranh nhau GPU (SPEC mục 2).

Tiến trình Studio không nạp model: các stage bọc DriveStudio chạy script trong tiến trình con
(ARCHITECTURE mục 2.9), nên luồng nền chỉ điều phối và ghi ``run.json``.

File này chứa:
- ``RunQueue``: nhận run cần chạy, chạy bằng ``execute_run``, chuyển sự kiện tới ``EventBroker``.
- ``BackgroundJobs``: chạy việc nền dài (ingest, Kaggle) ở luồng riêng, giữ trạng thái từng việc;
  việc nào biết tiến độ (tải từ Drive) thì báo qua ``BackgroundJobs.report``. Việc tải từ Drive
  huỷ được (``cancel``) và chạy lại được (``retry``): phần đã tải nằm trong ``.part`` nên lần sau
  đi tiếp từ đó.
- ``JobCancelledError``: lỗi ``report`` ném ra trong việc đã bị huỷ.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from drive2sim.core import subprocs
from drive2sim.engine.registry import REGISTRY, StageRegistry
from drive2sim.engine.run import RunActiveError, execute_run, request_cancel, run_is_active
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
        """Dừng nhận run mới, bỏ các run còn chờ, huỷ run đang chạy; gọi khi tắt Studio.

        Run bị bỏ vẫn ở ``pending`` trong ``run.json``; run đang chạy dừng trong khoảng một giây
        (``request_cancel``, như nút huỷ) và thành ``cancelled``. Cả hai bấm chạy tiếp khi mở lại
        Studio; stage đã xong không chạy lại.
        """
        with self._lock:
            busy = [Path(key) for key in self._busy]
        for run_dir in busy:
            if run_is_active(run_dir):
                request_cancel(run_dir)
        self._executor.shutdown(wait=True, cancel_futures=True)


# Loại việc huỷ và chạy lại được: dừng giữa chừng không để lại gì sai (tải tiếp từ phần đã có).
CANCELLABLE = frozenset({"drive_pull", "drive_push"})
ACTIVE = ("queued", "running")


class JobCancelledError(RuntimeError):
    """Việc nền đã bị huỷ từ giao diện; ``report`` ném ra để việc dừng ở chỗ an toàn tiếp theo."""


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
        self._stopping = threading.Event()
        self._local = threading.local()
        self._works: dict[str, Callable[[], Optional[str]]] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._threads: dict[str, int] = {}

    def submit(
        self,
        kind: str,
        label: str,
        work: Callable[[], Optional[str]],
        target: Optional[str] = None,
    ) -> dict:
        """Xếp một việc vào hàng đợi.

        Args:
            kind: Loại việc: ``ingest``, ``kaggle_push``, ``kaggle_pull``, ``colab_push``,
                ``colab_pull``, ``colab_update``, ``colab_session``, ``drive_pull``,
                ``drive_push``.
            label: Mô tả ngắn hiện trên giao diện, ví dụ ``nuscenes scene-0061``.
            work: Hàm làm việc, trả về ``scene_id`` hoặc ``run_id`` liên quan (để giao diện mở;
                ``drive_pull`` trả thư mục đã tải); lỗi gì thì việc thành ``failed`` kèm thông báo
                lỗi. Trong ``work`` gọi ``report`` để báo tiến độ.
            target: Thư mục ``uploads/<tên>/`` việc ghi vào (``drive_pull``), để trang Lưu trữ
                nối việc với bộ dữ liệu; việc khác để ``None``.

        Returns:
            Trạng thái ban đầu của việc (``queued``); ``cancellable`` cho biết huỷ, chạy lại được.
        """
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "kind": kind,
            "label": label,
            "status": "queued",
            "result": None,
            "error": None,
            "progress": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "target": target,
            "cancellable": kind in CANCELLABLE,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._works[job_id] = work
            self._cancels[job_id] = threading.Event()
        self._executor.submit(self._run, job_id, work)
        return dict(job)

    def _run(self, job_id: str, work: Callable[[], Optional[str]]) -> None:
        """Chạy một việc và ghi kết quả vào trạng thái."""
        cancel = self._cancels[job_id]
        with self._lock:
            if cancel.is_set():  # huỷ khi còn chờ: không chạy
                return
            self._threads[job_id] = threading.get_ident()
        self._update(job_id, status="running", started_at=time.time())
        self._local.job_id = job_id
        try:
            result = work()
        except Exception as error:
            message = str(error)
            if cancel.is_set():
                self._update(job_id, status="cancelled", error=None, finished_at=time.time())
                return
            if self._stopping.is_set():
                # Bị dừng vì tắt Studio (shutdown): không phải lỗi, không in traceback.
                logger.warning("dừng việc %s vì tắt Studio", self._jobs[job_id]["label"])
                message = "dừng vì tắt Studio"
            else:
                logger.exception("job %s failed", job_id)
            self._update(job_id, status="failed", error=message, finished_at=time.time())
        else:
            self._update(job_id, status="succeeded", result=result, finished_at=time.time())
        finally:
            self._local.job_id = None
            with self._lock:
                self._threads.pop(job_id, None)
            subprocs.clear_thread(threading.get_ident())

    def cancel(self, job_id: str) -> dict:
        """Huỷ một việc huỷ được (``CANCELLABLE``) đang chờ hoặc đang chạy.

        Việc đang chờ thành ``cancelled`` ngay. Việc đang chạy: lệnh ngoài nó đang chạy (``colab``)
        bị dừng, ``report`` sau ném ``JobCancelledError``; dừng xong thì ``cancelled``.

        Returns:
            Trạng thái của việc sau khi huỷ.

        Raises:
            KeyError: Không có việc này.
            ValueError: Loại việc không huỷ được.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["kind"] not in CANCELLABLE:
                raise ValueError(f"việc {job['kind']} không huỷ được")
            if job["status"] not in ACTIVE:
                return dict(job)
            self._cancels[job_id].set()
            thread = self._threads.get(job_id)
            if thread is None:
                job.update(status="cancelled", finished_at=time.time())
        if thread is not None:
            subprocs.terminate_thread(thread)
        return self.get(job_id) or {}

    def retry(self, job_id: str) -> dict:
        """Chạy lại việc đã huỷ hoặc lỗi (loại ``CANCELLABLE``) thành một việc mới.

        Returns:
            Trạng thái của việc mới (``queued``).

        Raises:
            KeyError: Không có việc này.
            ValueError: Loại việc không chạy lại được, hoặc việc chưa dừng.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["kind"] not in CANCELLABLE or job["status"] not in ("cancelled", "failed"):
                raise ValueError("chỉ chạy lại được việc tải đã huỷ hoặc lỗi")
            work = self._works[job_id]
        active = self.active_for(job["target"]) if job["target"] else []
        if active:
            return active[0]
        return self.submit(job["kind"], job["label"], work, target=job["target"])

    def active_for(self, target: str) -> list[dict]:
        """Các việc đang chờ hoặc đang chạy ghi vào ``uploads/<target>/``, cũ nhất trước."""
        with self._lock:
            found = [
                dict(j)
                for j in self._jobs.values()
                if j.get("target") == target and j["status"] in ACTIVE
            ]
        return sorted(found, key=lambda j: j["created_at"])

    def wait_stopped(self, job_ids: list[str], timeout: float = 30.0) -> bool:
        """Chờ các việc rời trạng thái chờ, chạy; ``False`` nếu quá ``timeout`` giây."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if all(self._jobs[j]["status"] not in ACTIVE for j in job_ids if j in self._jobs):
                    return True
            time.sleep(0.05)
        return False

    def report(self, done: float, total: float, detail: str = "") -> None:
        """Báo tiến độ của việc đang chạy trên luồng hiện tại (gọi trong ``work``).

        Args:
            done: Phần đã xong (byte, số file...).
            total: Tổng; giao diện tính phần trăm và thời gian còn lại từ ``started_at``.
            detail: Mô tả ngắn phần đang làm, ví dụ tên file.
        """
        job_id = getattr(self._local, "job_id", None)
        if not job_id:
            return
        if self._cancels[job_id].is_set():
            raise JobCancelledError("đã huỷ")
        with self._lock:
            job = self._jobs[job_id]
            # ``first``: phần đã có lúc bắt đầu (tải tiếp), để tốc độ chỉ tính phần làm lần này
            first = job["progress"]["first"] if job["progress"] else done
            job["progress"] = {
                "done": done, "total": total, "detail": detail, "at": time.time(), "first": first,
            }  # fmt: skip

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
        """Bỏ các việc còn chờ, dừng lệnh ``kaggle``/``colab`` đang chạy, rồi chờ việc kết thúc.

        Gọi khi tắt Studio. Việc đẩy, tải Kaggle hay Colab đang chạy kết thúc với lỗi ngay
        (``subprocs.terminate_all``); tải dở không ghép gì vào run, mở lại Studio thì tự tải lại.
        Ingest chạy trong Python nên vẫn chờ xong.
        """
        self._stopping.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        subprocs.terminate_all()
        self._executor.shutdown(wait=True)
