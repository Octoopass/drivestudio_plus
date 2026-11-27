"""Cài môi trường chính ngay từ Studio (ARCHITECTURE mục 2.9, CONTRACTS mục 11.1).

Bảng kiểm tra môi trường (``environment.py``) chỉ in lệnh cài; người dùng mới thường không biết
chạy ở đâu, trong Python nào. Ở đây Studio chạy chính ``envs/install_<env>.sh`` với
``--python`` là Python đang chạy Studio (môi trường mà run chạy trên máy dùng), trong nhóm tiến
trình riêng, ghi log ra ``<thư mục làm việc>/logs/install-<env>.log`` và đọc dòng
``=== hh:mm:ss N/M ...`` của script để biết đang ở bước nào. Cài lỗi giữa chừng thì cài tiếp từ
bước đó (``--from N``), bỏ qua các bước đã xong.

Không cài khi máy không có GPU NVIDIA (cài xong vẫn không chạy được) hoặc Python khác bản mà
``envs/<env>.yml`` chốt. Không đi qua hàng việc nền của Studio: cài mất hàng chục phút (build
gsplat-geer, nvdiffrast, pytorch3d), trong lúc đó ingest và tải Drive vẫn phải chạy được.

File này chứa:
- ``SCRIPTS``, ``PYTHON``: script và bản Python của từng môi trường.
- ``InstallFlow``: một lần cài đang chạy hoặc vừa xong (Studio giữ một bản); cài môi trường
  (``start``), một lệnh của bảng thư viện (``start_command``, ``libraries.py``), hoặc lần lượt mọi
  mục còn thiếu (``start_all``, nút "Cài tất cả", ``d2s libraries install --all``).

``start_all`` chạy một hàng đợi ở luồng riêng: với từng mục, hỏi lại trạng thái ngay lúc tới lượt
(mục đã có nhờ mục trước, như renderer sau môi trường chính, thì bỏ qua), mục mà mục nó cần chưa
có hoặc máy không cài được (không GPU, Python khác bản, không phải Linux) thì ghi lý do và đi
tiếp; mục lỗi không dừng cả hàng. Dừng thì dừng mục đang cài và bỏ các mục sau.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from drive2sim.engine import environment
from drive2sim.engine.machine import _gpus
from drive2sim.paths import PACKAGE_DIR

REPO_DIR = PACKAGE_DIR.parent
SCRIPTS = {
    "main": "envs/install_main.sh",
    "nurec": "envs/install_nurec.sh",
    "xsim": "envs/install_xsim.sh",
}
# Python mà Studio phải chạy bằng: môi trường chính cài vào chính Python đó (envs/main.yml);
# NuRec dựng venv riêng bằng uv, nên Python của Studio chỉ chạy script cài.
PYTHON: dict[str, Optional[tuple]] = {"main": (3, 10), "nurec": None, "xsim": None}
STEP = re.compile(r"^=== \d\d:\d\d:\d\d (\d+)/(\d+) (.*)$")
LOG_TAIL = 6000


class InstallFlow:
    """Chạy ``envs/install_<env>.sh`` ở nền và giữ trạng thái: bước, log cuối, kết quả.

    Mỗi lúc một lần cài. Trạng thái giữ trong bộ nhớ; log đầy đủ nằm trong file.
    """

    def __init__(
        self,
        log_dir: Path,
        python: str = sys.executable,
        repo: Path = REPO_DIR,
        popen: Callable[..., "subprocess.Popen[str]"] = subprocess.Popen,
        gpus: Callable[[], list] = _gpus,
        python_version: tuple = tuple(sys.version_info[:2]),
    ) -> None:
        self._log_dir = log_dir
        self._python = python
        self._repo = repo
        self._popen = popen
        self._gpus = gpus
        self._python_version = python_version
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._state: dict[str, Any] = {"state": "idle"}
        self._log = ""
        self._cancelled = False
        self._reader: Optional[threading.Thread] = None
        self._queue: Optional[threading.Thread] = None
        self._all: Optional[dict[str, Any]] = None
        self._all_stop = False
        self.echo: Optional[Callable[[str], None]] = None  # CLI: in log ra màn hình

    @property
    def python(self) -> str:
        """Python mà các lệnh cài (pip) dùng: Python đang chạy Studio."""
        return self._python

    def view(self) -> dict:
        """Trạng thái lần cài.

        Returns:
            ``{"state", "env", "python", "step", "steps", "label", "from_step", "started_at",
            "step_started_at", "finished_at", "log", "log_path"}``; ``state`` là ``idle``,
            ``running``, ``succeeded``, ``failed`` hoặc ``cancelled``.
        """
        with self._lock:
            view = {
                "state": "idle", "env": None, "python": self._python, "step": None,
                "steps": None, "label": None, "from_step": None, "started_at": None,
                "step_started_at": None, "finished_at": None, "log_path": None,
            }  # fmt: skip
            view.update(self._state)
            view["log"] = self._log[-LOG_TAIL:]
            view["all"] = None
            if self._all is not None:
                view["all"] = {**self._all, "items": [dict(i) for i in self._all["items"]]}
            return view

    def preflight(self, env: str) -> Path:
        """Kiểm tra cài được trên máy này; trả đường dẫn script.

        Raises:
            ValueError: Môi trường lạ, thiếu script, Python sai bản, hoặc không có GPU NVIDIA.
        """
        if env not in SCRIPTS:
            raise ValueError(f"không có môi trường {env}; có: {', '.join(SCRIPTS)}")
        if not sys.platform.startswith("linux"):
            raise ValueError(
                f"môi trường {env} chỉ cài được trên Linux có GPU NVIDIA (Windows: dùng WSL2); "
                f"{environment.REMOTE_HINT}"
            )
        script = self._repo / SCRIPTS[env]
        if not script.is_file():
            raise ValueError(
                f"không thấy {SCRIPTS[env]}: Studio phải chạy từ mã nguồn drive2sim (git clone) "
                "mới cài được môi trường"
            )
        want = PYTHON[env]
        if want is not None and tuple(self._python_version) != want:
            have = ".".join(map(str, self._python_version))
            raise ValueError(
                f"Studio đang chạy bằng Python {have} ({self._python}), môi trường {env} cần "
                f"Python {want[0]}.{want[1]}: tạo env bằng envs/{env}.yml rồi mở Studio bằng "
                "Python của env đó"
            )
        if not self._gpus():
            raise ValueError(
                f"máy này không có GPU NVIDIA (nvidia-smi): cài môi trường {env} xong vẫn không "
                f"train được; {environment.REMOTE_HINT}"
            )
        return script

    def start(self, env: str = "main", from_step: int = 1) -> dict:
        """Bắt đầu cài (đang cài thì giữ nguyên lần đang chạy).

        Args:
            env: Môi trường: một khoá của ``SCRIPTS``.
            from_step: Cài tiếp từ bước này (``--from``); 1 là cài từ đầu.

        Raises:
            ValueError: Theo ``preflight``.
        """
        if self.busy():
            return self.view()
        script = self.preflight(env)
        command = ["bash", str(script), "--python", self._python, "--from", str(int(from_step))]
        return self._launch(env, command, int(from_step))

    def start_command(self, name: str, command: list[str]) -> dict:
        """Chạy một lệnh cài (bảng thư viện) như một lần cài: log, dừng được, mỗi lúc một việc.

        Args:
            name: Tên mục (``env`` trong trạng thái, tên file log ``install-<name>.log``).
            command: Lệnh cài.
        """
        if self.busy():
            return self.view()
        return self._launch(name, command, 1)

    def busy(self) -> bool:
        """Đang có lần cài chạy, hoặc hàng "Cài tất cả" chưa xong."""
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            queued = self._queue is not None and self._queue.is_alive()
        return running or queued

    def start_all(self, item_ids: list[str]) -> dict:
        """Cài lần lượt các mục còn thiếu (đang cài thì giữ nguyên lần đang chạy).

        Args:
            item_ids: Mục của bảng thư viện theo thứ tự cài (``libraries.bulk_ids()``).

        Returns:
            Trạng thái như ``view``; ``all`` có ``state`` và từng mục ``{"id", "name",
            "result": "waiting" | "running" | "installed" | "present" | "skipped" | "failed" |
            "cancelled", "reason"}``.
        """
        from drive2sim.engine import libraries

        if self.busy():
            return self.view()
        with self._lock:
            self._all_stop = False
            self._all = {
                "state": "running", "current": None, "started_at": time.time(),
                "finished_at": None,
                "items": [
                    {"id": i, "name": libraries.BY_ID[i]["name"], "result": "waiting",
                     "reason": None}
                    for i in item_ids
                ],
            }  # fmt: skip
            self._queue = threading.Thread(target=self._run_all, daemon=True)
            self._queue.start()
        return self.view()

    def wait(self) -> dict:
        """Chờ lần cài (hoặc hàng "Cài tất cả") xong; trả trạng thái cuối."""
        for thread in (self._queue, self._reader):
            if thread is not None:
                thread.join()
        if self._queue is not None:
            self._queue.join()
        return self.view()

    def _set_item(self, index: int, result: str, reason: Optional[str] = None) -> None:
        with self._lock:
            assert self._all is not None
            self._all["items"][index].update(result=result, reason=reason)
            self._all["current"] = self._all["items"][index]["id"] if result == "running" else None

    def _run_all(self) -> None:
        from drive2sim.engine import libraries

        assert self._all is not None
        for index, entry in enumerate(list(self._all["items"])):
            if self._all_stop:
                self._set_item(index, "cancelled")
                continue
            item = libraries.BY_ID[entry["id"]]
            if libraries.item_status(item)["status"] == "ok":
                self._set_item(index, "present")
                continue
            missing = [
                libraries.BY_ID[r]["name"]
                for r in item.get("requires", [])
                if libraries.item_status(libraries.BY_ID[r])["status"] != "ok"
            ]
            if missing:
                self._set_item(index, "skipped", "cần " + ", ".join(missing) + " trước")
                continue
            try:
                plan = libraries.install_plan(entry["id"], self._python)
                if plan[0] == "env":
                    script = self.preflight(plan[1])
                    command = ["bash", str(script), "--python", self._python,
                               "--from", str(plan[2])]  # fmt: skip
                    self._launch(plan[1], command, plan[2])
                else:
                    self._launch(entry["id"], plan[1], 1)
            except (ValueError, OSError) as error:
                self._set_item(index, "skipped", str(error))
                continue
            self._set_item(index, "running")
            if self._reader is not None:
                self._reader.join()
            state = self.view()["state"]
            result = {"succeeded": "installed", "cancelled": "cancelled"}.get(state, "failed")
            self._set_item(index, result, None if result != "failed" else "xem log")
        with self._lock:
            results = {i["result"] for i in self._all["items"]}
            self._all["state"] = (
                "cancelled" if self._all_stop else "failed" if "failed" in results else "succeeded"
            )
            self._all["finished_at"] = time.time()
            self._all["current"] = None

    def _launch(self, env: str, command: list[str], from_step: int) -> dict:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"install-{env}.log"
        env_vars = dict(os.environ, PYTHONUNBUFFERED="1")
        with self._lock:
            self._process = self._popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                cwd=str(self._repo), env=env_vars, start_new_session=True,
            )  # fmt: skip
            now = time.time()
            self._state = {
                "state": "running", "env": env, "step": None, "steps": None, "label": None,
                "from_step": from_step, "started_at": now, "step_started_at": now,
                "finished_at": None, "log_path": str(log_path),
            }  # fmt: skip
            self._log = ""
            self._cancelled = False
            process = self._process
        self._reader = threading.Thread(target=self._read, args=(process, log_path), daemon=True)
        self._reader.start()
        return self.view()

    def _read(self, process: "subprocess.Popen[str]", log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(process.args)}\n")
            for line in process.stdout or ():
                log.write(line)
                log.flush()
                if self.echo is not None:
                    self.echo(line)
                with self._lock:
                    self._log = (self._log + line)[-4 * LOG_TAIL :]
                    found = STEP.match(line.strip())
                    if found:
                        self._state.update(
                            step=int(found.group(1)), steps=int(found.group(2)),
                            label=found.group(3), step_started_at=time.time(),
                        )  # fmt: skip
        code = process.wait()
        if self._python == sys.executable:
            from drive2sim.engine import libraries

            for stray in libraries.remove_stray_packages():
                note = f"đã xoá {stray}: gói cài nhầm vào site-packages, che thư mục của repo\n"
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(note)
                with self._lock:
                    self._log += note
        with self._lock:
            state = "cancelled" if self._cancelled else "succeeded" if code == 0 else "failed"
            self._state.update(state=state, finished_at=time.time())
        # PyTorch, NuRec vừa đổi: bảng kiểm tra môi trường phải hỏi lại, không dùng kết quả đã nhớ
        environment._cuda_cache.clear()
        from drive2sim.external import nurec, xsim

        nurec.clear_cache()
        xsim.clear_cache()

    def cancel(self) -> dict:
        """Dừng lần cài đang chạy (cả nhóm tiến trình: pip, nvcc...)."""
        with self._lock:
            self._all_stop = self._all is not None and self._all["state"] == "running"
            process = self._process
            if process is not None and process.poll() is None:
                self._cancelled = True
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (OSError, AttributeError):
                    process.terminate()
        return self.view()
