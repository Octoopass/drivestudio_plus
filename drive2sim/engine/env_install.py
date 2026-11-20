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
  (``start``) hoặc một lệnh của bảng thư viện (``start_command``, ``libraries.py``).
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
SCRIPTS = {"main": "envs/install_main.sh", "nurec": "envs/install_nurec.sh"}
# Python mà Studio phải chạy bằng: môi trường chính cài vào chính Python đó (envs/main.yml);
# NuRec dựng venv riêng bằng uv, nên Python của Studio chỉ chạy script cài.
PYTHON: dict[str, Optional[tuple]] = {"main": (3, 10), "nurec": None}
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
            return view

    def preflight(self, env: str) -> Path:
        """Kiểm tra cài được trên máy này; trả đường dẫn script.

        Raises:
            ValueError: Môi trường lạ, thiếu script, Python sai bản, hoặc không có GPU NVIDIA.
        """
        if env not in SCRIPTS:
            raise ValueError(f"không có môi trường {env}; có: {', '.join(SCRIPTS)}")
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
            env: Môi trường, hiện chỉ có ``main``.
            from_step: Cài tiếp từ bước này (``--from``); 1 là cài từ đầu.

        Raises:
            ValueError: Theo ``preflight``.
        """
        with self._lock:
            running = self._process is not None and self._process.poll() is None
        if running:
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
        with self._lock:
            running = self._process is not None and self._process.poll() is None
        if running:
            return self.view()
        return self._launch(name, command, 1)

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
        threading.Thread(target=self._read, args=(process, log_path), daemon=True).start()
        return self.view()

    def _read(self, process: "subprocess.Popen[str]", log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(process.args)}\n")
            for line in process.stdout or ():
                log.write(line)
                log.flush()
                with self._lock:
                    self._log = (self._log + line)[-4 * LOG_TAIL :]
                    found = STEP.match(line.strip())
                    if found:
                        self._state.update(
                            step=int(found.group(1)), steps=int(found.group(2)),
                            label=found.group(3), step_started_at=time.time(),
                        )  # fmt: skip
        code = process.wait()
        with self._lock:
            state = "cancelled" if self._cancelled else "succeeded" if code == 0 else "failed"
            self._state.update(state=state, finished_at=time.time())
        # PyTorch, NuRec vừa đổi: bảng kiểm tra môi trường phải hỏi lại, không dùng kết quả đã nhớ
        environment._cuda_cache.clear()
        from drive2sim.external import nurec

        nurec.clear_cache()

    def cancel(self) -> dict:
        """Dừng lần cài đang chạy (cả nhóm tiến trình: pip, nvcc...)."""
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                self._cancelled = True
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (OSError, AttributeError):
                    process.terminate()
        return self.view()
