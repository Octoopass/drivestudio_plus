"""Bật viewer GPU của DriveStudio cho một run, để trang Xem 3D nhúng vào (mode "Render GPU").

Viewer là ``drive2sim/recon/tools/viewer.py`` (bản vá của Bosch, viser + nerfview): nạp checkpoint
bằng trainer của DriveStudio và vẽ bằng renderer gốc, nên thấy đủ mọi phần của model (nền, xe,
người, vật biến dạng, bầu trời), tự xoay camera, chọn khung, chọn loại Gaussian, xem độ sâu.
Studio chạy nó thành một tiến trình con, nghe trên ``127.0.0.1`` ở một cổng trống, rồi trang Xem
3D nhúng địa chỉ đó.

Viewer cần dữ liệu cảnh theo cấu trúc DriveStudio. Thư mục cảnh ảo mà stage train dựng
(``scene_view/``) có thể không dùng được trên máy này (run chạy trên Kaggle thì symlink trỏ tới
đường dẫn Kaggle), nên mỗi lần bật, Studio dựng lại một cảnh ảo trong ``d2s/viewer/`` từ thư mục
cảnh và cache trên máy, rồi ghi một bản ``config.yaml`` trỏ ``data.data_root`` vào đó.

Mỗi lúc chỉ chạy một viewer (GPU của máy cá nhân thường ít VRAM); bật viewer khác thì viewer cũ
bị tắt. Tắt Studio thì tắt viewer.

File này chứa ``GpuViewers``, ``GpuViewerError``.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Any, Optional

import yaml

from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine import gpu_process
from drive2sim.engine.registry import produces_artifact
from drive2sim.engine.render_check import CHECKPOINT, render_readiness
from drive2sim.engine.run import D2S_DIR, RUN_FILE
from drive2sim.engine.workdir import KAGGLE_SCENE_FILE, Workdir
from drive2sim.kaggle.scene_files import missing_view_files
from drive2sim.paths import SMPL_MODEL_ENV, find_smpl_model
from drive2sim.stages.scene_view import build_view
from drive2sim.stages.train.drivestudio import MASK_ENTRIES, SMPL_ENTRY

HOST = "127.0.0.1"
VIEWER_DIR = "viewer"
CONFIG = "config.yaml"
LOG = "viewer.log"
LOG_TAIL_LINES = 40
# Tiến trình GPU theo loại: viewer (viser, vẽ liên tục) hoặc frame server (vẽ một ảnh khi được hỏi,
# cho chế độ Kết hợp của trang Xem 3D). Mỗi lúc chỉ chạy một tiến trình: GPU nhỏ (4 GB) không giữ
# nổi hai bản model.
MODULES = {
    "viewer": "drive2sim.recon.tools.viewer",
    "frames": "drive2sim.recon.tools.frame_server",
}
RENDER_TIMEOUT_S = 30


class GpuViewerError(RuntimeError):
    """Không bật được viewer GPU: máy thiếu điều kiện hoặc run thiếu checkpoint, config."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def _listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((HOST, port)) == 0


def _tail(path: Path) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-LOG_TAIL_LINES:])


def prepare(workdir: Workdir, run_dir: Path) -> tuple[Path, Path, Path]:
    """Tìm checkpoint, dựng cảnh ảo trên máy này và ghi config cho viewer.

    Args:
        workdir: Thư mục làm việc.
        run_dir: Thư mục run.

    Returns:
        ``(checkpoint, config cho viewer, thư mục viewer)``.

    Raises:
        GpuViewerError: Nếu run không có stage train với ``checkpoint_final.pth`` và
            ``config.yaml``, cảnh trên máy thiếu pose, calib (cảnh ingest trên Kaggle chưa tải
            file nhỏ: ``d2s kaggle scene-files``), hoặc thiếu thư mục cảnh hay mask, SMPL trong
            cache.
    """
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    stages_dir = run_dir / D2S_DIR / "stages"
    train = next(
        (
            stages_dir / s.stage_id
            for s in record.stages
            if s.stage_id.startswith("train.") and (stages_dir / s.stage_id / CHECKPOINT).is_file()
        ),
        None,
    )
    if train is None:
        raise GpuViewerError(f"run {record.run_id} chưa có {CHECKPOINT}")
    if not (train / CONFIG).is_file():
        raise GpuViewerError(f"thư mục {train.name} không có {CONFIG} của DriveStudio")
    extra: dict[str, Path] = {}
    for stage in record.stages:
        if not stage.cache_key:
            continue
        folder = workdir.cache_dir / stage.cache_key
        names = MASK_ENTRIES if stage.stage_id.startswith("prep.masks") else ()
        if produces_artifact([stage.stage_id], "smpl"):
            names = (SMPL_ENTRY,)
        extra.update({name: folder / name for name in names if (folder / name).exists()})
    # Viewer không đọc ảnh, LiDAR nhưng cần pose, calib của cảnh; cảnh ingest trên Kaggle chỉ có
    # chúng trên máy sau khi tải file nhỏ (kaggle/scene_files.py).
    scene_dir = workdir.scene_dir(record.scene_id)
    missing = missing_view_files(scene_dir)
    if missing:
        hint = (
            "cảnh ingest trên Kaggle nên máy chưa có các file này; tải về bằng: "
            f"d2s kaggle scene-files {record.scene_id}"
            if (scene_dir / KAGGLE_SCENE_FILE).is_file()
            else "ingest lại cảnh"
        )
        raise GpuViewerError(
            f"cảnh {record.scene_id} trên máy thiếu {', '.join(missing)} để vẽ bằng renderer gốc; "
            + hint
        )
    out = run_dir / D2S_DIR / VIEWER_DIR
    out.mkdir(parents=True, exist_ok=True)
    try:
        build_view(
            workdir.scene_dir(record.scene_id),
            out / "scene_view",
            exclude=(*MASK_ENTRIES, SMPL_ENTRY),
            extra=extra,
        )
    except FileNotFoundError as error:
        raise GpuViewerError(f"không dựng được dữ liệu cảnh cho viewer: {error}") from None
    config = yaml.safe_load((train / CONFIG).read_text(encoding="utf-8"))
    config.setdefault("data", {})["data_root"] = str(out / "scene_view")
    (out / CONFIG).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return train / CHECKPOINT, out / CONFIG, out


class GpuViewers:
    """Giữ tối đa một tiến trình viewer GPU cho cả Studio.

    Args:
        workdir: Thư mục làm việc.
        python: Trình thông dịch chạy viewer; mặc định chính Python đang chạy Studio (cùng môi
            trường đã kiểm bằng ``render_readiness``).
    """

    def __init__(self, workdir: Workdir, python: str = sys.executable) -> None:
        self.workdir = workdir
        self.python = python
        self._lock = threading.Lock()
        # Cả lượt bật (dựng lại cảnh ảo bằng rmtree rồi mkdir, rồi chạy tiến trình) đi một lần một:
        # bấm "Bật viewer GPU" hai lần nhanh từng làm hai luồng cùng dựng thư mục đó và một bên
        # lỗi FileExistsError hoặc báo nhầm "không dựng được dữ liệu cảnh".
        self._start_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[IO[str]] = None
        self._run_id: Optional[str] = None
        self._port: Optional[int] = None
        self._log: Optional[Path] = None
        self._kind: Optional[str] = None

    def start(self, run_id: str, kind: str = "viewer") -> dict[str, Any]:
        """Tắt tiến trình đang chạy (nếu có) rồi bật ``kind`` (``viewer``, ``frames``) cho run.

        Frame server đang chạy đúng run đó thì giữ nguyên (khỏi nạp lại model).

        Returns:
            Trạng thái như ``status``.

        Raises:
            GpuViewerError: Nếu máy chưa đủ điều kiện render GPU hoặc run thiếu dữ liệu.
        """
        if kind not in MODULES:
            raise GpuViewerError(f"không có loại {kind}")
        with self._start_lock:
            current = self.status()
            if (
                kind == "frames"
                and current["kind"] == kind
                and current["run_id"] == run_id
                and current["state"] in ("starting", "running")
            ):
                return current
            return self._start(run_id, kind)

    def _start(self, run_id: str, kind: str) -> dict[str, Any]:
        run_dir = self.workdir.find_run(run_id)
        readiness = render_readiness(run_dir)
        if not readiness["ready"]:
            missing = ", ".join(c["id"] for c in readiness["checks"] if not c["ok"])
            raise GpuViewerError(f"máy chưa đủ điều kiện render GPU, còn thiếu: {missing}")
        checkpoint, config, out = prepare(self.workdir, run_dir)
        with self._lock:
            self._stop_locked()
            port = _free_port()
            log = out / LOG
            self._log_file = log.open("w", encoding="utf-8")
            command = [
                self.python,
                "-m",
                MODULES[kind],
                "--ckpt",
                str(checkpoint),
                "--config",
                str(config),
                "--port",
                str(port),
                "--host",
                HOST,
                "--output_dir",
                str(out),
            ]
            # Model có người (SMPL) nạp file mô hình SMPL theo biến D2S_SMPL_MODEL, mặc định là
            # đường dẫn tương đối "smpl_models/..." tính từ thư mục hiện tại (ở đây là ``out``)
            # nên không thấy. Tìm file như stage train rồi truyền đường dẫn tuyệt đối.
            env = dict(os.environ)
            smpl_model = find_smpl_model(self.workdir.root)
            if smpl_model is not None:
                env[SMPL_MODEL_ENV] = str(smpl_model)
            self._process = subprocess.Popen(  # noqa: S603  (danh sách tham số, không qua shell)
                command,
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(out),
            )
            self._run_id, self._port, self._log, self._kind = run_id, port, log, kind
            # Ghi lại để train trên máy (kể cả lệnh d2s ở tiến trình khác) tắt nó trước khi chạy.
            gpu_process.record(self.workdir, self._process.pid, MODULES[kind], run_id)
        return self.status()

    def render_frame(self, run_id: str, body: bytes) -> tuple[bytes, str]:
        """Chuyển yêu cầu vẽ một ảnh (JSON của ``frame_server``) tới frame server của run.

        Returns:
            ``(JPEG, thời gian vẽ ms)``.

        Raises:
            GpuViewerError: Nếu frame server của run này chưa chạy hoặc vẽ lỗi.
        """
        with self._lock:
            ready = (
                self._kind == "frames"
                and self._run_id == run_id
                and self._process is not None
                and self._process.poll() is None
            )
            port = self._port
        if not ready or port is None:
            raise GpuViewerError(f"frame server của run {run_id} chưa chạy")
        request = urllib.request.Request(
            f"http://{HOST}:{port}/render",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=RENDER_TIMEOUT_S) as response:  # noqa: S310
                return response.read(), response.headers.get("X-Render-Ms", "")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise GpuViewerError(f"frame server vẽ lỗi: {detail}") from None
        except (urllib.error.URLError, OSError) as error:
            raise GpuViewerError(f"frame server chưa vẽ được: {error}") from None

    def status(self) -> dict[str, Any]:
        """Trạng thái viewer.

        Returns:
            ``{"state": "stopped" | "starting" | "running" | "exited", "run_id", "url",
            "exit_code", "log"}``. ``starting``: tiến trình còn chạy nhưng chưa mở cổng (đang
            nạp checkpoint và dữ liệu); ``exited``: tiến trình đã dừng, ``log`` là phần cuối log.
        """
        with self._lock:
            if self._process is None:
                return {
                    "state": "stopped",
                    "kind": None,
                    "run_id": None,
                    "url": None,
                    "exit_code": None,
                    "log": "",
                }
            code = self._process.poll()
            if code is not None:
                state = "exited"
            elif self._port is not None and _listening(self._port):
                state = "running"
            else:
                state = "starting"
            return {
                "state": state,
                "kind": self._kind,
                "run_id": self._run_id,
                "url": f"http://{HOST}:{self._port}/" if state == "running" else None,
                "exit_code": code,
                "log": _tail(self._log) if self._log else "",
            }

    def stop(self) -> None:
        """Tắt viewer đang chạy, nếu có."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        process = self._process
        if process is not None:
            gpu_process.clear(self.workdir, process.pid)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self._log_file is not None:
            self._log_file.close()
        self._process = self._log_file = None
        self._run_id = self._port = None
