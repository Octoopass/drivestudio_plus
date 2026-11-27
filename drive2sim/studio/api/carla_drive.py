"""Phiên lái thử trong CARLA cho trang Lái thử của Studio.

Một phiên gồm ba tiến trình trên máy chạy Studio: CARLA 1.0 không màn hình (chỉ bật nếu cổng 2000
chưa có ai nghe, và chỉ tắt nếu phiên đã bật nó), server render Gaussian
(``drive2sim.sim.server.sensorsim``) cho PLY mới nhất của run (``train.nurec_refine`` nếu có, không
thì ``train.nurec``), và ``drive2sim.sim.carla_drive`` (vòng tick, nhận lệnh lái, giữ ảnh mới
nhất). Studio bật chúng ở một luồng nền rồi chuyển tiếp lệnh lái và ảnh qua HTTP cục bộ; trình
duyệt chỉ nói chuyện với Studio. File cảnh ``scene.usdz`` lấy của ``render.carla`` nếu run đã
chạy stage đó, không thì dựng vào ``d2s/drive/``.

Mỗi lúc chỉ một phiên (CARLA chỉ có một thế giới); bật phiên mới thì phiên cũ bị tắt. Tắt Studio
thì tắt phiên.

File này chứa ``CarlaDrive``, ``CarlaDriveError``.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine.machine import gpu_usage
from drive2sim.engine.run import D2S_DIR, RUN_FILE
from drive2sim.engine.workdir import Workdir
from drive2sim.stages.commands import run_logged
from drive2sim.stages.prep.ncore import CLIP_FILE
from drive2sim.stages.render import carla as carla_stage
from drive2sim.stages.train.nurec import MERGED_PLY

DRIVE_DIR = "drive"
MODEL_STAGES = ("train.nurec_refine", "train.nurec")  # PLY mới nhất trước
START_TIMEOUT_S = 180.0
FRAME_KINDS = ("gaussian", "carla")
LOG_TAIL_LINES = 30


class CarlaDriveError(RuntimeError):
    """Không bật được phiên lái thử hoặc phiên chưa chạy."""


class CarlaDrive:
    """Quản lý phiên lái thử duy nhất của Studio."""

    def __init__(self, workdir: Workdir) -> None:
        """Tạo bộ quản lý rỗng cho thư mục làm việc ``workdir``."""
        self.workdir = workdir
        self.lock = threading.Lock()
        self.state = "idle"
        self.message = ""
        self.run_id: Optional[str] = None
        self.drive_dir: Optional[Path] = None
        self.http_port: Optional[int] = None
        self.processes: dict[str, object] = {}
        self.generation = 0  # tăng mỗi lần start/stop, để luồng bật cũ biết mình đã bị thay
        # GPU của phiên: server render (và CARLA nếu phiên này bật nó); CARLA đã chạy sẵn từ trước
        # thì không biết nó ở GPU nào.
        self.gpus: dict[str, Optional[int]] = {}
        self._usage: tuple[float, list[dict]] = (0.0, [])

    # ------------------------------------------------------------------ bật / tắt
    def start(
        self,
        run_id: str,
        camera: str = "",
        ratio: float = 0.5,
        fps: int = 20,
        objects: bool = True,
    ) -> dict:
        """Bật phiên lái thử cho ``run_id`` ở nền (tắt phiên cũ nếu có).

        ``objects`` bật thì xe, người từ box 3D của cảnh thành actor trong CARLA.

        Returns:
            Trạng thái như ``status()`` (``state`` là ``starting``).

        Raises:
            CarlaDriveError: Run không phải NuRec, thiếu PLY hoặc clip NCore, hoặc thiếu CARLA.
        """
        run_dir = self.workdir.find_run(run_id)
        record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
        ply = self.model_ply(run_dir, record)
        clip_dir = self.clip_dir(record)
        root, python = carla_stage.carla_root(), carla_stage.carla_python()
        if not (root / carla_stage.CARLA_SCRIPT).is_file():
            raise CarlaDriveError(f"không thấy CARLA 1.0 ở {root} ({carla_stage.CARLA_ROOT_ENV})")
        if not python.is_file():
            raise CarlaDriveError(
                f"không thấy Python có gói carla ở {python} ({carla_stage.CARLA_PYTHON_ENV})"
            )
        self.stop()
        with self.lock:
            self.generation += 1
            generation = self.generation
            self.state, self.message, self.run_id = "starting", "chuẩn bị", run_id
            self.drive_dir = run_dir / D2S_DIR / DRIVE_DIR
            self.drive_dir.mkdir(parents=True, exist_ok=True)
            for name in ("carla.log", "server.log", "drive.log", "scene.log"):
                (self.drive_dir / name).unlink(missing_ok=True)
        threading.Thread(
            target=self._start,
            args=(generation, run_dir, record, ply, clip_dir, camera, ratio, fps, objects),
            daemon=True,
        ).start()
        return self.status()

    def _start(
        self,
        generation: int,
        run_dir: Path,
        record: RunRecord,
        ply: Path,
        clip_dir: Path,
        camera: str,
        ratio: float,
        fps: int,
        objects: bool = True,
    ) -> None:
        """Luồng nền: dựng ``scene.usdz`` nếu cần, bật CARLA, server render và vòng lái."""
        assert self.drive_dir is not None
        out = self.drive_dir

        def cancelled() -> bool:
            return generation != self.generation

        try:
            # Dựng lại mỗi lần (vài giây): bản đồ, quỹ đạo và xe, người từ box của cảnh.
            usdz = out / carla_stage.SCENE_FILE
            self._set(generation, "dựng bản đồ, quỹ đạo, xe và người cho CARLA")
            clip = json.loads((clip_dir / CLIP_FILE).read_text(encoding="utf-8"))
            split = self.workdir.load_split(record.scene_id, record.split_id)
            sensors = [clip["sensors"][c] for c in split.cameras if c in clip["sensors"]]
            run_logged(
                carla_stage.scene_command(
                    clip_dir / clip["clip"],
                    usdz,
                    sensors,
                    carla_stage.CarlaParams(objects=objects),
                    self.workdir.scene_dir(record.scene_id),
                ),
                out / "scene.log",
                should_stop=cancelled,
            )
            gpu = carla_stage.pick_gpu()
            env = carla_stage.carla_env(gpu)
            with self.lock:
                self.gpus = {"render": gpu, "carla": None}
            if not carla_stage.port_open(2000):
                with self.lock:
                    self.gpus["carla"] = gpu
                self._set(generation, f"bật CARLA trên GPU {gpu} (lần đầu cỡ 30 giây)")
                self._keep(
                    generation,
                    "carla",
                    carla_stage.start_carla(
                        carla_stage.carla_root(), 2000, gpu, out / "carla.log", cancelled
                    ),
                )
            render_port = carla_stage.free_port()
            self._set(generation, f"nạp {ply.parent.name}/{ply.name} vào server render")
            self._keep(
                generation,
                "server",
                carla_stage.start_render_server(
                    carla_stage.carla_python(),
                    usdz,
                    ply,
                    render_port,
                    env,
                    out / "server.log",
                    cancelled,
                ),  # fmt: skip
            )
            http_port = carla_stage.free_port()
            command = [
                str(carla_stage.carla_python()), "-u", "-m", "drive2sim.sim.carla_drive",
                "--usdz", str(usdz), "--render-port", str(render_port),
                "--http-port", str(http_port), "--ratio", str(ratio), "--fps", str(fps),
                "--carla-root", str(carla_stage.carla_root()),
            ]  # fmt: skip
            if camera:
                command += ["--camera", camera]
            self._set(generation, "dựng thế giới CARLA và xe")
            drive = carla_stage._spawn(command, carla_stage.REPO_ROOT, env, out / "drive.log")
            self._keep(generation, "drive", drive)
            with self.lock:
                self.http_port = http_port
            deadline = time.monotonic() + START_TIMEOUT_S
            while True:
                if cancelled():
                    return
                if drive.poll() is not None:
                    raise CarlaDriveError("vòng lái thoát:\n" + self._tail(out / "drive.log"))
                info = self._get_json("/state")
                if info and info.get("phase") == "running":
                    break
                if info and info.get("phase") == "error":
                    raise CarlaDriveError(info.get("message") or "vòng lái lỗi")
                if time.monotonic() > deadline:
                    raise CarlaDriveError(f"vòng lái chưa sẵn sàng sau {START_TIMEOUT_S:.0f} giây")
                time.sleep(0.5)
            with self.lock:
                if generation == self.generation:
                    self.state, self.message = "running", ""
        except Exception as error:  # noqa: BLE001 - báo mọi lỗi lên trang Lái thử
            if cancelled():
                return
            self._shutdown()
            with self.lock:
                self.state, self.message = "error", str(error)

    def stop(self) -> dict:
        """Tắt phiên đang chạy (nếu có): vòng lái, server render, CARLA nếu phiên đã bật nó."""
        with self.lock:
            self.generation += 1
            port = self.http_port
        if port is not None:
            try:
                self._post("/stop", {}, timeout=2.0)
            except CarlaDriveError:
                pass
        self._shutdown()
        with self.lock:
            self.state, self.message = "idle", ""
        return self.status()

    def _shutdown(self) -> None:
        with self.lock:
            processes = dict(self.processes)
            self.processes.clear()
            self.http_port = None
        for name in ("drive", "server", "carla"):
            process = processes.get(name)
            if process is not None:
                carla_stage.stop_process(process)  # type: ignore[arg-type]

    def _keep(self, generation: int, name: str, process: object) -> None:
        with self.lock:
            stale = generation != self.generation
            if not stale:
                self.processes[name] = process
        if stale:  # phiên đã bị tắt trong lúc bật: đừng để tiến trình mồ côi
            carla_stage.stop_process(process)  # type: ignore[arg-type]
            raise CarlaDriveError("phiên đã bị tắt")

    def _set(self, generation: int, message: str) -> None:
        with self.lock:
            if generation == self.generation:
                self.message = message

    # ------------------------------------------------------------------ trạng thái, chuyển tiếp
    def status(self) -> dict:
        """Trạng thái phiên.

        Returns:
            ``state`` (idle, starting, running, error), ``run_id``, ``message``, ``info``
            (tốc độ, vị trí, khung/giây... khi đang chạy), ``log_tail``.
        """
        with self.lock:
            state, message, run_id = self.state, self.message, self.run_id
            drive = self.processes.get("drive")
            out = self.drive_dir
        info = None
        if state == "running":
            if drive is not None and drive.poll() is not None:  # type: ignore[attr-defined]
                tail = self._tail(out / "drive.log") if out else ""
                self._shutdown()
                with self.lock:
                    self.state, self.message = "error", "vòng lái đã thoát:\n" + tail
                state, message = self.state, self.message
            else:
                info = self._get_json("/state")
                if info is not None:
                    info["gpus"] = self._gpu_load()
        log_tail = ""
        if out is not None and state in ("starting", "error"):
            log_tail = "\n".join(
                self._tail(out / name, 8)
                for name in ("scene.log", "carla.log", "server.log", "drive.log")
                if (out / name).is_file()
            )
        return {
            "state": state,
            "run_id": run_id,
            "message": message,
            "info": info,
            "log_tail": log_tail,
        }

    def _gpu_load(self) -> list[dict]:
        """Tải các GPU phiên đang dùng (``nvidia-smi``, đọc lại tối đa mỗi giây một lần).

        Returns:
            Mỗi GPU một dict như ``gpu_usage`` thêm ``roles`` (``render``, ``carla``). Số đo là
            của cả GPU, gồm cả tiến trình khác chạy trên đó.
        """
        with self.lock:
            roles = dict(self.gpus)
            stamp, rows = self._usage
        if time.monotonic() - stamp > 1.0:
            rows = gpu_usage()
            with self.lock:
                self._usage = (time.monotonic(), rows)
        used: dict[int, list[str]] = {}
        for role, index in roles.items():
            if index is not None:
                used.setdefault(index, []).append(role)
        return [{**row, "roles": used[row["index"]]} for row in rows if row["index"] in used]

    def control(self, body: dict) -> None:
        """Chuyển lệnh lái tới vòng lái."""
        self._post("/control", body)

    def reset(self) -> None:
        """Đưa xe về đầu quỹ đạo."""
        self._post("/reset", {})

    def frame(self, kind: str) -> Optional[bytes]:
        """JPEG mới nhất của ``kind`` (``gaussian`` hoặc ``carla``), ``None`` nếu chưa có."""
        if kind not in FRAME_KINDS:
            raise CarlaDriveError(f"không có loại ảnh {kind!r}")
        port = self._port()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/frame/{kind}", timeout=3.0
            ) as reply:
                return reply.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise CarlaDriveError(f"vòng lái trả lỗi {error.code}") from None
        except (urllib.error.URLError, OSError) as error:
            raise CarlaDriveError(f"không nối được vòng lái: {error}") from None

    # ------------------------------------------------------------------ tiện ích
    def model_ply(self, run_dir: Path, record: RunRecord) -> Path:
        """PLY mới nhất của run (``train.nurec_refine`` rồi ``train.nurec``)."""
        done = {s.stage_id for s in record.stages if s.status == "succeeded"}
        for stage_id in MODEL_STAGES:
            ply = run_dir / D2S_DIR / "stages" / stage_id / MERGED_PLY
            if stage_id in done and ply.is_file():
                return ply
        raise CarlaDriveError("run chưa có PLY của Instant NuRec (train.nurec chưa xong)")

    def clip_dir(self, record: RunRecord) -> Path:
        """Thư mục clip NCore của run (kết quả ``prep.ncore`` trong cache)."""
        stage = next((s for s in record.stages if s.stage_id == "prep.ncore"), None)
        if stage is None or not stage.cache_key:
            raise CarlaDriveError("run không có clip NCore (prep.ncore): chỉ lái được run NuRec")
        path = self.workdir.cache_dir / stage.cache_key
        if not (path / CLIP_FILE).is_file():
            raise CarlaDriveError(f"không thấy {CLIP_FILE} trong {path}")
        return path

    def _port(self) -> int:
        with self.lock:
            port = self.http_port if self.state in ("starting", "running") else None
        if port is None:
            raise CarlaDriveError("chưa có phiên lái thử nào đang chạy")
        return port

    def _get_json(self, path: str) -> Optional[dict]:
        with self.lock:
            port = self.http_port
        if port is None:
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=1.5) as reply:
                return json.loads(reply.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _post(self, path: str, body: dict, timeout: float = 2.0) -> None:
        port = self._port()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout):
                pass
        except (urllib.error.URLError, OSError) as error:
            raise CarlaDriveError(f"không nối được vòng lái: {error}") from None

    @staticmethod
    def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
        if not path.is_file():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
