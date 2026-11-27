"""Stage ``render.carla``: chạy cảnh Instant NuRec trong CARLA 1.0, ghi video ``carla.mp4``.

Các bước, đều trên máy chạy Studio:
1. ``ncore_carla_scene.py`` (môi trường NuRec) dựng ``scene.usdz`` từ clip NCore: bản đồ OpenDRIVE
   theo đoạn đường đã đi, quỹ đạo xe ego, camera của split.
2. Bật CARLA ở chế độ không màn hình (``CarlaUnreal.sh -RenderOffScreen``) nếu cổng ``carla_port``
   chưa có ai nghe; CARLA do stage bật thì stage tắt khi xong.
3. Bật server render ``drive2sim.sim.server.sensorsim`` với PLY của ``train.nurec`` ở cổng trống.
4. ``drive2sim.sim.carla_nurec`` chạy ví dụ NuRec của CARLA: xe ego chạy trong thế giới CARLA,
   mỗi tick CARLA hỏi server ảnh của từng camera; ảnh ghép thành ``carla.mp4`` (tab Video của
   Studio liệt kê mọi MP4 trong thư mục stage).

Cần CARLA 1.0 (``D2S_CARLA_ROOT``, mặc định ``~/opt/carla/Carla-1.0-Linux-Shipping``) và Python có
gói ``carla``, ``grpcio``, protobuf 5
(``D2S_CARLA_PYTHON``, mặc định ``~/venvs/carla/bin/python``).
Thêm vào run NuRec đã xong bằng nút "Chạy trong CARLA" trên Studio.

File này chứa ``CarlaParams``, các hàm bật CARLA và server render (``start_carla``,
``start_render_server``, dùng cả cho trang Lái thử của Studio) và ``RenderCarlaStage``.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageCancelledError, StageContext
from drive2sim.external import nurec
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.prep.ncore import CLIP_FILE
from drive2sim.stages.train.nurec import MERGED_PLY

CARLA_ROOT_ENV = "D2S_CARLA_ROOT"
CARLA_PYTHON_ENV = "D2S_CARLA_PYTHON"
DEFAULT_CARLA_ROOT = Path.home() / "opt" / "carla" / "Carla-1.0-Linux-Shipping"
DEFAULT_CARLA_PYTHON = Path.home() / "venvs" / "carla" / "bin" / "python"
CARLA_SCRIPT = "CarlaUnreal.sh"
SCENE_FILE = "scene.usdz"
VIDEO_FILE = "carla.mp4"
STATS_FILE = "carla.json"
FRAMES_DIR = "frames"
CARLA_LOG = "carla_server.log"
SERVER_LOG = "sensorsim.log"
CARLA_START_S = 180.0  # CARLA nạp shader lần đầu khá lâu
CARLA_SETTLE_S = 5.0  # cổng mở rồi vẫn cần chút thời gian mới nhận lệnh
SERVER_START_S = 120.0
REPO_ROOT = Path(__file__).resolve().parents[3]


class CarlaParams(BaseModel):
    """Tham số của ``render.carla``.

    Attributes:
        ego: Cách lái xe ego: ``replay`` đặt xe đúng quỹ đạo gốc; ``trajectory`` để vật lý CARLA
            lái bám quỹ đạo; ``autopilot`` giao cho Traffic Manager của CARLA.
        max_seconds: Dừng sau chừng ấy giây thời gian mô phỏng; 0 là chạy hết quỹ đạo.
        carla_port: Cổng RPC của CARLA; có CARLA đang nghe thì dùng luôn, không thì stage bật.
        rig_height: Độ cao gốc rig so với mặt đường (m), để đặt mặt đường của bản đồ.
        lane_width: Bề rộng một làn của bản đồ OpenDRIVE (m).
        resolution_ratio: Tỉ lệ độ phân giải ảnh camera so với ảnh gốc của cảnh (1 là đủ cỡ,
            nét nhất nhưng chậm nhất).
        fps: Số tick camera mỗi giây, cũng là FPS của video.
        video_height: Chiều cao (pixel) mỗi ô camera trong ``carla.mp4``.
        keep_frames: Giữ ảnh từng camera (``frames/``) sau khi ghép video.
        objects: Đưa xe, người từ box 3D của cảnh (``instances/``) vào CARLA làm actor có vật lý.
    """

    model_config = ConfigDict(extra="forbid")

    ego: Literal["replay", "trajectory", "autopilot"] = "trajectory"
    max_seconds: float = Field(default=30, ge=0, le=600)
    carla_port: int = Field(default=2000, ge=1024, le=65000)
    rig_height: float = Field(default=1.65, ge=0, le=5)
    lane_width: float = Field(default=3.5, ge=2, le=6)
    resolution_ratio: float = Field(default=0.5, ge=0.1, le=1)
    fps: float = Field(default=30, ge=5, le=60)
    video_height: int = Field(default=540, ge=120, le=1080)
    keep_frames: bool = False
    objects: bool = True


def carla_root() -> Path:
    """Thư mục CARLA 1.0 (``D2S_CARLA_ROOT`` hoặc mặc định)."""
    return Path(os.environ.get(CARLA_ROOT_ENV, DEFAULT_CARLA_ROOT)).expanduser()


def carla_python() -> Path:
    """Python có gói ``carla`` (``D2S_CARLA_PYTHON`` hoặc mặc định)."""
    return Path(os.environ.get(CARLA_PYTHON_ENV, DEFAULT_CARLA_PYTHON)).expanduser()


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Có tiến trình đang nghe ở ``host:port`` không."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def free_port() -> int:
    """Một cổng TCP đang trống trên máy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_process(process: Optional[subprocess.Popen], grace_s: float = 15.0) -> None:
    """Dừng cả nhóm tiến trình của ``process`` (SIGINT, rồi SIGKILL nếu quá ``grace_s``)."""
    if process is None or process.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGINT, grace_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def scene_command(
    clip_json: Path,
    out: Path,
    cameras: list[str],
    params: CarlaParams,
    scene_root: Optional[Path] = None,
) -> list[str]:
    """Câu lệnh ``ncore_carla_scene.py`` bằng Python của môi trường NuRec.

    ``scene_root`` (thư mục cảnh) có và ``params.objects`` bật thì xe, người lấy từ box của cảnh.
    """
    command = [
        str(nurec.nurec_python()), "-u", str(nurec.TOOLS_DIR / "ncore_carla_scene.py"),
        str(clip_json), "--out", str(out), "--cameras", ",".join(cameras),
        "--rig-height", str(params.rig_height), "--lane-width", str(params.lane_width),
    ]  # fmt: skip
    if scene_root is not None and params.objects:
        command += ["--scene-root", str(scene_root)]
    return command


def runner_command(usdz: Path, render_port: int, out_dir: Path, params: CarlaParams) -> list[str]:
    """Câu lệnh ``drive2sim.sim.carla_nurec`` (Python có gói ``carla``)."""
    command = [
        str(carla_python()), "-u", "-m", "drive2sim.sim.carla_nurec",
        "--usdz", str(usdz), "--render-port", str(render_port),
        "--carla-port", str(params.carla_port), "--carla-root", str(carla_root()),
        "--out-dir", str(out_dir / FRAMES_DIR), "--video", str(out_dir / VIDEO_FILE),
        "--ego", params.ego, "--resolution-ratio", str(params.resolution_ratio),
        "--fps", str(params.fps), "--video-height", str(params.video_height),
    ]  # fmt: skip
    if params.max_seconds > 0:
        command += ["--max-seconds", str(params.max_seconds)]
    return command


def pick_gpu() -> int:
    """GPU còn nhiều VRAM trống nhất (0 nếu không đọc được ``nvidia-smi``)."""
    from drive2sim.engine.machine import freest_gpu

    gpu = freest_gpu()
    return 0 if gpu is None else gpu


def carla_env(gpu: int) -> dict[str, str]:
    """Biến môi trường cho tiến trình Python của CARLA: GPU và ``PYTHONPATH`` tới repo."""
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p
        ),
    }


def wait_until(
    ready: Callable[[], bool],
    process: subprocess.Popen,
    timeout_s: float,
    what: str,
    log: Path,
    should_stop: Callable[[], bool] = lambda: False,
) -> None:
    """Chờ ``ready()`` đúng; lỗi nếu tiến trình chết, quá giờ hoặc bị huỷ (``should_stop``).

    Raises:
        StageCancelledError: Nếu ``should_stop()`` đúng.
        RuntimeError: Nếu tiến trình thoát hoặc quá ``timeout_s`` giây.
    """
    deadline = time.monotonic() + timeout_s
    while not ready():
        if should_stop():
            raise StageCancelledError(f"huỷ khi đang chờ {what}")
        if process.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            raise RuntimeError(f"{what} thoát (mã {process.returncode}):\n" + "\n".join(tail))
        if time.monotonic() > deadline:
            raise RuntimeError(f"{what} không lên sau {timeout_s:.0f} giây (xem {log.name})")
        time.sleep(1.0)


def _spawn(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> subprocess.Popen:
    """Chạy ``command`` trong nhóm tiến trình riêng, nối đầu ra vào ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(command) + "\n")
        f.flush()
        return subprocess.Popen(
            command,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
            start_new_session=True,
        )


def start_carla(
    root: Path,
    port: int,
    gpu: int,
    log: Path,
    should_stop: Callable[[], bool] = lambda: False,
) -> subprocess.Popen:
    """Bật ``CarlaUnreal.sh -RenderOffScreen`` trên ``gpu``, chờ cổng ``port`` mở."""
    command = [
        str(root / CARLA_SCRIPT), "-RenderOffScreen", "-nosound",
        f"-carla-rpc-port={port}", f"-graphicsadapter={gpu}",
    ]  # fmt: skip
    process = _spawn(command, root, {"CUDA_VISIBLE_DEVICES": str(gpu)}, log)
    try:
        wait_until(lambda: port_open(port), process, CARLA_START_S, "CARLA", log, should_stop)
        time.sleep(CARLA_SETTLE_S)
    except BaseException:
        stop_process(process)
        raise
    return process


def start_render_server(
    python: Path,
    usdz: Path,
    ply: Path,
    port: int,
    env: dict[str, str],
    log: Path,
    should_stop: Callable[[], bool] = lambda: False,
) -> subprocess.Popen:
    """Bật ``drive2sim.sim.server.sensorsim`` cho ``ply``, chờ tới khi nó in ``Serving on``."""
    command = [
        str(python), "-u", "-m", "drive2sim.sim.server.sensorsim",
        "--usdz", str(usdz), "--ply", str(ply), "--port", str(port),
        "--carla-root", str(carla_root()),
    ]  # fmt: skip
    start = log.stat().st_size if log.is_file() else 0
    process = _spawn(command, REPO_ROOT, env, log)

    def serving() -> bool:
        with log.open("rb") as f:
            f.seek(start)
            return b"Serving on" in f.read()

    try:
        wait_until(serving, process, SERVER_START_S, "server render", log, should_stop)
    except BaseException:
        stop_process(process)
        raise
    return process


@register_stage("render.carla")
class RenderCarlaStage(Stage):
    """Chạy cảnh NuRec trong CARLA (server render drive2sim), ghi ``carla.mp4``."""

    version: ClassVar[str] = "0.3.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model", "clip": "ncore_clip", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"videos": "videos"}
    Params: ClassVar[type[BaseModel]] = CarlaParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Dựng ``scene.usdz``, bật CARLA và server render, chạy mô phỏng, ghép video.

        Returns:
            ``{"videos": <thư mục stage>}``.

        Raises:
            RuntimeError: Thiếu CARLA hoặc Python của CARLA, clip không có camera của split,
                CARLA hay server không lên, hoặc một bước lỗi.
        """
        assert isinstance(params, CarlaParams)
        root, python = carla_root(), carla_python()
        if not (root / CARLA_SCRIPT).is_file():
            raise RuntimeError(f"không thấy CARLA 1.0 ở {root}: đặt {CARLA_ROOT_ENV}")
        if not python.is_file():
            raise RuntimeError(
                f"không thấy Python có gói carla ở {python}: đặt {CARLA_PYTHON_ENV}"
            )
        clip = json.loads((inputs["clip"] / CLIP_FILE).read_text(encoding="utf-8"))
        sensors = [clip["sensors"][c] for c in ctx.split.cameras if c in clip["sensors"]]
        if not sensors:
            raise RuntimeError("clip NCore không có camera nào của split")
        out = ctx.output_dir
        log = out / LOG_FILE
        usdz = out / SCENE_FILE
        ctx.progress(0.02, f"dựng bản đồ và quỹ đạo cho CARLA ({len(sensors)} camera)")
        run_logged(
            scene_command(inputs["clip"] / clip["clip"], usdz, sensors, params, ctx.scene.root),
            log,
            should_stop=ctx.should_stop,
        )
        gpu = pick_gpu()
        env = carla_env(gpu)
        carla_process = server = None
        began = time.monotonic()
        try:
            if port_open(params.carla_port):
                ctx.progress(0.1, f"dùng CARLA đang chạy ở cổng {params.carla_port}")
            else:
                ctx.progress(0.1, f"bật CARLA (GPU {gpu})")
                carla_process = start_carla(
                    root, params.carla_port, gpu, out / CARLA_LOG, ctx.should_stop
                )
            render_port = free_port()
            ctx.progress(0.2, f"bật server render ở cổng {render_port}")
            server = start_render_server(python, usdz, inputs["model"] / MERGED_PLY,
                                         render_port, env, out / SERVER_LOG,
                                         ctx.should_stop)  # fmt: skip
            ctx.progress(0.3, f"mô phỏng trong CARLA (xe ego: {params.ego})")
            run_logged(
                runner_command(usdz, render_port, out, params),
                log,
                cwd=REPO_ROOT,
                should_stop=ctx.should_stop,
                env=env,
            )
        finally:
            stop_process(server)
            stop_process(carla_process)
            if not params.keep_frames:
                shutil.rmtree(out / FRAMES_DIR, ignore_errors=True)
        if not (out / VIDEO_FILE).is_file():
            raise RuntimeError(f"mô phỏng xong nhưng không có {VIDEO_FILE}")
        (out / STATS_FILE).write_text(
            json.dumps(
                {
                    "ego": params.ego,
                    "max_seconds": params.max_seconds,
                    "resolution_ratio": params.resolution_ratio,
                    "fps": params.fps,
                    "video_height": params.video_height,
                    "cameras": sensors,
                    "gpu": gpu,
                    "carla_started_by_stage": carla_process is not None,
                    "wall_seconds": round(time.monotonic() - began, 1),
                    "video_mb": round((out / VIDEO_FILE).stat().st_size / 2**20, 1),
                },
                indent=2,
            )
            + "\n"
        )
        return {"videos": out}
