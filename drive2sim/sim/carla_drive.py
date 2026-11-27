"""Lái tay xe ego trong CARLA 1.0, nhìn bằng ảnh Gaussian của cảnh NuRec (trang Lái thử).

    ~/venvs/carla/bin/python -m drive2sim.sim.carla_drive --usdz scene.usdz --render-port 46435
        --http-port 47000 [--carla-port 2000] [--camera camera_front_left_00] [--ratio 0.5]
        [--fps 20] [--carla-root ...]

Dựng thế giới và xe ego bằng ``NurecScenario`` của ví dụ NuRec trong CARLA (bản đồ ``map.xodr``,
xe đặt ở pose đầu của quỹ đạo, server render ``drive2sim.sim.server.sensorsim`` đã chạy sẵn ở
``--render-port``), rồi chạy vòng tick đồng bộ theo thời gian thật. Mỗi tick:

- lệnh lái mới nhất (ga, phanh, lái, lùi, phanh tay) đặt lên xe có vật lý của CARLA;
- camera ``--camera`` của cảnh vẽ ảnh Gaussian từ pose xe (tỉ lệ ``--ratio``, ``--fps`` ảnh/giây);
- một camera RGB của chính CARLA gắn sau xe chụp thế giới 3D của CARLA (đường, xe vật lý).

Không có màn hình: một server HTTP nhỏ ở ``127.0.0.1:--http-port`` cho Studio gọi:
``GET /state`` (tốc độ, vị trí, khung/giây), ``GET /frame/gaussian`` và ``GET /frame/carla`` (JPEG
mới nhất), ``POST /control`` (JSON ``throttle``, ``steer``, ``brake``, ``reverse``,
``hand_brake``), ``POST /reset`` (đưa xe về đầu quỹ đạo), ``POST /stop``. Không nhận lệnh lái
trong ``CONTROL_TIMEOUT_S`` giây (mất kết nối, đổi tab) thì nhả ga và phanh. Quỹ đạo gốc hết thì
vẫn lái tiếp được (thời gian kịch bản không dừng).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_CARLA_ROOT = Path.home() / "opt" / "carla" / "Carla-1.0-Linux-Shipping"
NUREC_EXAMPLES = Path("PythonAPI") / "examples" / "nvidia" / "nurec"
CONTROL_TIMEOUT_S = 1.0
CHASE_SIZE = (960, 540)
JPEG_QUALITY = 82

log = logging.getLogger("drive2sim.carla_drive")


class DriveState:
    """Dữ liệu dùng chung giữa vòng tick và server HTTP (khoá ``lock``)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0, "reverse": False,
                        "hand_brake": False}  # fmt: skip
        self.control_time = 0.0
        self.frames: dict[str, bytes] = {}
        self.frame_ids = {"gaussian": 0, "carla": 0}
        self.info: dict = {}
        self.reset = False
        self.stop = threading.Event()

    def set_frame(self, kind: str, jpeg: bytes) -> None:
        """Lưu JPEG mới nhất của ``kind``."""
        with self.lock:
            self.frames[kind] = jpeg
            self.frame_ids[kind] += 1


def encode_jpeg(bgr: object) -> bytes:
    """Ảnh BGR (numpy) thành JPEG."""
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def make_handler(state: DriveState) -> type[BaseHTTPRequestHandler]:
    """Lớp xử lý request HTTP cho ``state``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # không in mỗi request
            return

        def reply(self, code: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/state":
                with state.lock:
                    info = {**state.info, "frames": dict(state.frame_ids),
                            "control": dict(state.control)}  # fmt: skip
                self.reply(200, json.dumps(info).encode())
            elif self.path.startswith("/frame/"):
                kind = self.path.split("/")[2].split("?")[0]
                with state.lock:
                    jpeg = state.frames.get(kind)
                if jpeg is None:
                    self.reply(404, json.dumps({"error": "chưa có ảnh"}).encode())
                else:
                    self.reply(200, jpeg, "image/jpeg")
            else:
                self.reply(404, b"{}")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            if self.path == "/control":
                try:
                    data = json.loads(body or b"{}")
                except ValueError:
                    self.reply(400, json.dumps({"error": "JSON hỏng"}).encode())
                    return
                clamp = lambda v, lo, hi: max(lo, min(hi, float(v)))  # noqa: E731
                with state.lock:
                    c = state.control
                    c["throttle"] = clamp(data.get("throttle", 0), 0, 1)
                    c["brake"] = clamp(data.get("brake", 0), 0, 1)
                    c["steer"] = clamp(data.get("steer", 0), -1, 1)
                    c["reverse"] = bool(data.get("reverse", False))
                    c["hand_brake"] = bool(data.get("hand_brake", False))
                    state.control_time = time.monotonic()
                self.reply(200, b"{}")
            elif self.path == "/reset":
                with state.lock:
                    state.reset = True
                self.reply(200, b"{}")
            elif self.path == "/stop":
                state.stop.set()
                self.reply(200, b"{}")
            else:
                self.reply(404, b"{}")

    return Handler


def main(argv: list[str] | None = None) -> int:
    """Dựng thế giới, bật server HTTP, chạy vòng tick tới khi ``/stop`` hoặc Ctrl+C."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--usdz", required=True, type=Path)
    p.add_argument("--render-port", type=int, required=True)
    p.add_argument("--http-port", type=int, required=True)
    p.add_argument("--carla-host", default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--camera", default="", help="camera của cảnh (mặc định: camera đầu tiên)")
    p.add_argument("--ratio", type=float, default=0.5, help="tỉ lệ độ phân giải ảnh Gaussian")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--carla-root", type=Path, default=Path(os.environ.get("CARLA_ROOT", DEFAULT_CARLA_ROOT))
    )
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = a.carla_root.expanduser() / NUREC_EXAMPLES
    sys.path.insert(0, str(examples))
    import carla
    import numpy as np
    from nurec_integration import EGO_TRACK_ID, NurecScenario, mat_to_carla_transform

    from drive2sim.sim.carla_nurec import patch_render_service

    patch_render_service("127.0.0.1", a.render_port)
    state = DriveState()
    server = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with state.lock:
        state.info = {"phase": "loading", "message": "dựng thế giới CARLA từ map.xodr"}

    client = carla.Client(a.carla_host, a.carla_port)
    client.set_timeout(60.0)
    chase = None
    world = None
    try:
        with NurecScenario(
            client, str(a.usdz.resolve()), port=a.render_port, fps=a.fps
        ) as scenario:
            world = client.get_world()
            world.set_weather(carla.WeatherParameters.ClearNoon)
            cameras = scenario.get_available_cameras()
            front = [c for c in sorted(cameras) if "front" in c]  # mặc định: camera trước
            camera = a.camera if a.camera in cameras else (front or sorted(cameras))[0]

            def on_gaussian(image: np.ndarray) -> None:
                state.set_frame("gaussian", encode_jpeg(np.ascontiguousarray(image[..., ::-1])))

            scenario.add_camera(camera, on_gaussian, framerate=a.fps, resolution_ratio=a.ratio)
            ego = scenario.actor_mapping[EGO_TRACK_ID].actor_inst
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(CHASE_SIZE[0]))
            bp.set_attribute("image_size_y", str(CHASE_SIZE[1]))
            bp.set_attribute("fov", "90")
            chase = world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=-7.0, z=3.2), carla.Rotation(pitch=-14.0)),
                attach_to=ego,
            )

            def on_carla(image: carla.Image) -> None:
                bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
                bgra = bgra.reshape(image.height, image.width, 4)
                state.set_frame("carla", encode_jpeg(np.ascontiguousarray(bgra[..., :3])))

            chase.listen(on_carla)
            spawn = mat_to_carla_transform(
                scenario.t_scenario_carla @ scenario.scenario.ego_poses.poses[0]
            )
            # Hết quỹ đạo gốc thì kịch bản tự dừng và camera thôi vẽ: lái tay thì không dừng.
            scenario._check_done = lambda: None
            scenario.start_replay()
            tick_s = 1.0 / (2 * a.fps)
            for _ in range(3):  # camera chỉ vẽ khi thời gian kịch bản đã chạy
                scenario.tick()
            scenario.apply_ego_control(carla.VehicleControl(brake=1.0))
            ego.set_target_velocity(carla.Vector3D())
            log.info("sẵn sàng: camera %s, tỉ lệ %.2f, %d khung/giây", camera, a.ratio, a.fps)
            # Đếm trong cửa sổ 1 giây: ảnh Gaussian, ảnh camera CARLA, số tick mô phỏng.
            window_start = time.monotonic()
            counted = {"gaussian": 0, "carla": 0}
            fps = {"gaussian": 0.0, "carla": 0.0, "sim": 0.0}
            ticks = 0
            while not state.stop.is_set():
                began = time.monotonic()
                with state.lock:
                    c = dict(state.control)
                    stale = began - state.control_time > CONTROL_TIMEOUT_S
                    do_reset, state.reset = state.reset, False
                if stale:
                    c.update(throttle=0.0, brake=0.5)
                if do_reset:
                    ego.set_transform(spawn)
                    ego.set_target_velocity(carla.Vector3D())
                    ego.set_target_angular_velocity(carla.Vector3D())
                ego.apply_control(
                    carla.VehicleControl(
                        throttle=c["throttle"], steer=c["steer"], brake=c["brake"],
                        reverse=c["reverse"], hand_brake=c["hand_brake"],
                    )
                )  # fmt: skip
                scenario.tick()
                velocity = ego.get_velocity()
                location = ego.get_location()
                ticks += 1
                elapsed = time.monotonic() - window_start
                if elapsed >= 1.0:
                    with state.lock:
                        frames = dict(state.frame_ids)
                    for kind in counted:
                        fps[kind] = (frames[kind] - counted[kind]) / elapsed
                        counted[kind] = frames[kind]
                    fps["sim"] = ticks / elapsed
                    ticks, window_start = 0, time.monotonic()
                with state.lock:
                    state.info = {
                        "phase": "running",
                        "camera": camera,
                        "cameras": sorted(cameras),
                        "speed_kmh": round(3.6 * math.sqrt(
                            velocity.x**2 + velocity.y**2 + velocity.z**2), 1),
                        "location": [round(location.x, 1), round(location.y, 1),
                                     round(location.z, 1)],
                        "sim_time_s": round(scenario.seconds_since_start(), 1),
                        "render_fps": round(fps["gaussian"], 1),
                        "carla_fps": round(fps["carla"], 1),
                        "sim_fps": round(fps["sim"], 1),
                        "tick_ms": round(1000 * (time.monotonic() - began), 1),
                        "ratio": a.ratio,
                    }  # fmt: skip
                spare = tick_s - (time.monotonic() - began)
                if spare > 0:  # không chạy nhanh hơn thời gian thật
                    time.sleep(spare)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        log.exception("lái thử lỗi")
        with state.lock:
            state.info = {"phase": "error", "message": str(error)}
        time.sleep(2.0)  # cho Studio kịp đọc lỗi
        return 1
    finally:
        if chase is not None:
            try:
                chase.stop()
                chase.destroy()
            except RuntimeError:
                pass
        if world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except RuntimeError:
                pass
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
