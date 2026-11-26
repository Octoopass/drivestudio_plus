"""Chạy ví dụ NuRec của CARLA 1.0 (``nurec_runner.py``) với server render của drive2sim.

    ~/venvs/carla/bin/python -m drive2sim.sim.carla_nurec --usdz scene.usdz --render-port 46435
        [--carla-port 2000] [--out-dir frames] [--video out.mp4] [--max-seconds 20]
        [--ego replay|trajectory|autopilot] [--config demo.yaml]
        [--carla-root ~/opt/carla/Carla-1.0-Linux-Shipping]

``nurec_runner.py`` tự bật container NRE của NVIDIA bằng Docker rồi nối gRPC vào nó. Ở đây
``NuRecRenderService.start`` được thay để trỏ thẳng vào server đã chạy sẵn
(``python -m drive2sim.sim.server.sensorsim``), không đụng tới Docker; phần còn lại (dựng thế giới
từ ``map.xodr``, đặt xe ego theo quỹ đạo, camera theo ``T_sensor_rig``, tick đồng bộ, ghi ảnh) là
code của CARLA, không sửa.

Cần CARLA server đang chạy (``CarlaUnreal.sh -RenderOffScreen``). Không có màn hình thì pygame chạy
ở chế độ ``dummy``. ``--max-seconds`` dừng sau chừng ấy giây thời gian mô phỏng (mặc định: hết
quỹ đạo). ``--ego``: ``replay`` đặt xe ego đúng quỹ đạo gốc; ``trajectory`` (mặc định, như ví dụ
của CARLA) để vật lý CARLA lái xe bám quỹ đạo sau 1 giây; ``autopilot`` giao cho Traffic Manager.
Không có ``--config`` thì dùng cấu hình của ví dụ, tắt LiDAR (server chưa render LiDAR).
``--resolution-ratio``: tỉ lệ độ phân giải ảnh camera xin server render (mặc định 0.5; ví dụ của
CARLA tự chọn cỡ ô pygame, cỡ 0.25, nên ảnh mờ). ``--fps``: tick camera mỗi giây và FPS video.
``--video-height``: chiều cao mỗi ô camera trong video.
``--video``: ghép ảnh các camera (mỗi tick một hàng) thành MP4 H.264 sau khi chạy xong (cần
``--out-dir``).
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
from pathlib import Path

import yaml

DEFAULT_CARLA_ROOT = Path.home() / "opt" / "carla" / "Carla-1.0-Linux-Shipping"
NUREC_EXAMPLES = Path("PythonAPI") / "examples" / "nvidia" / "nurec"

log = logging.getLogger("drive2sim.carla_nurec")


def patch_render_service(render_host: str, render_port: int) -> None:
    """``NuRecRenderService`` dùng server có sẵn thay vì bật container NRE."""
    import nurec_render_service as service

    def start(self: object) -> dict:
        self.final_port = render_port
        self.container_running = True
        self.container_reused = True
        self.container_name = f"drive2sim-sensorsim@{render_host}:{render_port}"
        log.info("dùng server render drive2sim ở %s:%d", render_host, render_port)
        return {
            "container_name": self.container_name,
            "host": render_host,
            "port": render_port,
            "status": "external",
        }

    def stop(self: object) -> None:
        self.container_running = False

    service.NuRecRenderService.start = start
    service.NuRecRenderService.stop = stop


def patch_duration(max_seconds: float) -> None:
    """Dừng phát lại sau ``max_seconds`` giây mô phỏng."""
    import nurec_integration as integration

    original = integration.NurecScenario.is_done

    def is_done(self: object) -> bool:
        return original(self) or (self.running and self.seconds_since_start() >= max_seconds)

    integration.NurecScenario.is_done = is_done


def frames_to_video(frames_dir: Path, video: Path, fps: float = 30.0, height: int = 240) -> int:
    """Ghép ``<frames_dir>/<camera>/NNNNN.jpg`` thành MP4, mỗi khung là hàng ảnh các camera."""
    import cv2
    import numpy as np

    cameras = sorted(p for p in frames_dir.iterdir() if p.is_dir() and any(p.glob("*.jpg")))
    order = [
        "front_left",
        "front_right",
        "front",
        "fisheye_left",
        "cross_left",
        "fisheye_right",
        "cross_right",
    ]
    cameras.sort(
        key=lambda p: next((i for i, key in enumerate(order) if key in p.name), len(order))
    )
    files = {c.name: sorted(c.glob("*.jpg")) for c in cameras}
    count = min((len(v) for v in files.values()), default=0)
    if not count:
        raise RuntimeError(f"không có ảnh trong {frames_dir}")
    writer = None
    for index in range(count):
        tiles = []
        for name, paths in files.items():
            image = cv2.imread(str(paths[index]))
            image = cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
            cv2.putText(
                image,
                name,
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(image)
        row = np.concatenate(tiles, 1)
        row = row[:, : row.shape[1] // 2 * 2]
        if writer is None:
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (row.shape[1], row.shape[0])
            )
        writer.write(row)
    writer.release()
    to_h264(video)
    return count


def to_h264(video: Path, crf: int = 18) -> bool:
    """Đổi MP4 (mp4v của OpenCV) sang H.264 yuv420p để trình duyệt phát; lỗi thì giữ file cũ."""
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return False
    tmp = video.with_suffix(".h264.mp4")
    done = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
         "-movflags", "+faststart", str(tmp)],
        check=False,
    )  # fmt: skip
    if done.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(video)
    return True


def main(argv: list[str] | None = None) -> int:
    """Chạy ``nurec_runner.py`` của CARLA với server render drive2sim."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--usdz", required=True, type=Path)
    p.add_argument("--render-host", default="127.0.0.1")
    p.add_argument("--render-port", type=int, default=46435)
    p.add_argument("--carla-host", default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--out-dir", type=Path, default=None, help="ghi ảnh từng camera vào đây")
    p.add_argument("--video", type=Path, default=None, help="ghép ảnh thành MP4 (cần --out-dir)")
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--resolution-ratio", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--video-height", type=int, default=540)
    p.add_argument("--ego", choices=["replay", "trajectory", "autopilot"], default="trajectory")
    p.add_argument("--config", type=Path, default=None, help="file cấu hình demo của nurec_runner")
    p.add_argument(
        "--carla-root", type=Path, default=Path(os.environ.get("CARLA_ROOT", DEFAULT_CARLA_ROOT))
    )
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = a.carla_root.expanduser() / NUREC_EXAMPLES
    runner = examples / "nurec_runner.py"
    if not runner.is_file():
        raise SystemExit(f"không thấy {runner}: đặt --carla-root tới thư mục CARLA 1.0")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, str(examples))
    patch_render_service(a.render_host, a.render_port)
    if a.max_seconds > 0:
        patch_duration(a.max_seconds)

    args = [
        "nurec_runner.py",
        "--host",
        a.carla_host,
        "-p",
        str(a.carla_port),
        "-u",
        str(a.usdz.resolve()),
        "-np",
        str(a.render_port),
    ]
    if a.out_dir is not None:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        args += ["--saveimages", "-o", str(a.out_dir.resolve())]
    if a.video is not None and a.out_dir is None:
        raise SystemExit("--video cần --out-dir")
    config = a.config
    if config is None:  # cấu hình của ví dụ, tắt LiDAR, chọn cách lái xe ego
        config = (a.out_dir or Path.cwd()) / "drive2sim_carla.yaml"
        settings = {
            "lidar": {"enabled": False},
            "control": {"ego": a.ego},
            "cameras": {"resolution_ratio": a.resolution_ratio, "framerate": a.fps},
        }
        config.write_text(yaml.safe_dump(settings))
    args += ["-c", str(config.resolve())]
    sys.argv = args
    runpy.run_path(str(runner), run_name="__main__")
    if a.video is not None:
        count = frames_to_video(a.out_dir, a.video, fps=a.fps, height=a.video_height)
        log.info("ghi %s (%d khung)", a.video, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
