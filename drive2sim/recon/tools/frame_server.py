r"""Vẽ từng ảnh theo yêu cầu bằng renderer gốc của model, cho chế độ "Kết hợp" của trang Xem 3D.

Viewer GPU (``viewer.py``) vẽ liên tục theo camera của viser rồi gửi ảnh qua websocket, nên trên
GPU yếu phải hạ độ phân giải khi kéo. Tiến trình này nạp model một lần như ``viewer.py`` rồi chỉ
vẽ khi được hỏi: trình duyệt vẽ Gói xem trước khi người xem kéo, dừng tay thì gửi pose camera tới
đây và đè ảnh trả về lên. Ảnh vẽ bằng đúng hàm của viewer GPU (``make_viewer_render_fn``: mọi node,
trời, chỉnh màu Affine), nên giống ảnh trong video của stage train.

HTTP nội bộ (chỉ ``127.0.0.1``):
- ``GET /health``: ``{"ready": true, "num_frames": F}``.
- ``POST /render`` với JSON ``{"c2w": 16 số theo hàng (OpenCV, hệ world của DriveStudio),
  "K": 9 số theo hàng, "width", "height", "frame"}``: trả JPEG, header ``X-Render-Ms`` là thời gian
  vẽ.

    python -m drive2sim.recon.tools.frame_server --ckpt <checkpoint_final.pth> \\
        --config <config.yaml> --port 8090 --host 127.0.0.1 --output_dir <thư mục>
"""

import argparse
import io
import json
import os
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from drive2sim.recon.data.driving_dataset import DrivingDataset
from drive2sim.recon.tools.viewer import (
    GsplatRenderTabState,
    _dataset_camera_tools,
    _initial_render_settings_from_trainer,
    _set_frame_on_trainer,
    make_viewer_render_fn,
)
from drive2sim.recon.utils.misc import import_str

MAX_PIXELS = 1920 * 1080  # giới hạn cỡ ảnh mỗi lần vẽ
JPEG_QUALITY = 92


class _Camera:
    """Đủ cho ``viewer_render_fn``: ``c2w`` và ``get_K`` (K do trình duyệt gửi, đủ cx, cy)."""

    def __init__(self, c2w: np.ndarray, K: np.ndarray) -> None:
        self.c2w = c2w
        self._K = K

    def get_K(self, _size: tuple) -> np.ndarray:
        return self._K


def load(ckpt: str, config: str) -> tuple:
    """Nạp dữ liệu cảnh và model như ``viewer.main``; trả ``(trainer, dataset, số khung)``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(config)
    cfg.data.viewer = True
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=os.path.abspath(ckpt), load_only_model=True)
    trainer.set_eval()
    return trainer, dataset, int(dataset.num_img_timesteps), device


def make_renderer(
    trainer: object, dataset: DrivingDataset, num_frames: int, device: torch.device
) -> Callable[..., np.ndarray]:
    """Hàm ``render(c2w, K, width, height, frame) -> ảnh uint8`` dùng chung code với viewer GPU."""
    frame = {"value": 0}
    tools = _dataset_camera_tools(dataset)
    render_fn = make_viewer_render_fn(
        trainer,
        device,
        num_frames,
        frame_getter=lambda: frame["value"],
        image_context_getter=tools.get("image_context_getter"),
    )
    state = GsplatRenderTabState()
    for name, value in _initial_render_settings_from_trainer(trainer).items():
        setattr(state, name, value)
    state.selected_gaussian_types = tuple(trainer.gaussian_classes.keys())
    state.background_mode = "sky"
    state.render_mode = "rgb"
    state.preview_render = False
    lock = threading.Lock()

    def render(c2w: np.ndarray, K: np.ndarray, width: int, height: int, at: int) -> np.ndarray:
        with lock:  # một ảnh một lúc: model giữ khung hiện tại trong trạng thái
            frame["value"] = max(0, min(int(at), num_frames - 1))
            _set_frame_on_trainer(trainer, frame["value"])
            state.viewer_width, state.viewer_height = int(width), int(height)
            image = render_fn(_Camera(c2w.astype(np.float32), K.astype(np.float32)), state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return (np.clip(image, 0, 1) * 255).round().astype(np.uint8)

    return render


def serve(render: Callable[..., np.ndarray], num_frames: int, host: str, port: int) -> None:
    """Chạy HTTP server tới khi tiến trình bị dừng."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:  # không in mỗi request
            return

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"ready": True, "num_frames": num_frames})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/render":
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                width, height = int(body["width"]), int(body["height"])
                if width < 16 or height < 16 or width * height > MAX_PIXELS:
                    raise ValueError(f"cỡ ảnh {width}x{height} không hợp lệ")
                c2w = np.asarray(body["c2w"], dtype=np.float64).reshape(4, 4)
                K = np.asarray(body["K"], dtype=np.float64).reshape(3, 3)
                started = time.perf_counter()
                image = render(c2w, K, width, height, int(body.get("frame", 0)))
                elapsed_ms = (time.perf_counter() - started) * 1000
            except (KeyError, ValueError, TypeError) as error:
                return self._json(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001  (vẽ lỗi, ví dụ hết VRAM: báo, không sập)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._json(500, {"error": f"{type(error).__name__}: {error}"})
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=JPEG_QUALITY)
            data = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Render-Ms", f"{elapsed_ms:.0f}")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"frame server sẵn sàng ở http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    """Đọc tham số, nạp model, chạy server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--output_dir", default=".")  # cùng tham số với viewer.py, không dùng
    args = parser.parse_args()
    trainer, dataset, num_frames, device = load(args.ckpt, args.config)
    serve(make_renderer(trainer, dataset, num_frames, device), num_frames, args.host, args.port)


if __name__ == "__main__":
    main()
