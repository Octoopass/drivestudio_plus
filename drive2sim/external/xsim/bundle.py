"""Đọc gói quan sát do stage ``prep.observations`` xuất (CONTRACTS mục 5.4, ``observations``).

Worker XSIM chỉ đọc dữ liệu qua ``Observations``: ảnh, mask, pose, LiDAR, box và SMPL trong gói là
đúng những gì loader DriveStudio đưa cho OmniRe, nên hai bên nhận cùng đầu vào. File này chỉ dùng
numpy và Pillow để kiểm được trên CPU, không cần runtime XSIM.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from drive2sim.external.xsim import OBSERVATIONS_INDEX

FORMAT = "d2s-observations"
VERSION = 1


@dataclass(frozen=True)
class CameraInfo:
    """Calibration của một camera ở độ phân giải đã nạp.

    Attributes:
        index: Chỉ số camera trong manifest (cũng là số trong tên file ảnh).
        camera_id: Tên camera trong manifest.
        model: ``pinhole`` hoặc ``fisheye`` (Kannala–Brandt 4 hệ số, như OpenCV fisheye).
        width, height: Kích thước ảnh trong gói.
        intrinsics: ``fx, fy, cx, cy`` theo pixel.
        distortion: Hệ số méo: 4 cho mắt cá, 5 (k1, k2, p1, p2, k3) cho pinhole.
        train_mask, eval_mask: Gói có mask train, mask đánh giá cho camera này không.
    """

    index: int
    camera_id: str
    model: str
    width: int
    height: int
    intrinsics: tuple[float, float, float, float]
    distortion: tuple[float, ...]
    train_mask: bool
    eval_mask: bool


class Observations:
    """Gói quan sát trên đĩa, mở lười từng phần."""

    def __init__(self, root: Path) -> None:
        """Đọc ``observations.json`` và kiểm định dạng.

        Raises:
            ValueError: Nếu sai định dạng, phiên bản, hoặc thiếu trường bắt buộc.
        """
        self.root = Path(root)
        index = json.loads((self.root / OBSERVATIONS_INDEX).read_text("utf-8"))
        if index.get("format") != FORMAT or index.get("version") != VERSION:
            raise ValueError(f"không phải gói quan sát {FORMAT} v{VERSION}: {self.root}")
        self.index = index
        self.frames: list[int] = [int(f) for f in index["frames"]]
        stamps = [int(s) for s in index["timestamps_us"]]
        if len(stamps) != len(self.frames) or any(a >= b for a, b in zip(stamps, stamps[1:])):
            raise ValueError("timestamp của gói quan sát thiếu hoặc không tăng")
        self.timestamps_us = stamps
        self.cameras: dict[int, CameraInfo] = {}
        for item in index["cameras"]:
            camera = CameraInfo(
                index=int(item["index"]),
                camera_id=str(item["camera_id"]),
                model=str(item["model"]),
                width=int(item["width"]),
                height=int(item["height"]),
                intrinsics=tuple(float(v) for v in item["intrinsics"]),
                distortion=tuple(float(v) for v in item["distortion"]),
                train_mask=bool(item["train_mask"]),
                eval_mask=bool(item["eval_mask"]),
            )
            if camera.model not in ("pinhole", "fisheye"):
                raise ValueError(f"camera {camera.camera_id}: mô hình {camera.model} chưa hỗ trợ")
            self.cameras[camera.index] = camera
        self.lidar_frames: list[int] = [int(f) for f in index.get("lidar_frames", [])]
        self._position = {frame: t for t, frame in enumerate(self.frames)}

    @property
    def split(self) -> dict:
        """Split của run mà gói được xuất cho (dạng JSON của ``Split``)."""
        return self.index["split"]

    @property
    def camera_indices(self) -> dict[str, int]:
        """Tên camera của split → chỉ số manifest."""
        return {name: int(i) for name, i in self.index["camera_indices"].items()}

    def position(self, frame: int) -> int:
        """Vị trí của khung cảnh ``frame`` trong gói (0 là khung đầu)."""
        return self._position[frame]

    def time_s(self, frame: int) -> float:
        """Thời điểm của khung tính bằng giây, gốc là khung đầu của gói."""
        return (self.timestamps_us[self.position(frame)] - self.timestamps_us[0]) / 1e6

    def source_frame(self, frame: int) -> int:
        """Số hiệu khung trong dữ liệu gốc."""
        return int(self.index["source_frames"][self.position(frame)])

    @cached_property
    def _poses(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "cameras.npz") as data:
            return {key: data[key] for key in data.files}

    def cam_to_world(self, camera: int, frame: int) -> np.ndarray:
        """Pose camera → world (trục OpenCV) của camera ``camera`` tại khung ``frame``."""
        return self._poses[str(camera)][self.position(frame)]

    def image_path(self, frame: int, camera: int) -> Path:
        """Ảnh RGB của cặp (khung, camera)."""
        return self.root / "images" / f"{frame:03d}_{camera}.png"

    def sky_path(self, frame: int, camera: int) -> Optional[Path]:
        """Mask bầu trời của cặp, ``None`` nếu gói không có."""
        path = self.root / "sky" / f"{frame:03d}_{camera}.png"
        return path if path.is_file() else None

    def _mask(self, kind: str, camera: int) -> Optional[np.ndarray]:
        path = self.root / "masks" / f"{kind}_{camera}.png"
        if not path.is_file():
            return None
        with Image.open(path) as image:
            return np.asarray(image.convert("L")) > 127

    def train_mask(self, camera: int) -> np.ndarray:
        """Pixel được dùng khi train (H, W) bool; cả ảnh khi loader không che gì."""
        info = self.cameras[camera]
        mask = self._mask("train", camera)
        return mask if mask is not None else np.ones((info.height, info.width), dtype=bool)

    def eval_mask(self, camera: int) -> Optional[np.ndarray]:
        """Pixel dùng để chấm điểm, ``None`` khi loader chấm cả ảnh (như ``eval.standard``)."""
        return self._mask("eval", camera)

    def lidar(self, frame: int) -> dict[str, np.ndarray]:
        """Lượt quét tại khung: ``points``, ``origins`` (world, N×3), ``sensor_to_world`` (4×4)."""
        with np.load(self.root / "lidar" / f"{frame:03d}.npz") as data:
            return {key: data[key] for key in data.files}

    def instances(self) -> Optional[dict[str, np.ndarray]]:
        """Box theo loader: ``poses`` (F, I, 4, 4), ``sizes``, ``visible``, ``model_types``."""
        if not self.index.get("instances"):
            return None
        with np.load(self.root / "instances.npz") as data:
            return {key: data[key] for key in data.files}

    def smpl(self) -> Optional[dict[str, np.ndarray]]:
        """SMPL theo cột box: ``columns``, ``rotations`` (H, F, 24, 3, 3), ``betas``, ``valid``."""
        if not self.index.get("smpl"):
            return None
        with np.load(self.root / "smpl.npz") as data:
            return {key: data[key] for key in data.files}
