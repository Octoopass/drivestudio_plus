"""Công cụ vẽ mask thân xe trong Studio: bấm vài điểm lên ảnh trung vị, SAM2.1 tách mask.

Dataset không kèm mask thân xe thì người dùng tạo mask ở đây thay vì vẽ bằng phần mềm khác:

1. ``median_image``: ảnh trung vị của khoảng ``MEDIAN_FRAMES`` khung trải đều trong cảnh. Cảnh thay
   đổi nên bị xoá mờ; thân xe gắn cứng với camera nên hiện rõ, dễ bấm đúng chỗ.
2. ``segment``: các điểm người dùng bấm (thân xe / không phải) gửi cho SAM2.1 (``sam2_worker.py``,
   tiến trình riêng vì cần transformers mới hơn môi trường ``drive2sim``); mask nháp ghi ở
   ``reports/ego_mask/draft_<camera>.png``.
3. ``save``: chép mask nháp sang ``ego_masks/<camera>.png`` của cảnh, chỗ mọi loader đọc
   (``recon/data/base/pixel_source.py``, ``scene_ego_mask``). ``delete`` xoá mask của cảnh.

Model và Python của SAM2: ``D2S_SAM2_MODEL`` (mặc định ``<repo>/models/sam2.1-hiera-large``),
``D2S_SAM2_PYTHON`` (mặc định ``<repo>/models/sam2-venv/bin/python``). Thiếu một trong hai thì
``status`` báo chưa sẵn sàng và Studio hiện hướng dẫn cài; xem và xoá mask vẫn dùng được.

File này chứa ``EgoMaskError``, ``status``, ``median_image``, ``segment``, ``save``, ``delete``,
``mark_none``, ``overlay_png``.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.core.schema import Manifest
from drive2sim.ingest import ego_masks

MEDIAN_FRAMES = 40
TOOL_DIR = Path("reports") / "ego_mask"
SEGMENT_TIMEOUT_S = 300.0
REPO_DIR = paths.PACKAGE_DIR.parent


# Các route giữ khoá này suốt thao tác đọc-sửa-ghi để kiểm tra revision có hiệu lực.
EDIT_LOCK = threading.RLock()


def revision(scene_root: Path, camera_id: str) -> str:
    """Băm trạng thái đã lưu và nháp để phát hiện một tab đang sửa bản cũ.

    Args:
        scene_root: Thư mục cảnh.
        camera_id: Camera đã được kiểm tra trong manifest.

    Returns:
        Chuỗi SHA-256 dùng làm revision của editor.
    """
    digest = hashlib.sha256()
    for path in (
        scene_root / "ego_masks" / f"{camera_id}.png",
        scene_root / TOOL_DIR / f"draft_{camera_id}.png",
        scene_root / ego_masks.STATUS_FILE,
    ):
        digest.update(path.read_bytes() if path.is_file() else b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    """Thay một file bằng các byte hoàn chỉnh để reader không đọc trúng PNG đang ghi.

    Args:
        path: File đích.
        content: Nội dung đầy đủ.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def discard_draft(scene_root: Path, manifest: Manifest, camera_id: str) -> None:
    """Xoá nháp của camera khi người dùng bỏ các điểm, giữ nguyên mask chính thức.

    Args:
        scene_root: Thư mục cảnh.
        manifest: Manifest dùng kiểm tra camera.
        camera_id: Camera cần bỏ nháp.
    """
    _index(manifest, camera_id)
    (scene_root / TOOL_DIR / f"draft_{camera_id}.png").unlink(missing_ok=True)


class EgoMaskError(ValueError):
    """Không làm được: camera lạ, thiếu ảnh, SAM2 chưa cài hoặc lỗi."""


def sam2_model_dir() -> Path:
    """Thư mục weight SAM2.1 (``D2S_SAM2_MODEL``)."""
    return Path(os.environ.get("D2S_SAM2_MODEL", REPO_DIR / "models" / "sam2.1-hiera-large"))


def sam2_python() -> Path:
    """Python chạy ``sam2_worker.py`` (``D2S_SAM2_PYTHON``)."""
    default = REPO_DIR / "models" / "sam2-venv" / "bin" / "python"
    return Path(os.environ.get("D2S_SAM2_PYTHON", default))


def _index(manifest: Manifest, camera_id: str) -> int:
    ids = [camera.camera_id for camera in manifest.cameras]
    if camera_id not in ids:
        raise EgoMaskError(f"cảnh không có camera {camera_id}")
    return ids.index(camera_id)


def _frame_images(scene_root: Path, index: int) -> list[Path]:
    images = []
    for path in sorted((scene_root / "images").glob(f"*_{index}.*")):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            images.append(path)
    return images


def status(scene_root: Path, manifest: Manifest) -> dict:
    """Trạng thái mask thân xe của từng camera và SAM2 đã sẵn sàng chưa.

    Returns:
        ``{"sam2_ready": bool, "sam2_hint": str, "cameras": [{"camera_id", "state", "source",
        "has_mask", "coverage", "has_draft"}]}``: ``state``, ``source`` như
        ``ingest.ego_masks.camera_states``; ``has_mask``, ``coverage`` (tỉ lệ ảnh bị che) nói về
        mask trong cảnh, thứ công cụ này sửa được.
    """
    import cv2

    cameras = []
    for item in ego_masks.camera_states(scene_root, manifest):
        camera_id = str(item["camera_id"])
        path = scene_root / "ego_masks" / f"{camera_id}.png"
        coverage = None
        if path.is_file():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            coverage = float((mask > 127).mean()) if mask is not None else None
        draft = scene_root / TOOL_DIR / f"draft_{camera_id}.png"
        draft_mask = cv2.imread(str(draft), cv2.IMREAD_GRAYSCALE) if draft.is_file() else None
        cameras.append(
            {
                **item,
                "has_mask": path.is_file(),
                "coverage": coverage,
                "has_draft": draft_mask is not None,
                "draft_coverage": float((draft_mask > 127).mean())
                if draft_mask is not None
                else None,
                "revision": revision(scene_root, camera_id),
            }
        )
    missing = [str(p) for p in (sam2_python(), sam2_model_dir()) if not p.exists()]
    hint = "" if not missing else "chưa có " + ", ".join(missing) + " (xem envs/install_sam2.sh)"
    return {"sam2_ready": not missing, "sam2_hint": hint, "cameras": cameras}


def median_image(scene_root: Path, manifest: Manifest, camera_id: str) -> Path:
    """Ảnh trung vị của camera (JPEG, kích thước gốc), tính một lần rồi giữ trong ``reports/``.

    Raises:
        EgoMaskError: Camera lạ hoặc không có ảnh.
    """
    import cv2
    import numpy as np

    index = _index(manifest, camera_id)
    target = scene_root / TOOL_DIR / f"median_{camera_id}.jpg"
    if target.is_file():
        return target
    images = _frame_images(scene_root, index)
    if not images:
        raise EgoMaskError(f"không có ảnh của camera {camera_id}")
    step = max(1, len(images) // MEDIAN_FRAMES)
    stack = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in images[::step][:MEDIAN_FRAMES]]
    stack = [image for image in stack if image is not None]
    if not stack:
        raise EgoMaskError(f"không đọc được ảnh của camera {camera_id}")
    median = np.median(np.stack(stack), axis=0).astype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), median, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return target


class _Worker:
    """Tiến trình ``sam2_worker.py`` dùng chung cho cả Studio, khởi động khi cần lần đầu."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None

    def _start(self) -> subprocess.Popen:
        python, model = sam2_python(), sam2_model_dir()
        if not python.exists() or not model.exists():
            raise EgoMaskError(f"SAM2 chưa cài: cần {python} và {model} (envs/install_sam2.sh)")
        process = subprocess.Popen(
            [str(python), str(Path(__file__).with_name("sam2_worker.py")), str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        ready = self._read(process)
        if not ready.get("ready"):
            process.kill()
            raise EgoMaskError(f"SAM2 không khởi động được: {ready}")
        return process

    @staticmethod
    def _read(process: subprocess.Popen) -> dict:
        assert process.stdout is not None
        result: dict = {}

        def read() -> None:
            line = process.stdout.readline()
            result.update(json.loads(line) if line else {"ok": False, "error": "SAM2 đã thoát"})

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(SEGMENT_TIMEOUT_S)
        if thread.is_alive():
            process.kill()
            raise EgoMaskError("SAM2 không trả lời")
        return result

    def run(self, request: dict) -> dict:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.process = self._start()
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            return self._read(self.process)


_WORKER = _Worker()


def segment(
    scene_root: Path, manifest: Manifest, camera_id: str, points: list[tuple[float, float, int]]
) -> float:
    """Tách mask theo các điểm bấm trên ảnh trung vị, ghi mask nháp.

    Args:
        scene_root: Thư mục cảnh.
        manifest: Manifest của cảnh.
        camera_id: Camera.
        points: ``(x, y, nhãn)``, ``x``, ``y`` là pixel của ảnh gốc, nhãn 1 thân xe, 0 không phải.

    Returns:
        Tỉ lệ ảnh mask nháp che.

    Raises:
        EgoMaskError: Không có điểm thân xe, SAM2 chưa cài hoặc lỗi.
    """
    if not any(label == 1 for _, _, label in points):
        raise EgoMaskError("cần ít nhất một điểm trên thân xe")
    image = median_image(scene_root, manifest, camera_id)
    draft = scene_root / TOOL_DIR / f"draft_{camera_id}.png"
    with tempfile.TemporaryDirectory(dir=draft.parent) as directory:
        temporary = Path(directory) / "mask.png"
        result = _WORKER.run({"image": str(image), "points": points, "out": str(temporary)})
        if not result.get("ok"):
            raise EgoMaskError(f"SAM2 lỗi: {result.get('error')}")
        _write_atomic(draft, temporary.read_bytes())
    return float(result["coverage"])


def erase(
    scene_root: Path,
    manifest: Manifest,
    camera_id: str,
    points: list[tuple[float, float]],
    radius: int,
    source: Optional[str] = None,
) -> float:
    """Tẩy một nét trên mask nháp hoặc mask đã lưu, giữ bản đã lưu tới khi xác nhận.

    Args:
        scene_root: Thư mục cảnh.
        manifest: Manifest để xác nhận tên camera.
        camera_id: Tên camera cần sửa mask.
        points: Các điểm liên tiếp của nét tẩy, theo pixel ảnh gốc.
        radius: Bán kính cọ từ 1 đến 256 pixel.
        source: saved hoặc draft đúng overlay đang hiện; None giữ tương thích client cũ.

    Returns:
        Tỉ lệ pixel còn thuộc mask sau nét tẩy.

    Raises:
        EgoMaskError: Nếu thiếu mask, nét rỗng, kích thước cọ hoặc toạ độ không hợp lệ.
    """
    import math

    import cv2
    import numpy as np

    _index(manifest, camera_id)
    draft = scene_root / TOOL_DIR / f"draft_{camera_id}.png"
    if source not in (None, "saved", "draft"):
        raise EgoMaskError("nguồn mask phải là saved hoặc draft")
    path = (
        draft
        if source == "draft" or (source is None and draft.is_file())
        else scene_root / "ego_masks" / f"{camera_id}.png"
    )
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    if mask is None:
        raise EgoMaskError(f"camera {camera_id} chưa có mask để tẩy")
    if not points or not 1 <= radius <= 256:
        raise EgoMaskError("cần nét tẩy và bán kính từ 1 đến 256 pixel")
    height, width = mask.shape
    if any(
        not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height)
        for x, y in points
    ):
        raise EgoMaskError("điểm tẩy phải nằm trong ảnh")
    mask = (mask > 127).astype(np.uint8) * 255
    previous = None
    for x, y in points:
        point = (round(x), round(y))
        if previous is not None:
            cv2.line(mask, previous, point, 0, thickness=2 * radius)
        cv2.circle(mask, point, radius, 0, thickness=-1)
        previous = point
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise EgoMaskError("không ghi được mask nháp")
    _write_atomic(draft, encoded.tobytes())
    return float((mask > 127).mean())


def save(scene_root: Path, manifest: Manifest, camera_id: str) -> None:
    """Lưu bản nháp hoàn chỉnh thành mask chính thức bằng phép thay file nguyên tử.

    Args:
        scene_root: Thư mục cảnh.
        manifest: Manifest dùng kiểm tra camera.
        camera_id: Camera có nháp cần lưu vào ego_masks/<camera>.png.

    Raises:
        EgoMaskError: Nếu camera lạ hoặc không có nháp.
    """
    _index(manifest, camera_id)
    draft = scene_root / TOOL_DIR / f"draft_{camera_id}.png"
    if not draft.is_file():
        raise EgoMaskError(f"camera {camera_id} chưa có mask nháp")
    (scene_root / "ego_masks").mkdir(exist_ok=True)
    _write_atomic(scene_root / "ego_masks" / f"{camera_id}.png", draft.read_bytes())
    draft.unlink()
    ego_masks.set_state(scene_root, camera_id, None)  # có file mask là đủ, bỏ "none" cũ nếu có


def delete(scene_root: Path, manifest: Manifest, camera_id: str) -> None:
    """Xoá mask của cảnh, mask nháp và lựa chọn đã ghi: camera về lại trạng thái suy ra."""
    _index(manifest, camera_id)
    for path in (
        scene_root / "ego_masks" / f"{camera_id}.png",
        scene_root / TOOL_DIR / f"draft_{camera_id}.png",
    ):
        path.unlink(missing_ok=True)
    ego_masks.set_state(scene_root, camera_id, None)


def mark_none(scene_root: Path, manifest: Manifest, camera_id: str) -> None:
    """Xác nhận camera không thấy thân xe: xoá mask của cảnh (nếu có), ghi ``none``."""
    delete(scene_root, manifest, camera_id)
    ego_masks.set_state(scene_root, camera_id, "none")


def overlay_png(scene_root: Path, manifest: Manifest, camera_id: str, draft: bool) -> bytes:
    """Mask dạng PNG trong suốt (xanh lá: thân xe) để chồng lên ảnh trung vị trên trình duyệt.

    Raises:
        EgoMaskError: Không có mask (nháp hoặc của cảnh).
    """
    import cv2
    import numpy as np

    _index(manifest, camera_id)
    path = (
        scene_root / TOOL_DIR / f"draft_{camera_id}.png"
        if draft
        else scene_root / "ego_masks" / f"{camera_id}.png"
    )
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    if mask is None:
        raise EgoMaskError(f"camera {camera_id} chưa có mask")
    on = mask > 127
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[on] = (60, 220, 60, 120)  # BGRA
    edge = cv2.morphologyEx(on.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    rgba[edge] = (255, 0, 255, 255)
    ok, png = cv2.imencode(".png", rgba)
    if not ok:
        raise EgoMaskError("không mã hoá được ảnh mask")
    return png.tobytes()
