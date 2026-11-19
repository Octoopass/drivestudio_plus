"""Bản xem trước của video run để trình duyệt phát được (CONTRACTS mục 11.1).

Video do ``train.py`` ghi ghép mọi camera vào một khung: KITTI-360 bốn camera là 4224x2304 điểm
ảnh, vượt mức H.264 mà trình duyệt giải mã được (khoảng 4096 điểm ảnh chiều ngang), nên thẻ
``<video>`` hiện khung đen. File này tạo một bản H.264 ``yuv420p`` rộng tối đa ``MAX_WIDTH``
điểm ảnh, có chỉ mục ở đầu file để phát khi đang tải, lưu ở ``d2s/previews/`` của run; video
gốc giữ nguyên để tải về.

Video đang được ghi (``train.py`` render cả đoạn sau bước cuối, mất vài phút) chưa có hộp ``moov``:
``is_complete`` trả ``False`` và Studio báo đang ghi thay vì phát.

File này chứa:
- ``is_complete``: video MP4 đã ghi xong chưa.
- ``preview_path``: đường dẫn bản xem trước của một video trong run.
- ``build_preview``: tạo (hoặc dùng lại) bản xem trước bằng ffmpeg.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

MAX_WIDTH = 1920
"""Chiều ngang tối đa của bản xem trước, điểm ảnh."""

PREVIEW_DIR = "previews"
"""Thư mục bản xem trước, nằm trong ``d2s/`` của run."""

_TAIL_BYTES = 1 << 20


class PreviewError(RuntimeError):
    """Không tạo được bản xem trước (video chưa ghi xong hoặc ffmpeg lỗi)."""


def is_complete(path: Path) -> bool:
    """Video MP4 đã ghi xong chưa, xét hộp ``moov`` (imageio ghi nó khi đóng file).

    Args:
        path: File MP4.

    Returns:
        ``True`` nếu thấy ``moov`` ở đầu hoặc cuối file.
    """
    try:
        size = path.stat().st_size
        if size < 64:
            return False
        with path.open("rb") as stream:
            head = stream.read(1 << 16)
            stream.seek(max(0, size - _TAIL_BYTES))
            tail = stream.read()
    except OSError:
        return False
    return b"moov" in head or b"moov" in tail


def preview_path(d2s_dir: Path, stage_id: str, name: str) -> Path:
    """Đường dẫn bản xem trước của video ``name`` thuộc stage ``stage_id``.

    Args:
        d2s_dir: Thư mục ``d2s/`` của run.
        stage_id: Stage chứa video.
        name: Đường dẫn video tương đối với thư mục của stage.

    Returns:
        ``<d2s_dir>/previews/<stage_id>/<name>``.
    """
    return d2s_dir / PREVIEW_DIR / stage_id / name


def _ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_preview(source: Path, target: Path) -> Path:
    """Tạo bản xem trước của ``source`` ở ``target``; dùng lại nếu đã có và mới hơn video gốc.

    Không có ffmpeg (môi trường Studio không cài ``imageio-ffmpeg``) thì trả về chính video gốc.

    Args:
        source: Video gốc.
        target: Nơi ghi bản xem trước.

    Returns:
        File để phát.

    Raises:
        PreviewError: Video gốc chưa ghi xong, hoặc ffmpeg lỗi.
    """
    if not is_complete(source):
        raise PreviewError("video đang được ghi, thử lại sau ít phút")
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        return source
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.stem}.partial.mp4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale='min({MAX_WIDTH},iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(partial),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise PreviewError(f"ffmpeg lỗi: {result.stderr.strip()[-300:]}")
    partial.replace(target)
    return target
