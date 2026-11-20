"""Ghép video của hai run thành một file (trên dưới hoặc cạnh nhau, có nhãn) (CONTRACTS mục 11.1).

Trang So sánh đặt hai video cạnh nhau và phát đồng bộ trong trình duyệt; như Fisheye Fusion Lab,
người dùng còn cần một file MP4 ghép sẵn để tải về, gửi đi hoặc đưa vào báo cáo. File ghép bằng
ffmpeg (cùng ffmpeg với bản xem trước, ``preview.py``): hai video co về cùng chiều rộng (trên
dưới) hoặc cùng chiều cao (cạnh nhau), mỗi bên một nhãn, H.264, dừng ở video ngắn hơn. Kết quả
lưu ở ``<thư mục làm việc>/compare/<khoá>_<bố cục>.mp4``, khoá theo hai file nguồn và thời điểm
sửa, nên ghép lại cùng cặp thì dùng lại.

File này chứa:
- ``video_path``: video của một run theo ``stage_id`` và ``name`` (như ``GET .../videos``).
- ``compose``: ghép hai video.
- ``compare_video``: tìm hoặc tạo file ghép, trả tên file.
"""

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Literal

from drive2sim.studio.api import preview

COMPARE_DIR = "compare"
FILE_PATTERN = r"^[0-9a-f]{16}_(vstack|hstack)\.mp4$"
WIDTH = 1280  # trên dưới: chiều rộng chung
HEIGHT = 480  # cạnh nhau: chiều cao chung
Layout = Literal["vstack", "hstack"]


class CompareError(RuntimeError):
    """Không ghép được (không có ffmpeg, video đang ghi, ffmpeg lỗi)."""


def video_path(stages_dir: Path, stage_id: str, name: str) -> Path:
    """Video ``<stages_dir>/<stage_id>/<name>``, kiểm tra không ra ngoài thư mục stage.

    Raises:
        LookupError: Không có video đó.
    """
    stage_dir = (stages_dir / stage_id).resolve()
    source = (stage_dir / name).resolve()
    if stages_dir.resolve() not in stage_dir.parents or stage_dir not in source.parents:
        raise LookupError(f"không có video {stage_id}/{name}")
    if source.suffix.lower() != ".mp4" or not source.is_file():
        raise LookupError(f"không có video {stage_id}/{name}")
    return source


def _label(text: str) -> str:
    """Nhãn an toàn cho ``drawtext``: chỉ giữ chữ, số, ``_ . -`` và khoảng trắng."""
    return re.sub(r"[^A-Za-z0-9_. -]", "", text)[:60]


def _filter(layout: Layout, labels: tuple[str, str], with_text: bool) -> str:
    size = f"{WIDTH}:-2" if layout == "vstack" else f"-2:{HEIGHT}"
    parts = []
    for index, label in enumerate(labels):
        chain = f"[{index}:v]scale={size},setsar=1"
        if with_text and label:
            chain += (
                f",drawtext=text='{_label(label)}':x=12:y=12:fontsize=24:fontcolor=white"
                ":box=1:boxcolor=black@0.6:boxborderw=6"
            )
        parts.append(chain + f"[v{index}]")
    parts.append(f"[v0][v1]{layout}=inputs=2,scale=trunc(iw/2)*2:trunc(ih/2)*2[out]")
    return ";".join(parts)


def compose(a: Path, label_a: str, b: Path, label_b: str, out: Path, layout: Layout) -> Path:
    """Ghép ``a`` và ``b`` vào ``out``; ffmpeg không vẽ được chữ thì ghép không nhãn.

    Raises:
        CompareError: Không có ffmpeg, video đang được ghi, hoặc ffmpeg lỗi.
    """
    ffmpeg = preview._ffmpeg()
    if ffmpeg is None:
        raise CompareError("không có ffmpeg (cài imageio-ffmpeg vào môi trường chạy Studio)")
    for source in (a, b):
        if not preview.is_complete(source):
            raise CompareError(f"video {source.name} đang được ghi, thử lại sau ít phút")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f"{out.stem}.partial.mp4")
    last = ""
    for with_text in (True, False):
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(a), "-i", str(b),
            "-filter_complex", _filter(layout, (label_a, label_b), with_text),
            "-map", "[out]", "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(partial),
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
        if result.returncode == 0 and partial.is_file() and partial.stat().st_size:
            partial.replace(out)
            return out
        last = (result.stderr or "").strip()[-300:]
    partial.unlink(missing_ok=True)
    raise CompareError(f"ffmpeg không ghép được hai video: {last}")


def compare_video(
    workdir_root: Path, a: Path, label_a: str, b: Path, label_b: str, layout: Layout
) -> str:
    """Tên file ghép trong ``compare/`` (tạo nếu chưa có hoặc nguồn đã đổi).

    Raises:
        ValueError: Hai bên là cùng một video.
        CompareError: Theo ``compose``.
    """
    if a.resolve() == b.resolve():
        raise ValueError("chọn hai video khác nhau để ghép")
    seed = (
        "|".join(f"{path.resolve()}:{path.stat().st_mtime_ns}" for path in (a, b))
        + f"|{label_a}|{label_b}|{layout}"
    )
    name = f"{hashlib.sha1(seed.encode()).hexdigest()[:16]}_{layout}.mp4"
    out = workdir_root / COMPARE_DIR / name
    if not out.is_file():
        compose(a, label_a, b, label_b, out, layout)
    return name
