"""Instant NuRec của NVIDIA chạy như một codebase ngoài của drive2sim (ARCHITECTURE mục 2.9).

Instant NuRec dựng cảnh 3DGS bằng một mạng feed-forward (không train lại theo cảnh): đầu vào là
một clip NCore V4, đầu ra là PLY Gaussian của cả cảnh. Nó chạy trong môi trường riêng (mã nguồn
chốt revision, venv Python 3.11 do ``uv sync`` dựng, có ``nvidia-ncore``), gọi qua tiến trình con.
Các script chạy trong môi trường đó nằm ở ``tools/`` (lấy từ Fisheye Fusion Lab, không theo chuẩn
ruff của repo vì chạy bằng Python khác).

Nơi cài: biến ``D2S_NUREC_DIR``, mặc định ``~/.cache/drive2sim/instant-nurec``; cài bằng
``bash envs/install_nurec.sh`` hoặc nút cài trên Studio.

File này chứa:
- ``REVISION``, ``WEIGHTS``, ``MODELS``, ``SENSORS``: revision chốt, checkpoint, model, tên camera.
- ``nurec_dir``, ``nurec_python``, ``checkpoint_path``: đường dẫn của bản cài.
- ``environment_checks``: các mục kiểm tra môi trường NuRec cho ``environment_check``.

Đọc, ghi và đổi hệ PLY ở ``ply.py`` (cần numpy, chỉ nạp khi stage chạy).
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path(__file__).with_name("tools")
NUREC_DIR_ENV = "D2S_NUREC_DIR"
REVISION = "f2ecafcdf28f2dc8f2cc17447a772571cde558be"
# model → (file trong repo Hugging Face nvidia/instant-nurec, số byte); khớp tools/nurec_setup.py
WEIGHTS = {
    "pa-multiview": ("pth/instant_nurec_pa_multiview_1.1.0.pth", 730202407),
    "pa-front": ("pth/instant_nurec_pa_front_1.1.0.pth", 730200087),
    "pq-front": ("pth/instant_nurec_pq_road_1.0.0.pth", 1141125047),
}
MODELS = tuple(WEIGHTS)
# camera KITTI-360 → tên sensor trong clip NCore (tools/kitti360_to_ncore.py, bảng CAMERAS)
SENSORS = {
    "image_00": "camera_front_left_00",
    "image_01": "camera_front_right_01",
    "image_02": "camera_fisheye_left_02",
    "image_03": "camera_fisheye_right_03",
}


def nurec_dir() -> Path:
    """Thư mục mã nguồn Instant NuRec (``D2S_NUREC_DIR`` hoặc ``~/.cache/drive2sim/...``)."""
    default = Path.home() / ".cache" / "drive2sim" / "instant-nurec"
    return Path(os.environ.get(NUREC_DIR_ENV, default)).expanduser()


def nurec_python() -> Path:
    """Python của venv Instant NuRec (``<thư mục>/.venv/bin/python``)."""
    return nurec_dir() / ".venv" / "bin" / "python"


def checkpoint_path(model: str) -> Path:
    """Checkpoint của một model (``<thư mục>/checkpoints/pth/<file>``).

    Raises:
        ValueError: Nếu model lạ.
    """
    if model not in WEIGHTS:
        raise ValueError(f"model NuRec không có: {model}; chọn một trong {', '.join(MODELS)}")
    return nurec_dir() / "checkpoints" / WEIGHTS[model][0]


def _revision(repo: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _probe(python: Path) -> Optional[str]:
    """``<python> <torch> <cuda có hay không>`` của venv, hoặc ``None`` nếu không chạy được."""
    code = (
        "import sys, torch, ncore; print('%d.%d' % sys.version_info[:2], "
        "torch.__version__, torch.cuda.is_available())"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    return lines[-1] if result.returncode == 0 and lines else None


_CACHE_S = 120  # chạy thử venv NuRec mất vài giây: giữ kết quả một lúc
_cache: dict = {}


def clear_cache() -> None:
    """Bỏ kết quả kiểm tra đã nhớ (sau khi cài xong)."""
    _cache.clear()


def environment_checks() -> list[dict]:
    """Kiểm tra bản cài Instant NuRec: mã nguồn đúng revision, venv chạy được, có checkpoint.

    Kết quả được nhớ 2 phút theo thư mục cài.

    Returns:
        ``[{"id", "level", "found", "expected"}]`` với ``id`` là ``nurec_source``, ``nurec_env``,
        ``nurec_checkpoint``; ``level`` là ``ok``, ``warn`` hoặc ``missing``.
    """
    key = str(nurec_dir())
    now = time.monotonic()
    if _cache.get("key") == key and now - _cache.get("at", -1e9) < _CACHE_S:
        return [dict(check) for check in _cache["value"]]
    value = _environment_checks()
    _cache.update(key=key, at=now, value=value)
    return [dict(check) for check in value]


def _environment_checks() -> list[dict]:
    repo = nurec_dir()
    checks = []
    revision = _revision(repo) if (repo / "run_inference.py").is_file() else None
    checks.append(
        {
            "id": "nurec_source",
            "level": "missing" if not revision else "ok" if revision == REVISION else "warn",
            "found": f"{repo} @ {revision[:10]}" if revision else None,
            "expected": REVISION[:10],
        }
    )
    python = nurec_python()
    probe = _probe(python) if python.is_file() else None
    parts = (probe or "").split()
    env_level = "missing"
    if len(parts) == 3:
        env_level = "ok" if parts[0] == "3.11" and parts[2] == "True" else "warn"
    checks.append(
        {
            "id": "nurec_env",
            "level": env_level,
            "found": f"Python {parts[0]}, torch {parts[1]}, CUDA {parts[2]}" if parts else None,
            "expected": "Python 3.11, torch có CUDA, nvidia-ncore",
        }
    )
    have = [model for model in MODELS if checkpoint_path(model).is_file()]
    checks.append(
        {
            "id": "nurec_checkpoint",
            "level": "ok" if have else "missing",
            "found": ", ".join(have) or None,
            "expected": "ít nhất một trong " + ", ".join(MODELS),
        }
    )
    return checks
