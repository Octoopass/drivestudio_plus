"""XSIM chạy như một codebase ngoài của drive2sim (ARCHITECTURE mục 2.9).

XSIM (avision-team/XSIM) và renderer xsimgs chạy trong một venv riêng (Python 3.11, PyTorch 2.8
CUDA 12.8), gọi qua tiến trình con ``drive2sim.external.xsim.worker``. Mã gốc được giữ nguyên ở
revision chốt, chỉ thêm bản vá ``patches/runtime.patch``. Dữ liệu đưa vào XSIM là gói quan sát do
stage ``prep.observations`` xuất từ đường dữ liệu chung của tool (CONTRACTS mục 5.4), nên XSIM nhận
cùng ảnh, mask, pose, LiDAR, box và SMPL như các approach DriveStudio.

Nơi cài: biến ``D2S_XSIM_DIR``, mặc định ``~/.cache/drive2sim/xsim``; ``bash envs/install_xsim.sh``
dựng ``<thư mục>/XSIM`` (mã nguồn) và ``<thư mục>/.venv`` (Python của worker). Biến
``D2S_XSIM_PYTHON`` đặt thẳng Python của worker khi cài theo cách khác.

File này chỉ dùng thư viện chuẩn, để engine và Studio gọi được khi máy không có XSIM:
- ``REVISION``: commit XSIM đã chốt.
- ``WORKER_ENV``: biến môi trường cho tiến trình chạy Python của venv XSIM.
- ``OBSERVATIONS_INDEX``: tên file chỉ mục của gói quan sát.
- ``xsim_dir``, ``source_dir``, ``xsim_python``: đường dẫn của bản cài.
- ``environment_checks``: các mục kiểm tra cho ``engine.environment.environment_check``.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

# Tên file chỉ mục của gói quan sát; đặt ở đây để engine đọc được mà không cần numpy.
OBSERVATIONS_INDEX = "observations.json"
XSIM_DIR_ENV = "D2S_XSIM_DIR"
XSIM_PYTHON_ENV = "D2S_XSIM_PYTHON"
REVISION = "5c86a33a036f3163f78eff5dcd8d3fc75c05d2a3"
PYTHON_VERSIONS = ("3.11", "3.12")
# README của XSIM khuyến nghị GPU từ 24 GB để train; dưới mức này chỉ cảnh báo.
RECOMMENDED_VRAM_MB = 24 * 1024
# Biến môi trường cho mọi tiến trình chạy Python của venv XSIM. ``import xsim`` nạp
# matplotlib.pyplot; kernel Jupyter (Kaggle, Colab) đặt MPLBACKEND trỏ tới backend inline mà venv
# XSIM không có, nên matplotlib báo lỗi ngay khi import. Worker không vẽ lên màn hình: dùng Agg.
WORKER_ENV = {"MPLBACKEND": "Agg"}


def xsim_dir() -> Path:
    """Thư mục cài XSIM (``D2S_XSIM_DIR`` hoặc ``~/.cache/drive2sim/xsim``)."""
    default = Path.home() / ".cache" / "drive2sim" / "xsim"
    return Path(os.environ.get(XSIM_DIR_ENV, default)).expanduser()


def source_dir() -> Path:
    """Mã nguồn XSIM đã checkout (``<thư mục cài>/XSIM``)."""
    return xsim_dir() / "XSIM"


def xsim_python() -> Path:
    """Python của worker: ``D2S_XSIM_PYTHON`` nếu có, không thì ``<thư mục cài>/.venv``."""
    override = os.environ.get(XSIM_PYTHON_ENV)
    if override:
        return Path(override).expanduser()
    return xsim_dir() / ".venv" / "bin" / "python"


def _revision(repo: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


_PROBE = (
    "import json, sys, torch, toast, xsimgs._C, xsim, drive2sim, pathlib; "
    "print(json.dumps({'python': '%d.%d' % sys.version_info[:2], 'torch': torch.__version__, "
    "'cuda': torch.cuda.is_available(), "
    "'drive2sim': str(pathlib.Path(drive2sim.__file__).resolve().parent), "
    "'source': str(pathlib.Path(xsim.__file__).resolve().parents[2])}))"
)


def _probe(python: Path) -> Optional[dict]:
    """Thông tin runtime của venv worker, ``None`` nếu không chạy hoặc không import được."""
    try:
        result = subprocess.run(
            [str(python), "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env={**os.environ, **WORKER_ENV},
        )
        lines = result.stdout.strip().splitlines()
        if result.returncode != 0 or not lines:
            return None
        return json.loads(lines[-1])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


_CACHE_S = 120  # import torch và extension CUDA của venv mất vài giây: giữ kết quả một lúc
_cache: dict = {}


def clear_cache() -> None:
    """Bỏ kết quả kiểm tra đã nhớ (sau khi cài xong)."""
    _cache.clear()


def environment_checks() -> list[dict]:
    """Kiểm tra bản cài XSIM: mã nguồn đúng revision, venv import được runtime, đúng drive2sim.

    Returns:
        ``[{"id", "level", "found", "expected"}]`` với ``id`` là ``xsim_source`` và ``xsim_env``;
        ``level`` là ``ok``, ``warn`` hoặc ``missing``.
    """
    key = (str(xsim_dir()), str(xsim_python()))
    now = time.monotonic()
    if _cache.get("key") == key and now - _cache.get("at", -1e9) < _CACHE_S:
        return [dict(check) for check in _cache["value"]]
    value = _environment_checks()
    _cache.update(key=key, at=now, value=value)
    return [dict(check) for check in value]


def _environment_checks() -> list[dict]:
    python = xsim_python()
    runtime = _probe(python) if python.is_file() else None
    # Mã nguồn là bản worker thật sự import (đúng cả khi cài nơi khác qua D2S_XSIM_PYTHON);
    # chưa import được thì xét thư mục cài mặc định.
    repo = Path(runtime["source"]) if runtime else source_dir()
    revision = _revision(repo) if (repo / "setup.py").is_file() else None
    checks = [
        {
            "id": "xsim_source",
            "level": "missing" if not revision else "ok" if revision == REVISION else "warn",
            "found": f"{repo} @ {revision[:10]}" if revision else None,
            "expected": REVISION[:10],
        }
    ]
    level, found = "missing", None
    if runtime:
        here = Path(__file__).resolve().parents[2]
        found = (
            f"Python {runtime['python']}, torch {runtime['torch']}, CUDA {runtime['cuda']}, "
            f"drive2sim {runtime['drive2sim']}"
        )
        # drive2sim phải là bản cài editable của repo này, không thì worker chạy mã cũ.
        usable = runtime["python"] in PYTHON_VERSIONS and runtime["cuda"]
        level = "ok" if usable and Path(runtime["drive2sim"]) == here else "warn"
    checks.append(
        {
            "id": "xsim_env",
            "level": level,
            "found": found,
            "expected": "Python 3.11, torch có CUDA, toast, xsimgs, xsim, drive2sim editable",
        }
    )
    return checks
