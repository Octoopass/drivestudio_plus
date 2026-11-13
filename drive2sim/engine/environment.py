"""Kiểm tra môi trường Python của máy này có đúng môi trường mà approach cần không.

Approach dòng DriveStudio chạy trong môi trường chính, mô tả ở ``envs/main.yml`` và cài bằng
``envs/install_main.sh`` (cùng một script cho máy cá nhân và notebook Kaggle). Lệch phiên bản
PyTorch hay gsplat có thể vẫn chạy được nhưng cho kết quả khác đi, nên bảng kiểm tra tách ba mức:
``ok``, ``warn`` (có nhưng khác phiên bản: vẫn cho chạy, kết quả có thể lệch so với run khác) và
``missing`` (thiếu: run sẽ lỗi).

Chỉ kiểm cho run chạy trên máy này; run trên Kaggle cài môi trường trong notebook.

File này chứa ``EXPECTED``, ``environment_check``.
"""

import time
from typing import Any, Optional

from drive2sim.engine.machine import _gpus, package_versions
from drive2sim.engine.render_check import _torch_cuda

# Phiên bản của môi trường chính; phải khớp envs/main.yml và envs/install_main.sh (có test kiểm).
EXPECTED = {"main": {"pytorch": "2.4.1", "cuda": "12.1", "gsplat": "1.5.3"}}
# Codebase của approach → môi trường cần; codebase không có ở đây (ví dụ approach mẫu) thì không
# cần môi trường GPU nào.
CODEBASE_ENV = {"drivestudio": "main"}
# Hỏi PyTorch tốn vài giây (tiến trình con), nên giữ kết quả một lúc.
_CUDA_CACHE_S = 120
_cuda_cache: dict[str, Any] = {}


def _cached_torch_cuda() -> tuple[bool, str]:
    now = time.monotonic()
    if _cuda_cache and now - _cuda_cache["at"] < _CUDA_CACHE_S:
        return _cuda_cache["value"]
    value = _torch_cuda()
    _cuda_cache.update(at=now, value=value)
    return value


def _level(found: Optional[str], expected: str) -> str:
    if not found:
        return "missing"
    return "ok" if found.split("+")[0] == expected else "warn"


def environment_check(codebase: str) -> dict[str, Any]:
    """Bảng kiểm tra môi trường cho một codebase.

    Args:
        codebase: Trường ``codebase`` của file approach, ví dụ ``drivestudio``.

    Returns:
        ``{"env": tên hoặc None, "file", "install", "level": "ok" | "warn" | "missing",
        "checks": [{"id", "level", "found", "expected"}]}``. ``env`` là ``None`` khi codebase
        không cần môi trường GPU (``checks`` rỗng, ``level`` ``ok``).
    """
    env = CODEBASE_ENV.get(codebase)
    if env is None:
        return {"env": None, "file": None, "install": None, "level": "ok", "checks": []}
    expected = EXPECTED[env]
    versions = package_versions()
    gpus = _gpus()
    checks = [
        {
            "id": "gpu",
            "level": "ok" if gpus else "missing",
            "found": gpus[0][0] if gpus else None,
            "expected": None,
        },
        {
            "id": "pytorch",
            "level": _level(versions["pytorch"], expected["pytorch"]),
            "found": versions["pytorch"],
            "expected": expected["pytorch"],
        },
    ]
    if versions["pytorch"]:
        ok, info = _cached_torch_cuda()
        cuda_level = _level(versions["cuda"], expected["cuda"]) if ok else "missing"
        checks.append(
            {
                "id": "cuda",
                "level": cuda_level,
                "found": versions["cuda"] if ok else (info or None),
                "expected": expected["cuda"],
            }
        )
    else:
        checks.append(
            {"id": "cuda", "level": "missing", "found": None, "expected": expected["cuda"]}
        )
    checks.append(
        {
            "id": "gsplat",
            "level": _level(versions["gsplat"], expected["gsplat"]),
            "found": versions["gsplat"],
            "expected": expected["gsplat"],
        }
    )
    levels = {check["level"] for check in checks}
    level = "missing" if "missing" in levels else "warn" if "warn" in levels else "ok"
    return {
        "env": env,
        "file": f"envs/{env}.yml",
        "install": f"bash envs/install_{env}.sh",
        "level": level,
        "checks": checks,
    }
