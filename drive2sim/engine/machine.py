"""Đọc cấu hình máy đang chạy run và phiên bản các gói, để ghi vào ``d2s/env.json``.

Thời gian train, VRAM và tốc độ render phụ thuộc phần cứng, nên chỉ so được giữa các run chạy
trên cùng loại máy (CONTRACTS mục 6.4). File này đọc thông tin mà không cần nạp PyTorch: CPU và
RAM từ ``/proc`` (Linux), GPU qua lệnh ``nvidia-smi``, phiên bản gói từ metadata đã cài.

File này chứa:
- ``current_machine``: cấu hình phần cứng của máy đang chạy.
- ``package_versions``: phiên bản PyTorch, CUDA, gsplat.
- ``same_machine``: hai bản ghi có phải cùng một cấu hình máy không.
"""

import os
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

from drive2sim.core.schema import Machine

NVIDIA_SMI = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]


def _cpu_name() -> Optional[str]:
    """Tên CPU từ ``/proc/cpuinfo``; máy không phải Linux thì dùng ``platform.processor``."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return re.sub(r"\s+", " ", line.split(":", 1)[1]).strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> Optional[float]:
    """Dung lượng RAM, GB, từ dòng ``MemTotal`` của ``/proc/meminfo``."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / (1024 * 1024), 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gpus() -> list[tuple[str, Optional[int], Optional[str]]]:
    """Danh sách ``(tên, VRAM MB, driver)`` của các GPU NVIDIA; rỗng khi thiếu nvidia-smi."""
    try:
        result = subprocess.run(NVIDIA_SMI, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0]:
            continue
        memory = int(float(parts[1])) if re.fullmatch(r"[\d.]+", parts[1]) else None
        gpus.append((parts[0], memory, parts[2] or None))
    return gpus


def current_machine() -> Machine:
    """Cấu hình phần cứng của máy đang chạy; trường nào không đọc được thì để trống.

    Returns:
        Bản ghi ``Machine`` (chưa có ``first_seen_utc``).
    """
    gpus = _gpus()
    name, memory, driver = gpus[0] if gpus else (None, None, None)
    return Machine(
        cpu=_cpu_name(),
        cpu_cores=os.cpu_count(),
        ram_gb=_ram_gb(),
        gpu=name,
        gpu_count=len(gpus),
        gpu_memory_mb=memory,
        gpu_driver=driver,
        os=f"{platform.system()} {platform.release()}".strip() or None,
    )


def _version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def package_versions() -> dict[str, Optional[str]]:
    """Phiên bản PyTorch, CUDA mà PyTorch dùng, và gsplat, đọc mà không nạp PyTorch.

    CUDA lấy từ đuôi phiên bản của bánh xe PyTorch (``2.0.0+cu117`` → ``11.7``).

    Returns:
        ``{"pytorch", "cuda", "gsplat"}``; gói chưa cài thì giá trị là ``None``.
    """
    torch = _version("torch")
    cuda = None
    match = re.search(r"\+cu(\d+)(\d)$", torch or "")
    if match:
        cuda = f"{match.group(1)}.{match.group(2)}"
    return {"pytorch": torch, "cuda": cuda, "gsplat": _version("gsplat")}


def same_machine(a: Machine, b: Machine) -> bool:
    """Hai bản ghi có cùng cấu hình máy không (bỏ qua thời điểm ghi)."""
    keep = set(Machine.model_fields) - {"first_seen_utc"}
    return a.model_dump(include=keep) == b.model_dump(include=keep)
