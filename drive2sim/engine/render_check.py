"""Kiểm tra máy đang chạy Studio có render được model của một run bằng GPU không.

Mode "Render GPU" của trang Xem 3D vẽ cảnh từ checkpoint bằng renderer của DriveStudio, nên máy
chạy Studio cần: GPU NVIDIA, PyTorch có CUDA và gsplat trong môi trường Python của tool, và
checkpoint cuối của stage train. File này chỉ kiểm tra, không nạp model.

PyTorch được kiểm trong một tiến trình con (``python -c "import torch"``), để Studio không phải
nạp PyTorch (vài giây, vài trăm MB RAM) chỉ để hỏi GPU có dùng được không.

File này chứa ``render_readiness``.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from drive2sim.engine.machine import _gpus, package_versions

CHECKPOINT = "checkpoint_final.pth"
TORCH_PROBE = (
    "import torch; print(torch.cuda.is_available()); "
    "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
)


def _torch_cuda() -> tuple[bool, str]:
    """Hỏi PyTorch trong tiến trình con xem có dùng được CUDA không.

    Returns:
        ``(dùng được, thông tin)``; thông tin là tên GPU PyTorch thấy, hoặc lý do lỗi.
    """
    try:
        result = subprocess.run(  # noqa: S603  (danh sách tham số, không qua shell)
            [sys.executable, "-c", TORCH_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        last = (result.stderr.strip().splitlines() or ["không chạy được PyTorch"])[-1]
        return False, last
    return lines[0] == "True", lines[1] if len(lines) > 1 else ""


def _checkpoint(run_dir: Path) -> Optional[Path]:
    """Checkpoint cuối của stage train trong run, nếu có."""
    found = sorted((run_dir / "d2s" / "stages").glob(f"train.*/{CHECKPOINT}"))
    return found[0] if found else None


def render_readiness(run_dir: Optional[Path] = None) -> dict[str, Any]:
    """Kiểm tra từng điều kiện để render GPU, theo thứ tự nên sửa.

    Args:
        run_dir: Thư mục run cần xem; ``None`` thì bỏ qua bước kiểm tra checkpoint.

    Returns:
        ``{"ready": bool, "checks": [{"id", "ok", "value"}]}``. ``id`` là một trong ``gpu``,
        ``pytorch``, ``cuda``, ``gsplat``, ``checkpoint``; ``value`` là thứ đã thấy (tên GPU,
        phiên bản, đường dẫn) hoặc lý do lỗi, ``None`` nếu không có. Bước sau không chạy khi
        bước trước nó cần đã lỗi (không có PyTorch thì không hỏi CUDA).
    """
    checks: list[dict[str, Any]] = []
    gpus = _gpus()
    gpu = gpus[0] if gpus else None
    checks.append(
        {
            "id": "gpu",
            "ok": gpu is not None,
            "value": f"{gpu[0]}, {gpu[1]} MB, driver {gpu[2]}" if gpu else None,
            "memory_mb": gpu[1] if gpu else None,
        }
    )
    versions = package_versions()
    checks.append({"id": "pytorch", "ok": bool(versions["pytorch"]), "value": versions["pytorch"]})
    if versions["pytorch"]:
        ok, info = _torch_cuda()
        checks.append({"id": "cuda", "ok": ok, "value": info or None})
    else:
        checks.append({"id": "cuda", "ok": False, "value": None})
    checks.append({"id": "gsplat", "ok": bool(versions["gsplat"]), "value": versions["gsplat"]})
    if run_dir is not None:
        checkpoint = _checkpoint(run_dir)
        checks.append(
            {
                "id": "checkpoint",
                "ok": checkpoint is not None,
                "value": str(checkpoint.relative_to(run_dir)) if checkpoint else None,
            }
        )
    return {"ready": all(check["ok"] for check in checks), "checks": checks}
