"""Tìm và đọc các file approach đi kèm package.

File approach nằm ở ``drive2sim/approaches/<approach_id>.yaml`` (CONTRACTS mục 4.1). File này
chứa hai hàm mà CLI và Studio dùng:
- ``list_approach_ids``: liệt kê các approach có sẵn (mặc định bỏ approach bị tắt).
- ``load_approach``: đọc một approach theo ``approach_id`` và kiểm tra theo schema.
- ``ApproachDisabledError``, ``require_enabled``: từ chối dùng approach bị tắt (``disabled``).
"""

from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import Approach
from drive2sim.engine.resolve import ApproachDisabledError, require_enabled  # noqa: F401
from drive2sim.engine.workdir import WorkdirError


def list_approach_ids(
    directory: Optional[Path] = None, include_disabled: bool = False
) -> list[str]:
    """Liệt kê ``approach_id`` của các file approach có sẵn.

    Args:
        directory: Thư mục chứa file approach; mặc định là ``drive2sim/approaches/``.
        include_disabled: Lấy cả approach bị tắt (``disabled``); mặc định bỏ, để Studio, CLI
            và Trợ lý không gợi ý chúng.

    Returns:
        Danh sách ``approach_id`` sắp theo chữ cái, ví dụ ``["omnire", "streetgs"]``.
    """
    directory = directory or paths.APPROACHES_DIR
    ids = sorted(path.stem for path in directory.glob("*.yaml"))
    if include_disabled:
        return ids
    return [i for i in ids if not files.load(directory / f"{i}.yaml", Approach).disabled]


def load_approach(approach_id: str, directory: Optional[Path] = None) -> Approach:
    """Đọc file approach theo ``approach_id`` và kiểm tra theo schema.

    Args:
        approach_id: Định danh approach, ví dụ ``"omnire"``.
        directory: Thư mục chứa file approach; mặc định là ``drive2sim/approaches/``.

    Returns:
        Approach đã kiểm tra.

    Raises:
        WorkdirError: Nếu không có file approach này; thông báo liệt kê các approach có sẵn.
        ValueError: Nếu file sai schema.
    """
    directory = directory or paths.APPROACHES_DIR
    path = directory / f"{approach_id}.yaml"
    if not path.is_file():
        known = (
            ", ".join(list_approach_ids(directory, include_disabled=True))
            or "(chưa có approach nào)"
        )
        raise WorkdirError(f"không có approach {approach_id!r}; các approach có sẵn: {known}")
    return files.load(path, Approach)
