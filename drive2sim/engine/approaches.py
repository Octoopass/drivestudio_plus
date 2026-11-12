"""Tìm và đọc các file approach đi kèm package.

File approach nằm ở ``drive2sim/approaches/<approach_id>.yaml`` (CONTRACTS mục 4.1). File này
chứa hai hàm mà CLI và Studio dùng:
- ``list_approach_ids``: liệt kê các approach có sẵn.
- ``load_approach``: đọc một approach theo ``approach_id`` và kiểm tra theo schema.
"""

from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import Approach
from drive2sim.engine.workdir import WorkdirError


def list_approach_ids(directory: Optional[Path] = None) -> list[str]:
    """Liệt kê ``approach_id`` của mọi file approach có sẵn.

    Args:
        directory: Thư mục chứa file approach; mặc định là ``drive2sim/approaches/``.

    Returns:
        Danh sách ``approach_id`` sắp theo chữ cái, ví dụ ``["omnire", "streetgs"]``.
    """
    directory = directory or paths.APPROACHES_DIR
    return sorted(path.stem for path in directory.glob("*.yaml"))


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
        known = ", ".join(list_approach_ids(directory)) or "(chưa có approach nào)"
        raise WorkdirError(f"không có approach {approach_id!r}; các approach có sẵn: {known}")
    return files.load(path, Approach)
