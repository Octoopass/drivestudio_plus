"""Đường dẫn tới các file đi kèm package ``drive2sim``.

Code tìm assets và configs qua các hằng ở đây, không dùng đường dẫn tương đối theo thư mục đang
chạy, để package chạy đúng dù được gọi từ đâu (ARCHITECTURE mục 3).
"""

import os
from pathlib import Path
from typing import Optional

# Thư mục của package drive2sim.
PACKAGE_DIR = Path(__file__).resolve().parent
# File tĩnh đi kèm: mask thân xe và danh sách cảnh của từng dataset.
ASSETS_DIR = PACKAGE_DIR / "assets"
EGO_MASKS_DIR = ASSETS_DIR / "ego_masks"
SCENE_LISTS_DIR = ASSETS_DIR / "scene_lists"
# Config method và config dataset của DriveStudio.
CONFIGS_DIR = PACKAGE_DIR / "configs"
# File khai báo approach: <approach_id>.yaml (CONTRACTS mục 4.1).
APPROACHES_DIR = PACKAGE_DIR / "approaches"

# Mô hình người SMPL (giới tính trung tính) mà OmniRe cần để dựng người. File có bản quyền nên
# không đi kèm repo; tải ở https://smpl.is.tue.mpg.de (cần tài khoản, đồng ý giấy phép).
SMPL_MODEL_ENV = "D2S_SMPL_MODEL"
SMPL_MODEL_NAME = "SMPL_NEUTRAL.pkl"
KAGGLE_INPUT = Path("/kaggle/input")


def find_smpl_model(workdir_root: Optional[Path] = None) -> Optional[Path]:
    """Tìm file mô hình SMPL, theo thứ tự ưu tiên.

    1. Biến môi trường ``D2S_SMPL_MODEL`` (đường dẫn tới file).
    2. ``<thư mục làm việc>/smpl_models/SMPL_NEUTRAL.pkl``.
    3. ``<repo>/smpl_models/SMPL_NEUTRAL.pkl`` (chỗ README của DriveStudio hướng dẫn đặt).
    4. Trên Kaggle: file cùng tên trong một dataset đã gắn dưới ``/kaggle/input``.

    Args:
        workdir_root: Thư mục làm việc, nếu có.

    Returns:
        Đường dẫn tuyệt đối, hoặc ``None`` nếu không thấy.
    """
    candidates = []
    if os.environ.get(SMPL_MODEL_ENV):
        candidates.append(Path(os.environ[SMPL_MODEL_ENV]).expanduser())
    if workdir_root is not None:
        candidates.append(Path(workdir_root) / "smpl_models" / SMPL_MODEL_NAME)
    candidates.append(PACKAGE_DIR.parent / "smpl_models" / SMPL_MODEL_NAME)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    if KAGGLE_INPUT.is_dir():
        found = sorted(KAGGLE_INPUT.glob(f"**/{SMPL_MODEL_NAME}"))
        if found:
            return found[0].resolve()
    return None
