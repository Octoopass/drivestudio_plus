"""Đường dẫn tới các file đi kèm package ``drive2sim``.

Code tìm assets và configs qua các hằng ở đây, không dùng đường dẫn tương đối theo thư mục đang
chạy, để package chạy đúng dù được gọi từ đâu (ARCHITECTURE mục 3).
"""

from pathlib import Path

# Thư mục của package drive2sim.
PACKAGE_DIR = Path(__file__).resolve().parent
# File tĩnh đi kèm: mask thân xe và danh sách cảnh của từng dataset.
ASSETS_DIR = PACKAGE_DIR / "assets"
EGO_MASKS_DIR = ASSETS_DIR / "ego_masks"
SCENE_LISTS_DIR = ASSETS_DIR / "scene_lists"
# Config method và config dataset của DriveStudio.
CONFIGS_DIR = PACKAGE_DIR / "configs"
