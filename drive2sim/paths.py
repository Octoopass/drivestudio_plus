"""Locations of files shipped inside the drive2sim package."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
EGO_MASKS_DIR = ASSETS_DIR / "ego_masks"
SCENE_LISTS_DIR = ASSETS_DIR / "scene_lists"
CONFIGS_DIR = PACKAGE_DIR / "configs"
