"""Báo cáo cần PyTorch chạy sau ingest: sơ đồ camera, chiếu LiDAR lên ảnh, rồi kiểm tra lại.

``scene ingest`` (CLI, Studio) và notebook ingest trên Kaggle gọi ``scene_reports`` sau khi ghi
cảnh. Hai báo cáo chạy bằng module GPU ở tiến trình con (``engine/rig.py``); lỗi ở đây chỉ là
cảnh báo, không làm hỏng cảnh vừa ingest. Kiểm tra lại (``validate_scene``) để báo cáo có các mục
``projection.*`` từ ``reports/lidar_check.json``.
"""

from pathlib import Path

from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.rig import RigError, generate_lidar_check, generate_rig
from drive2sim.ingest.validate import validate_scene


def revalidate(scene_dir: Path) -> None:
    """Kiểm tra lại cảnh và ghi ``reports/validation.json``."""
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    files.dump(validate_scene(scene_dir, manifest), scene_dir / "reports" / "validation.json")


def scene_reports(scene_dir: Path) -> list[str]:
    """Tạo sơ đồ camera và ảnh chồng LiDAR, rồi kiểm tra lại cảnh.

    Returns:
        Cảnh báo (bước nào không chạy được); rỗng nếu mọi bước xong.
    """
    warnings = []
    try:
        generate_rig(scene_dir)
    except RigError as error:
        warnings.append(str(error))
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    if manifest.available.lidar:
        try:
            generate_lidar_check(scene_dir)
        except RigError as error:
            warnings.append(str(error))
    revalidate(scene_dir)
    return warnings
