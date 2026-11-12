"""Sinh ``manifest.yaml`` từ một thư mục cảnh DriveStudio đã tiền xử lý.

Sau khi script tiền xử lý của một dataset ghi xong thư mục cảnh, ingest gọi file này để ghi
manifest: cảnh có những camera nào, độ phân giải, bao nhiêu khung, có LiDAR và box không
(CONTRACTS mục 2.2). Engine dựa vào manifest để khoá approach và chọn config dataset; Studio dựa
vào nó để hiện tổng quan cảnh.

File này chứa:
- ``frame_indices``: các chỉ số khung có ảnh của một camera.
- ``converter_version``: phiên bản converter ghi vào manifest.
- ``build_manifest``: dựng ``Manifest`` từ thư mục cảnh.

Mọi thông tin đọc từ chính các file trong thư mục cảnh (tên file ảnh, header ảnh, thư mục
``lidar/``, ``instances/``), không đọc lại dữ liệu gốc.
"""

import json
from importlib import metadata
from pathlib import Path
from typing import Optional

from drive2sim.core.schema import SCHEMA_VERSIONS, Manifest
from drive2sim.engine.run import code_commit
from drive2sim.ingest.images import IMAGE_SUFFIXES, read_image_size


def frame_indices(scene_dir: Path, camera_index: int) -> list[int]:
    """Liệt kê các chỉ số khung có ảnh của một camera.

    Ảnh DriveStudio đặt tên ``images/<khung 3 chữ số>_<chỉ số camera>.<đuôi>``.

    Args:
        scene_dir: Thư mục cảnh.
        camera_index: Chỉ số camera của DriveStudio.

    Returns:
        Các chỉ số khung, tăng dần, ví dụ ``[0, 1, 2]``.
    """
    indices = []
    for path in (scene_dir / "images").glob(f"*_{camera_index}.*"):
        stem_frame = path.stem.rsplit("_", 1)[0]
        if path.suffix.lower() in IMAGE_SUFFIXES and stem_frame.isdigit():
            indices.append(int(stem_frame))
    return sorted(indices)


def converter_version() -> str:
    """Trả về phiên bản converter dạng ``<phiên bản drive2sim>+<commit>``, ví dụ ``0.1.0+e567e29``.

    Phiên bản này nằm trong khoá cache của mọi stage nhận cảnh: ingest lại bằng code khác thì
    cache cũ không bị dùng nhầm (ARCHITECTURE mục 2.7).
    """
    try:
        package = metadata.version("drive2sim")
    except metadata.PackageNotFoundError:
        package = "0.0.0"
    return f"{package}+{code_commit()}"


def build_manifest(
    scene_dir: Path,
    scene_id: str,
    dataset: str,
    sequence_id: str,
    camera_ids: list[str],
    converter: str,
    lidar_ids: Optional[list[str]] = None,
    data_license: Optional[str] = None,
) -> Manifest:
    """Dựng manifest cho một thư mục cảnh DriveStudio.

    Số khung lấy theo camera đầu tiên. Độ phân giải đọc từ ảnh đầu tiên của từng camera. Cảnh có
    LiDAR khi ``lidar/`` có file ``.bin``; có box khi ``instances/instances_info.json`` không rỗng.
    Mô hình camera ghi ``opencv_pinhole`` và shutter ``GLOBAL``, đúng với cách loader DriveStudio
    của các dataset pinhole đọc ảnh.

    Args:
        scene_dir: Thư mục cảnh đã tiền xử lý.
        scene_id: Định danh cảnh, ví dụ ``nuscenes_scene-0061``.
        dataset: Dataset gốc, ví dụ ``nuscenes``.
        sequence_id: Id của chuỗi trong nguồn, ví dụ ``scene-0061``.
        camera_ids: Tên camera theo đúng thứ tự chỉ số camera của DriveStudio cho dataset đó.
        converter: Tên converter, ví dụ ``drivestudio.nuscenes``.
        lidar_ids: Tên các LiDAR nếu cảnh có LiDAR, ví dụ ``["LIDAR_TOP"]``.
        data_license: License của dữ liệu nguồn.

    Returns:
        Manifest đã kiểm tra theo schema.

    Raises:
        ValueError: Nếu camera đầu tiên không có ảnh nào, hoặc ảnh đầu của một camera không đọc
            được kích thước.
    """
    frames = frame_indices(scene_dir, 0)
    if not frames:
        raise ValueError(f"{scene_dir}: không có ảnh nào của camera 0 trong images/")
    cameras = []
    for index, camera_id in enumerate(camera_ids):
        images = sorted((scene_dir / "images").glob(f"{frames[0]:03d}_{index}.*"))
        size = read_image_size(images[0]) if images else None
        if size is None:
            raise ValueError(f"{scene_dir}: không đọc được ảnh khung {frames[0]} của {camera_id}")
        cameras.append(
            {
                "camera_id": camera_id,
                "model": "opencv_pinhole",
                "resolution": list(size),
                "shutter": "GLOBAL",
            }
        )
    has_lidar = any((scene_dir / "lidar").glob("*.bin"))
    instances_info = scene_dir / "instances" / "instances_info.json"
    has_cuboids = False
    if instances_info.is_file():
        has_cuboids = bool(json.loads(instances_info.read_text(encoding="utf-8")))
    lidars = list(lidar_ids or []) if has_lidar else []
    return Manifest.model_validate(
        {
            "schema_version": SCHEMA_VERSIONS["manifest"],
            "scene_id": scene_id,
            "source": {
                "dataset": dataset,
                "sequence_id": sequence_id,
                "converter": converter,
                "converter_version": converter_version(),
                "data_license": data_license,
            },
            "track": "pinhole",
            "cameras": cameras,
            "lidars": lidars,
            "timeline": {"reference_sensor": camera_ids[0], "num_frames": frames[-1] + 1},
            "available": {"lidar": bool(lidars), "cuboids": "gt" if has_cuboids else "none"},
        }
    )
