"""Ingest một cảnh nuScenes: gọi script tiền xử lý có sẵn của DriveStudio rồi ghi manifest.

Không viết lại việc chuyển dữ liệu. File này gọi ``python -m drive2sim.recon.preprocess.run
--dataset nuscenes`` (cần ``nuscenes-devkit``, như hướng dẫn ``docs/datasets/nuscenes.md``) để
chuyển một cảnh sang thư mục DriveStudio, chuyển thư mục đó tới ``data/<scene_id>/``, rồi ghi
``manifest.yaml`` và ``reports/validation.json`` (ARCHITECTURE mục 4.2, CONTRACTS mục 2).

File này chứa:
- ``NUSCENES_CAMERAS``: tên camera theo thứ tự chỉ số camera của DriveStudio.
- ``find_scene_index``: đổi tên cảnh (``scene-0061``) sang số thứ tự mà script dùng.
- ``build_command``: dựng câu lệnh tiền xử lý; tách riêng để test được không cần nuScenes thật.
- ``ingest_nuscenes``: hàm chính mà ``d2s scene ingest`` gọi.

Script tiền xử lý thay **mọi** chữ ``processed`` trong đường dẫn đích bằng ``processed_10Hz``
khi nội suy (``NuScenesProcessor.__init__``), nên hàm chạy script trong một thư mục tạm với đường
dẫn đích tương đối không chứa chữ đó, để đường dẫn của người dùng không bị đổi nhầm.

Chưa chạy thật: cần dữ liệu nuScenes và ``nuscenes-devkit``.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from drive2sim.core import files, identifiers
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest.manifest import build_manifest
from drive2sim.ingest.validate import validate_scene
from drive2sim.stages.commands import run_logged

# Thứ tự chỉ số camera của DriveStudio cho nuScenes: images/<khung>_<chỉ số>.jpg
# (cam_list trong drive2sim/recon/preprocess/nuscenes/process.py).
NUSCENES_CAMERAS: list[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
]
NUSCENES_LIDARS: list[str] = ["LIDAR_TOP"]
NUSCENES_LICENSE = "CC BY-NC-SA 4.0"
CONVERTER = "drivestudio.nuscenes"
_TARGET = "out"  # đường dẫn đích tương đối, không chứa chữ "processed"


class IngestError(RuntimeError):
    """Lỗi khi ingest: nguồn thiếu, cảnh không có, thư mục đích đã có, hoặc dữ liệu hỏng."""


def find_scene_index(raw_dir: Path, version: str, scene_name: str) -> int:
    """Đổi tên cảnh nuScenes sang số thứ tự mà script tiền xử lý dùng.

    Script chọn cảnh bằng chỉ số trong danh sách ``nusc.scene``, có cùng thứ tự với
    ``<version>/scene.json``.

    Args:
        raw_dir: Thư mục gốc của nuScenes (chứa ``v1.0-mini/``...).
        version: Phiên bản, ví dụ ``v1.0-mini``.
        scene_name: Tên cảnh, ví dụ ``scene-0061``.

    Returns:
        Chỉ số của cảnh, ví dụ ``0``.

    Raises:
        IngestError: Nếu không đọc được ``scene.json`` hoặc không có cảnh này.
    """
    path = raw_dir / version / "scene.json"
    try:
        scenes = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise IngestError(f"không đọc được {path}: {error}") from None
    names = [scene.get("name") for scene in scenes]
    if scene_name not in names:
        raise IngestError(
            f"{version} không có cảnh {scene_name}; các cảnh có: {', '.join(map(str, names[:20]))}"
        )
    return names.index(scene_name)


def build_command(
    raw_dir: Path, version: str, scene_index: int, interpolate_n: int, workers: int
) -> list[str]:
    """Dựng câu lệnh tiền xử lý nuScenes cho một cảnh.

    Câu lệnh chạy trong thư mục tạm và ghi vào ``out/<phần sau dấu - của version>/<chỉ số>/``,
    ví dụ ``out/mini/000/``.

    Args:
        raw_dir: Thư mục gốc của nuScenes.
        version: Phiên bản, ví dụ ``v1.0-mini``.
        scene_index: Chỉ số cảnh (``find_scene_index``).
        interpolate_n: Số khung nội suy giữa hai keyframe 2 Hz; ``4`` cho 10 Hz như teammate
            đang dùng, ``0`` để giữ 2 Hz.
        workers: Số tiến trình song song của script.

    Returns:
        Câu lệnh dạng danh sách.
    """
    return [
        sys.executable,
        "-m",
        "drive2sim.recon.preprocess.run",
        "--dataset",
        "nuscenes",
        "--data_root",
        str(raw_dir.resolve()),
        "--split",
        version,
        "--target_dir",
        _TARGET,
        "--scene_ids",
        str(scene_index),
        "--interpolate_N",
        str(interpolate_n),
        "--workers",
        str(workers),
    ]


def ingest_nuscenes(
    raw_dir: Path,
    workdir: Workdir,
    scene_name: str,
    version: str,
    scene_id: Optional[str] = None,
    interpolate_n: int = 4,
    workers: int = 4,
) -> Path:
    """Ingest một cảnh nuScenes vào ``data/<scene_id>/``.

    Các bước: tìm chỉ số cảnh; chạy script tiền xử lý trong ``data/.ingest/<scene_id>/``; chuyển
    thư mục kết quả tới ``data/<scene_id>/``; ghi ``manifest.yaml``, ``reports/validation.json``
    và lưu log của script ở ``reports/ingest_log.txt``. Thư mục tạm bị xoá khi xong; nếu lỗi thì
    giữ lại để xem log.

    Args:
        raw_dir: Thư mục gốc của nuScenes.
        workdir: Thư mục làm việc.
        scene_name: Tên cảnh trong nuScenes, ví dụ ``scene-0061``.
        version: Phiên bản, ví dụ ``v1.0-mini``.
        scene_id: Định danh cảnh trong tool; mặc định ``nuscenes_<scene_name>``.
        interpolate_n: Xem ``build_command``.
        workers: Xem ``build_command``.

    Returns:
        Thư mục cảnh ``data/<scene_id>/``.

    Raises:
        IngestError: Nếu cảnh đã có, nguồn không có cảnh này, hoặc script không ghi ra thư mục
            cảnh.
        RuntimeError: Nếu script tiền xử lý lỗi; thông báo kèm cuối log.
    """
    scene_id = scene_id or f"nuscenes_{scene_name.lower()}"
    identifiers.check_scene_id(scene_id)
    scene_dir = workdir.scene_dir(scene_id)
    if scene_dir.exists():
        raise IngestError(
            f"cảnh {scene_id} đã có ở {workdir.relative(scene_dir)}; xoá thư mục đó hoặc đặt "
            "--scene khác"
        )
    scene_index = find_scene_index(raw_dir, version, scene_name)

    temp_dir = workdir.data_dir / ".ingest" / scene_id
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True)
    log_path = temp_dir / "ingest_log.txt"
    run_logged(
        build_command(raw_dir, version, scene_index, interpolate_n, workers),
        log_path,
        cwd=temp_dir,
    )
    produced = temp_dir / _TARGET / version.split("-")[-1] / f"{scene_index:03d}"
    if not produced.is_dir():
        raise IngestError(f"script chạy xong nhưng không có {produced}; xem {log_path}")

    scene_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(scene_dir))
    (scene_dir / "reports").mkdir(exist_ok=True)
    shutil.move(str(log_path), str(scene_dir / "reports" / "ingest_log.txt"))
    shutil.rmtree(temp_dir, ignore_errors=True)

    manifest = build_manifest(
        scene_dir,
        scene_id=scene_id,
        dataset="nuscenes",
        sequence_id=scene_name,
        camera_ids=NUSCENES_CAMERAS,
        converter=CONVERTER,
        lidar_ids=NUSCENES_LIDARS,
        data_license=NUSCENES_LICENSE,
    )
    files.dump(manifest, scene_dir / "manifest.yaml")
    files.dump(validate_scene(scene_dir, manifest), scene_dir / "reports" / "validation.json")
    return scene_dir
