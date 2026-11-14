"""Converter chung cho mọi dataset mà DriveStudio đã có script tiền xử lý.

Mỗi dataset (nuScenes, Waymo, Argoverse 2, PandaSet, KITTI, nuPlan, PhysicalAI) đã có script
chuyển dữ liệu gốc thành thư mục cảnh DriveStudio, gọi qua ``python -m
drive2sim.recon.preprocess.run --dataset <tên>``. Định dạng chuẩn của tool là đúng thư mục đó cộng
``manifest.yaml`` (ARCHITECTURE mục 2.2), nên ingest không viết lại việc chuyển dữ liệu: nó gọi
script, tìm thư mục cảnh vừa sinh ra, chuyển tới ``data/<scene_id>/``, rồi ghi manifest và báo
cáo kiểm tra (ARCHITECTURE mục 4.2).

File này chứa:
- ``DriveStudioSource``: những gì khác nhau giữa các dataset: tên dataset của script, cách đổi
  tên chuỗi sang tham số ``--scene_ids``, danh sách camera theo thứ tự chỉ số, tên LiDAR, license.
- ``SOURCES``: bảng các dataset được hỗ trợ.
- ``build_command``: dựng câu lệnh tiền xử lý.
- ``find_scene_dir``: tìm thư mục cảnh script vừa ghi (thư mục duy nhất có ``images/``), để không
  phải đoán đường dẫn riêng của từng script.
- ``ingest_drivestudio``: hàm chính mà ``d2s scene ingest`` và Studio gọi.

Danh sách camera chép từ ``cam_list`` (hoặc ``CAMERAS``) trong script của từng dataset; thứ tự là
chỉ số camera trong tên ảnh ``<khung>_<chỉ số>.jpg``. Tên LiDAR lấy từ script khi script có ghi
(nuScenes, Waymo, KITTI), còn lại dùng tên chung ``LIDAR``.

Script chạy trong một thư mục tạm với đường dẫn đích tương đối không chứa chữ ``processed``, vì
script nuScenes thay mọi chữ đó trong đường dẫn đích khi nội suy.

Chưa chạy thật với dữ liệu nào: cần dữ liệu gốc và devkit của từng dataset (``docs/datasets/``).
"""

import json
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from drive2sim.core import files, identifiers
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest.manifest import build_manifest
from drive2sim.ingest.validate import validate_scene
from drive2sim.paths import SCENE_LISTS_DIR
from drive2sim.stages.commands import run_logged

_TARGET = "out"  # đường dẫn đích tương đối, không chứa chữ "processed"
# DriveStudio lưu một đám mây điểm gộp mỗi khung (lidar/<khung>.bin). Với dataset mà script không
# ghi tên cảm biến gốc, manifest ghi một LiDAR tên chung thay vì đoán tên.
GENERIC_LIDAR = ("LIDAR",)


class IngestError(RuntimeError):
    """Lỗi khi ingest.

    Gồm: nguồn không có chuỗi, thư mục đích đã có, script lỗi hoặc không ghi ra thư mục cảnh.
    """


def _index_in(names: Sequence[str], name: str, where: str) -> str:
    """Trả về chỉ số của ``name`` trong ``names`` dạng chuỗi.

    Raises:
        IngestError: Nếu không có ``name``; thông báo liệt kê vài tên có.
    """
    if name not in names:
        shown = ", ".join(list(names)[:10])
        raise IngestError(f"{where} không có {name}; ví dụ tên có: {shown}")
    return str(list(names).index(name))


def _scene_list(file_name: str) -> list[str]:
    """Đọc một danh sách cảnh trong ``drive2sim/assets/scene_lists/``."""
    return (SCENE_LISTS_DIR / file_name).read_text().splitlines()


def _nuscenes_arg(raw_dir: Path, version: str, sequence: str) -> str:
    """nuScenes: chỉ số của cảnh trong ``<version>/scene.json`` (cùng thứ tự ``nusc.scene``)."""
    path = raw_dir / version / "scene.json"
    try:
        scenes = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise IngestError(f"không đọc được {path}: {error}") from None
    return _index_in([s.get("name", "") for s in scenes], sequence, version)


def _waymo_arg(raw_dir: Path, version: str, sequence: str) -> str:
    """Waymo: chỉ số của segment trong ``waymo_train_list.txt`` (script đọc tfrecord theo list)."""
    return _index_in(_scene_list("waymo_train_list.txt"), sequence, "waymo_train_list.txt")


def _argoverse_arg(raw_dir: Path, version: str, sequence: str) -> str:
    """Argoverse 2: chỉ số của log trong ``argoverse_train_list.txt``."""
    return _index_in(_scene_list("argoverse_train_list.txt"), sequence, "argoverse_train_list.txt")


def _nuplan_arg(raw_dir: Path, version: str, sequence: str) -> str:
    """nuPlan: chỉ số của log trong split.

    Giả định ``NuPlanDBWrapper.log_names`` sắp theo tên file ``.db`` trong
    ``nuplan-v1.1/splits/<split>/``; chưa kiểm với nuplan-devkit.
    """
    split_dir = raw_dir / "nuplan-v1.1" / "splits" / version
    return _index_in(sorted(p.stem for p in split_dir.glob("*.db")), sequence, str(split_dir))


def _physicalai_arg(raw_dir: Path, version: str, sequence: str) -> str:
    """PhysicalAI: chỉ số trong danh sách ``pai_*.json`` đã sắp (``PhysicalAIProcessor.paths``)."""
    stems = [p.stem for p in sorted(raw_dir.glob("**/pai_*.json"))]
    return _index_in(stems, sequence, str(raw_dir))


def _same(raw_dir: Path, version: str, sequence: str) -> str:
    """PandaSet, KITTI: script nhận thẳng tên chuỗi."""
    return sequence


@dataclass(frozen=True)
class DriveStudioSource:
    """Mô tả một dataset mà DriveStudio có script tiền xử lý.

    Attributes:
        dataset: Tên trong manifest (``source.dataset``), ví dụ ``argoverse2``.
        script_dataset: Giá trị ``--dataset`` của ``preprocess.run``, ví dụ ``argoverse``.
        cameras: Tên camera theo thứ tự chỉ số trong tên ảnh.
        lidars: Tên LiDAR ghi vào manifest khi cảnh có LiDAR.
        scene_arg: Hàm ``(thư mục gốc, phiên bản, tên chuỗi) → giá trị --scene_ids``.
        uses_version: ``True`` nếu script cần ``--split <phiên bản>``.
        data_license: License của dữ liệu, nếu đã biết chắc.
        extra_args: Tham số thêm cho script, ví dụ nội suy của nuScenes.
        camera_model: Mô hình ống kính của ảnh sau tiền xử lý. Các dataset pinhole là
            ``opencv_pinhole``; PhysicalAI giữ ảnh F-theta gốc (docs/datasets/physicalai.md) nên
            là ``ftheta``. Track của cảnh suy ra từ mô hình này.
    """

    dataset: str
    script_dataset: str
    cameras: tuple[str, ...]
    lidars: tuple[str, ...]
    scene_arg: Callable[[Path, str, str], str]
    uses_version: bool = False
    data_license: Optional[str] = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    camera_model: str = "opencv_pinhole"


SOURCES: dict[str, DriveStudioSource] = {
    "nuscenes": DriveStudioSource(
        "nuscenes",
        "nuscenes",
        (
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
        ),
        ("LIDAR_TOP",),
        _nuscenes_arg,
        uses_version=True,
        data_license="CC BY-NC-SA 4.0",
    ),
    "waymo": DriveStudioSource(
        "waymo",
        "waymo",
        ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"),
        ("TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"),
        _waymo_arg,
        uses_version=True,
    ),
    "argoverse2": DriveStudioSource(
        "argoverse2",
        "argoverse",
        (
            "ring_front_center",
            "ring_front_left",
            "ring_front_right",
            "ring_side_left",
            "ring_side_right",
            "ring_rear_left",
            "ring_rear_right",
        ),
        GENERIC_LIDAR,
        _argoverse_arg,
        data_license="CC BY-NC-SA 4.0",
    ),
    "pandaset": DriveStudioSource(
        "pandaset",
        "pandaset",
        (
            "front_camera",
            "front_left_camera",
            "front_right_camera",
            "left_camera",
            "right_camera",
            "back_camera",
        ),
        GENERIC_LIDAR,
        _same,
    ),
    "kitti": DriveStudioSource(
        "kitti",
        "kitti",
        ("CAM_LEFT", "CAM_RIGHT"),
        ("velodyne",),
        _same,
        uses_version=True,
        data_license="CC BY-NC-SA 3.0",
    ),
    "nuplan": DriveStudioSource(
        "nuplan",
        "nuplan",
        ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"),
        GENERIC_LIDAR,
        _nuplan_arg,
        uses_version=True,
    ),
    "physicalai": DriveStudioSource(
        "physicalai",
        "physicalai",
        (
            "camera_front_wide_120fov",
            "camera_front_tele_30fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_rear_left_70fov",
            "camera_rear_right_70fov",
            "camera_rear_tele_30fov",
        ),
        GENERIC_LIDAR,
        _physicalai_arg,
        camera_model="ftheta",
    ),
}


def default_scene_id(dataset: str, sequence: str) -> str:
    """Đặt ``scene_id`` mặc định ``<dataset>_<chuỗi>`` theo CONTRACTS mục 10.1.

    Chữ hoa thành chữ thường; ký tự ngoài ``a-z 0-9 _ -`` thành ``-``.

    Args:
        dataset: Dataset, ví dụ ``waymo``.
        sequence: Tên chuỗi trong nguồn.

    Returns:
        Ví dụ ``waymo_segment-10017090168044687777_6380_000_6400_000_with_camera_labels``.
    """
    return f"{dataset}_{re.sub(r'[^a-z0-9_-]', '-', sequence.lower())}"


def build_command(
    source: DriveStudioSource,
    raw_dir: Path,
    version: Optional[str],
    scene_arg: str,
    workers: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Dựng câu lệnh ``python -m drive2sim.recon.preprocess.run`` cho một chuỗi.

    Args:
        source: Dataset.
        raw_dir: Thư mục dữ liệu gốc.
        version: Phiên bản hoặc split (nuScenes ``v1.0-mini``, Waymo ``training``, KITTI ngày,
            nuPlan ``mini``); bỏ qua với dataset không dùng.
        scene_arg: Giá trị ``--scene_ids``.
        workers: Số tiến trình song song của script.
        extra_args: Tham số thêm, ví dụ ``["--interpolate_N", "4"]``.

    Returns:
        Câu lệnh dạng danh sách; ghi vào thư mục tương đối ``out``.
    """
    command = [
        sys.executable,
        "-m",
        "drive2sim.recon.preprocess.run",
        "--dataset",
        source.script_dataset,
        "--data_root",
        str(raw_dir.resolve()),
        "--target_dir",
        _TARGET,
        "--scene_ids",
        scene_arg,
        "--workers",
        str(workers),
    ]
    if source.uses_version and version:
        command += ["--split", version]
    return command + list(source.extra_args) + list(extra_args)


def find_scene_dir(output_root: Path) -> Path:
    """Tìm thư mục cảnh mà script vừa ghi: thư mục duy nhất có ``images/``.

    Mỗi script đặt cảnh ở một chỗ khác nhau (``<đích>/<split>/<số>/``, ``<đích>/<tên log>/``...);
    tìm theo nội dung thì không phải giữ bảng đường dẫn cho từng script.

    Args:
        output_root: Thư mục đích đã truyền cho script.

    Returns:
        Thư mục cảnh.

    Raises:
        IngestError: Nếu không có, hoặc có nhiều hơn một thư mục như vậy.
    """
    found = sorted(path.parent for path in output_root.glob("**/images") if path.is_dir())
    if len(found) != 1:
        raise IngestError(
            f"cần đúng một thư mục cảnh có images/ trong {output_root}, thấy {len(found)}"
        )
    return found[0]


def ingest_drivestudio(
    raw_dir: Path,
    workdir: Workdir,
    dataset: str,
    sequence: str,
    version: Optional[str] = None,
    scene_id: Optional[str] = None,
    workers: int = 4,
    extra_args: Sequence[str] = (),
) -> Path:
    """Ingest một chuỗi của dataset vào ``data/<scene_id>/``.

    Các bước: đổi tên chuỗi sang ``--scene_ids``; chạy script trong ``data/.ingest/<scene_id>/``;
    tìm thư mục cảnh và chuyển tới ``data/<scene_id>/``; ghi ``manifest.yaml``,
    ``reports/validation.json``, và lưu log của script ở ``reports/ingest_log.txt``. Thư mục tạm
    bị xoá khi xong; nếu lỗi thì giữ lại để xem log.

    Args:
        raw_dir: Thư mục dữ liệu gốc.
        workdir: Thư mục làm việc.
        dataset: Khoá trong ``SOURCES``, ví dụ ``nuscenes``.
        sequence: Tên chuỗi trong nguồn, ví dụ ``scene-0061``.
        version: Phiên bản hoặc split, với dataset cần.
        scene_id: Định danh cảnh trong tool; mặc định ``default_scene_id``.
        workers: Số tiến trình song song của script.
        extra_args: Tham số thêm cho script.

    Returns:
        Thư mục cảnh ``data/<scene_id>/``.

    Raises:
        IngestError: Nếu dataset chưa hỗ trợ, cảnh đã có, nguồn không có chuỗi này, script lỗi
            (thông báo kèm cuối log) hoặc không ghi ra thư mục cảnh.
    """
    if dataset not in SOURCES:
        raise IngestError(f"chưa hỗ trợ ingest {dataset}; hỗ trợ: {', '.join(SOURCES)}")
    source = SOURCES[dataset]
    scene_id = scene_id or default_scene_id(dataset, sequence)
    identifiers.check_scene_id(scene_id)
    scene_dir = workdir.scene_dir(scene_id)
    if scene_dir.exists():
        raise IngestError(
            f"cảnh {scene_id} đã có ở {workdir.relative(scene_dir)}; xoá cảnh đó hoặc đặt "
            "scene_id khác"
        )
    scene_arg = source.scene_arg(raw_dir, version or "", sequence)

    temp_dir = workdir.data_dir / ".ingest" / scene_id
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True)
    log_path = temp_dir / "ingest_log.txt"
    command = build_command(source, raw_dir, version, scene_arg, workers, extra_args)
    try:
        run_logged(command, log_path, cwd=temp_dir)
    except RuntimeError as error:
        raise IngestError(
            f"script tiền xử lý lỗi; log giữ ở {workdir.relative(log_path)}. {error}"
        ) from None
    produced = find_scene_dir(temp_dir / _TARGET)

    scene_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(scene_dir))
    (scene_dir / "reports").mkdir(exist_ok=True)
    shutil.move(str(log_path), str(scene_dir / "reports" / "ingest_log.txt"))
    shutil.rmtree(temp_dir, ignore_errors=True)

    manifest = build_manifest(
        scene_dir,
        scene_id=scene_id,
        dataset=source.dataset,
        sequence_id=sequence,
        camera_ids=list(source.cameras),
        converter=f"drivestudio.{source.script_dataset}",
        lidar_ids=list(source.lidars),
        data_license=source.data_license,
        camera_model=source.camera_model,
    )
    files.dump(manifest, scene_dir / "manifest.yaml")
    files.dump(validate_scene(scene_dir, manifest), scene_dir / "reports" / "validation.json")
    return scene_dir
