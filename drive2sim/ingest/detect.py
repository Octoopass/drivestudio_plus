"""Xem trước một thư mục dữ liệu gốc: nhận ra dataset và báo trong đó có gì.

Đây là phần việc của lệnh ``d2s source inspect``; ``d2s scene ingest`` không có ``--source-type``
cũng dùng nó để tự nhận dạng (ARCHITECTURE mục 4.2). Hàm chỉ đọc, không ghi gì.

File này chứa:
- ``SourceReport``: kết quả xem trước: dataset nhận ra được, các chuỗi (cảnh gốc), camera, có
  LiDAR và box không, ingest đã hỗ trợ chưa, lỗi sẽ làm ingest hỏng (nuScenes thiếu ``sweeps/``),
  thư mục con hoặc cha có dataset (khi chính thư mục không có), và danh sách thư mục, file với
  dữ liệu lạ.
- ``inspect_source``: hàm chính.
- Các hàm ``_detect_*``: mỗi hàm nhận ra một loại nguồn theo dấu hiệu trong thư mục và liệt kê
  phiên bản, chuỗi có trong đó.
- ``_find_candidates``: khi không nhận ra gì, tìm dataset ở thư mục con (tới độ sâu 3) và thư mục
  cha (hai cấp), để báo đúng thư mục cần truyền thay vì để người dùng tự đoán.

Người dùng mới hay trỏ vào thư mục cha (``~/datasets`` thay vì ``~/datasets/nuscenes``) hoặc vào
thư mục con (``nuscenes/v1.0-mini``). Tool **không tự đổi** sang thư mục tìm được mà chỉ báo tên
nó: script tiền xử lý đọc đường dẫn theo đúng gốc dataset, thư mục cha có thể chứa nhiều dataset,
và đường dẫn đã chọn hiện rõ trong lệnh tiếp theo (ARCHITECTURE mục 4.2).

Với dữ liệu lạ, hàm **không đoán quy ước** (trục camera, chiều của pose, đơn vị), vì những thứ đó
không suy ra được từ file; nó chỉ liệt kê thư mục ảnh và các loại file để người viết converter
biết bắt đầu từ đâu.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from drive2sim.ingest import kitti360, phenikaa
from drive2sim.ingest.drivestudio import SOURCES
from drive2sim.ingest.images import IMAGE_SUFFIXES, read_image_size
from drive2sim.paths import SCENE_LISTS_DIR

# Nguồn mà scene ingest chạy được: mọi dataset có script tiền xử lý của DriveStudio, cộng các
# nguồn có converter tự viết trong drive2sim/ingest/ (KITTI-360).
INGEST_SUPPORTED: frozenset[str] = frozenset(SOURCES) | {kitti360.DATASET, phenikaa.DATASET}
_MAX_DEPTH = 3
_MAX_INVENTORY = 20
_MAX_SEARCH_DIRS = 200  # số thư mục con tối đa được xem khi tìm dataset nằm sâu hơn
_MAX_CANDIDATES = 10
_PARENT_LEVELS = 2
# Script nuScenes ở chế độ nội suy (--interpolate_N > 0, mặc định của tool là 4 cho 10 Hz) đọc
# ảnh và LiDAR giữa hai keyframe: các sample_data không phải keyframe nằm trong sweeps/<kênh>/
# (recon/preprocess/nuscenes/process.py: find_closest_img_tokens, find_cloest_lidar_tokens).
NUSCENES_SWEEP_CHANNELS: tuple[str, ...] = (
    *SOURCES["nuscenes"].cameras,
    *SOURCES["nuscenes"].lidars,
)


@dataclass
class SourceReport:
    """Kết quả xem trước một thư mục dữ liệu gốc.

    Attributes:
        path: Thư mục đã xem.
        dataset: Dataset nhận ra được (``nuscenes``, ``waymo``, ``kitti``, ``colmap``, ``ncore``),
            hoặc ``None`` nếu không nhận ra.
        versions: Phiên bản → danh sách chuỗi (cảnh gốc) trong phiên bản đó, ví dụ
            ``{"v1.0-mini": ["scene-0061", ...]}``; rỗng nếu dataset không chia phiên bản.
        cameras: Tên các camera, nếu đọc được.
        has_lidar: Có LiDAR không; ``None`` nếu không biết.
        has_cuboids: Có box 3D không; ``None`` nếu không biết.
        inventory: Với dữ liệu lạ: mỗi mục là một thư mục ảnh hoặc một nhóm file cùng đuôi.
        notes: Ghi chú cho người dùng, tiếng Việt.
        errors: Lỗi sẽ làm ingest hỏng với cách ingest mặc định, kèm cách sửa; tiếng Việt.
        missing_sweeps: nuScenes: các ``sweeps/<kênh>`` còn thiếu mà ingest nội suy cần; rỗng
            nếu đủ hoặc không phải nuScenes.
        candidates: Khi không nhận ra dataset ở ``path``: các thư mục con hoặc cha có dataset,
            mỗi mục ``{"path", "dataset", "relation": "child" | "parent"}``.
    """

    path: Path
    dataset: Optional[str] = None
    versions: dict[str, list[str]] = field(default_factory=dict)
    cameras: list[str] = field(default_factory=list)
    has_lidar: Optional[bool] = None
    has_cuboids: Optional[bool] = None
    inventory: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing_sweeps: list[str] = field(default_factory=list)
    candidates: list[dict[str, str]] = field(default_factory=list)

    @property
    def ingest_supported(self) -> bool:
        """Cho biết ``scene ingest`` đã chạy được với nguồn này chưa."""
        return self.dataset in INGEST_SUPPORTED

    def to_dict(self) -> dict[str, object]:
        """Đổi sang dict dạng JSON, cho ``--json``."""
        return {
            "path": str(self.path),
            "dataset": self.dataset,
            "versions": self.versions,
            "cameras": self.cameras,
            "has_lidar": self.has_lidar,
            "has_cuboids": self.has_cuboids,
            "ingest_supported": self.ingest_supported,
            "inventory": self.inventory,
            "notes": self.notes,
            "errors": self.errors,
            "missing_sweeps": self.missing_sweeps,
            "candidates": self.candidates,
        }


def _read_json_list(path: Path) -> list[dict]:
    """Đọc một file JSON dạng danh sách; trả danh sách rỗng nếu không đọc được."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _detect_nuscenes(root: Path, report: SourceReport) -> bool:
    """Nhận ra nuScenes: có thư mục ``v1.0-*`` chứa ``scene.json`` và ``sensor.json``.

    Đọc tên cảnh từ ``scene.json``, camera và LiDAR từ ``sensor.json`` (trường ``modality``),
    và có box hay không từ ``sample_annotation.json`` (rỗng với bản ``v1.0-test``). Kiểm thêm
    ``sweeps/<kênh>`` cho 6 camera và ``LIDAR_TOP``: thiếu thì ghi ``missing_sweeps`` và một lỗi,
    vì ingest mặc định (nội suy 10 Hz) đọc ảnh, LiDAR giữa các keyframe trong đó và sẽ hỏng ở
    giữa script (``cp`` ảnh lỗi lặng lẽ, đọc LiDAR thì ném lỗi). Chỉ ingest keyframe 2 Hz
    (``--interpolate-n 0``) thì không cần ``sweeps/``.

    Args:
        root: Thư mục nguồn.
        report: Kết quả, được điền thêm nếu nhận ra.

    Returns:
        ``True`` nếu là nuScenes.
    """
    version_dirs = sorted(
        path
        for path in root.glob("v1.0-*")
        if (path / "scene.json").is_file() and (path / "sensor.json").is_file()
    )
    if not version_dirs:
        return False
    report.dataset = "nuscenes"
    for version_dir in version_dirs:
        scenes = _read_json_list(version_dir / "scene.json")
        report.versions[version_dir.name] = [scene.get("name", "?") for scene in scenes]
    sensors = _read_json_list(version_dirs[0] / "sensor.json")
    report.cameras = [s["channel"] for s in sensors if s.get("modality") == "camera"]
    report.has_lidar = any(s.get("modality") == "lidar" for s in sensors)
    report.has_cuboids = bool(_read_json_list(version_dirs[0] / "sample_annotation.json"))
    report.missing_sweeps = [
        channel for channel in NUSCENES_SWEEP_CHANNELS if not (root / "sweeps" / channel).is_dir()
    ]
    if report.missing_sweeps:
        what = (
            "thiếu thư mục sweeps/"
            if len(report.missing_sweeps) == len(NUSCENES_SWEEP_CHANNELS)
            else "sweeps/ thiếu " + ", ".join(report.missing_sweeps)
        )
        report.errors.append(
            f"{what} (ảnh và LiDAR giữa các keyframe) mà ingest mặc định (--interpolate-n 4, "
            "10 Hz) cần: tải đủ dữ liệu cảm biến của nuScenes (phần sweeps, cùng gói với "
            f"samples/) rồi giải nén vào {root}; hoặc chỉ ingest keyframe 2 Hz bằng "
            "--interpolate-n 0"
        )
    return True


def _dirs(root: Path, pattern: str) -> list[Path]:
    """Các thư mục con khớp ``pattern``, sắp theo tên."""
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def _detect_waymo(root: Path, report: SourceReport) -> bool:
    """Nhận ra Waymo: có file ``.tfrecord`` nằm ngay trong thư mục.

    Script tiền xử lý đọc ``<thư mục>/<segment>.tfrecord`` (``recon/preprocess/waymo/process.py``)
    và chỉ đọc các segment có trong ``waymo_train_list.txt``, nên chuỗi liệt kê là các segment vừa
    có file vừa có trong danh sách đó. File nằm sâu hơn (``training/*.tfrecord``) thì không nhận ở
    đây mà ``_find_candidates`` báo thư mục con cần truyền: truyền thư mục cha thì script không
    thấy file.
    """
    stems = sorted(path.stem for path in root.glob("*.tfrecord"))
    if not stems:
        return False
    report.dataset = "waymo"
    known = set((SCENE_LISTS_DIR / "waymo_train_list.txt").read_text().splitlines())
    report.versions["training"] = [stem for stem in stems if stem in known]
    if len(report.versions["training"]) < len(stems):
        report.notes.append(
            f"{len(stems) - len(report.versions['training'])} segment không có trong "
            "waymo_train_list.txt nên script tiền xử lý không đọc được"
        )
    report.has_lidar = report.has_cuboids = True
    return True


def _detect_argoverse(root: Path, report: SourceReport) -> bool:
    """Nhận ra Argoverse 2: các thư mục log có ``sensors/cameras/ring_front_center``."""
    logs = [p.name for p in _dirs(root, "*") if (p / "sensors" / "cameras").is_dir()]
    if not logs:
        return False
    report.dataset = "argoverse2"
    known = set((SCENE_LISTS_DIR / "argoverse_train_list.txt").read_text().splitlines())
    report.versions[""] = [log for log in logs if log in known]
    report.cameras = list(SOURCES["argoverse2"].cameras)
    report.has_lidar = True
    report.has_cuboids = any((root / log / "annotations.feather").is_file() for log in logs)
    return True


def _detect_pandaset(root: Path, report: SourceReport) -> bool:
    """Nhận ra PandaSet: thư mục ``<3 chữ số>/camera/front_camera``."""
    sequences = [p.name for p in _dirs(root, "[0-9][0-9][0-9]") if (p / "camera").is_dir()]
    if not sequences:
        return False
    report.dataset = "pandaset"
    report.versions[""] = sequences
    report.cameras = list(SOURCES["pandaset"].cameras)
    report.has_lidar = any((root / s / "lidar").is_dir() for s in sequences)
    report.has_cuboids = any((root / s / "annotations").is_dir() for s in sequences)
    return True


def _detect_kitti(root: Path, report: SourceReport) -> bool:
    """Nhận ra KITTI raw: ``<ngày>/<ngày>_drive_<số>_sync/image_02``; phiên bản là ngày."""
    for date_dir in _dirs(root, "????_??_??"):
        drives = [p.name for p in _dirs(date_dir, "*_drive_*_sync") if (p / "image_02").is_dir()]
        if drives:
            report.versions[date_dir.name] = drives
    if not report.versions:
        return False
    report.dataset = "kitti"
    report.cameras = list(SOURCES["kitti"].cameras)
    report.has_lidar = True
    return True


def _detect_kitti360(root: Path, report: SourceReport) -> bool:
    """Nhận ra KITTI-360: ``calibration/calib_cam_to_pose.txt`` cùng ảnh trong ``data_2d_raw/``.

    Chuỗi lấy từ tên thư mục hoặc file ``.tar`` trong ``data_2d_raw/`` (``<chuỗi>_image_0X.tar``).
    Có box 3D khi có box chính thức (``data_3d_bboxes``) hoặc metadata ROSA
    (``ImageSets-rosa-6class/``); ở đây chỉ xem file có không, không mở pickle (converter đọc nó
    bằng unpickler chỉ cho mảng numpy, ``kitti360_boxes``).

    Args:
        root: Thư mục nguồn.
        report: Kết quả, được điền thêm nếu nhận ra.

    Returns:
        ``True`` nếu là KITTI-360.
    """
    if not (root / "calibration" / "calib_cam_to_pose.txt").is_file():
        return False
    sequences = kitti360.find_sequences(root)
    if not sequences:
        return False
    report.dataset = kitti360.DATASET
    report.versions[""] = sequences
    report.cameras = list(kitti360.CAMERAS)
    report.has_lidar = any(kitti360.has_lidar(root, sequence) for sequence in sequences)
    official = (root / "data_3d_bboxes").is_dir() or (root / "data_3d_bboxes.zip").is_file()
    rosa = any((root / "ImageSets-rosa-6class" / name).is_file() for name in (
        "kitti360_rosa_infos_train.pkl", "kitti360_rosa_infos_val.pkl"
    ))  # fmt: skip
    report.has_cuboids = official or rosa
    if rosa and not official:
        report.notes.append(
            "box 3D lấy từ metadata ROSA (ImageSets-rosa-6class/), đọc bằng unpickler chỉ cho "
            "mảng numpy; chuỗi không có trong ROSA thì cảnh không có box"
        )
    return True


def _detect_phenikaa(root: Path, report: SourceReport) -> bool:
    """Nhận ra PHENIKAA: thư mục chuỗi có ``camera/CAM_*``, ``ego_pose/``, ``lidar/*.laz``.

    ``root`` là một chuỗi hoặc thư mục chứa các chuỗi (thường kèm ``calib/``). Chưa có ID theo dõi
    cho box nên chưa có ``instances/``.

    Args:
        root: Thư mục nguồn.
        report: Kết quả, được điền thêm nếu nhận ra.

    Returns:
        ``True`` nếu là PHENIKAA.
    """
    sequences = phenikaa.find_sequences(root)
    if not sequences or not any(
        (path / "camera").glob("CAM_*") and any((path / "lidar").glob("*.laz"))
        for path in sequences.values()
    ):
        return False
    report.dataset = phenikaa.DATASET
    report.versions[""] = list(sequences)
    report.cameras = list(phenikaa.CAMERAS)
    report.has_lidar = True
    report.has_cuboids = False
    report.notes.append(
        "box 3D của PHENIKAA chưa có ID theo dõi: cảnh ingest chưa có instances/ (cần bước "
        "tracking); camera pinhole được khử méo khi ingest"
    )
    try:
        phenikaa.find_calib(next(iter(sequences.values())))
    except phenikaa.IngestError as error:
        report.errors.append(str(error))
    return True


def _detect_nuplan(root: Path, report: SourceReport) -> bool:
    """Nhận ra nuPlan: ``nuplan-v1.1/splits/<split>/*.db``; phiên bản là split."""
    for split_dir in _dirs(root / "nuplan-v1.1" / "splits", "*"):
        logs = sorted(p.stem for p in split_dir.glob("*.db"))
        if logs:
            report.versions[split_dir.name] = logs
    if not report.versions:
        return False
    report.dataset = "nuplan"
    report.cameras = list(SOURCES["nuplan"].cameras)
    report.has_lidar = report.has_cuboids = True
    return True


def _detect_physicalai(root: Path, report: SourceReport) -> bool:
    """Nhận ra PhysicalAI: file ``pai_<clip>.json`` (kho NCore của bộ dữ liệu PhysicalAI)."""
    stems = [p.stem for p in sorted(root.glob("**/pai_*.json"))]
    if not stems:
        return False
    report.dataset = "physicalai"
    report.versions[""] = stems
    report.cameras = list(SOURCES["physicalai"].cameras)
    report.has_lidar = True
    return True


def _detect_colmap(root: Path, report: SourceReport) -> bool:
    """Nhận ra COLMAP (chưa có ingest): có ``sparse/0/cameras.bin`` hoặc ``cameras.txt``."""
    if any((root / "sparse" / "0" / name).is_file() for name in ("cameras.bin", "cameras.txt")):
        report.dataset = "colmap"
        return True
    return False


def _detect_ncore(root: Path, report: SourceReport) -> bool:
    """Nhận ra NCore khác PhysicalAI (chưa có ingest): có ``.zarr.itar`` hoặc ``.zarr``."""
    if any(root.rglob("*.zarr.itar")) or any(root.rglob("*.zarr")):
        report.dataset = "ncore"
        return True
    return False


_DETECTORS = (
    _detect_phenikaa,
    _detect_nuscenes,
    _detect_waymo,
    _detect_physicalai,
    _detect_nuplan,
    _detect_kitti360,
    _detect_kitti,
    _detect_argoverse,
    _detect_pandaset,
    _detect_colmap,
    _detect_ncore,
)
# Dùng khi tìm dataset ở thư mục con, cha: chỉ các dấu hiệu nhìn ngay trong thư mục (không đi
# sâu), để mỗi thư mục xem tốn ít. PhysicalAI và NCore tìm đệ quy nên nếu có ở thư mục con thì
# đã được nhận ra ở chính thư mục người dùng truyền.
_SHALLOW_DETECTORS = (
    _detect_phenikaa,
    _detect_nuscenes,
    _detect_waymo,
    _detect_nuplan,
    _detect_kitti360,
    _detect_kitti,
    _detect_argoverse,
    _detect_pandaset,
    _detect_colmap,
)


def _shallow_dataset(directory: Path) -> Optional[str]:
    """Dataset nhận ra ngay ở ``directory`` theo ``_SHALLOW_DETECTORS``; ``None`` nếu không có."""
    probe = SourceReport(path=directory)
    try:
        found = any(detect(directory, probe) for detect in _SHALLOW_DETECTORS)
        return probe.dataset if found else None
    except OSError:  # thư mục không đọc được (quyền, symlink hỏng): coi như không có
        return None


def _find_candidates(root: Path) -> list[dict[str, str]]:
    """Tìm dataset ở thư mục con và thư mục cha của ``root``.

    Thư mục con: duyệt theo chiều rộng tới độ sâu ``_MAX_DEPTH``, bỏ thư mục ẩn, không đi vào
    thư mục đã nhận ra dataset, xem tối đa ``_MAX_SEARCH_DIRS`` thư mục để lệnh vẫn nhanh khi
    người dùng trỏ vào một thư mục rất lớn. Thư mục cha: ``_PARENT_LEVELS`` cấp, cho trường hợp
    trỏ vào bên trong dataset (``nuscenes/v1.0-mini``, ``kitti360/data_2d_raw``).

    Args:
        root: Thư mục người dùng truyền, nơi không nhận ra dataset nào.

    Returns:
        Tối đa ``_MAX_CANDIDATES`` mục ``{"path", "dataset", "relation"}``, thư mục con trước
        (gần trước, theo tên), rồi thư mục cha (gần trước).
    """
    found: list[dict[str, str]] = []
    queue = [(root, 0)]
    visited = 0
    while queue and visited < _MAX_SEARCH_DIRS and len(found) < _MAX_CANDIDATES:
        directory, depth = queue.pop(0)
        if depth >= _MAX_DEPTH:
            continue
        try:
            children = sorted(
                p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
            )
        except OSError:
            continue
        for child in children:
            visited += 1
            if visited > _MAX_SEARCH_DIRS or len(found) >= _MAX_CANDIDATES:
                break
            dataset = _shallow_dataset(child)
            if dataset is not None:
                found.append({"path": str(child), "dataset": dataset, "relation": "child"})
            else:
                queue.append((child, depth + 1))
    parent = root.resolve()
    for _ in range(_PARENT_LEVELS):
        if parent.parent == parent or len(found) >= _MAX_CANDIDATES:
            break
        parent = parent.parent
        dataset = _shallow_dataset(parent)
        if dataset is not None:
            found.append({"path": str(parent), "dataset": dataset, "relation": "parent"})
            break
    return found


def _inventory(root: Path) -> list[dict[str, object]]:
    """Liệt kê thư mục ảnh và các nhóm file cùng đuôi, tới độ sâu 3.

    Args:
        root: Thư mục nguồn.

    Returns:
        Mỗi thư mục có ảnh một mục ``{"kind": "images", "path", "count", "resolution"}``; mỗi
        đuôi file khác một mục ``{"kind": "files", "suffix", "count"}``. Tối đa 20 thư mục ảnh.
    """
    image_dirs: dict[Path, list[Path]] = {}
    suffixes: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file() or len(path.relative_to(root).parts) > _MAX_DEPTH + 1:
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image_dirs.setdefault(path.parent, []).append(path)
        else:
            suffixes["".join(path.suffixes) or path.name] += 1
    items: list[dict[str, object]] = []
    for directory, images in sorted(image_dirs.items())[:_MAX_INVENTORY]:
        size = read_image_size(sorted(images)[0])
        items.append(
            {
                "kind": "images",
                "path": directory.relative_to(root).as_posix() or ".",
                "count": len(images),
                "resolution": list(size) if size else None,
            }
        )
    for suffix, count in suffixes.most_common(_MAX_INVENTORY):
        items.append({"kind": "files", "suffix": suffix, "count": count})
    return items


def inspect_source(path: Path) -> SourceReport:
    """Xem trước một thư mục dữ liệu gốc.

    Thử lần lượt từng loại nguồn theo dấu hiệu trong thư mục (``_DETECTORS``); PhysicalAI được thử
    trước NCore chung vì nó cũng là NCore. Không nhận ra thì tìm dataset ở thư mục con, cha
    (``_find_candidates``) và ghi chú tên thư mục cần truyền; không thấy cả ở đó thì liệt kê thư
    mục ảnh và loại file.

    Args:
        path: Thư mục dữ liệu gốc.

    Returns:
        ``SourceReport`` mô tả thư mục.

    Raises:
        FileNotFoundError: Nếu ``path`` không phải thư mục.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"không có thư mục {root}")
    report = SourceReport(path=root)
    if any(detect(root, report) for detect in _DETECTORS):
        if not report.ingest_supported:
            report.notes.append(
                f"nhận ra {report.dataset} nhưng scene ingest chưa hỗ trợ nguồn này"
            )
        return report
    report.candidates = _find_candidates(root)
    if report.candidates:
        for candidate in report.candidates:
            where = "thư mục con" if candidate["relation"] == "child" else "thư mục cha"
            report.notes.append(
                f"không nhận ra dataset ở đúng thư mục này, nhưng thấy {candidate['dataset']} ở "
                f"{where} {candidate['path']}; truyền thư mục đó: d2s source inspect "
                f"{candidate['path']}"
            )
        return report
    report.inventory = _inventory(root)
    report.notes.append(
        "không nhận ra dataset nào đã biết; cần viết converter mới, bắt đầu từ converter gần "
        "nhất về hình dạng dữ liệu (ARCHITECTURE mục 4.2)"
    )
    return report
