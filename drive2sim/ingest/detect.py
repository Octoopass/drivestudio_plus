"""Xem trước một thư mục dữ liệu gốc: nhận ra dataset và báo trong đó có gì.

Đây là phần việc của lệnh ``d2s source inspect``; ``d2s scene ingest`` không có ``--source-type``
cũng dùng nó để tự nhận dạng (ARCHITECTURE mục 4.2). Hàm chỉ đọc, không ghi gì.

File này chứa:
- ``SourceReport``: kết quả xem trước: dataset nhận ra được, các chuỗi (cảnh gốc), camera, có
  LiDAR và box không, ingest đã hỗ trợ chưa, và danh sách thư mục, file với dữ liệu lạ.
- ``inspect_source``: hàm chính.
- Các hàm ``_detect_*``: mỗi hàm nhận ra một loại nguồn theo dấu hiệu trong thư mục.

Với dữ liệu lạ, hàm **không đoán quy ước** (trục camera, chiều của pose, đơn vị), vì những thứ đó
không suy ra được từ file; nó chỉ liệt kê thư mục ảnh và các loại file để người viết converter
biết bắt đầu từ đâu.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from drive2sim.ingest.images import IMAGE_SUFFIXES, read_image_size

# Nguồn mà scene ingest đã chạy được.
INGEST_SUPPORTED: frozenset[str] = frozenset({"nuscenes"})
_MAX_DEPTH = 3
_MAX_INVENTORY = 20


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
    """

    path: Path
    dataset: Optional[str] = None
    versions: dict[str, list[str]] = field(default_factory=dict)
    cameras: list[str] = field(default_factory=list)
    has_lidar: Optional[bool] = None
    has_cuboids: Optional[bool] = None
    inventory: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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
    và có box hay không từ ``sample_annotation.json`` (rỗng với bản ``v1.0-test``).

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
    return True


def _detect_by_signature(root: Path, report: SourceReport) -> bool:
    """Nhận ra các nguồn chưa có ingest theo dấu hiệu trong thư mục.

    Dấu hiệu: file ``.tfrecord`` là Waymo; ``sparse/0/cameras.bin`` hoặc ``cameras.txt`` là
    COLMAP; file ``.zarr.itar`` hoặc thư mục ``.zarr`` là NCore; thư mục ``calib`` cùng
    ``image_02`` là KITTI.

    Args:
        root: Thư mục nguồn.
        report: Kết quả, được điền thêm nếu nhận ra.

    Returns:
        ``True`` nếu nhận ra một nguồn.
    """
    tfrecords = sorted(root.rglob("*.tfrecord"))
    if tfrecords:
        report.dataset = "waymo"
        report.versions[""] = [path.stem for path in tfrecords]
        return True
    if any((root / "sparse" / "0" / name).is_file() for name in ("cameras.bin", "cameras.txt")):
        report.dataset = "colmap"
        return True
    if any(root.rglob("*.zarr.itar")) or any(root.rglob("*.zarr")):
        report.dataset = "ncore"
        return True
    if any(root.rglob("image_02")) and any(root.rglob("calib*")):
        report.dataset = "kitti"
        return True
    return False


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

    Thử nhận ra nuScenes trước (đọc được nhiều thông tin nhất), rồi các nguồn khác theo dấu hiệu.
    Không nhận ra thì liệt kê thư mục ảnh và loại file.

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
    if _detect_nuscenes(root, report) or _detect_by_signature(root, report):
        if not report.ingest_supported:
            report.notes.append(
                f"nhận ra {report.dataset} nhưng scene ingest chưa hỗ trợ nguồn này"
            )
        return report
    report.inventory = _inventory(root)
    report.notes.append(
        "không nhận ra dataset nào đã biết; cần viết converter mới, bắt đầu từ converter gần "
        "nhất về hình dạng dữ liệu (ARCHITECTURE mục 4.2)"
    )
    return report
