"""Tập file nhỏ của một cảnh ingest trên Kaggle: những gì máy cần để xem model, trừ ảnh và LiDAR.

Cảnh ingest bằng ``d2s kaggle ingest`` (CONTRACTS mục 9.4) nằm trong output của notebook ingest.
Ảnh, LiDAR, mask nặng (KITTI-360 0003: khoảng 8 GB) ở lại Kaggle. Máy vẫn cần pose và calib của
camera để xem model bằng renderer gốc (Render GPU, ảnh chính xác của Gói xem trước): loader ở chế
độ viewer không đọc ảnh, LiDAR hay box (``DrivingDataset`` với ``data.viewer``) nhưng đọc
``intrinsics/``, ``extrinsics/`` và ``ego_pose/`` hoặc ``lidar_pose/``. Các thư mục này chỉ vài MB.

Notebook in các file này vào log (đọc bằng một lệnh API), vì ``kaggle kernels output`` liệt kê mọi
file của output trước khi lọc, kể cả với ``--file-pattern``, và output của cảnh lớn đủ làm Kaggle
trả 429 (đã gặp với KITTI-360 0003).

Chỉ dùng thư viện chuẩn: notebook trên Kaggle import file này từ repo vừa clone, không cài gói.

File này chứa ``light_scene_files``, ``missing_view_files``, ``write_scene_files``.
"""

from pathlib import Path

# Thư mục pose, calib mà loader đọc ở chế độ viewer (CONTRACTS mục 2.1).
VIEW_DIRS = ("intrinsics", "extrinsics", "ego_pose", "lidar_pose")
POSE_DIRS = ("ego_pose", "lidar_pose")
# File ở cấp gốc của cảnh: manifest.yaml, frame_ids.txt, timestamps.txt...
TOP_LEVEL_SUFFIXES = (".yaml", ".txt", ".json")
# Giới hạn tổng dung lượng in vào log; nuScenes 1 cảnh khoảng 5 MB (extrinsics theo từng khung).
MAX_BYTES = 32 * 1024 * 1024


def light_scene_files(scene: Path) -> dict[str, str]:
    """Đọc tập file nhỏ của cảnh.

    Args:
        scene: Thư mục cảnh (dữ liệu chuẩn, CONTRACTS mục 2.1).

    Returns:
        ``{đường dẫn trong cảnh: nội dung}``: file text ở cấp gốc, ``reports/*.json`` và mọi file
        trong ``VIEW_DIRS``.

    Raises:
        ValueError: Nếu tổng dung lượng quá ``MAX_BYTES``.
    """
    scene = Path(scene)
    paths = sorted(p for p in scene.iterdir() if p.is_file() and p.suffix in TOP_LEVEL_SUFFIXES)
    paths += sorted((scene / "reports").glob("*.json"))
    for name in VIEW_DIRS:
        paths += sorted(p for p in (scene / name).rglob("*") if p.is_file())
    files: dict[str, str] = {}
    total = 0
    for path in paths:
        text = path.read_text("utf-8")
        total += len(text.encode("utf-8"))
        files[path.relative_to(scene).as_posix()] = text
    if total > MAX_BYTES:
        raise ValueError(f"file nhỏ của cảnh {scene.name} quá {MAX_BYTES} byte ({total})")
    return files


def missing_view_files(scene: Path) -> list[str]:
    """Những gì cảnh trên máy còn thiếu để xem model bằng renderer gốc.

    Args:
        scene: Thư mục cảnh trên máy.

    Returns:
        Tên các thư mục thiếu, ví dụ ``["intrinsics/", "ego_pose/ hoặc lidar_pose/"]``; rỗng là đủ.
    """
    scene = Path(scene)
    missing = [f"{name}/" for name in ("intrinsics", "extrinsics") if not (scene / name).is_dir()]
    if not any((scene / name).is_dir() for name in POSE_DIRS):
        missing.append(" hoặc ".join(f"{name}/" for name in POSE_DIRS))
    return missing


def write_scene_files(scene: Path, files: dict[str, str]) -> list[str]:
    """Ghi file nhỏ vào cảnh trên máy; không ghi đè file đã có.

    File đã có (``manifest.yaml``, ``splits/``...) là bản máy đang dùng, nên giữ nguyên.

    Args:
        scene: Thư mục cảnh trên máy.
        files: ``{đường dẫn trong cảnh: nội dung}`` như ``light_scene_files``.

    Returns:
        Các đường dẫn vừa ghi.

    Raises:
        ValueError: Nếu một đường dẫn trỏ ra ngoài thư mục cảnh; khi đó chưa ghi file nào.
    """
    root = Path(scene).resolve()
    targets = {}
    for name, text in files.items():
        target = (root / name).resolve()
        if root not in target.parents:
            raise ValueError(f"đường dẫn lạ trong file nhỏ của cảnh: {name}")
        targets[name] = (target, text)
    written = []
    for name, (target, text) in targets.items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, "utf-8")
        written.append(name)
    return written
