"""Dựng "thư mục cảnh ảo" bằng symlink, để chạy script DriveStudio mà không sửa dữ liệu chuẩn.

Script của DriveStudio (sinh mask bằng SegFormer, sinh SMPL, train) đều đọc và ghi trong một
thư mục cảnh dạng ``<data_root>/<số 3 chữ số>/``. Nhưng theo thiết kế, thư mục cảnh
``data/<scene_id>/`` không được sửa, và stage chỉ được ghi vào thư mục của nó (CONTRACTS mục
5.1). Cách làm: trong thư mục của stage, dựng ``<view>/000/`` gồm các symlink trỏ tới từng mục
của thư mục cảnh (cùng các artifact cần thêm, như ``sky_masks/`` từ cache), rồi cho script chạy
trên đó. Mục nào script tạo mới là thư mục thật nằm trong thư mục của stage. Nhờ vậy không phải
sửa dòng nào trong mã DriveStudio (ARCHITECTURE mục 4.3).

File này chứa:
- ``VIEW_SCENE``: tên thư mục cảnh trong view (``000``), đúng dạng script DriveStudio chờ.
- ``DRIVESTUDIO_DATASETS``: dataset gốc trong manifest → tên dataset mà script và config dùng.
- ``build_view``: dựng view.
- ``collect_outputs``: chuyển các mục script vừa tạo từ view ra thư mục của stage.

An toàn: view **không** chứa symlink tới các mục mà stage sắp ghi (tham số ``exclude``), nên kể cả
khi thư mục cảnh đã có sẵn ``humanpose/`` hay ``sky_masks/``, script cũng không ghi qua symlink
vào dữ liệu thật.
"""

import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

VIEW_SCENE = "000"
# Dataset gốc trong manifest → tên mà script DriveStudio (--dataset) và thư mục config dùng.
DRIVESTUDIO_DATASETS: dict[str, str] = {
    "nuscenes": "nuscenes",
    "waymo": "waymo",
    "argoverse2": "argoverse",
    "pandaset": "pandaset",
    "kitti": "kitti",
    "nuplan": "nuplan",
    "physicalai": "physicalai",
}


def build_view(
    scene_dir: Path,
    view_root: Path,
    exclude: Iterable[str] = (),
    extra: Mapping[str, Path] = {},
) -> Path:
    """Dựng thư mục cảnh ảo ``<view_root>/000/`` bằng symlink.

    Xoá view cũ nếu có rồi dựng lại, nên gọi lại nhiều lần (ví dụ khi chạy tiếp một run) vẫn ra
    cùng kết quả.

    Args:
        scene_dir: Thư mục cảnh thật ``data/<scene_id>/``.
        view_root: Thư mục chứa view, nằm trong thư mục của stage; đây là ``data_root`` truyền
            cho script DriveStudio.
        exclude: Tên các mục của thư mục cảnh không đưa vào view, thường là các mục stage sắp
            ghi, ví dụ ``["sky_masks", "fine_dynamic_masks"]``.
        extra: Tên mục → đường dẫn cần thêm vào view (đè lên mục cùng tên của thư mục cảnh), ví
            dụ ``{"humanpose": <cache>/humanpose}``.

    Returns:
        Thư mục cảnh ảo ``<view_root>/000/``.

    Raises:
        FileNotFoundError: Nếu thư mục cảnh hoặc một mục trong ``extra`` không tồn tại.
    """
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"không có thư mục cảnh {scene_dir}")
    view = view_root / VIEW_SCENE
    if view_root.exists():
        shutil.rmtree(view_root)
    view.mkdir(parents=True)
    skipped = set(exclude) | set(extra) | {"manifest.yaml", "splits", "reports"}
    for entry in sorted(scene_dir.iterdir()):
        if entry.name not in skipped:
            os.symlink(entry.resolve(), view / entry.name)
    for name, target in extra.items():
        if not target.exists():
            raise FileNotFoundError(f"không có {target} để đưa vào view")
        os.symlink(target.resolve(), view / name)
    return view


def collect_outputs(view: Path, output_dir: Path, names: Iterable[str]) -> list[str]:
    """Chuyển các mục script vừa tạo trong view ra thư mục của stage.

    Chỉ chuyển mục là thư mục hoặc file thật (không phải symlink), để không bao giờ chuyển nhầm
    dữ liệu của thư mục cảnh.

    Args:
        view: Thư mục cảnh ảo do ``build_view`` trả về.
        output_dir: Thư mục của stage.
        names: Tên các mục cần lấy, ví dụ ``["sky_masks", "fine_dynamic_masks"]``.

    Returns:
        Tên các mục đã chuyển; mục nào script không tạo thì không có trong danh sách.
    """
    moved = []
    for name in names:
        source = view / name
        if source.exists() and not source.is_symlink():
            shutil.move(str(source), str(output_dir / name))
            moved.append(name)
    return moved
