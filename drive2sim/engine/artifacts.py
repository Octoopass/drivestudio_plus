"""Đọc những gì một run đã sinh ra để Studio hiển thị: artifact từng stage và đường cong train.

Kết quả của một stage nằm ở một trong hai chỗ (CONTRACTS mục 5, 6.1):
- stage có cache (``prep.*``, ...): ``cache/<scene_id>/<stage_id>@<version>-<hash8>/``, dùng chung
  giữa các run;
- stage khác: ``runs/<scene_id>/<approach_id>/<run_id>/d2s/stages/<stage_id>/``.

Run chạy trên Kaggle sau ``d2s kaggle pull`` cũng nằm đúng các chỗ đó (``pull`` thay ``d2s/`` của
run bằng bản trên Kaggle và chép thêm các mục cache máy chưa có), nên không có chỗ lưu riêng cho
Kaggle.

File này chứa:
- ``run_artifacts``: liệt kê thư mục kết quả của từng stage và các file cấp một trong đó.
- ``run_downloads``: các file nên tải về máy (PLY, ``scene.usdz`` cho CARLA, checkpoint, video).
- ``train_curve``: đọc ``metrics.json`` mà ``train.py`` ghi trong lúc train (mỗi dòng một JSON).
"""

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

from drive2sim.core.schema import RunRecord
from drive2sim.engine.workdir import Workdir

D2S_DIR = "d2s"
TRAIN_METRICS = "metrics.json"
MAX_CURVE_POINTS = 400
DRIVE_DIR = "drive"  # d2s/drive của run: scene.usdz mà trang Lái thử dựng
# (loại, stage (mẫu fnmatch; None là d2s/drive), tên file), theo thứ tự ưu tiên. Loại trong
# ``SINGLE_DOWNLOADS`` chỉ lấy file đầu tiên tìm thấy; ``video`` lấy hết.
DOWNLOADS: tuple[tuple[str, Optional[str], str], ...] = (
    ("ply", "export.nurec_ply", "scene.ply"),
    ("ply", "export.ply_world", "scene.ply"),
    ("nurec_ply", "train.nurec_refine", "scene_nurec.ply"),
    ("nurec_ply", "train.nurec", "scene_nurec.ply"),
    ("usdz", "render.carla", "scene.usdz"),
    ("usdz", None, "scene.usdz"),
    ("checkpoint", "train.*", "checkpoint_final.pth"),
    ("video", "render.carla", "carla.mp4"),
    ("video", "render.nurec_video", "flythrough.mp4"),
)
SINGLE_DOWNLOADS = frozenset({"ply", "nurec_ply", "usdz", "checkpoint"})


def _size(path: Path) -> int:
    """Dung lượng của file, hoặc tổng dung lượng mọi file trong thư mục (không theo symlink)."""
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def _entries(folder: Path, workdir: Workdir) -> list[dict[str, Any]]:
    """Các file và thư mục cấp một trong ``folder``, thư mục trước, kèm dung lượng."""
    items = []
    for child in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        items.append(
            {
                "name": child.name,
                "is_dir": child.is_dir(),
                "size_bytes": _size(child),
                "path": workdir.relative(child),
            }
        )
    return items


def run_artifacts(workdir: Workdir, record: RunRecord, run_dir: Path) -> dict[str, Any]:
    """Liệt kê nơi lưu kết quả của từng stage trong run và các file ở đó.

    Args:
        workdir: Thư mục làm việc.
        record: ``run.json`` của run.
        run_dir: Thư mục run.

    Returns:
        ``{"run_dir", "run_dir_abs", "files": [...], "stages": [{"stage_id", "status",
        "location": "run" | "cache", "path", "path_abs", "size_bytes", "files": [...]}]}``.
        ``files`` của run là các file của tool trong ``d2s/`` (``run.json``, ``env.json``, log...).
        Đường dẫn ``path`` tương đối với thư mục làm việc; stage chưa có kết quả thì ``path`` là
        ``None``.
    """
    stages = []
    for stage in record.stages:
        folder, location = _stage_folder(workdir, stage.stage_id, stage.cache_key, run_dir)
        stages.append(
            {
                "stage_id": stage.stage_id,
                "status": stage.status,
                "location": location,
                "path": workdir.relative(folder) if folder else None,
                "path_abs": str(folder.resolve()) if folder else None,
                "size_bytes": _size(folder) if folder else 0,
                "files": _entries(folder, workdir) if folder else [],
            }
        )
    tool_dir = run_dir / D2S_DIR
    files = [item for item in _entries(tool_dir, workdir) if not item["is_dir"]]
    return {
        "run_dir": workdir.relative(run_dir),
        "run_dir_abs": str(run_dir.resolve()),
        "files": files,
        "stages": stages,
    }


def _stage_folder(
    workdir: Workdir, stage_id: str, cache_key: Optional[str], run_dir: Path
) -> tuple[Optional[Path], str]:
    """Thư mục kết quả của một stage (cache nếu có, không thì trong run) và nơi lưu."""
    if cache_key:
        candidate = workdir.cache_dir / cache_key
        if candidate.is_dir():
            return candidate, "cache"
    candidate = run_dir / D2S_DIR / "stages" / stage_id
    return (candidate if candidate.is_dir() else None), "run"


def run_downloads(workdir: Workdir, record: RunRecord, run_dir: Path) -> list[dict[str, Any]]:
    """Các file của run nên tải về máy, theo ``DOWNLOADS``.

    - ``ply``: PLY Gaussian dạng 3DGS chuẩn (``export.*``), mở được bằng viewer khác;
    - ``nurec_ply``: PLY của Instant NuRec mà server render của CARLA đọc;
    - ``usdz``: ``scene.usdz`` cho CARLA (bản đồ OpenDRIVE, quỹ đạo, camera, xe và người);
    - ``checkpoint``: checkpoint drivestudio; ``video``: video CARLA và video đi dọc đường.

    Returns:
        ``[{"kind", "stage_id", "name", "path", "size_bytes"}]``; ``stage_id`` là ``None`` với
        file trong ``d2s/drive``; ``path`` tương đối với thư mục làm việc (dùng với ``/files/``).
    """
    folders: list[tuple[Optional[str], Optional[Path]]] = [
        (s.stage_id, _stage_folder(workdir, s.stage_id, s.cache_key, run_dir)[0])
        for s in record.stages
    ]
    folders.append((None, run_dir / D2S_DIR / DRIVE_DIR))
    items: list[dict[str, Any]] = []
    for kind, pattern, name in DOWNLOADS:
        if kind in SINGLE_DOWNLOADS and any(item["kind"] == kind for item in items):
            continue
        for stage_id, folder in folders:
            matches = stage_id is None if pattern is None else (
                stage_id is not None and fnmatch(stage_id, pattern)
            )  # fmt: skip
            target = folder / name if folder is not None and matches else None
            if target is not None and target.is_file():
                items.append({
                    "kind": kind, "stage_id": stage_id, "name": name,
                    "path": workdir.relative(target), "size_bytes": target.stat().st_size,
                })  # fmt: skip
                break
    return items


def train_curve(train_dir: Path) -> dict[str, list[list[float]]]:
    """Đọc đường cong train từ ``metrics.json`` của DriveStudio.

    Mỗi dòng của file là một JSON ``{"iteration", "train_metrics/psnr", "losses/<tên>",
    "train_stats/gaussian_num_<node>", ...}`` ghi sau mỗi ``print_freq`` bước. Số bước đếm lại
    từ 0 khi chạy tiếp từ checkpoint, nên hàm cộng dồn để trục bước luôn tăng (gần đúng: các bước
    bị làm lại sau khi ngắt vẫn được tính).

    Args:
        train_dir: Thư mục của stage ``train.*``.

    Returns:
        ``{"psnr": [[bước, giá trị], ...], "loss": [...], "gaussians": [...]}``: PSNR trên ảnh
        train, tổng các loss, tổng số Gaussian mọi node; tối đa ``MAX_CURVE_POINTS`` điểm mỗi
        đường. Rỗng nếu chưa có file.
    """
    path = train_dir / TRAIN_METRICS
    if not path.is_file():
        return {}
    series: dict[str, list[list[float]]] = {"psnr": [], "loss": [], "gaussians": []}
    offset = previous = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or "iteration" not in row:
            continue
        step = float(row["iteration"])
        if step + offset < previous:
            offset = previous
        step += offset
        previous = step
        if isinstance(row.get("train_metrics/psnr"), (int, float)):
            series["psnr"].append([step, float(row["train_metrics/psnr"])])
        losses = [
            v for k, v in row.items() if k.startswith("losses/") and isinstance(v, (int, float))
        ]
        if losses:
            series["loss"].append([step, float(sum(losses))])
        counts = [
            v
            for k, v in row.items()
            if k.startswith("train_stats/gaussian_num_") and isinstance(v, (int, float))
        ]
        if counts:
            series["gaussians"].append([step, float(sum(counts))])
    for name, points in series.items():
        if len(points) > MAX_CURVE_POINTS:
            stride = len(points) / MAX_CURVE_POINTS
            series[name] = [points[int(i * stride)] for i in range(MAX_CURVE_POINTS)] + [
                points[-1]
            ]
    return {name: points for name, points in series.items() if points}
