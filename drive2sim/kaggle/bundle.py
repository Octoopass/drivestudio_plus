"""Gói job cho Kaggle và dựng lại thư mục làm việc từ gói trên Kaggle (CONTRACTS mục 9.2).

Gói gồm vài file JSON nhỏ, được nhúng thẳng vào notebook (không upload riêng):
- ``run.json``: run đã tạo trên máy, đủ để chạy (approach, lựa chọn, tham số từng stage).
- ``split.json``: split của run.
- ``scene_ref.json``: cảnh nằm ở dataset Kaggle nào.
- ``resume.json``: notebook của lần chạy trước có output cần chạy tiếp, nếu có.

Trên Kaggle, ``d2s kaggle run <gói>`` gọi ``materialize`` để dựng thư mục làm việc: cảnh lấy từ
dataset đã gắn (tìm theo ``manifest.yaml``, không giả định đường dẫn gắn của Kaggle), split ghi
mới, run và cache của lần chạy trước chép vào nếu có; rồi chạy run như ``d2s run resume``.

File này chứa ``write_bundle``, ``read_bundle`` và ``materialize``.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from drive2sim.core import files
from drive2sim.core.schema import Manifest, RunRecord
from drive2sim.engine.run import D2S_DIR, RUN_FILE
from drive2sim.engine.workdir import Workdir

BUNDLE_FILES = ("run.json", "split.json", "scene_ref.json", "resume.json")


def write_bundle(
    run_dir: Path,
    workdir: Workdir,
    scene_dataset: str,
    resume_kernel: Optional[str] = None,
) -> dict[str, dict]:
    """Dựng gói job của một run.

    Args:
        run_dir: Thư mục run trên máy.
        workdir: Thư mục làm việc trên máy.
        scene_dataset: Dataset Kaggle chứa cảnh, ví dụ ``alice/d2s-1a2b3c4d``.
        resume_kernel: Notebook của lần chạy trước, để chạy tiếp từ output của nó.

    Returns:
        ``{tên file: nội dung JSON}``.
    """
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    scene = workdir.load_scene(record.scene_id)
    split = workdir.load_split(record.scene_id, record.split_id)
    return {
        "run.json": record.model_dump(mode="json"),
        "split.json": split.model_dump(mode="json"),
        "scene_ref.json": {
            "scene_id": record.scene_id,
            "converter_version": scene.manifest.source.converter_version,
            "kaggle_dataset": scene_dataset,
        },
        "resume.json": {"kernel": resume_kernel},
    }


def read_bundle(bundle_dir: Path) -> dict[str, dict]:
    """Đọc gói job đã ghi ra thư mục.

    Raises:
        FileNotFoundError: Nếu thiếu file nào trong ``BUNDLE_FILES``.
    """
    return {name: json.loads((bundle_dir / name).read_text("utf-8")) for name in BUNDLE_FILES}


def _find_scene(input_root: Path, scene_id: str) -> Path:
    """Tìm thư mục cảnh có ``manifest.yaml`` mang ``scene_id`` dưới ``input_root``."""
    for manifest_path in sorted(input_root.glob("**/manifest.yaml")):
        try:
            if files.load(manifest_path, Manifest).scene_id == scene_id:
                return manifest_path.parent
        except ValueError:
            continue
    raise FileNotFoundError(f"không thấy cảnh {scene_id} (manifest.yaml) dưới {input_root}")


def _find_previous_run(input_root: Path, record: RunRecord) -> Optional[Path]:
    """Tìm thư mục run của lần chạy trước trong output đã gắn, nếu có."""
    pattern = f"**/runs/{record.scene_id}/{record.approach.approach_id}/{record.run_id}"
    found = [p for p in sorted(input_root.glob(pattern)) if (p / D2S_DIR / RUN_FILE).is_file()]
    return found[0] if found else None


def materialize(bundle: dict[str, dict], workdir: Workdir, input_root: Path) -> Path:
    """Dựng thư mục làm việc trên Kaggle từ gói job.

    Cảnh được dựng bằng symlink tới từng mục của thư mục cảnh trong dataset (dataset chỉ đọc),
    trừ ``splits/`` ghi mới. Nếu ``resume.json`` có notebook trước, run và cache trong output của
    nó được chép vào, nên stage đã xong không chạy lại và train chạy tiếp từ checkpoint.

    Args:
        bundle: Gói job (``read_bundle``).
        workdir: Thư mục làm việc trên Kaggle, ví dụ ``/kaggle/working``.
        input_root: Nơi Kaggle gắn dataset và output notebook, ví dụ ``/kaggle/input``.

    Returns:
        Thư mục run sẵn sàng cho ``execute_run``.

    Raises:
        FileNotFoundError: Nếu không thấy cảnh dưới ``input_root``.
    """
    record = RunRecord.model_validate(bundle["run.json"])
    source = _find_scene(input_root, record.scene_id)
    scene_dir = workdir.scene_dir(record.scene_id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        link = scene_dir / item.name
        if item.name != "splits" and not link.exists():
            link.symlink_to(item.resolve())
    split_path = workdir.split_path(record.scene_id, record.split_id)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(bundle["split.json"], indent=2), "utf-8")

    run_dir = workdir.run_dir(record.scene_id, record.approach.approach_id, record.run_id)
    previous = _find_previous_run(input_root, record) if bundle["resume.json"]["kernel"] else None
    if previous is not None and not run_dir.exists():
        shutil.copytree(previous, run_dir, symlinks=True)
        previous_cache = previous.parents[2].parent / "cache" / record.scene_id
        if previous_cache.is_dir():
            shutil.copytree(
                previous_cache, workdir.cache_dir / record.scene_id, dirs_exist_ok=True
            )
    if not (run_dir / D2S_DIR / RUN_FILE).is_file():
        (run_dir / D2S_DIR).mkdir(parents=True, exist_ok=True)
        files.dump(record, run_dir / D2S_DIR / RUN_FILE)
    return run_dir
