"""Route của nhóm cảnh và split: ``/api/scenes/...`` (CONTRACTS mục 11.1; SPEC mục 8 nhóm A).

File này chứa:
- ``GET /api/scenes``: danh sách cảnh (manifest).
- ``GET /api/scenes/{scene_id}``: manifest, báo cáo kiểm tra, danh sách split của một cảnh và
  cảnh có được ingest bằng converter cũ không.
- ``GET /api/scenes/{scene_id}/frames/{frame_index}``: đường dẫn ảnh của mọi camera ở một khung,
  cho lưới ảnh đồng bộ.
- ``GET /api/scenes/{scene_id}/splits``, ``GET .../splits/{split_id}``: đọc split.
- ``POST /api/scenes/{scene_id}/splits``: tạo split (như ``d2s split create``).
- ``POST /api/scenes/{scene_id}/validate``: kiểm tra lại cảnh, ghi ``reports/validation.json``.
- ``GET /api/scenes/{scene_id}/rig``: sơ đồ camera và quỹ đạo xe (``reports/rig.json``) hoặc
  ``null``; ``POST`` cùng đường dẫn tạo lại file đó (``engine/rig.py``, cần PyTorch).
- ``GET /api/scenes/{scene_id}/lidar``: ảnh chồng LiDAR (url) và số liệu
  ``reports/lidar_check.json`` hoặc ``null``; ``POST`` cùng đường dẫn chiếu lại và kiểm tra lại
  cảnh (cần PyTorch, ảnh trên máy).
- ``DELETE /api/scenes/{scene_id}``: xoá cảnh cùng split, cache, run của nó.
- ``DELETE /api/scenes/{scene_id}/splits/{split_id}``: xoá split (còn run dùng thì ``409``).
"""

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from drive2sim.core import files
from drive2sim.engine import housekeeping, scenes
from drive2sim.engine.rig import (
    LIDAR_OVERLAY_DIR,
    RigError,
    generate_lidar_check,
    generate_rig,
    read_lidar_check,
    read_rig,
)
from drive2sim.engine.workdir import KAGGLE_SCENE_FILE
from drive2sim.ingest.images import IMAGE_SUFFIXES
from drive2sim.ingest.manifest import outdated_output_version
from drive2sim.ingest.reports import revalidate
from drive2sim.ingest.validate import validate_scene
from drive2sim.studio.api.state import state_of

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


class CreateSplitBody(BaseModel):
    """Thân của ``POST /api/scenes/{scene_id}/splits``; cùng ý nghĩa với ``d2s split create``.

    Attributes:
        strategy: Chiến lược, ví dụ ``every_nth_frame:8``.
        cameras: Camera dùng; bỏ trống là mọi camera.
        frame_range: Khung đầu và khung cuối; bỏ trống là cả cảnh.
        split_id: Tên split; bỏ trống thì đặt theo CONTRACTS mục 10.1.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str
    cameras: Optional[list[str]] = None
    frame_range: Optional[tuple[int, int]] = None
    split_id: Optional[str] = None


@router.get("")
def list_scenes(request: Request) -> dict:
    """Liệt kê các cảnh đã ingest.

    Returns:
        ``{"items": [manifest, ...], "total": n}``.
    """
    items = [m.model_dump(mode="json") for m in scenes.list_scenes(state_of(request).workdir)]
    return {"items": items, "total": len(items)}


@router.get("/{scene_id}")
def get_scene(scene_id: str, request: Request) -> dict:
    """Trả về manifest, báo cáo kiểm tra và danh sách split của một cảnh.

    Returns:
        ``{"manifest", "validation", "splits", "split_cameras", "kaggle",
        "converter_outdated"}``; ``split_cameras``
        là camera của từng split (``{split_id: [camera_id]}``), để giao diện hiện loại ống kính và
        xét approach theo camera của split; ``validation`` là ``null`` nếu chưa có
        báo cáo; ``kaggle`` là danh sách notebook ingest (``kaggle_scene.json``) nếu cảnh được nạp
        trên Kaggle (máy chỉ có manifest và báo cáo, không có ảnh), ngược lại ``null``;
        ``converter_outdated`` là ``{"output_version", "current_output_version"}`` khi cảnh được
        ingest bằng converter cũ (``outdated_output_version``), ngược lại ``null``.
    """
    workdir = state_of(request).workdir
    scene = workdir.load_scene(scene_id)
    report_path = scene.root / "reports" / "validation.json"
    validation = json.loads(report_path.read_text("utf-8")) if report_path.is_file() else None
    kaggle_path = scene.root / KAGGLE_SCENE_FILE
    kaggle = None
    if kaggle_path.is_file():
        kaggle = json.loads(kaggle_path.read_text("utf-8"))["kernels"]
    splits = scenes.list_splits(workdir, scene_id)
    outdated = outdated_output_version(scene.manifest.source)
    return {
        "manifest": scene.manifest.model_dump(mode="json"),
        "validation": validation,
        "splits": [split.split_id for split in splits],
        "split_cameras": {split.split_id: list(split.cameras) for split in splits},
        "kaggle": kaggle,
        "converter_outdated": (
            {"output_version": outdated[0], "current_output_version": outdated[1]}
            if outdated
            else None
        ),
    }


@router.get("/{scene_id}/frames/{frame_index}")
def get_frame(scene_id: str, frame_index: int, request: Request) -> dict:
    """Trả về đường dẫn ảnh của mọi camera ở một khung, theo thứ tự camera trong manifest.

    Ảnh DriveStudio đặt tên ``images/<khung 3 chữ số>_<chỉ số camera>.<đuôi>``.

    Returns:
        ``{"frame_index", "images": [{"camera_id", "url"}]}``; ``url`` là ``null`` nếu thiếu ảnh.
    """
    workdir = state_of(request).workdir
    scene = workdir.load_scene(scene_id)
    images = []
    for index, camera in enumerate(scene.manifest.cameras):
        found = [
            path
            for path in sorted((scene.root / "images").glob(f"{frame_index:03d}_{index}.*"))
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        url = f"/files/{workdir.relative(found[0])}" if found else None
        images.append({"camera_id": camera.camera_id, "url": url})
    return {"frame_index": frame_index, "images": images}


@router.get("/{scene_id}/splits")
def list_splits(scene_id: str, request: Request) -> dict:
    """Liệt kê split của một cảnh.

    Returns:
        ``{"items": [split, ...], "total": n}``.
    """
    items = [
        s.model_dump(mode="json") for s in scenes.list_splits(state_of(request).workdir, scene_id)
    ]
    return {"items": items, "total": len(items)}


@router.get("/{scene_id}/splits/{split_id}")
def get_split(scene_id: str, split_id: str, request: Request) -> dict:
    """Trả về nội dung một split."""
    return state_of(request).workdir.load_split(scene_id, split_id).model_dump(mode="json")


@router.post("/{scene_id}/splits", status_code=201)
def create_split(scene_id: str, body: CreateSplitBody, request: Request) -> dict:
    """Tạo split ``every_nth_frame:N`` (như ``d2s split create``).

    Returns:
        Split vừa tạo; mã ``201``.
    """
    split = scenes.create_split(
        state_of(request).workdir,
        scene_id,
        body.strategy,
        cameras=body.cameras,
        frame_range=body.frame_range,
        split_id=body.split_id,
    )
    return split.model_dump(mode="json")


@router.get("/{scene_id}/rig")
def rig(scene_id: str, request: Request) -> dict:
    """Sơ đồ camera và quỹ đạo xe của cảnh.

    Returns:
        ``{"rig": nội dung reports/rig.json | null}``.
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    return {"rig": read_rig(scene.root)}


@router.post("/{scene_id}/rig")
def make_rig(scene_id: str, request: Request) -> dict:
    """Tạo lại ``reports/rig.json`` (như ``d2s scene rig``); vài chục giây.

    Raises:
        HTTPException: ``500`` kèm phần cuối log nếu không tạo được (thiếu PyTorch, thiếu pose).
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    try:
        rig = generate_rig(scene.root)
    except RigError as error:
        raise HTTPException(status_code=500, detail=str(error)) from None
    if not (scene.root / KAGGLE_SCENE_FILE).is_file():
        revalidate(scene.root)  # projection.camera_tilt đọc tilt_deg của rig.json
    return {"rig": rig}


def _lidar_payload(workdir, scene_root) -> dict:  # noqa: ANN001
    data = read_lidar_check(scene_root)
    folder = scene_root / "reports" / LIDAR_OVERLAY_DIR
    overlays = []
    for item in (data or {}).get("overlays", []):
        path = folder / str(item.get("file", ""))
        if path.is_file():
            overlays.append({**item, "url": f"/files/{workdir.relative(path)}"})
    return {"check": data, "overlays": overlays}


@router.get("/{scene_id}/lidar")
def lidar(scene_id: str, request: Request) -> dict:
    """Ảnh chồng LiDAR và số liệu kiểm tra thô của cảnh.

    Returns:
        ``{"check": nội dung reports/lidar_check.json | null, "overlays": [{"frame",
        "camera_id", "file", "url"}]}``; cảnh ingest trên Kaggle có ``check`` nhưng không có ảnh.
    """
    state = state_of(request)
    scene = state.workdir.load_scene(scene_id)
    return _lidar_payload(state.workdir, scene.root)


@router.post("/{scene_id}/lidar")
def make_lidar(scene_id: str, request: Request) -> dict:
    """Chiếu lại LiDAR lên ảnh (như ``d2s scene lidar-check``) rồi kiểm tra lại cảnh.

    Raises:
        HTTPException: ``409`` nếu cảnh chỉ có trên Kaggle hoặc không có LiDAR; ``500`` kèm log nếu
            không chạy được.
    """
    state = state_of(request)
    scene = state.workdir.load_scene(scene_id)
    if (scene.root / KAGGLE_SCENE_FILE).is_file() or not scene.manifest.available.lidar:
        raise HTTPException(status_code=409, detail="cảnh không có ảnh hoặc LiDAR trên máy")
    try:
        generate_lidar_check(scene.root)
    except RigError as error:
        raise HTTPException(status_code=500, detail=str(error)) from None
    revalidate(scene.root)
    return _lidar_payload(state.workdir, scene.root)


@router.post("/{scene_id}/validate")
def validate(scene_id: str, request: Request) -> dict:
    """Kiểm tra lại cảnh (như ``d2s scene validate``) và ghi báo cáo.

    Returns:
        Báo cáo kiểm tra mới.
    """
    scene = state_of(request).workdir.load_scene(scene_id)
    report = validate_scene(scene.root, scene.manifest)
    files.dump(report, scene.root / "reports" / "validation.json")
    return report.model_dump(mode="json")


@router.delete("/{scene_id}")
def delete_scene(scene_id: str, request: Request) -> dict:
    """Xoá cảnh cùng dữ liệu chuẩn, split, cache và run; có run đang chạy thì ``409``.

    Returns:
        ``{"removed": [đường dẫn đã xoá]}``.
    """
    return {"removed": housekeeping.delete_scene(state_of(request).workdir, scene_id)}


@router.delete("/{scene_id}/splits/{split_id}")
def delete_split(scene_id: str, split_id: str, request: Request) -> dict:
    """Xoá một split; còn run dùng split thì ``409``.

    Returns:
        ``{"removed": [đường dẫn đã xoá]}``.
    """
    return {"removed": housekeeping.delete_split(state_of(request).workdir, scene_id, split_id)}
