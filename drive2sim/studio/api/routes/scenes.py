"""Route của nhóm cảnh và split: ``/api/scenes/...`` (CONTRACTS mục 11.1; SPEC mục 8 nhóm A).

File này chứa:
- ``GET /api/scenes``: danh sách cảnh (manifest).
- ``GET /api/scenes/{scene_id}``: manifest, báo cáo kiểm tra và danh sách split của một cảnh.
- ``GET /api/scenes/{scene_id}/frames/{frame_index}``: đường dẫn ảnh của mọi camera ở một khung,
  cho lưới ảnh đồng bộ.
- ``GET /api/scenes/{scene_id}/splits``, ``GET .../splits/{split_id}``: đọc split.
- ``POST /api/scenes/{scene_id}/splits``: tạo split (như ``d2s split create``).
"""

import json
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from drive2sim.engine import scenes
from drive2sim.ingest.images import IMAGE_SUFFIXES
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
        ``{"manifest", "validation", "splits"}``; ``validation`` là ``null`` nếu chưa có báo cáo.
    """
    workdir = state_of(request).workdir
    scene = workdir.load_scene(scene_id)
    report_path = scene.root / "reports" / "validation.json"
    validation = json.loads(report_path.read_text("utf-8")) if report_path.is_file() else None
    return {
        "manifest": scene.manifest.model_dump(mode="json"),
        "validation": validation,
        "splits": [split.split_id for split in scenes.list_splits(workdir, scene_id)],
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
