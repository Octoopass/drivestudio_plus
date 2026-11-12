"""Route quản lý cache: ``/api/cache`` (như ``d2s cache list`` và ``d2s cache gc``).

File này chứa:
- ``GET /api/cache``: các kết quả trong ``cache/`` kèm dung lượng và run nào còn dùng.
- ``POST /api/cache/gc``: xoá kết quả không run nào dùng và thư mục ``.partial`` bỏ dở;
  ``dry_run`` để xem trước.
"""

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from drive2sim.engine.housekeeping import cache_entries, gc_cache
from drive2sim.studio.api.state import state_of

router = APIRouter(prefix="/api/cache", tags=["cache"])


class GcBody(BaseModel):
    """Thân của ``POST /api/cache/gc``.

    Attributes:
        scene_id: Chỉ dọn cache của cảnh này.
        dry_run: ``True`` thì chỉ trả về danh sách sẽ xoá.
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: Optional[str] = None
    dry_run: bool = False


@router.get("")
def list_cache(request: Request, scene_id: Optional[str] = None) -> dict:
    """Liệt kê cache.

    Returns:
        ``{"items": [{"cache_key", "scene_id", "stage_id", "size_bytes", "partial", "stale",
        "used_by"}], "total": n, "size_bytes": tổng}``.
    """
    items = [asdict(e) for e in cache_entries(state_of(request).workdir, scene_id)]
    return {"items": items, "total": len(items), "size_bytes": sum(i["size_bytes"] for i in items)}


@router.post("/gc")
def collect(body: GcBody, request: Request) -> dict:
    """Dọn cache.

    Returns:
        ``{"removed": [...], "freed_bytes": n, "dry_run": bool}``.
    """
    removed = [asdict(e) for e in gc_cache(state_of(request).workdir, body.scene_id, body.dry_run)]
    return {
        "removed": removed,
        "freed_bytes": sum(e["size_bytes"] for e in removed),
        "dry_run": body.dry_run,
    }
