"""Route bảng thư viện: ``/api/libraries`` (CONTRACTS mục 11.1; ``drive2sim/engine/libraries.py``).

File này chứa:
- ``GET /api/libraries``: danh mục kèm trạng thái trên máy chạy Studio và lần cài hiện tại.
- ``POST /api/libraries/{item_id}/install``: cài một mục; tiến độ ở
  ``GET /api/environment/install``.
- ``POST /api/libraries/install-all``: "Cài tất cả": cài lần lượt mọi mục còn thiếu
  (``libraries.bulk_ids``), tiến độ từng mục ở khoá ``all`` của ``GET /api/environment/install``.
"""

from fastapi import APIRouter, Request

from drive2sim.engine import libraries
from drive2sim.studio.api.routes.environment import install_flow

router = APIRouter(prefix="/api/libraries", tags=["libraries"])


@router.get("")
def list_libraries(request: Request) -> dict:
    """Danh mục thư viện, trạng thái từng mục và lần cài đang chạy (``install``)."""
    return {**libraries.catalog(), "install": install_flow(request).view()}


@router.post("/install-all")
def install_all(request: Request) -> dict:
    """Cài lần lượt mọi mục còn thiếu (bỏ mục ``bulk: False``).

    Dừng bằng ``POST /api/environment/install/cancel``.

    Returns:
        Trạng thái lần cài (như ``GET /api/environment/install``), có ``all``.
    """
    return install_flow(request).start_all(libraries.bulk_ids())


@router.post("/{item_id}/install")
def install_library(item_id: str, request: Request) -> dict:
    """Cài một mục; mục cài tay, mục lạ, máy không cài được thì ``400``.

    Returns:
        Trạng thái lần cài (như ``GET /api/environment/install``).
    """
    flow = install_flow(request)
    plan = libraries.install_plan(item_id, flow.python)
    if plan[0] == "env":
        return flow.start(plan[1], plan[2])
    return flow.start_command(item_id, plan[1])
