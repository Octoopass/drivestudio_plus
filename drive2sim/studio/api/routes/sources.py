"""Route nhận dạng và ingest dữ liệu gốc: ``/api/sources/...``, ``/api/jobs/...``.

Cùng ý nghĩa với ``d2s source inspect`` và ``d2s scene ingest`` (CONTRACTS mục 11.1). Đường dẫn
là đường dẫn trên máy chạy Studio; Studio chỉ nghe ở ``127.0.0.1`` nên đó cũng là máy người dùng.

File này chứa:
- ``POST /api/sources/inspect``: xem thư mục có dataset gì, phiên bản, chuỗi nào.
- ``POST /api/sources/ingest``: chọn chuỗi (lỗi chọn trả ``400`` ngay), rồi ingest nền; trả
  ``202`` kèm việc.
- ``GET /api/jobs``, ``GET /api/jobs/{job_id}``: trạng thái các việc nền (ingest, Kaggle).
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.ingest.detect import SourceReport, inspect_source
from drive2sim.ingest.dispatch import plan_ingest, run_ingest
from drive2sim.studio.api.state import state_of

router = APIRouter(tags=["sources"])


class InspectBody(BaseModel):
    """Thân của ``POST /api/sources/inspect``.

    Attributes:
        path: Thư mục dữ liệu gốc trên máy chạy Studio.
    """

    model_config = ConfigDict(extra="forbid")

    path: str


class IngestBody(InspectBody):
    """Thân của ``POST /api/sources/ingest``; cùng ý nghĩa với ``d2s scene ingest``.

    Attributes:
        source_type: Dataset; bỏ trống thì tự nhận dạng.
        version: Phiên bản hoặc split; bỏ trống khi chỉ có một.
        sequence: Chuỗi; bỏ trống khi chỉ có một.
        scene_id: Định danh cảnh; bỏ trống thì đặt ``<dataset>_<chuỗi>``.
        interpolate_n: nuScenes: số khung nội suy giữa hai keyframe.
    """

    source_type: Optional[str] = None
    version: Optional[str] = None
    sequence: Optional[str] = None
    scene_id: Optional[str] = None
    interpolate_n: int = Field(4, ge=0)


def _inspect(path: str) -> SourceReport:
    """Nhận dạng thư mục; thư mục không có thì trả ``404``."""
    try:
        return inspect_source(Path(path).expanduser())
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from None


@router.post("/api/sources/inspect")
def inspect(body: InspectBody) -> dict:
    """Xem thư mục dữ liệu gốc có gì.

    Returns:
        Kết quả nhận dạng: ``dataset``, ``versions``, ``cameras``, ``ingest_supported``,
        ``notes``, ``inventory``...
    """
    return _inspect(body.path).to_dict()


@router.post("/api/sources/ingest", status_code=202)
def ingest(body: IngestBody, request: Request) -> dict:
    """Ingest một chuỗi ở nền.

    Returns:
        Việc vừa xếp hàng: ``{"job_id", "status": "queued", "label", ...}``.
    """
    state = state_of(request)
    raw_dir = Path(body.path).expanduser()
    target = plan_ingest(_inspect(body.path), body.source_type, body.version, body.sequence)

    def work() -> str:
        scene_dir = run_ingest(
            raw_dir,
            state.workdir,
            target,
            scene_id=body.scene_id,
            interpolate_n=body.interpolate_n,
        )
        return scene_dir.name

    return state.jobs.submit("ingest", f"{target.dataset} {target.sequence}", work)


@router.get("/api/jobs")
def list_jobs(request: Request) -> dict:
    """Liệt kê các việc nền từ lúc mở Studio, mới nhất trước."""
    items = state_of(request).jobs.list()
    return {"items": items, "total": len(items)}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    """Trạng thái một việc nền.

    Trạng thái là ``queued``, ``running``, ``succeeded`` (``result`` là ``scene_id`` hoặc
    ``run_id`` liên quan) hoặc ``failed`` (kèm ``error``).
    """
    job = state_of(request).jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"không có việc {job_id}")
    return job
