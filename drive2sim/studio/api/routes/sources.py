"""Route nhận dạng và ingest dữ liệu gốc: ``/api/sources/...``, ``/api/jobs/...``.

Cùng ý nghĩa với ``d2s source inspect`` và ``d2s scene ingest`` (CONTRACTS mục 11.1). Đường dẫn
là đường dẫn trên máy chạy Studio; Studio chỉ nghe ở ``127.0.0.1`` nên đó cũng là máy người dùng.

File này chứa:
- ``POST /api/sources/inspect``: xem thư mục có dataset gì, phiên bản, chuỗi nào.
- ``POST /api/sources/ingest``: chọn chuỗi (lỗi chọn trả ``400`` ngay), rồi ingest nền; trả
  ``202`` kèm việc.
- ``GET /api/jobs``, ``GET /api/jobs/{job_id}``: trạng thái các việc nền (ingest, Kaggle).
- ``GET /api/sources/uploads``, ``POST /api/sources/uploads``,
  ``POST /api/sources/uploads/{name}/chunk``, ``DELETE /api/sources/uploads/{name}``: tải dữ liệu
  gốc từ máy người dùng lên ``uploads/<tên>/`` qua trình duyệt (``studio/api/uploads.py``).
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.ingest.detect import SourceReport, inspect_source
from drive2sim.ingest.dispatch import plan_ingest, run_ingest
from drive2sim.studio.api import uploads
from drive2sim.studio.api.state import state_of

router = APIRouter(tags=["sources"])


class InspectBody(BaseModel):
    """Thân của ``POST /api/sources/inspect``.

    Attributes:
        path: Thư mục dữ liệu gốc trên máy chạy Studio.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


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
    target = plan_ingest(
        _inspect(body.path),
        body.source_type,
        body.version,
        body.sequence,
        interpolate_n=body.interpolate_n,
    )

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


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    """Huỷ việc tải từ Drive đang chờ hoặc đang chạy; phần đã tải giữ lại để tải tiếp.

    Returns:
        Trạng thái việc (``cancelled`` ngay nếu còn chờ; đang chạy thì thành ``cancelled`` sau
        vài giây). Không có việc trả ``404``, loại việc không huỷ được trả ``400``.
    """
    try:
        return state_of(request).jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(404, f"không có việc {job_id}") from None


@router.post("/api/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str, request: Request) -> dict:
    """Tải tiếp việc tải từ Drive đã huỷ hoặc lỗi (một việc mới, đi tiếp từ phần đã có).

    Returns:
        Việc mới (hoặc việc đang chạy cho cùng thư mục). Không có việc trả ``404``.
    """
    try:
        return state_of(request).jobs.retry(job_id)
    except KeyError:
        raise HTTPException(404, f"không có việc {job_id}") from None


class UploadFile(BaseModel):
    """Một file trong lần tải: đường dẫn tương đối trong thư mục người dùng chọn và kích thước.

    Attributes:
        path: Đường dẫn tương đối, dấu ``/``.
        size: Số byte.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0)


class UploadPlanBody(BaseModel):
    """Thân của ``POST /api/sources/uploads``.

    Attributes:
        name: Tên lần tải, thành thư mục ``uploads/<tên>/``.
        files: Các file sẽ gửi.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=uploads.NAME_PATTERN)
    files: list[UploadFile] = Field(min_length=1, max_length=50000)


@router.get("/api/sources/uploads")
def list_uploads(request: Request) -> dict:
    """Các thư mục đã tải lên.

    Returns:
        ``{"items": [{"name", "path", "path_abs", "files", "size_bytes", "partial",
        "partial_bytes", "modified", "used_by"}], "total"}`` (``uploads.list_uploads``).
    """
    items = uploads.list_uploads(state_of(request).workdir)
    return {"items": items, "total": len(items)}


@router.post("/api/sources/uploads")
def plan_upload(body: UploadPlanBody, request: Request) -> dict:
    """Lập kế hoạch tải: file đã có bỏ qua, file dở tải tiếp từ ``offset``.

    Returns:
        Kết quả ``uploads.plan_upload``; tên, đường dẫn sai hoặc hết chỗ trả ``400``.
    """
    files = [f.model_dump() for f in body.files]
    return uploads.plan_upload(state_of(request).workdir, body.name, files)


@router.post("/api/sources/uploads/{name}/chunk")
async def upload_chunk(name: str, path: str, offset: int, total: int, request: Request) -> dict:
    """Ghi một khối (thân là byte thô) vào file ``path`` của lần tải ``name``.

    Returns:
        ``{"path", "offset", "done"}``; khối không nối tiếp phần đã có trả ``409`` (trình duyệt
        gọi lại ``POST /api/sources/uploads`` để biết vị trí rồi gửi tiếp).
    """
    data = await request.body()
    workdir = state_of(request).workdir
    try:
        return await run_in_threadpool(
            uploads.write_chunk, workdir, name, path, offset, total, data
        )
    except uploads.UploadOffsetError as error:
        raise HTTPException(409, str(error)) from None


@router.delete("/api/sources/uploads/{name}")
def delete_upload(name: str, request: Request) -> dict:
    """Xoá thư mục ``uploads/<tên>/``.

    Việc tải từ Drive đang ghi vào thư mục này bị huỷ và chờ dừng trước; giao diện dừng việc tải
    từ trình duyệt trước, và cảnh báo khi có cảnh đọc dữ liệu gốc từ đây.

    Returns:
        ``{"removed": [...]}``; việc tải không dừng kịp trả ``409``.
    """
    state = state_of(request)
    active = [job["job_id"] for job in state.jobs.active_for(name)]
    for job_id in active:
        state.jobs.cancel(job_id)
    if active and not state.jobs.wait_stopped(active):
        raise HTTPException(409, "việc tải từ Drive chưa dừng kịp; thử xoá lại sau ít giây")
    return uploads.delete_upload(state.workdir, name)
