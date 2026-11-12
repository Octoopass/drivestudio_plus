"""Route của nhóm run: ``/api/runs/...`` và ``/api/events``.

Theo CONTRACTS mục 11; phục vụ SPEC mục 8 nhóm B (tiến trình) và C (kết quả một run).

File này chứa:
- ``POST /api/runs``: tạo run rồi xếp vào hàng đợi chạy nền; trả ``202`` kèm ``run.json``.
- ``GET /api/runs``, ``GET /api/runs/{run_id}``: liệt kê run; xem ``run.json`` kèm ``summary``.
- ``POST /api/runs/{run_id}/resume``: chạy tiếp một run.
- ``POST /api/runs/{run_id}/cancel``: huỷ run đang chạy hoặc đang chờ.
- ``DELETE /api/runs/{run_id}``: xoá thư mục run (run đang chạy thì ``409``).
- ``POST /api/runs/{run_id}/export``: chạy lại một stage export (như ``d2s export``).
- ``GET /api/runs/{run_id}/events``, ``GET /api/events``: tiến độ qua Server-Sent Events.
- ``GET /api/runs/{run_id}/log``: log của engine và phần cuối log của từng stage.
- ``GET /api/runs/{run_id}/videos``: các video MP4 trong thư mục run, kèm đường dẫn để phát.
"""

from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine import housekeeping
from drive2sim.engine.run import (
    D2S_DIR,
    LOG_FILE,
    RUN_FILE,
    create_run,
    list_runs,
    load_summary,
    request_cancel,
    reset_stage,
    run_is_active,
)
from drive2sim.studio.api.events import stream_events
from drive2sim.studio.api.routes.approaches import PlanBody, plan_from_body
from drive2sim.studio.api.state import state_of

router = APIRouter(tags=["runs"])
LOG_TAIL_LINES = 200
FINISHED = ("succeeded", "failed", "cancelled")
EXPORT_STAGES = {"ply": "export.ply_world", "preview": "export.preview"}


class ExportBody(BaseModel):
    """Thân của ``POST /api/runs/{run_id}/export``.

    Attributes:
        kind: ``ply`` (``export.ply_world``) hoặc ``preview`` (``export.preview``).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ply", "preview"]


class CreateRunBody(PlanBody):
    """Thân của ``POST /api/runs``; cùng ý nghĩa với ``d2s run start``.

    Attributes:
        approach_id: Approach cần chạy.
        executor: Nơi chạy; hiện chỉ có ``local``.
    """

    approach_id: str
    executor: Literal["local", "kaggle"] = "local"


def _load(request: Request, run_id: str) -> tuple[Path, RunRecord]:
    """Tìm thư mục run và đọc ``run.json``."""
    run_dir = state_of(request).workdir.find_run(run_id)
    return run_dir, files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """Đọc ``lines`` dòng cuối của một file văn bản; trả chuỗi rỗng nếu không có file."""
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "".join(text.splitlines(True)[-lines:])


@router.post("/api/runs", status_code=202)
def start_run(body: CreateRunBody, request: Request) -> dict:
    """Tạo một run và xếp vào hàng đợi chạy nền.

    Returns:
        ``run.json`` vừa tạo (trạng thái ``pending``); theo dõi tiến độ qua
        ``/api/runs/{run_id}/events``.
    """
    if body.executor != "local":
        raise HTTPException(400, "chạy trên Kaggle chưa có; dùng executor local")
    state = state_of(request)
    scene, split, plan = plan_from_body(state, body.approach_id, body)
    run_dir = create_run(plan, state.workdir, scene, split, executor=body.executor)
    state.queue.submit(run_dir)
    return files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord).model_dump(mode="json")


@router.get("/api/runs")
def get_runs(
    request: Request, scene_id: Optional[str] = None, status: Optional[str] = None
) -> dict:
    """Liệt kê run, mới nhất trước; lọc theo ``scene_id`` và ``status``.

    Returns:
        ``{"items": [run.json, ...], "total": n}``.
    """
    records = list_runs(state_of(request).workdir, scene_id)
    items = [r.model_dump(mode="json") for r in records if status in (None, r.status)]
    return {"items": items, "total": len(items)}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    """Trả về ``run.json`` cộng ``summary`` (nội dung ``summary.json``, nếu đã có)."""
    run_dir, record = _load(request, run_id)
    summary = load_summary(run_dir)
    data = record.model_dump(mode="json")
    data["summary"] = summary.model_dump(mode="json") if summary else None
    data["run_dir"] = state_of(request).workdir.relative(run_dir)
    return data


@router.post("/api/runs/{run_id}/resume", status_code=202)
def resume_run(run_id: str, request: Request) -> dict:
    """Chạy tiếp một run; stage đã xong không chạy lại.

    Returns:
        ``run.json`` hiện tại.
    """
    run_dir, record = _load(request, run_id)
    if run_is_active(run_dir):
        raise HTTPException(409, f"run {run_id} đang chạy")
    state_of(request).queue.submit(run_dir)
    return record.model_dump(mode="json")


@router.post("/api/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str, request: Request) -> dict:
    """Huỷ một run.

    Run đang chạy dừng trước stage kế tiếp, hoặc ngay khi stage đang chạy kiểm tra yêu cầu huỷ
    (tiến trình con bị dừng); run đang chờ trong hàng đợi bị huỷ khi tới lượt. Sự kiện ``done``
    với trạng thái ``cancelled`` báo khi xong.

    Returns:
        ``run.json`` hiện tại.
    """
    run_dir, _ = _load(request, run_id)
    return request_cancel(run_dir).model_dump(mode="json")


@router.delete("/api/runs/{run_id}")
def delete_run(run_id: str, request: Request) -> dict:
    """Xoá thư mục run; run đang chạy thì trả ``409``.

    Returns:
        ``{"removed": [đường dẫn đã xoá]}``.
    """
    return {"removed": housekeeping.delete_run(state_of(request).workdir, run_id)}


@router.post("/api/runs/{run_id}/export", status_code=202)
def export_run(run_id: str, body: ExportBody, request: Request) -> dict:
    """Chạy lại một stage export của run (thêm stage nếu run chưa có), các stage khác giữ nguyên.

    Returns:
        ``run.json`` sau khi đặt lại stage; run được xếp vào hàng đợi.
    """
    state = state_of(request)
    run_dir, _ = _load(request, run_id)
    if run_is_active(run_dir):
        raise HTTPException(409, f"run {run_id} đang chạy")
    stage_id = EXPORT_STAGES[body.kind]
    if stage_id not in state.registry:
        raise HTTPException(400, f"stage {stage_id} chưa có")
    record = reset_stage(run_dir, stage_id, state.registry)
    state.queue.submit(run_dir)
    return record.model_dump(mode="json")


def _initial_events(record: RunRecord) -> list[dict]:
    """Dựng các sự kiện mô tả trạng thái hiện tại của run, gửi ngay khi client kết nối."""
    events: list[dict] = [
        {"event": "status", "run_id": record.run_id, "stage_id": None, "status": record.status}
    ]
    events += [
        {"event": "status", "run_id": record.run_id, "stage_id": s.stage_id, "status": s.status}
        for s in record.stages
    ]
    if record.status in FINISHED:
        events.append({"event": "done", "run_id": record.run_id, "status": record.status})
    return events


@router.get("/api/runs/{run_id}/events")
def run_events(
    run_id: str, request: Request, last_event_id: Optional[int] = Header(default=None)
) -> StreamingResponse:
    """Phát tiến độ của một run qua SSE; đóng kết nối sau sự kiện ``done``."""
    _, record = _load(request, run_id)
    state = state_of(request)
    initial = [] if last_event_id is not None else _initial_events(record)
    stream = stream_events(state.broker, run_id, initial, last_event_id)
    return StreamingResponse(stream, media_type="text/event-stream")


@router.get("/api/events")
def all_events(request: Request) -> StreamingResponse:
    """Phát tiến độ của mọi run qua SSE, cho màn hình tổng quan."""
    stream = stream_events(state_of(request).broker, None, [])
    return StreamingResponse(stream, media_type="text/event-stream")


@router.get("/api/runs/{run_id}/log")
def run_log(run_id: str, request: Request) -> dict:
    """Trả về log của engine và 200 dòng cuối log của từng stage.

    Returns:
        ``{"engine": "...", "stages": {stage_id: "..."}}``.
    """
    run_dir, record = _load(request, run_id)
    stages_dir = run_dir / D2S_DIR / "stages"
    return {
        "engine": _tail(run_dir / D2S_DIR / LOG_FILE),
        "stages": {
            s.stage_id: _tail(stages_dir / s.stage_id / LOG_FILE)
            for s in record.stages
            if (stages_dir / s.stage_id / LOG_FILE).is_file()
        },
    }


@router.get("/api/runs/{run_id}/videos")
def run_videos(run_id: str, request: Request) -> dict:
    """Liệt kê các video MP4 trong thư mục run.

    Returns:
        ``{"items": [{"name", "stage_id", "url"}], "total": n}``; ``url`` phát qua ``/files/``.
    """
    run_dir, _ = _load(request, run_id)
    workdir = state_of(request).workdir
    stages_dir = run_dir / D2S_DIR / "stages"
    items = []
    for path in sorted(stages_dir.glob("*/**/*.mp4")):
        stage_id = path.relative_to(stages_dir).parts[0]
        if "scene_view" in path.relative_to(stages_dir).parts:
            continue
        items.append(
            {
                "name": path.relative_to(stages_dir / stage_id).as_posix(),
                "stage_id": stage_id,
                "url": f"/files/{workdir.relative(path)}",
            }
        )
    return {"items": items, "total": len(items)}
