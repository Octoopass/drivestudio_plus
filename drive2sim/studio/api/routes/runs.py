"""Route của nhóm run: ``/api/runs/...`` và ``/api/events``.

Theo CONTRACTS mục 11; phục vụ SPEC mục 8 nhóm B (tiến trình) và C (kết quả một run).

File này chứa:
- ``POST /api/runs``: tạo run rồi xếp vào hàng đợi chạy nền (hoặc đẩy lên Kaggle); trả ``202``
  kèm ``run.json``.
- ``GET /api/runs``, ``GET /api/runs/{run_id}``: liệt kê run; xem ``run.json`` kèm ``summary``.
- ``POST /api/runs/{run_id}/resume``: chạy tiếp một run.
- ``POST /api/runs/{run_id}/cancel``: huỷ run đang chạy hoặc đang chờ.
- ``DELETE /api/runs/{run_id}``: xoá thư mục run (run đang chạy thì ``409``).
- ``POST /api/runs/{run_id}/export``: chạy lại một stage export (như ``d2s export``).
- ``GET /api/runs/{run_id}/events``, ``GET /api/events``: tiến độ qua Server-Sent Events.
- ``GET /api/runs/{run_id}/log``: log của engine và phần cuối log của từng stage.
- ``GET /api/runs/{run_id}/videos``: các video MP4 trong thư mục run, kèm đường dẫn để phát.
- ``GET /api/runs/{run_id}/artifacts``: nơi lưu kết quả của từng stage và các file trong đó.
- ``GET /api/runs/{run_id}/curve``: đường cong train (PSNR, loss, số Gaussian theo bước).
- ``GET /api/runs/{run_id}/splats``: file Gaussian xem được trong trình duyệt (gói xem trước,
  ``scene.ply``), kèm đường dẫn.
- ``GET /api/runs/{run_id}/render-check``: máy chạy Studio có render được run này bằng GPU không.
"""

from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from drive2sim.core import files
from drive2sim.core.schema import Preview, RunRecord
from drive2sim.engine import housekeeping
from drive2sim.engine.artifacts import run_artifacts, train_curve
from drive2sim.engine.render_check import render_readiness
from drive2sim.engine.run import (
    D2S_DIR,
    LOG_FILE,
    RUN_FILE,
    create_run,
    list_runs,
    load_env,
    load_summary,
    request_cancel,
    reset_stage,
    run_is_active,
)
from drive2sim.studio.api.events import stream_events
from drive2sim.studio.api.routes import kaggle
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
        executor: Nơi chạy: ``local`` (hàng đợi trên máy) hoặc ``kaggle`` (đẩy lên Kaggle ở nền).
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

    Với ``executor: kaggle``, run được tạo rồi đẩy lên Kaggle ở nền (``/api/jobs``).

    Returns:
        ``run.json`` vừa tạo (trạng thái ``pending``); theo dõi tiến độ qua
        ``/api/runs/{run_id}/events``.
    """
    state = state_of(request)
    scene, split, plan = plan_from_body(state, body.approach_id, body)
    run_dir = create_run(plan, state.workdir, scene, split, executor=body.executor)
    if body.executor == "kaggle":
        kaggle.submit_push(state, run_dir.name)
    else:
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
    """Trả về ``run.json`` cộng ``summary`` (nếu đã có) và ``env`` (nội dung ``env.json``)."""
    run_dir, record = _load(request, run_id)
    summary = load_summary(run_dir)
    data = record.model_dump(mode="json")
    data["summary"] = summary.model_dump(mode="json") if summary else None
    env = load_env(run_dir)
    data["env"] = env.model_dump(mode="json") if env else None
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


@router.get("/api/runs/{run_id}/artifacts")
def artifacts(run_id: str, request: Request) -> dict:
    """Nơi lưu kết quả của từng stage (thư mục run hay cache) và các file cấp một trong đó.

    Returns:
        Kết quả ``run_artifacts``.
    """
    run_dir, record = _load(request, run_id)
    return run_artifacts(state_of(request).workdir, record, run_dir)


@router.get("/api/runs/{run_id}/curve")
def curve(run_id: str, request: Request) -> dict:
    """Đường cong train đọc từ ``metrics.json`` của stage ``train.*``, cả khi run đã xong.

    Returns:
        ``{"stage_id", "series": {"psnr" | "loss" | "gaussians": [[bước, giá trị], ...]}}``;
        ``series`` rỗng nếu chưa có dữ liệu.
    """
    run_dir, record = _load(request, run_id)
    stage = next((s for s in record.stages if s.stage_id.startswith("train.")), None)
    if stage is None:
        return {"stage_id": None, "series": {}}
    return {
        "stage_id": stage.stage_id,
        "series": train_curve(run_dir / D2S_DIR / "stages" / stage.stage_id),
    }


@router.get("/api/runs/{run_id}/render-check")
def run_render_check(run_id: str, request: Request) -> dict:
    """Kiểm tra máy chạy Studio có đủ điều kiện render run này bằng GPU không.

    Returns:
        ``{"ready", "checks": [{"id", "ok", "value"}]}`` theo ``render_readiness``.
    """
    run_dir, _ = _load(request, run_id)
    return render_readiness(run_dir)


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


@router.get("/api/runs/{run_id}/splats")
def run_splats(run_id: str, request: Request) -> dict:
    """Liệt kê những gì trang xem 3D mở được của một run.

    Returns:
        ``{"preview": {...} | null, "ply": url | null}``. ``preview`` (khi đã chạy
        ``export.preview``) gồm nội dung ``preview.json`` cộng ``background_url``,
        ``actor_urls`` (``track_id`` → url) và ``poses_url``; ``ply`` là ``scene.ply`` của
        ``export.ply_world``.
    """
    run_dir, _ = _load(request, run_id)
    workdir = state_of(request).workdir
    stages_dir = run_dir / D2S_DIR / "stages"

    def url(path: Path) -> Optional[str]:
        return f"/files/{workdir.relative(path)}" if path.is_file() else None

    preview_dir = stages_dir / "export.preview"
    preview = None
    if (preview_dir / "preview.json").is_file() and (preview_dir / "done").is_file():
        data = files.load(preview_dir / "preview.json", Preview).model_dump(mode="json")
        preview = {
            **data,
            "background_url": url(preview_dir / "background.ply"),
            "actor_urls": {
                track: url(preview_dir / "actors" / f"{track}.ply")
                for track in data["rigid_actors"]
            },
            "poses_url": url(preview_dir / "actor_poses.json"),
        }
    return {"preview": preview, "ply": url(stages_dir / "export.ply_world" / "scene.ply")}
