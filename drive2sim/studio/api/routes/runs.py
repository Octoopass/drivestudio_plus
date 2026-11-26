"""Route của nhóm run: ``/api/runs/...`` và ``/api/events``.

Theo CONTRACTS mục 11; phục vụ SPEC mục 8 nhóm B (tiến trình) và C (kết quả một run).

File này chứa:
- ``POST /api/runs``: tạo run rồi xếp vào hàng đợi chạy nền (hoặc đẩy lên Kaggle, Colab); trả
  ``202`` kèm ``run.json``. Chạy trên máy mà máy thiếu môi trường thì trả ``422 env_missing``
  trước khi tạo run, cùng câu với ``d2s run start``.
- ``GET /api/runs``, ``GET /api/runs/{run_id}``: liệt kê run; xem ``run.json`` kèm ``summary``.
- ``POST /api/runs/{run_id}/resume``: chạy tiếp một run.
- ``POST /api/runs/{run_id}/cancel``: huỷ run đang chạy hoặc đang chờ.
- ``DELETE /api/runs/{run_id}``: xoá thư mục run (run đang chạy thì ``409``).
- ``POST /api/runs/{run_id}/export``: chạy lại một stage export (như ``d2s export``).
- ``GET /api/runs/{run_id}/events``, ``GET /api/events``: tiến độ qua Server-Sent Events.
- ``GET /api/runs/{run_id}/log``: log của engine và phần cuối log của từng stage.
- ``GET /api/runs/{run_id}/videos``: các video MP4 trong thư mục run, kèm đường dẫn để phát, đường
  dẫn bản xem trước và cờ đã ghi xong.
- ``GET /api/runs/{run_id}/videos/preview``: bản xem trước trình duyệt phát được của một video
  (``drive2sim/studio/api/preview.py``).
- ``GET /api/runs/{run_id}/video_sets``: video đánh giá gom theo bộ (test, full) và kênh (gốc,
  render, sai khác, độ sâu, lớp), kèm khung từng camera nếu có (``engine/videos.py``).
- ``GET /api/runs/{run_id}/artifacts``: nơi lưu kết quả của từng stage và các file trong đó.
- ``GET /api/runs/{run_id}/curve``: đường cong train (PSNR, loss, số Gaussian theo bước).
- ``GET /api/runs/{run_id}/splats``: file Gaussian xem được trong trình duyệt (gói xem trước,
  ``scene.ply``), kèm đường dẫn.
- ``GET /api/runs/{run_id}/render-check``: máy chạy Studio có render được run này bằng GPU không.
"""

import re
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from drive2sim.core import files
from drive2sim.core.schema import Preview, RunRecord
from drive2sim.engine import housekeeping, remote
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.artifacts import run_artifacts, train_curve
from drive2sim.engine.densify_info import load_densify_info
from drive2sim.engine.environment import require_local_environment
from drive2sim.engine.machine import gpu_usage
from drive2sim.engine.registry import produces_artifact
from drive2sim.engine.render_check import gpu_render_supported, render_readiness
from drive2sim.engine.report import train_params
from drive2sim.engine.resolve import ResolveError
from drive2sim.engine.run import (
    D2S_DIR,
    LOG_FILE,
    RUN_FILE,
    create_run,
    insert_stage,
    list_runs,
    load_env,
    load_summary,
    ply_export_stage,
    request_cancel,
    reset_stage,
    run_is_active,
)
from drive2sim.engine.videos import video_sets
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle import executor as kaggle_executor
from drive2sim.stages.commands import CAMERAS_FILE
from drive2sim.studio.api import preview, video_compare
from drive2sim.studio.api.events import stream_events
from drive2sim.studio.api.routes import colab, kaggle
from drive2sim.studio.api.routes.approaches import PlanBody, plan_from_body
from drive2sim.studio.api.state import state_of

router = APIRouter(tags=["runs"])
LOG_TAIL_LINES = 200
FINISHED = ("succeeded", "failed", "cancelled")
EXPORT_STAGES = {
    "ply": "export.ply_world",
    "preview": "export.preview",
    "video": "render.nurec_video",
    "carla": "render.carla",
    "refine": "train.nurec_refine",
}
# Stage chèn vào giữa run (ngay sau stage này) thay vì thêm vào cuối; các stage dùng kết quả
# của nó chạy lại.
INSERT_AFTER = {"train.nurec_refine": "train.nurec"}


class ExportBody(BaseModel):
    """Thân của ``POST /api/runs/{run_id}/export``.

    Attributes:
        kind: ``ply`` (``export.ply_world``), ``preview`` (``export.preview``), ``video``
            (``render.nurec_video``, video đi dọc đoạn đường của run Instant NuRec) hoặc ``carla``
            (``render.carla``, chạy cảnh NuRec trong CARLA rồi ghi video) hoặc ``refine``
            (``train.nurec_refine``, tối ưu tiếp PLY của NuRec; chèn sau ``train.nurec`` và chạy
            lại chấm điểm, PLY, video).
        params: Tham số đổi cho lần chạy này, ghép lên tham số đã lưu của stage (ví dụ
            ``{"ego": "replay", "max_seconds": 20}`` cho ``carla``).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ply", "preview", "video", "carla", "refine"]
    params: Optional[dict[str, object]] = None


class CreateRunBody(PlanBody):
    """Thân của ``POST /api/runs``; cùng ý nghĩa với ``d2s run start``.

    Attributes:
        approach_id: Approach cần chạy.
        executor: Nơi chạy: ``local`` (hàng đợi trên máy), ``kaggle`` hoặc ``colab`` (đẩy lên
            Kaggle, Colab ở nền).
    """

    approach_id: str
    executor: Literal["local", "kaggle", "colab"] = "local"


def _load(request: Request, run_id: str) -> tuple[Path, RunRecord]:
    """Tìm thư mục run và đọc ``run.json``."""
    run_dir = state_of(request).workdir.find_run(run_id)
    return run_dir, files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)


def _remote(workdir: Workdir, record: RunRecord) -> Optional[dict]:
    """Trạng thái phía Kaggle của run chưa tải kết quả lần đẩy mới nhất; ``None`` với run khác.

    Run Kaggle giữ trạng thái cũ trên máy tới khi tải về: ``pending`` với lần đẩy đầu, hoặc
    ``failed`` của lần trước khi đã đẩy lại để chạy tiếp. Studio dựa vào đây để hiện run chưa
    đẩy, đang chạy hay đã xong trên Kaggle (cùng điều kiện với ``kaggle_watch.watched_runs``).
    ``status`` là trạng thái notebook lần hỏi gần nhất (``None`` nếu chưa hỏi được, ví dụ run
    đẩy bằng tài khoản Kaggle khác).
    """
    if record.executor != "kaggle" or record.status == "succeeded":
        return None
    try:
        state = kaggle_executor.load_state(workdir.find_run(record.run_id))
    except (OSError, ValueError, TypeError):
        state = None
    if state is None:
        return (
            {"pushed": False, "status": None, "checked_utc": None}
            if record.status == "pending"
            else None
        )
    if state.pulled_utc and state.pulled_utc >= state.pushed_utc:
        return None  # đã tải kết quả lần đẩy mới nhất: trạng thái trên máy là đúng
    return {"pushed": True, "status": state.remote_status, "checked_utc": state.checked_utc}


def _claim(request: Request, run_dir: Path, run_id: str) -> None:
    """Giữ chỗ cho run trước khi sửa nó, hoặc ``409`` nếu run đang chạy, xếp hàng hay tải về."""
    if run_is_active(run_dir) or not state_of(request).queue.claim(run_dir):
        raise HTTPException(409, f"run {run_id} đang chạy, đang xếp hàng hoặc đang tải về")


def _require_local(request: Request, approach_id: str, stage_ids: list[str]) -> None:
    """Dừng sớm (``422 env_missing``) nếu máy chạy Studio thiếu môi trường để chạy run trên máy.

    Raises:
        EnvironmentMissingError: Nếu thiếu GPU, gói của môi trường chính hoặc file SMPL.
    """
    require_local_environment(
        load_approach(approach_id).codebase,
        needs_smpl=produces_artifact(stage_ids, "smpl"),
        workdir_root=state_of(request).workdir.root,
    )


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """Đọc ``lines`` dòng cuối của một file văn bản; trả chuỗi rỗng nếu không có file."""
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "".join(text.splitlines(True)[-lines:])


@router.post("/api/runs", status_code=202)
def start_run(body: CreateRunBody, request: Request) -> dict:
    """Tạo một run và xếp vào hàng đợi chạy nền.

    Với ``executor: kaggle`` hoặc ``colab``, run được tạo rồi đẩy lên Kaggle hoặc Colab ở nền
    (``/api/jobs``).

    Returns:
        ``run.json`` vừa tạo (trạng thái ``pending``); theo dõi tiến độ qua
        ``/api/runs/{run_id}/events``.
    """
    state = state_of(request)
    scene, split, plan = plan_from_body(state, body.approach_id, body)
    if body.executor == "local":
        _require_local(request, plan.approach_id, [stage.stage_id for stage in plan.stages])
    else:
        remote.require_remote_codebase(plan.approach_id, body.executor.capitalize())
    run_dir = create_run(plan, state.workdir, scene, split, executor=body.executor)
    if body.executor == "kaggle":
        kaggle.submit_push(state, run_dir.name)
    elif body.executor == "colab":
        colab.submit_push(state, run_dir.name)
    else:
        state.queue.submit(run_dir)
    return files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord).model_dump(mode="json")


@router.get("/api/runs")
def get_runs(
    request: Request, scene_id: Optional[str] = None, status: Optional[str] = None
) -> dict:
    """Liệt kê run, mới nhất trước; lọc theo ``scene_id`` và ``status``.

    Returns:
        ``{"items": [run.json cộng "remote", "train_params", ...], "total": n}``; ``remote`` theo
        ``_remote``, ``train_params`` theo ``report.train_params``.
    """
    workdir = state_of(request).workdir
    records = list_runs(workdir, scene_id)
    items = [
        {
            **r.model_dump(mode="json"),
            "remote": _remote(workdir, r),
            "train_params": train_params(r),
        }
        for r in records
        if status in (None, r.status)
    ]
    return {"items": items, "total": len(items)}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    """Trả về ``run.json`` cộng ``remote``, ``summary`` (nếu đã có), ``env``, ``densify``.

    ``remote`` là trạng thái phía Kaggle (``_remote``).

    ``env`` là nội dung ``env.json``; ``densify`` là lịch densify hiệu lực đọc từ ``config.yaml``
    của stage train (``densify_info``), ``null`` khi train chưa bắt đầu.
    """
    run_dir, record = _load(request, run_id)
    summary = load_summary(run_dir, record)
    data = record.model_dump(mode="json")
    data["train_params"] = train_params(record)
    data["remote"] = _remote(state_of(request).workdir, record)
    data["summary"] = summary.model_dump(mode="json") if summary else None
    env = load_env(run_dir)
    data["env"] = env.model_dump(mode="json") if env else None
    data["densify"] = load_densify_info(run_dir)
    data["run_dir"] = state_of(request).workdir.relative(run_dir)
    return data


@router.post("/api/runs/{run_id}/resume", status_code=202)
def resume_run(run_id: str, request: Request) -> dict:
    """Chạy tiếp một run; stage đã xong không chạy lại.

    Returns:
        ``run.json`` hiện tại.
    """
    run_dir, record = _load(request, run_id)
    if record.executor == "local":
        _require_local(
            request, record.approach.approach_id, [stage.stage_id for stage in record.stages]
        )
    _claim(request, run_dir, run_id)
    state_of(request).queue.submit(run_dir, claimed=True)
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
    state = state_of(request)
    run_dir = state.workdir.find_run(run_id)
    _claim(request, run_dir, run_id)
    try:
        return {"removed": housekeeping.delete_run(state.workdir, run_id)}
    finally:
        state.queue.release(run_dir)


@router.post("/api/runs/{run_id}/export", status_code=202)
def export_run(run_id: str, body: ExportBody, request: Request) -> dict:
    """Chạy lại một stage export của run (thêm stage nếu run chưa có), các stage khác giữ nguyên.

    Returns:
        ``run.json`` sau khi đặt lại stage; run được xếp vào hàng đợi.
    """
    state = state_of(request)
    run_dir, current = _load(request, run_id)
    stage_id = EXPORT_STAGES[body.kind]
    if body.kind == "ply":  # run Instant NuRec: export.nurec_ply, không phải export.ply_world
        stage_id = ply_export_stage([s.stage_id for s in current.stages])
    if stage_id not in state.registry:
        raise HTTPException(400, f"stage {stage_id} chưa có")
    _claim(request, run_dir, run_id)
    try:
        if stage_id in INSERT_AFTER:
            record = insert_stage(
                run_dir, stage_id, INSERT_AFTER[stage_id], state.registry, params=body.params
            )
        else:
            record = reset_stage(run_dir, stage_id, state.registry, params=body.params)
    except (ResolveError, ValueError) as error:  # ví dụ video NuRec cho run không có clip NCore
        state.queue.release(run_dir)
        raise HTTPException(400, f"run {run_id} không xuất được {body.kind}: {error}") from None
    except BaseException:
        state.queue.release(run_dir)
        raise
    state.queue.submit(run_dir, claimed=True)
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
    run_dir, _ = _load(request, run_id)
    state = state_of(request)

    def current() -> list[dict]:
        # Đọc run.json sau khi stream đã đăng ký nhận sự kiện (xem stream_events).
        return _initial_events(files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord))

    stream = stream_events(state.broker, run_id, current, last_event_id)
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


LIVE_TAIL_LINES = 40
# Thanh tiến độ tqdm trong log, ví dụ "Predicting in chunks:  62%|██▎ | 5/8 [00:04<00:02, ...]".
TQDM_BAR = re.compile(r"([^\r\n|]*?):?\s*\d+%\|[^|\r\n]*\|\s*(\d+)/(\d+)\s*\[")


def log_progress(text: str) -> Optional[dict]:
    """Thanh tiến độ tqdm cuối cùng trong log: ``{"label", "done", "total"}``, hoặc ``None``."""
    found = TQDM_BAR.findall(text)
    if not found:
        return None
    label, done, total = found[-1]
    return {"label": label.strip(), "done": int(done), "total": int(total)}


@router.get("/api/runs/{run_id}/live")
def run_live(run_id: str, request: Request) -> dict:
    """Tình hình lúc này của run: stage đang chạy, log cuối của nó, log im bao lâu, tải GPU.

    Log không có dòng mới lâu mà GPU rảnh thường là lệnh con đã treo; Studio dựa vào đây để báo.

    Returns:
        ``{"active", "stage_id", "log_tail", "log_idle_s", "log_progress", "gpus"}``;
        ``stage_id`` là ``None`` khi không có stage nào đang chạy (khi đó log là log engine).
    """
    run_dir, record = _load(request, run_id)
    stage = next((s for s in record.stages if s.status == "running"), None)
    log_path = (
        run_dir / D2S_DIR / "stages" / stage.stage_id / LOG_FILE
        if stage
        else run_dir / D2S_DIR / LOG_FILE
    )
    idle = time.time() - log_path.stat().st_mtime if log_path.is_file() else None
    active = run_is_active(run_dir)
    return {
        "active": active,
        "stage_id": stage.stage_id if stage else None,
        "log_tail": _tail(log_path, LIVE_TAIL_LINES),
        "log_idle_s": round(idle, 1) if idle is not None else None,
        "log_progress": log_progress(_tail(log_path, LIVE_TAIL_LINES)) if stage else None,
        "gpus": gpu_usage() if active else [],
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
        ``{"items": [{"name", "stage_id", "url", "preview_url", "ready"}], "total": n}``; ``url``
        là video gốc qua ``/files/``, ``preview_url`` là bản trình duyệt phát được, ``ready`` là
        ``False`` khi video còn đang được ghi.
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
                "preview_url": (
                    f"/api/runs/{run_id}/videos/preview?stage_id={stage_id}"
                    f"&name={path.relative_to(stages_dir / stage_id).as_posix()}"
                ),
                "ready": preview.is_complete(path),
            }
        )
    return {"items": items, "total": len(items)}


@router.get("/api/runs/{run_id}/video_sets")
def run_video_sets(run_id: str, request: Request) -> dict:
    """Video đánh giá của run, gom theo bộ và kênh cho trình phát so sánh.

    Returns:
        ``{"items": [...], "total": n}`` theo ``engine.videos.video_sets``.
    """
    run_dir, _ = _load(request, run_id)
    workdir = state_of(request).workdir
    items = video_sets(
        run_dir / D2S_DIR / "stages", lambda path: f"/files/{workdir.relative(path)}"
    )
    return {"items": items, "total": len(items)}


@router.get("/api/runs/{run_id}/videos/preview")
def run_video_preview(run_id: str, stage_id: str, name: str, request: Request) -> FileResponse:
    """Bản xem trước của một video: H.264 rộng tối đa 1920 điểm ảnh, tạo lần đầu rồi dùng lại.

    Returns:
        File MP4 để phát; ``404`` nếu không có video, ``409`` khi video đang được ghi hoặc
        ffmpeg lỗi.
    """
    run_dir, _ = _load(request, run_id)
    stage_dir = (run_dir / D2S_DIR / "stages" / stage_id).resolve()
    source = (stage_dir / name).resolve()
    stages_dir = (run_dir / D2S_DIR / "stages").resolve()
    if stages_dir not in stage_dir.parents or stage_dir not in source.parents:
        raise HTTPException(404, f"không có video {stage_id}/{name}")
    if source.suffix.lower() != ".mp4" or not source.is_file():
        raise HTTPException(404, f"không có video {stage_id}/{name}")
    target = preview.preview_path(run_dir / D2S_DIR, stage_id, name)
    try:
        playable = preview.build_preview(source, target)
    except preview.PreviewError as exc:
        raise HTTPException(409, str(exc)) from exc
    return FileResponse(playable, media_type="video/mp4")


class CompareSide(BaseModel):
    """Một bên của video ghép: video ``stage_id``/``name`` của run ``run_id``."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    stage_id: str
    name: str


class CompareVideosBody(BaseModel):
    """Thân của ``POST /api/compare/videos``.

    Attributes:
        a: Bên trên (``vstack``) hoặc bên trái (``hstack``).
        b: Bên còn lại.
        layout: ``vstack`` trên dưới, ``hstack`` cạnh nhau.
    """

    model_config = ConfigDict(extra="forbid")

    a: CompareSide
    b: CompareSide
    layout: Literal["vstack", "hstack"] = "vstack"


@router.post("/api/compare/videos")
def compare_videos(body: CompareVideosBody, request: Request) -> dict:
    """Ghép video của hai run thành một file (``studio/api/video_compare.py``).

    Returns:
        ``{"url", "name", "layout"}``; ``404`` nếu không có video, ``409`` khi video đang ghi,
        không có ffmpeg hoặc ffmpeg lỗi, ``400`` khi hai bên là một video.
    """
    workdir = state_of(request).workdir
    sources = []
    for side in (body.a, body.b):
        run_dir, _ = _load(request, side.run_id)
        try:
            sources.append(
                video_compare.video_path(run_dir / D2S_DIR / "stages", side.stage_id, side.name)
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None
    try:
        name = video_compare.compare_video(
            workdir.root, sources[0], body.a.run_id, sources[1], body.b.run_id, body.layout
        )
    except video_compare.CompareError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"url": f"/api/compare/videos/{name}", "name": name, "layout": body.layout}


@router.get("/api/compare/videos/{name}")
def compare_video_file(name: str, request: Request) -> FileResponse:
    """File video ghép đã tạo; ``404`` nếu tên sai dạng hoặc không có."""
    path = state_of(request).workdir.root / video_compare.COMPARE_DIR / name
    if not re.match(video_compare.FILE_PATTERN, name) or not path.is_file():
        raise HTTPException(404, f"không có video ghép {name}")
    return FileResponse(path, media_type="video/mp4", filename=name)


@router.get("/api/runs/{run_id}/splats")
def run_splats(run_id: str, request: Request) -> dict:
    """Liệt kê những gì trang xem 3D mở được của một run.

    Returns:
        ``{"preview": {...} | null, "ply": url | null}``. ``preview`` (khi đã chạy
        ``export.preview``) gồm nội dung ``preview.json`` cộng ``background_url``,
        ``actor_urls`` (``track_id`` → url), ``poses_url``, ``human_urls`` (``track_id`` →
        ``{"splats", "skin"}``), ``human_poses_url``, ``human_bones_url``, ``sky_url`` và
        ``color_affine_url``;
        ``ply`` là ``scene.ply`` của ``export.ply_world``; ``cameras`` là url của
        ``cameras.json`` (pose, thông số camera theo khung, cùng hệ toạ độ với các file trên),
        ``null`` nếu chưa có.
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
        ext = data["splat_format"]
        preview = {
            **data,
            "background_url": url(preview_dir / f"background.{ext}"),
            "actor_urls": {
                track: url(preview_dir / "actors" / f"{track}.{ext}")
                for track in data["rigid_actors"]
            },
            "poses_url": url(preview_dir / "actor_poses.json"),
            "human_urls": {
                track: {
                    "splats": url(preview_dir / "humans" / f"{track}.{ext}"),
                    "skin": url(preview_dir / "humans" / f"{track}.skin"),
                }
                for track in data["humans"]
            },
            "human_poses_url": url(preview_dir / "human_poses.json"),
            "human_bones_url": url(preview_dir / "human_bones.bin"),
            "sky_url": url(preview_dir / data["sky"]) if data["sky"] else None,
            "color_affine_url": (
                url(preview_dir / data["color_affine"]) if data["color_affine"] else None
            ),
        }
    # Pose và thông số camera theo khung (train.py ghi, stage xuất chép theo): để mở cảnh ở đúng
    # góc của video. Lấy ở gói xem trước, rồi scene.ply, rồi thư mục train.
    cameras = next(
        (
            found
            for found in (
                url(preview_dir / CAMERAS_FILE),
                url(stages_dir / "export.ply_world" / CAMERAS_FILE),
                *(url(path) for path in sorted(stages_dir.glob(f"train.*/{CAMERAS_FILE}"))),
            )
            if found
        ),
        None,
    )
    return {
        "preview": preview,
        "ply": url(stages_dir / "export.ply_world" / "scene.ply")
        or url(stages_dir / "export.nurec_ply" / "scene.ply"),
        "cameras": cameras,
        "gpu": gpu_render_supported(run_dir),
    }
