"""Dựng ứng dụng FastAPI của Studio (ARCHITECTURE mục 2.10, 4.9; CONTRACTS mục 11).

API chỉ là một lớp mỏng trên engine: mỗi route gọi đúng hàm mà lệnh CLI tương ứng gọi, nên Studio
và CLI luôn kiểm tra giống nhau và báo cùng lỗi.

File này chứa:
- ``error_response``: dựng phản hồi lỗi theo hình dạng chung.
- ``create_app``: dựng app, gắn các nhóm route ``/api/...``, route ``/files/...`` phục vụ video và
  ảnh, giao diện tĩnh ở ``/`` nếu đã build, và các bộ đổi lỗi thành HTTP theo bảng CONTRACTS
  mục 11.3.

Lệnh ``d2s studio`` (``drive2sim/cli/studio.py``) chạy app này, chỉ nghe trên ``127.0.0.1``.
"""

import atexit
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.cache import CacheBusyError
from drive2sim.engine.housekeeping import InUseError
from drive2sim.engine.registry import REGISTRY, StageRegistry, load_builtin_stages
from drive2sim.engine.resolve import BlockedError, ResolveError
from drive2sim.engine.run import RunActiveError
from drive2sim.engine.scenes import SplitError, SplitExistsError
from drive2sim.engine.workdir import Workdir, WorkdirError
from drive2sim.ingest.dispatch import IngestRequestError
from drive2sim.kaggle.client import KaggleError
from drive2sim.kaggle.setup import CredentialExistsError
from drive2sim.studio.api.events import EventBroker
from drive2sim.studio.api.gpu_viewer import GpuViewerError, GpuViewers
from drive2sim.studio.api.jobs import BackgroundJobs, RunQueue
from drive2sim.studio.api.kaggle_watch import KaggleWatcher
from drive2sim.studio.api.picker import PickerError, pick_folder
from drive2sim.studio.api.reveal import RevealError, reveal
from drive2sim.studio.api.routes import approaches, cache, kaggle, report, runs, scenes, sources
from drive2sim.studio.api.state import StudioState

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


# Tên máy được phép trong Host và Origin: Studio chỉ phục vụ chính máy này (kể cả qua SSH port
# forward, trình duyệt vẫn gọi localhost). Vite ở chế độ dev chuyển tiếp nguyên Host của nó.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _hostname(value: str) -> str:
    """Phần tên máy của ``host[:port]`` hoặc của một URL ``scheme://host[:port]``."""
    value = value.split("://", 1)[-1].split("/", 1)[0]
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.rsplit(":", 1)[0] if ":" in value else value


def is_local_request(host: Optional[str], origin: Optional[str], method: str) -> bool:
    """Request có đến từ chính máy này không (chống DNS rebinding và request chéo trang).

    - ``Host`` phải là tên máy cục bộ: một trang web lạ dùng DNS rebinding trỏ tên miền của nó
      về 127.0.0.1 thì ``Host`` vẫn là tên miền đó.
    - Lệnh ghi (POST, DELETE...) có ``Origin`` thì ``Origin`` cũng phải là máy cục bộ: trình duyệt
      gửi POST "đơn giản" chéo trang mà không hỏi trước (CORS preflight), nên chỉ nghe trên
      127.0.0.1 không chặn được trang lạ gọi ``/api/runs/.../cancel``.

    Args:
        host: Header ``Host``.
        origin: Header ``Origin``, nếu có.
        method: Phương thức HTTP.

    Returns:
        ``True`` nếu cho qua.
    """
    if not host or _hostname(host) not in LOCAL_HOSTS:
        return False
    if method.upper() not in SAFE_METHODS and origin:
        # "null": trang mở từ file:// hoặc sandbox, không biết từ đâu, nên không cho ghi.
        return origin != "null" and _hostname(origin) in LOCAL_HOSTS
    return True


def error_response(
    status: int, code: str, message: str, details: Optional[dict] = None
) -> JSONResponse:
    """Dựng phản hồi lỗi theo hình dạng chung của CONTRACTS mục 11.3.

    Args:
        status: Mã HTTP.
        code: Mã lỗi trong bảng ``ERROR_CODES``.
        message: Thông báo tiếng Việt cho người dùng.
        details: Thông tin thêm, ví dụ danh sách lý do bị khoá.

    Returns:
        ``{"error": {"code", "message", "details"}}`` với mã HTTP tương ứng.
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


def _install_error_handlers(app: FastAPI) -> None:
    """Đổi các lỗi của engine thành phản hồi HTTP theo bảng CONTRACTS mục 11.3."""

    @app.exception_handler(BlockedError)
    async def blocked(request: Request, error: BlockedError) -> JSONResponse:
        reasons = [
            {"message": r.message, "option": r.option, "choice": r.choice} for r in error.reasons
        ]
        return error_response(422, "blocked", "approach không chạy được", {"reasons": reasons})

    @app.exception_handler(SplitExistsError)
    async def conflict_split(request: Request, error: SplitExistsError) -> JSONResponse:
        return error_response(409, "conflict", str(error))

    @app.exception_handler(CacheBusyError)
    async def conflict_cache(request: Request, error: CacheBusyError) -> JSONResponse:
        return error_response(409, "conflict", str(error))

    @app.exception_handler(InUseError)
    async def conflict_in_use(request: Request, error: InUseError) -> JSONResponse:
        return error_response(409, "conflict", str(error))

    @app.exception_handler(RunActiveError)
    async def conflict_run_active(request: Request, error: RunActiveError) -> JSONResponse:
        return error_response(409, "conflict", str(error))

    @app.exception_handler(IngestRequestError)
    async def invalid_ingest(request: Request, error: IngestRequestError) -> JSONResponse:
        return error_response(400, "invalid_request", str(error))

    @app.exception_handler(CredentialExistsError)
    async def credential_exists(request: Request, error: CredentialExistsError) -> JSONResponse:
        return error_response(409, "conflict", str(error))

    @app.exception_handler(KaggleError)
    async def kaggle_failed(request: Request, error: KaggleError) -> JSONResponse:
        return error_response(502, "kaggle_failed", str(error))

    @app.exception_handler(ResolveError)
    async def invalid_resolve(request: Request, error: ResolveError) -> JSONResponse:
        return error_response(400, "invalid_request", str(error))

    @app.exception_handler(SplitError)
    async def invalid_split(request: Request, error: SplitError) -> JSONResponse:
        return error_response(400, "invalid_request", str(error))

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request: Request, error: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        return error_response(400, "invalid_request", f"yêu cầu không hợp lệ: {problems}")

    @app.exception_handler(WorkdirError)
    async def not_found(request: Request, error: WorkdirError) -> JSONResponse:
        return error_response(404, "not_found", str(error))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        code = {404: "not_found", 409: "conflict", 400: "invalid_request"}.get(
            error.status_code, "internal"
        )
        return error_response(error.status_code, code, str(error.detail))

    @app.exception_handler(ValueError)
    async def invalid_data(request: Request, error: ValueError) -> JSONResponse:
        # Lỗi kỹ thuật tiếng Anh từ drive2sim.core (file sai schema...), bọc tiếng Việt.
        return error_response(400, "invalid_request", f"dữ liệu không hợp lệ: {error}")


def create_app(
    workdir: Workdir,
    registry: StageRegistry = REGISTRY,
    frontend_dir: Optional[Path] = FRONTEND_DIST,
    kaggle_watch_s: Optional[float] = None,
) -> FastAPI:
    """Dựng ứng dụng Studio.

    Args:
        workdir: Thư mục làm việc.
        registry: Bảng stage; test truyền bảng riêng có stage giả.
        frontend_dir: Thư mục giao diện đã build (``npm run build``); phục vụ ở ``/`` nếu có.
        kaggle_watch_s: Có thì cứ chừng ấy giây hỏi trạng thái các run trên Kaggle và tự tải kết
            quả về khi notebook xong (``KaggleWatcher``); ``None`` thì không theo dõi (test).

    Returns:
        Ứng dụng FastAPI.
    """
    if registry is REGISTRY:
        load_builtin_stages()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Bật theo dõi Kaggle khi mở; khi tắt Studio, chờ run đang chạy xong rồi mới thoát."""
        watcher = None
        if kaggle_watch_s:
            watcher = KaggleWatcher(
                workdir,
                app.state.studio.jobs,
                interval_s=kaggle_watch_s,
                queue=app.state.studio.queue,
            )
            watcher.start()
        yield
        if watcher is not None:
            watcher.stop()
        gpu_viewers.stop()
        app.state.studio.queue.shutdown()
        app.state.studio.jobs.shutdown()

    app = FastAPI(
        title="drive2sim Studio",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    broker = EventBroker()
    app.state.studio = StudioState(
        workdir, broker, RunQueue(workdir, broker, registry), registry, BackgroundJobs(workdir)
    )
    gpu_viewers = GpuViewers(workdir)
    # Viewer chạy trong nhóm tiến trình riêng nên không chết theo Studio: Ctrl+C lần hai (bỏ qua
    # bước dọn dẹp của lifespan) vẫn phải tắt nó, nếu không nó giữ VRAM và cổng.
    atexit.register(gpu_viewers.stop)
    _install_error_handlers(app)

    @app.middleware("http")
    async def local_only(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Từ chối request không đến từ chính máy này (``is_local_request``) bằng ``403``."""
        if not is_local_request(
            request.headers.get("host"), request.headers.get("origin"), request.method
        ):
            return error_response(
                403, "forbidden", "Studio chỉ nhận request từ chính máy này (localhost)"
            )
        return await call_next(request)

    for module in (sources, scenes, approaches, runs, report, cache, kaggle):
        app.include_router(module.router)

    class RevealBody(BaseModel):
        """Đường dẫn cần mở, tương đối với thư mục làm việc."""

        model_config = ConfigDict(extra="forbid")
        path: str = Field(min_length=1, max_length=1024)

    @app.post("/api/reveal")
    def reveal_path(body: RevealBody) -> dict:
        """Mở file hoặc thư mục trong trình quản lý file của máy chạy Studio.

        Returns:
            ``{"path": <đường dẫn tuyệt đối đã mở>}``; không mở được thì ``409`` kèm lý do.
        """
        try:
            return {"path": str(reveal(workdir, body.path))}
        except RevealError as error:
            raise HTTPException(409, str(error)) from None

    class PickFolderBody(BaseModel):
        start: Optional[str] = Field(default=None, max_length=1024)

    @app.post("/api/pick-folder")
    def pick_folder_route(body: PickFolderBody) -> dict:
        """Mở hộp thoại chọn thư mục của hệ điều hành trên máy chạy Studio và chờ người dùng chọn.

        Returns:
            ``{"path": <thư mục đã chọn>}``, ``path`` là ``null`` nếu người dùng đóng hộp thoại;
            máy không mở được hộp thoại thì ``409`` kèm lý do.
        """
        try:
            return {"path": pick_folder(body.start)}
        except PickerError as error:
            raise HTTPException(409, str(error)) from None

    @app.post("/api/runs/{run_id}/gpu-viewer")
    def start_gpu_viewer(run_id: str) -> dict:
        """Bật viewer GPU của DriveStudio cho run (tắt viewer đang chạy nếu có).

        Returns:
            Trạng thái viewer như ``GET /api/gpu-viewer``; máy chưa đủ điều kiện hoặc run thiếu
            checkpoint thì ``409`` kèm lý do.
        """
        try:
            return gpu_viewers.start(run_id)
        except GpuViewerError as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/gpu-viewer")
    def gpu_viewer_status() -> dict:
        """Trạng thái viewer GPU: ``state``, ``run_id``, ``url`` khi đã sẵn sàng, phần cuối log."""
        return gpu_viewers.status()

    @app.delete("/api/gpu-viewer")
    def stop_gpu_viewer() -> dict:
        """Tắt viewer GPU đang chạy."""
        gpu_viewers.stop()
        return gpu_viewers.status()

    @app.get("/api/studio")
    def studio_info() -> dict:
        """Thông tin của Studio đang chạy: thư mục làm việc, và có phải dữ liệu mẫu không.

        Returns:
            ``{"workdir": <đường dẫn>, "demo": <true nếu là thư mục dữ liệu mẫu>}``.
        """
        return {"workdir": str(workdir.root), "demo": workdir.is_demo}

    @app.get("/files/{path:path}")
    def serve_file(path: str) -> FileResponse:
        """Phục vụ một file trong thư mục làm việc (video, ảnh, PLY), đường dẫn tương đối.

        Chặn mọi đường dẫn đi ra ngoài thư mục làm việc (ví dụ ``../``).
        """
        target = (workdir.root / path).resolve()
        if workdir.root not in target.parents or not target.is_file():
            raise HTTPException(404, f"không có file {path}")
        return FileResponse(target)

    @app.middleware("http")
    async def no_cache_index(request: Request, call_next):  # noqa: ANN001, ANN202
        """Bắt trình duyệt hỏi lại ``index.html`` mỗi lần, để bản build mới có hiệu lực ngay.

        File trong ``assets/`` mang hash trong tên nên vẫn được cache bình thường.
        """
        response = await call_next(request)
        if request.url.path in ("/", "/index.html"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    if frontend_dir is not None and (frontend_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app
