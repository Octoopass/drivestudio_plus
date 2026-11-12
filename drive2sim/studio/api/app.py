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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from drive2sim.engine.cache import CacheBusyError
from drive2sim.engine.registry import REGISTRY, StageRegistry, load_builtin_stages
from drive2sim.engine.resolve import BlockedError, ResolveError
from drive2sim.engine.scenes import SplitError, SplitExistsError
from drive2sim.engine.workdir import Workdir, WorkdirError
from drive2sim.studio.api.events import EventBroker
from drive2sim.studio.api.jobs import RunQueue
from drive2sim.studio.api.routes import approaches, report, runs, scenes
from drive2sim.studio.api.state import StudioState

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


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
) -> FastAPI:
    """Dựng ứng dụng Studio.

    Args:
        workdir: Thư mục làm việc.
        registry: Bảng stage; test truyền bảng riêng có stage giả.
        frontend_dir: Thư mục giao diện đã build (``npm run build``); phục vụ ở ``/`` nếu có.

    Returns:
        Ứng dụng FastAPI.
    """
    if registry is REGISTRY:
        load_builtin_stages()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Khi tắt Studio, chờ run đang chạy xong rồi mới thoát."""
        yield
        app.state.studio.queue.shutdown()

    app = FastAPI(
        title="drive2sim Studio",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    broker = EventBroker()
    app.state.studio = StudioState(workdir, broker, RunQueue(workdir, broker, registry), registry)
    _install_error_handlers(app)
    for module in (scenes, approaches, runs, report):
        app.include_router(module.router)

    @app.get("/files/{path:path}")
    def serve_file(path: str) -> FileResponse:
        """Phục vụ một file trong thư mục làm việc (video, ảnh, PLY), đường dẫn tương đối.

        Chặn mọi đường dẫn đi ra ngoài thư mục làm việc (ví dụ ``../``).
        """
        target = (workdir.root / path).resolve()
        if workdir.root not in target.parents or not target.is_file():
            raise HTTPException(404, f"không có file {path}")
        return FileResponse(target)

    if frontend_dir is not None and (frontend_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app
