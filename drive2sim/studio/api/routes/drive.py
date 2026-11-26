"""API của trang Lái thử: lái tay xe ego trong CARLA, nhìn bằng ảnh Gaussian của run NuRec.

- ``POST /api/runs/{run_id}/drive``: bật phiên lái thử cho run (tắt phiên cũ), chạy nền.
- ``GET /api/drive``: trạng thái phiên (``starting``, ``running``...), tốc độ, khung/giây.
- ``POST /api/drive/control``: lệnh lái ``throttle``, ``steer``, ``brake``, ``reverse``,
  ``hand_brake``.
- ``GET /api/drive/frame/{kind}``: JPEG mới nhất, ``gaussian`` hoặc ``carla``.
- ``POST /api/drive/reset``: đưa xe về đầu quỹ đạo.
- ``DELETE /api/drive``: tắt phiên, trả GPU.

Phiên do ``drive2sim.studio.api.carla_drive.CarlaDrive`` quản lý (``app.state.carla_drive``).
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.studio.api.carla_drive import CarlaDrive, CarlaDriveError

router = APIRouter(tags=["drive"])


class DriveStartBody(BaseModel):
    """Thân của ``POST /api/runs/{run_id}/drive``.

    Attributes:
        camera: Camera của cảnh dùng để nhìn (rỗng: camera trước).
        ratio: Tỉ lệ độ phân giải ảnh Gaussian.
        fps: Số ảnh Gaussian mỗi giây (cũng định nhịp tick: hai tick mỗi ảnh).
    """

    model_config = ConfigDict(extra="forbid")

    camera: str = ""
    ratio: float = Field(default=0.5, ge=0.1, le=1)
    fps: int = Field(default=20, ge=5, le=30)


class DriveControlBody(BaseModel):
    """Lệnh lái, như ``carla.VehicleControl``."""

    model_config = ConfigDict(extra="forbid")

    throttle: float = Field(default=0, ge=0, le=1)
    steer: float = Field(default=0, ge=-1, le=1)
    brake: float = Field(default=0, ge=0, le=1)
    reverse: bool = False
    hand_brake: bool = False


def _drive(request: Request) -> CarlaDrive:
    return request.app.state.carla_drive


@router.post("/api/runs/{run_id}/drive", status_code=202)
def start_drive(run_id: str, body: DriveStartBody, request: Request) -> dict:
    """Bật phiên lái thử cho run; ``409`` nếu run không lái được (không phải NuRec, thiếu PLY)."""
    try:
        return _drive(request).start(run_id, body.camera, body.ratio, body.fps)
    except CarlaDriveError as error:
        raise HTTPException(409, str(error)) from None


@router.get("/api/drive")
def drive_status(request: Request) -> dict:
    """Trạng thái phiên lái thử."""
    return _drive(request).status()


@router.post("/api/drive/control")
def drive_control(body: DriveControlBody, request: Request) -> dict:
    """Chuyển lệnh lái; ``409`` nếu chưa có phiên."""
    try:
        _drive(request).control(body.model_dump())
    except CarlaDriveError as error:
        raise HTTPException(409, str(error)) from None
    return {}


@router.get("/api/drive/frame/{kind}")
async def drive_frame(kind: str, request: Request) -> Response:
    """JPEG mới nhất; ``204`` khi chưa có ảnh, ``409`` khi chưa có phiên."""
    try:
        image = await run_in_threadpool(_drive(request).frame, kind)
    except CarlaDriveError as error:
        raise HTTPException(409, str(error)) from None
    if image is None:
        return Response(status_code=204)
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/api/drive/reset")
def drive_reset(request: Request) -> dict:
    """Đưa xe về đầu quỹ đạo."""
    try:
        _drive(request).reset()
    except CarlaDriveError as error:
        raise HTTPException(409, str(error)) from None
    return {}


@router.delete("/api/drive")
def stop_drive(request: Request) -> dict:
    """Tắt phiên lái thử."""
    return _drive(request).stop()
