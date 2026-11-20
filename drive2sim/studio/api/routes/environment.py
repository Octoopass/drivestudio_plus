"""Route cài môi trường: ``/api/environment/install`` (CONTRACTS mục 11.1).

Cùng việc với ``bash envs/install_main.sh`` chạy tay, nhưng bấm được từ Studio khi bảng kiểm tra
môi trường báo thiếu (``drive2sim/engine/env_install.py``).

File này chứa:
- ``GET /api/environment/install``: trạng thái lần cài (bước, log cuối, kết quả).
- ``POST /api/environment/install``: bắt đầu cài, hoặc cài tiếp từ một bước.
- ``POST /api/environment/install/cancel``: dừng lần cài đang chạy.
"""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.engine.env_install import InstallFlow
from drive2sim.studio.api.state import state_of

router = APIRouter(prefix="/api/environment", tags=["environment"])


class InstallBody(BaseModel):
    """Thân của ``POST /api/environment/install``.

    Attributes:
        env: Môi trường cần cài.
        from_step: Cài tiếp từ bước này; 1 là từ đầu.
    """

    model_config = ConfigDict(extra="forbid")

    env: Literal["main"] = "main"
    from_step: int = Field(1, ge=1, le=20)


def install_flow(request: Request) -> InstallFlow:
    """Lần cài của Studio: một bản cho cả app, log ở ``<thư mục làm việc>/logs/``."""
    app_state = request.app.state
    if not hasattr(app_state, "env_install"):
        app_state.env_install = InstallFlow(state_of(request).workdir.root / "logs")
    return app_state.env_install


@router.get("/install")
def install_status(request: Request) -> dict:
    """Trạng thái lần cài (``idle``, ``running``, ``succeeded``, ``failed``, ``cancelled``)."""
    return install_flow(request).view()


@router.post("/install")
def install_start(body: InstallBody, request: Request) -> dict:
    """Bắt đầu cài môi trường bằng Python đang chạy Studio; máy không cài được thì ``400``."""
    return install_flow(request).start(body.env, body.from_step)


@router.post("/install/cancel")
def install_cancel(request: Request) -> dict:
    """Dừng lần cài đang chạy (cài tiếp sau bằng ``from_step``)."""
    return install_flow(request).cancel()
