"""Trạng thái dùng chung của Studio mà mọi route cần.

File này chứa ``StudioState`` (thư mục làm việc, nơi trung chuyển sự kiện, hàng đợi chạy run,
bảng stage, hàng đợi ingest) và ``state_of`` để lấy nó từ một request. Tách riêng khỏi
``app.py`` để các file route import được mà không tạo vòng import.
"""

from dataclasses import dataclass

from fastapi import Request

from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api.events import EventBroker
from drive2sim.studio.api.jobs import IngestJobs, RunQueue


@dataclass
class StudioState:
    """Trạng thái dùng chung của Studio, gắn vào ``app.state.studio`` khi dựng app.

    Attributes:
        workdir: Thư mục làm việc.
        broker: Nơi trung chuyển sự kiện tiến độ.
        queue: Hàng đợi chạy run nền.
        registry: Bảng stage.
        ingest_jobs: Hàng đợi ingest nền.
    """

    workdir: Workdir
    broker: EventBroker
    queue: RunQueue
    registry: StageRegistry
    ingest_jobs: IngestJobs


def state_of(request: Request) -> StudioState:
    """Lấy ``StudioState`` mà ``create_app`` gắn vào app.

    Args:
        request: Request đang xử lý.

    Returns:
        Trạng thái dùng chung.
    """
    return request.app.state.studio
