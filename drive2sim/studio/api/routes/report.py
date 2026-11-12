"""Route bảng so sánh: ``GET /api/report`` (SPEC mục 8 nhóm D).

Trả các dòng của ``drive2sim.engine.report.report_rows`` dạng JSON, hoặc CSV khi
``format=csv`` để xuất cho báo cáo.
"""

from dataclasses import asdict
from typing import Annotated, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse

from drive2sim.engine.report import report_rows, rows_to_csv
from drive2sim.studio.api.state import state_of

router = APIRouter(prefix="/api/report", tags=["report"])


@router.get("")
def get_report(
    request: Request,
    scene_id: Optional[str] = None,
    split_id: Optional[str] = None,
    run_id: Annotated[Optional[list[str]], Query()] = None,
    format: str = "json",
) -> object:
    """Dựng bảng so sánh các run.

    Args:
        request: Request đang xử lý.
        scene_id: Chỉ lấy run của cảnh này.
        split_id: Chỉ lấy run dùng split này.
        run_id: Chỉ lấy các run này; lặp lại tham số để chọn nhiều run.
        format: ``json`` (mặc định) hoặc ``csv``.

    Returns:
        ``{"items": [...], "total": n}``, hoặc văn bản CSV.
    """
    rows = report_rows(state_of(request).workdir, scene_id, split_id, run_id)
    if format == "csv":
        return PlainTextResponse(rows_to_csv(rows), media_type="text/csv")
    return {"items": [asdict(row) for row in rows], "total": len(rows)}
