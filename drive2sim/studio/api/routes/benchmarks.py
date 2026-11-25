"""Route benchmark: ``/api/benchmarks`` (CONTRACTS mục 11.1; benchmark ở CONTRACTS mục 6.5).

File này chứa:
- ``GET /api/benchmarks``: liệt kê các file định nghĩa ``benchmarks/<tên>.yaml`` trong thư mục
  làm việc.
- ``GET /api/benchmarks/{name}``: đọc file định nghĩa đó và trả ``d2s-benchresult`` do
  ``drive2sim.engine.benchmark.run_benchmark`` dựng, giống ``d2s benchmark`` nhưng không ghi file.

Chỉ đọc file kết quả của các run, không chạy GPU. Kết quả tính lại mỗi lần gọi, nên run mới xong
hiện ngay trong lần mở sau.
"""

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from drive2sim.engine.benchmark import run_benchmark
from drive2sim.studio.api.state import state_of

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])

# Tên file không có đường dẫn: chữ, số, gạch dưới, gạch ngang, chấm giữa; không bắt đầu bằng chấm.
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


@router.get("")
def list_benchmarks(request: Request) -> dict:
    """Liệt kê file định nghĩa benchmark trong ``benchmarks/``.

    Args:
        request: Request đang xử lý.

    Returns:
        ``{"items": [{"name": ...}], "total": n}``, sắp theo tên; rỗng khi chưa có thư mục.
    """
    directory = state_of(request).workdir.benchmarks_dir
    names = sorted(path.stem for path in directory.glob("*.yaml")) if directory.is_dir() else []
    return {"items": [{"name": name} for name in names], "total": len(names)}


@router.get("/{name}")
def get_benchmark(name: str, request: Request) -> dict:
    """Dựng kết quả của một file định nghĩa benchmark.

    Args:
        name: Tên file không có đuôi ``.yaml``.
        request: Request đang xử lý.

    Returns:
        ``d2s-benchresult`` dạng JSON (CONTRACTS mục 6.5).

    Raises:
        HTTPException: ``400`` nếu tên có ký tự lạ, file sai schema hoặc một run không hợp lệ;
            ``404`` nếu không có file.
    """
    if not NAME_RE.match(name):
        raise HTTPException(status_code=400, detail=f"tên benchmark không hợp lệ: {name!r}")
    workdir = state_of(request).workdir
    path: Path = workdir.benchmarks_dir / f"{name}.yaml"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"không có benchmark {name!r}")
    try:
        result = run_benchmark(workdir, path)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    return result.model_dump(mode="json")
