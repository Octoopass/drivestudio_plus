"""Dựng bảng so sánh nhiều run trên cùng cảnh và split (SPEC mục 8 nhóm D, mục 9).

Studio (bảng xếp hạng, biểu đồ chất lượng theo thời gian train và VRAM) và lệnh ``d2s report``
cùng dùng file này.

File này chứa:
- ``ReportRow``: một dòng của bảng: run, approach, lựa chọn của các ô, metric, thời gian, VRAM.
- ``report_rows``: đọc các run đã xong và dựng bảng.
- ``rows_to_csv``: xuất bảng ra CSV cho báo cáo.

Thời gian train và VRAM lấy từ ``run.json`` (engine đo cho từng stage), vì ``summary.json`` của
DriveStudio để trống hai trường này (CONTRACTS mục 6.3).
"""

import csv
import io
from dataclasses import asdict, dataclass, field
from typing import Optional

from drive2sim.engine.run import list_runs, load_summary
from drive2sim.engine.workdir import Workdir

TRAIN_PREFIX = "train."


@dataclass
class ReportRow:
    """Một dòng của bảng so sánh.

    Attributes:
        run_id: Run.
        scene_id: Cảnh.
        split_id: Split.
        approach_id: Approach.
        options: Lựa chọn đã quyết định của các ô, ví dụ ``{"renderer": "3dgeer"}``.
        status: Trạng thái run.
        psnr: PSNR trên tập test; ``None`` nếu run chưa có ``summary.json``.
        ssim: SSIM.
        lpips: LPIPS.
        human_psnr: PSNR vùng người, nếu có.
        vehicle_psnr: PSNR vùng xe, nếu có.
        train_time_s: Tổng thời gian của các stage ``train.*``, giây.
        peak_vram_mb: VRAM cao nhất trong các stage ``train.*``, megabyte.
    """

    run_id: str
    scene_id: str
    split_id: str
    approach_id: str
    options: dict[str, object] = field(default_factory=dict)
    status: str = ""
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    human_psnr: Optional[float] = None
    vehicle_psnr: Optional[float] = None
    train_time_s: Optional[float] = None
    peak_vram_mb: Optional[int] = None


def report_rows(
    workdir: Workdir,
    scene_id: Optional[str] = None,
    split_id: Optional[str] = None,
    run_ids: Optional[list[str]] = None,
) -> list[ReportRow]:
    """Dựng bảng so sánh các run.

    Chỉ so sánh được các run cùng cảnh và cùng split; người gọi lọc bằng ``scene_id``, ``split_id``
    hoặc chọn hẳn ``run_ids``. Dòng sắp theo PSNR giảm dần; run chưa có kết quả đánh giá ở cuối.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Chỉ lấy run của cảnh này.
        split_id: Chỉ lấy run dùng split này.
        run_ids: Chỉ lấy các run này.

    Returns:
        Các dòng của bảng.
    """
    rows = []
    for record in list_runs(workdir, scene_id):
        if split_id and record.split_id != split_id:
            continue
        if run_ids is not None and record.run_id not in run_ids:
            continue
        train_stages = [s for s in record.stages if s.stage_id.startswith(TRAIN_PREFIX)]
        row = ReportRow(
            run_id=record.run_id,
            scene_id=record.scene_id,
            split_id=record.split_id,
            approach_id=record.approach.approach_id,
            options=dict(record.approach.options),
            status=record.status,
            train_time_s=sum(s.duration_s for s in train_stages) if train_stages else None,
            peak_vram_mb=max((s.peak_vram_mb for s in train_stages), default=None),
        )
        run_dir = workdir.run_dir(record.scene_id, record.approach.approach_id, record.run_id)
        summary = load_summary(run_dir)
        if summary is not None:
            metrics = summary.overall
            row.psnr, row.ssim, row.lpips = metrics.psnr, metrics.ssim, metrics.lpips
            row.human_psnr, row.vehicle_psnr = metrics.human_psnr, metrics.vehicle_psnr
        rows.append(row)
    return sorted(rows, key=lambda row: (row.psnr is None, -(row.psnr or 0.0)))


def rows_to_csv(rows: list[ReportRow]) -> str:
    """Xuất bảng so sánh ra CSV; cột ``options`` viết dạng ``ô=giá trị;ô=giá trị``.

    Args:
        rows: Các dòng của bảng.

    Returns:
        Nội dung CSV, có dòng tiêu đề.
    """
    buffer = io.StringIO()
    names = list(ReportRow.__dataclass_fields__)
    writer = csv.DictWriter(buffer, fieldnames=names)
    writer.writeheader()
    for row in rows:
        data = asdict(row)
        data["options"] = ";".join(f"{key}={value}" for key, value in row.options.items())
        writer.writerow(data)
    return buffer.getvalue()
