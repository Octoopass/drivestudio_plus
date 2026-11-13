"""Dựng bảng so sánh nhiều run trên cùng cảnh và split (SPEC mục 8 nhóm D, mục 9).

Studio (bảng xếp hạng, biểu đồ chất lượng theo thời gian train và VRAM) và lệnh ``d2s report``
cùng dùng file này.

File này chứa:
- ``ReportRow``: một dòng của bảng: run, approach, lựa chọn của các mục cấu hình, metric,
  thời gian, VRAM.
- ``report_rows``: đọc các run đã xong và dựng bảng.
- ``rows_to_csv``: xuất bảng ra CSV cho báo cáo.

Hai loại thời gian (CONTRACTS mục 6.3):
- ``train_time_s``: thời gian train thực của model, lấy từ ``summary.json`` (``train.py`` chỉ đếm
  các bước train, cộng qua các lần chạy tiếp). Run cũ chưa có số này thì lấy thời gian chạy
  stage ``train.*`` trong ``run.json``.
- ``recon_time_s``: tổng thời gian các stage từ đầu run tới khi xuất xong ``scene.ply``. Stage
  dùng lại kết quả từ cache tính thời gian của lần chạy gốc, để các run cùng cách làm có cùng tổng
  dù chạy trước hay sau.
"""

import csv
import io
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from drive2sim.core.schema import RunRecord
from drive2sim.engine.run import D2S_DIR, list_runs, load_env, load_summary
from drive2sim.engine.workdir import Workdir

TRAIN_PREFIX = "train."
PLY_STAGE = "export.ply_world"
PLY_FILE = "scene.ply"
PREVIEW_STAGE = "export.preview"


@dataclass
class ReportRow:
    """Một dòng của bảng so sánh.

    Attributes:
        run_id: Run.
        scene_id: Cảnh.
        split_id: Split.
        approach_id: Approach.
        options: Lựa chọn đã quyết định của các mục cấu hình, ví dụ ``{"renderer": "3dgeer"}``.
        quick: ``True`` nếu là run chạy thử nhanh; metric không so được với run đủ iteration.
        status: Trạng thái run.
        psnr: PSNR trên tập test; ``None`` nếu run chưa có ``summary.json``.
        ssim: SSIM.
        lpips: LPIPS.
        avg_error: Chỉ số gộp của Mip-NeRF: trung bình nhân của ``10^(-PSNR/10)`` (MSE),
            ``sqrt(1 - SSIM)`` và LPIPS; thấp là tốt; ``None`` nếu thiếu một trong ba metric.
        human_psnr: PSNR vùng người, nếu có.
        vehicle_psnr: PSNR vùng xe, nếu có.
        dynamic_psnr: PSNR vùng vật động (mọi người và xe), nếu có.
        occupied_psnr: PSNR vùng không phải bầu trời, nếu có.
        num_iters: Số iteration đã train.
        train_time_s: Thời gian train thực của model, giây.
        recon_time_s: Tổng thời gian từ stage đầu tới khi xuất xong ``scene.ply``, giây;
            ``None`` nếu run chưa xuất PLY hoặc thiếu thời gian của một stage dùng lại cache.
        peak_vram_mb: VRAM cao nhất khi train, megabyte.
        render_fps: Tốc độ render khi đánh giá, khung mỗi giây.
        num_gaussians: Số Gaussian của cả model.
        ply_gaussians: Số Gaussian trong ``scene.ply`` (nền, và xe nếu có; không có người).
        ply_size_mb: Dung lượng ``scene.ply``, megabyte.
        preview_gaussians: Tổng số Gaussian trong gói xem trước (nền và mọi xe), nếu đã xuất.
        preview_size_mb: Tổng dung lượng mọi file trong gói xem trước, megabyte.
        stage_times: ``stage_id`` → ``{"seconds", "reused"}``: thời gian từng stage; stage dùng
            lại cache ghi thời gian lần chạy gốc và ``reused`` là ``True``. Không xuất ra CSV.
        gpu: GPU đã chạy run (từ ``env.json``); nhiều GPU khác nhau qua các lần chạy tiếp thì nối
            bằng ``" + "``. Thời gian, VRAM, FPS chỉ so được giữa các run cùng GPU.
    """

    run_id: str
    scene_id: str
    split_id: str
    approach_id: str
    options: dict[str, object] = field(default_factory=dict)
    quick: bool = False
    status: str = ""
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    avg_error: Optional[float] = None
    human_psnr: Optional[float] = None
    vehicle_psnr: Optional[float] = None
    dynamic_psnr: Optional[float] = None
    occupied_psnr: Optional[float] = None
    num_iters: Optional[int] = None
    train_time_s: Optional[float] = None
    recon_time_s: Optional[float] = None
    peak_vram_mb: Optional[int] = None
    render_fps: Optional[float] = None
    num_gaussians: Optional[int] = None
    ply_gaussians: Optional[int] = None
    ply_size_mb: Optional[float] = None
    preview_gaussians: Optional[int] = None
    preview_size_mb: Optional[float] = None
    gpu: Optional[str] = None
    stage_times: dict[str, dict[str, object]] = field(default_factory=dict)


def avg_error(psnr: float, ssim: float, lpips: float) -> float:
    """Chỉ số gộp "Avg" của Mip-NeRF (Barron và cộng sự, 2021), thấp là tốt.

    Đưa ba metric về cùng dạng sai số rồi lấy trung bình nhân, nên không cần trọng số và điểm của
    một run không phụ thuộc các run khác trong bảng.

    Args:
        psnr: PSNR, dB.
        ssim: SSIM.
        lpips: LPIPS.

    Returns:
        ``(10^(-psnr/10) * sqrt(1 - ssim) * lpips) ** (1/3)``.
    """
    mse = 10 ** (-psnr / 10)
    return (mse * math.sqrt(max(0.0, 1.0 - ssim)) * max(0.0, lpips)) ** (1 / 3)


def recon_time(record: RunRecord) -> Optional[float]:
    """Tổng thời gian các stage từ đầu run tới hết ``export.ply_world``.

    Stage dùng lại cache tính ``reused_duration_s`` (thời gian lần chạy gốc).

    Returns:
        Số giây; ``None`` nếu run không có stage ``export.ply_world`` xong, hoặc một stage dùng
        lại cache mà không biết thời gian gốc.
    """
    total = 0.0
    for stage in record.stages:
        if stage.cache_hit:
            if stage.reused_duration_s is None:
                return None
            total += stage.reused_duration_s
        else:
            total += stage.duration_s
        if stage.stage_id == PLY_STAGE:
            return round(total, 3) if stage.status == "succeeded" else None
    return None


def preview_stats(folder: Path) -> tuple[Optional[int], Optional[float]]:
    """Tổng số Gaussian (mọi file ``.ply``) và tổng dung lượng của gói xem trước.

    Returns:
        ``(số Gaussian, megabyte)``; ``(None, None)`` nếu chưa xuất gói.
    """
    if not (folder / "preview.json").is_file():
        return None, None
    count = 0
    for ply in folder.rglob("*.ply"):
        number, _ = ply_stats(ply)
        count += number or 0
    size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    return count, round(size / (1024 * 1024), 2)


def ply_stats(ply: Path) -> tuple[Optional[int], Optional[float]]:
    """Đọc số Gaussian (dòng ``element vertex N`` trong phần đầu) và dung lượng của file PLY.

    Returns:
        ``(số Gaussian, megabyte)``; ``(None, None)`` nếu không có file.
    """
    try:
        size_mb = round(ply.stat().st_size / (1024 * 1024), 2)
        count = None
        with ply.open("rb") as handle:
            for _ in range(64):  # phần đầu của PLY chỉ vài chục dòng
                line = handle.readline().decode("ascii", "replace").strip()
                if line.startswith("element vertex "):
                    count = int(line.split()[2])
                if line == "end_header" or not line:
                    break
        return count, size_mb
    except (OSError, ValueError, IndexError):
        return None, None


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
        iters = next(
            (s.params["num_iters"] for s in train_stages if s.params.get("num_iters")), None
        )
        row = ReportRow(
            run_id=record.run_id,
            scene_id=record.scene_id,
            split_id=record.split_id,
            approach_id=record.approach.approach_id,
            options=dict(record.approach.options),
            quick=record.approach.quick,
            status=record.status,
            num_iters=iters,
            train_time_s=sum(s.duration_s for s in train_stages) if train_stages else None,
            recon_time_s=recon_time(record),
            peak_vram_mb=max((s.peak_vram_mb for s in train_stages), default=None) or None,
        )
        run_dir = workdir.run_dir(record.scene_id, record.approach.approach_id, record.run_id)
        row.ply_gaussians, row.ply_size_mb = ply_stats(
            run_dir / D2S_DIR / "stages" / PLY_STAGE / PLY_FILE
        )
        row.preview_gaussians, row.preview_size_mb = preview_stats(
            run_dir / D2S_DIR / "stages" / PREVIEW_STAGE
        )
        row.stage_times = {
            stage.stage_id: {
                "seconds": (stage.reused_duration_s or 0.0)
                if stage.cache_hit
                else stage.duration_s,
                "reused": stage.cache_hit,
            }
            for stage in record.stages
            if stage.status in ("succeeded", "failed", "cancelled", "running")
        }
        env = load_env(run_dir)
        if env is not None:
            names = list(dict.fromkeys(m.gpu for m in env.machines if m.gpu))
            row.gpu = " + ".join(names) or env.gpu
        summary = load_summary(run_dir)
        if summary is not None:
            metrics = summary.overall
            row.psnr, row.ssim, row.lpips = metrics.psnr, metrics.ssim, metrics.lpips
            row.avg_error = round(avg_error(metrics.psnr, metrics.ssim, metrics.lpips), 5)
            row.human_psnr, row.vehicle_psnr = metrics.human_psnr, metrics.vehicle_psnr
            row.dynamic_psnr, row.occupied_psnr = metrics.dynamic_psnr, metrics.occupied_psnr
            row.render_fps, row.num_gaussians = summary.render_fps, summary.num_gaussians
            # Số đo trong train.py (thời gian thực, VRAM của tiến trình train) chính xác hơn số
            # engine đo bên ngoài tiến trình con.
            row.train_time_s = summary.train_time_s or row.train_time_s
            row.peak_vram_mb = summary.peak_vram_mb or row.peak_vram_mb
            row.num_iters = summary.num_iters or row.num_iters
        rows.append(row)
    return sorted(rows, key=lambda row: (row.psnr is None, -(row.psnr or 0.0)))


def rows_to_csv(rows: list[ReportRow]) -> str:
    """Xuất bảng so sánh ra CSV; cột ``options`` viết dạng ``mục=giá trị;mục=giá trị``.

    Args:
        rows: Các dòng của bảng.

    Returns:
        Nội dung CSV, có dòng tiêu đề.
    """
    buffer = io.StringIO()
    names = [name for name in ReportRow.__dataclass_fields__ if name != "stage_times"]
    writer = csv.DictWriter(buffer, fieldnames=names)
    writer.writeheader()
    for row in rows:
        data = asdict(row)
        data.pop("stage_times")
        data["options"] = ";".join(f"{key}={value}" for key, value in row.options.items())
        writer.writerow(data)
    return buffer.getvalue()
