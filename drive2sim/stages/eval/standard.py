"""Stage ``eval.standard``: tóm tắt kết quả đánh giá của một run theo protocol chung.

Trainer DriveStudio đã tự đánh giá trên tập test ở cuối ``train.py`` và ghi
``metrics/images_test_<thời điểm>.json`` vào thư mục train. Stage này không đánh giá lại; nó đọc
file metric mới nhất đó, đổi sang định dạng ``summary.json`` của tool (CONTRACTS mục 6.3) và ghi
kèm protocol đã dùng, để mọi approach có cùng một file kết quả mà Studio và ``d2s report`` đọc
được (SPEC mục 9).

File này chứa:
- ``read_drivestudio_metrics``: tìm và đọc file metric test mới nhất của DriveStudio.
- ``make_summary``: dựng ``Summary`` từ metric đọc được.
- ``EvalStandardStage``: stage đăng ký với engine.

Giới hạn hiện tại: DriveStudio không tách metric theo camera nên ``per_camera`` để trống; thời
gian train và VRAM nằm trong ``run.json`` (engine đo), số Gaussian chưa đọc được khi không nạp
checkpoint, nên ba trường này để trống. Tách đánh giá khỏi ``train.py`` là việc sau
(ARCHITECTURE mục 7.2).
"""

import json
from importlib import metadata
from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import SCHEMA_VERSIONS, EvalProtocol, LpipsSetting, Metrics, Summary
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext

PROTOCOL_VERSION = "0.1"
METRIC_PREFIX = "image_metrics/test/"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "drivestudio_metrics.json"


def read_drivestudio_metrics(model_dir: Path) -> dict[str, float]:
    """Đọc file metric test mới nhất mà DriveStudio ghi trong thư mục train.

    Tên file có dạng ``metrics/images_test_<YYYY-MM-DD_HHMMSS>.json`` nên file mới nhất là file
    đứng cuối khi sắp theo tên. Khoá trong file có tiền tố ``image_metrics/test/``; hàm bỏ tiền tố.

    Args:
        model_dir: Thư mục kết quả của ``train.drivestudio``.

    Returns:
        Tên metric → giá trị, ví dụ ``{"psnr": 27.6, "ssim": 0.85, "lpips": 0.2}``.

    Raises:
        FileNotFoundError: Nếu chưa có file metric test nào; thường do split không có ảnh test.
    """
    candidates = sorted((model_dir / "metrics").glob("images_test_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"không có metrics/images_test_*.json trong {model_dir}; "
            "kiểm tra split có ảnh test không"
        )
    raw = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return {
        key[len(METRIC_PREFIX) :]: float(value)
        for key, value in raw.items()
        if key.startswith(METRIC_PREFIX)
    }


def _package_version(name: str) -> str:
    """Trả về phiên bản gói Python đã cài, hoặc ``"unknown"`` nếu không đọc được."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _resolution_of(model_dir: Path) -> str:
    """Mô tả độ phân giải khi đánh giá, đọc từ ``config.yaml`` DriveStudio ghi trong thư mục train.

    Returns:
        Ví dụ ``"downscale_when_loading=[2, 2, 2], downscale=1"``; ``"unknown"`` nếu không đọc
        được config.
    """
    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        return "unknown"
    pixel = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("data", {})
    pixel = pixel.get("pixel_source", {})
    return (
        f"downscale_when_loading={pixel.get('downscale_when_loading')}, "
        f"downscale={pixel.get('downscale')}"
    )


def make_summary(metrics: dict[str, float], split_id: str, model_dir: Path) -> Summary:
    """Dựng ``summary.json`` từ metric của DriveStudio.

    Metric theo vùng (người, xe, vật động) chỉ có khi cảnh có box; giá trị âm hoặc thiếu nghĩa là
    DriveStudio không tính được, được ghi ``null`` (CONTRACTS mục 10.3).

    Args:
        metrics: Kết quả ``read_drivestudio_metrics``.
        split_id: Split của run.
        model_dir: Thư mục train, để đọc độ phân giải trong config.

    Returns:
        Đối tượng ``Summary``.

    Raises:
        KeyError: Nếu thiếu ``psnr``, ``ssim`` hoặc ``lpips``.
    """

    def optional(name: str) -> Optional[float]:
        value = metrics.get(name)
        return value if value is not None and value >= 0 else None

    return Summary(
        schema_version=SCHEMA_VERSIONS["summary"],
        protocol=EvalProtocol(
            version=PROTOCOL_VERSION,
            image_dtype="uint8",
            resolution=_resolution_of(model_dir),
            # Trainer dùng LearnedPerceptualImagePatchSimilarity của torchmetrics, mặc định
            # backbone alex (drive2sim/recon/trainers/base.py).
            lpips=LpipsSetting(
                net="alex",
                package="torchmetrics",
                package_version=_package_version("torchmetrics"),
            ),
        ),
        split_id=split_id,
        overall=Metrics(
            psnr=metrics["psnr"],
            ssim=metrics["ssim"],
            lpips=metrics["lpips"],
            human_psnr=optional("human_psnr"),
            vehicle_psnr=optional("vehicle_psnr"),
            occupied_psnr=optional("occupied_psnr"),
        ),
    )


@register_stage("eval.standard")
class EvalStandardStage(Stage):
    """Đổi metric DriveStudio ghi khi train sang ``summary.json`` theo protocol chung."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"model": "model", "split": "split"}
    outputs: ClassVar[dict[str, str]] = {"report": "eval_report"}
    Params: ClassVar[type[BaseModel]] = NoParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Đọc metric trong thư mục train, ghi ``summary.json`` và bản sao metric gốc.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: ``{"model": <thư mục train>}``.
            params: Không có tham số.

        Returns:
            ``{"report": <thư mục stage>}``, chứa ``summary.json`` và ``drivestudio_metrics.json``.
        """
        metrics = read_drivestudio_metrics(inputs["model"])
        (ctx.output_dir / REPORT_FILE).write_text(json.dumps(metrics, indent=2) + "\n")
        files.dump(
            make_summary(metrics, ctx.split.split_id, inputs["model"]),
            ctx.output_dir / SUMMARY_FILE,
        )
        return {"report": ctx.output_dir}
