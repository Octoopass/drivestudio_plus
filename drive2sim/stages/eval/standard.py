"""Stage ``eval.standard``: tóm tắt kết quả đánh giá của một run theo protocol chung.

Trainer DriveStudio đã tự đánh giá trên tập test ở cuối ``train.py`` và ghi
``metrics/images_test_<thời điểm>.json`` vào thư mục train. Stage này không đánh giá lại; nó đọc
file metric mới nhất đó, đổi sang định dạng ``summary.json`` của tool (CONTRACTS mục 6.3) và ghi
kèm protocol đã dùng, để mọi approach có cùng một file kết quả mà Studio và ``d2s report`` đọc
được (SPEC mục 9).

Ngoài metric ảnh, stage đọc ``train_stats.json`` mà ``train.py`` ghi cùng mỗi checkpoint: thời
gian train thực (chỉ các bước train, cộng qua các lần chạy tiếp), VRAM cao nhất của tiến trình
train, số Gaussian của cả model và số iteration. Tốc độ render (``render_fps``) do ``train.py`` đo
khi render tập test và ghi cùng file metric test.

File này chứa:
- ``read_drivestudio_metrics``: tìm và đọc file metric test mới nhất của DriveStudio.
- ``read_train_stats``: đọc ``train_stats.json``.
- ``make_summary``: dựng ``Summary`` từ metric và thống kê train đọc được.
- ``EvalStandardStage``: stage đăng ký với engine.

Metric theo camera: ``train.py`` ghi khoá
``image_metrics/test/per_camera/<chỉ số>/<psnr|ssim|lpips>``
với chỉ số là vị trí của camera trong ``manifest.yaml`` (cách ``train.drivestudio`` đánh số
camera cho DriveStudio); stage đổi chỉ số sang ``camera_id`` để điền ``per_camera``. Khoá đã là
``camera_id`` (trainer khác ghi) thì giữ nguyên. Model train bằng ``train.py`` cũ không có khoá
này nên ``per_camera`` để trống.

Đo lại: file metric mới nhất chưa có khoá ``per_camera`` nghĩa là nó được ghi bằng code đo cũ
(trước protocol 0.3). Khi đó, nếu có ``checkpoint_final.pth``, stage render lại tập test từ
checkpoint bằng ``drive2sim.recon.tools.eval`` (cảnh ảo dựng lại như lúc train, gồm mask và SMPL
từ cache) và dùng metric mới đó; không có checkpoint thì giữ metric cũ.

Giới hạn hiện tại: tách đánh giá khỏi ``train.py`` là việc sau (ARCHITECTURE mục 7.2).
"""

import json
import os
import shutil
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import BaseModel

from drive2sim.core import files
from drive2sim.core.schema import (
    SCHEMA_VERSIONS,
    DepthMetrics,
    EvalProtocol,
    LpipsSetting,
    Metrics,
    ShiftedMetrics,
    Summary,
)
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import NoParams, Stage, StageContext
from drive2sim.paths import SMPL_MODEL_ENV
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.scene_view import build_view
from drive2sim.stages.train.drivestudio import (
    FINAL_CHECKPOINT,
    MASK_ENTRIES,
    SMPL_ENTRY,
    VIEW_DIR,
    smpl_model_for,
    view_extras,
)

# 0.5: ``frames.jsonl`` có thêm ``psnr_cc``, ``lpips_cc`` (sau chỉnh màu affine), ``psnr_static``;
# split có camera chỉ test (R2) thì ``overall`` chỉ tính trên ảnh test của camera đã train, còn
# camera chỉ test tính riêng vào ``shifted``. Split không có camera chỉ test cho số y như 0.4.
# 0.4: ghi thêm metric từng ảnh test (``frames_test_*.jsonl``) và độ sâu render so với LiDAR
# (``depth``); cách đo PSNR, SSIM, LPIPS không đổi so với 0.3, nên số cũ so được.
# 0.3: khi loader đánh dấu điểm ảnh không hợp lệ (``valid_masks``: camera mắt cá ngoài vùng nhìn
# dùng được, thân xe), SSIM và LPIPS cũng chỉ tính trên điểm ảnh hợp lệ như PSNR; ảnh không có điểm
# hợp lệ nào bị bỏ qua thay vì làm trung bình thành NaN; thêm metric theo camera.
# 0.2: ảnh render được làm tròn về 8 bit trước khi đo, như ảnh gốc (SPEC mục 9); 0.1 đo ảnh render
# ở dạng số thực.
PROTOCOL_VERSION = "0.6"
METRIC_PREFIX = "image_metrics/test/"
SUMMARY_FILE = "summary.json"
TRAIN_STATS_FILE = "train_stats.json"
REPORT_FILE = "drivestudio_metrics.json"
PER_CAMERA_PREFIX = "per_camera"
DEPTH_PREFIX = "depth"
FRAMES_PATTERN = "frames_test_*.jsonl"
FRAMES_FILE = "frames.jsonl"
REEVAL_DIR = "reeval"


def read_drivestudio_metrics(model_dir: Path, folder: str = "metrics") -> dict[str, float]:
    """Đọc file metric test mới nhất mà DriveStudio ghi trong thư mục train.

    Tên file có dạng ``metrics/images_test_<YYYY-MM-DD_HHMMSS>.json`` nên file mới nhất là file
    đứng cuối khi sắp theo tên. Khoá trong file có tiền tố ``image_metrics/test/``; hàm bỏ tiền tố.

    Args:
        model_dir: Thư mục kết quả của ``train.drivestudio``.
        folder: Thư mục con chứa file metric; ``metrics_eval`` với kết quả đo lại.

    Returns:
        Tên metric → giá trị, ví dụ ``{"psnr": 27.6, "ssim": 0.85, "lpips": 0.2}``.

    Raises:
        FileNotFoundError: Nếu chưa có file metric test nào; thường do split không có ảnh test.
    """
    candidates = sorted((model_dir / folder).glob("images_test_*.json"))
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


def has_per_camera(metrics: dict[str, float]) -> bool:
    """File metric có khoá theo camera không (ghi bằng code đo từ protocol 0.3)."""
    return any(key.startswith(PER_CAMERA_PREFIX + "/") for key in metrics)


def latest_frames_file(model_dir: Path, folder: str = "metrics") -> Optional[Path]:
    """File metric từng ảnh test mới nhất (``frames_test_*.jsonl``), nếu có.

    Code đo từ protocol 0.4 ghi file này cạnh ``images_test_*.json``; run cũ không có.

    Args:
        model_dir: Thư mục kết quả của ``train.drivestudio``.
        folder: Thư mục con chứa file metric; ``metrics_eval`` với kết quả đo lại.

    Returns:
        Đường dẫn file; ``None`` nếu chưa có.
    """
    candidates = sorted((model_dir / folder).glob(FRAMES_PATTERN))
    return candidates[-1] if candidates else None


def frames_have_color_match(path: Path) -> bool:
    """File metric từng ảnh có khoá ``psnr_cc`` không (do code đo từ protocol 0.5 ghi).

    Chỉ đọc dòng đầu: mọi dòng của một file cùng do một phiên bản code ghi.

    Args:
        path: File ``frames_test_*.jsonl`` hoặc ``frames.jsonl``.

    Returns:
        ``True`` nếu dòng đầu có ``psnr_cc``; ``False`` nếu thiếu khoá hoặc file rỗng.
    """
    with path.open(encoding="utf-8") as handle:
        first = handle.readline().strip()
    return bool(first) and "psnr_cc" in json.loads(first)


def read_frames(path: Path) -> list[dict]:
    """Đọc file metric từng ảnh test, mỗi dòng một bản ghi JSON.

    Args:
        path: File ``frames.jsonl``.

    Returns:
        Danh sách bản ghi, theo thứ tự trong file.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def split_frames_by_role(
    frames: list[dict], test_only_indices: set[int]
) -> tuple[list[dict], list[dict]]:
    """Chia bản ghi từng ảnh test thành ảnh của camera đã train và của camera chỉ test.

    Args:
        frames: Bản ghi của ``frames.jsonl``.
        test_only_indices: Vị trí trong ``manifest.yaml`` của các camera chỉ test; khớp với
            ``camera_index`` của bản ghi.

    Returns:
        Bộ ``(đã train, chỉ test)``, giữ thứ tự gốc.
    """
    trained = [item for item in frames if item["camera_index"] not in test_only_indices]
    shifted = [item for item in frames if item["camera_index"] in test_only_indices]
    return trained, shifted


def aggregate_frames(frames: list[dict]) -> dict[str, Optional[float]]:
    """Lấy trung bình theo ảnh các metric trong bản ghi từng ảnh test.

    ``psnr``, ``ssim``, ``lpips`` luôn có; ``psnr_cc``, ``lpips_cc`` có thể ``None`` (khớp màu
    suy biến) hoặc không có khoá (file do code đo cũ ghi): giá trị đó bị bỏ qua, và cả cột không
    có giá trị nào thì kết quả ``None``.

    Args:
        frames: Bản ghi của ``frames.jsonl``.

    Returns:
        Khoá ``psnr``, ``ssim``, ``lpips``, ``psnr_cc``, ``lpips_cc`` → trung bình hoặc ``None``.
    """
    result: dict[str, Optional[float]] = {}
    for name in ("psnr", "ssim", "lpips", "psnr_cc", "lpips_cc"):
        values = [item[name] for item in frames if item.get(name) is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


def reeval_command(checkpoint: Path, view_root: Path, log_dir: Path) -> list[str]:
    """Câu lệnh render lại tập test từ checkpoint bằng code đo hiện tại.

    Config lấy từ ``config.yaml`` cạnh checkpoint (ghi lúc train), chỉ đổi đường dẫn cảnh ảo và
    thư mục ghi; không render cả chuỗi và quỹ đạo mới (chỉ cần metric tập test).
    """
    return [
        sys.executable,
        "-m",
        "drive2sim.recon.tools.eval",
        "--resume_from",
        str(checkpoint),
        f"data.data_root={view_root}",
        f"log_dir={log_dir}",
        "render.render_full=false",
        "render.render_novel=null",
    ]


def read_train_stats(model_dir: Path) -> dict[str, object]:
    """Đọc ``train_stats.json`` trong thư mục train.

    Args:
        model_dir: Thư mục kết quả của ``train.drivestudio``.

    Returns:
        Nội dung file; rỗng nếu chưa có (model train bằng ``train.py`` cũ) hoặc đọc lỗi.
    """
    try:
        data = json.loads((model_dir / TRAIN_STATS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _package_version(name: str) -> str:
    """Trả về phiên bản gói Python đã cài, hoặc ``"unknown"`` nếu không đọc được."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _resolution_of(model_dir: Path) -> str:
    """Mô tả độ phân giải khi đánh giá, đọc từ ``config.yaml`` DriveStudio ghi trong thư mục train.

    Returns:
        Ví dụ ``"downscale_when_loading=[2, 2, 2], downscale=1"``; không có ``config.yaml`` thì
        lấy trường ``resolution`` của ``train_stats.json`` (ví dụ ``"downscale=2"``);
        ``"unknown"`` nếu không có cả hai.
    """
    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        # Trainer không phải DriveStudio có thể ghi mô tả độ phân giải vào
        # train_stats.json thay cho config.yaml.
        resolution = read_train_stats(model_dir).get("resolution")
        return resolution if isinstance(resolution, str) else "unknown"
    pixel = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("data", {})
    pixel = pixel.get("pixel_source", {})
    return (
        f"downscale_when_loading={pixel.get('downscale_when_loading')}, "
        f"downscale={pixel.get('downscale')}"
    )


def _camera_name(key: str, camera_ids: Optional[list[str]]) -> str:
    """Đổi khoá camera trong file metric sang ``camera_id``.

    ``train.py`` ghi chỉ số camera (``"2"``), là vị trí của camera trong ``manifest.yaml``.

    Args:
        key: Khoá camera đọc được, ví dụ ``"2"`` hoặc ``"CAM_FRONT"``.
        camera_ids: ``camera_id`` theo thứ tự manifest; ``None`` thì không đổi.

    Returns:
        ``camera_id`` ứng với chỉ số, ví dụ ``"image_02"``; khoá không phải chỉ số hợp lệ thì
        trả nguyên khoá.
    """
    if camera_ids is not None and key.isdigit() and int(key) < len(camera_ids):
        return camera_ids[int(key)]
    return key


def make_summary(
    metrics: dict[str, float],
    split_id: str,
    model_dir: Path,
    train_stats: Optional[dict[str, object]] = None,
    camera_ids: Optional[list[str]] = None,
    frames: Optional[list[dict]] = None,
    test_only_cameras: Sequence[str] = (),
) -> Summary:
    """Dựng ``summary.json`` từ metric của DriveStudio và ``train_stats.json``.

    Metric theo vùng (người, xe, vật động) chỉ có khi cảnh có box; giá trị âm hoặc thiếu nghĩa là
    DriveStudio không tính được, được ghi ``null`` (CONTRACTS mục 10.3).

    Split có camera chỉ test (R2): metric của DriveStudio trung bình cả camera đã train lẫn camera
    chỉ test, nên ``overall.psnr``, ``ssim``, ``lpips`` được tính lại từ ``frames`` chỉ trên ảnh
    của camera đã train (R1 vẫn là "khung giữ lại của camera đã train"), còn camera chỉ test vào
    ``shifted``. Metric vùng (người, xe, vật động, không phải trời) giữ số của DriveStudio, gồm
    mọi camera. Không có camera chỉ test thì ``frames`` không dùng tới và ``overall`` y như 0.4.

    Args:
        metrics: Kết quả ``read_drivestudio_metrics``.
        split_id: Split của run.
        model_dir: Thư mục train, để đọc độ phân giải trong config.
        train_stats: Kết quả ``read_train_stats``; thiếu trường nào thì trường đó để trống.
        camera_ids: ``camera_id`` theo thứ tự trong ``manifest.yaml``, để đổi khoá chỉ số camera
            (``per_camera/0/psnr``) sang ``camera_id``. ``None`` thì giữ nguyên khoá.
        frames: Bản ghi từng ảnh test (``read_frames``); bắt buộc khi có camera chỉ test.
        test_only_cameras: ``camera_id`` của camera chỉ test (``Split.test_only_cameras``).

    Returns:
        Đối tượng ``Summary``.

    Raises:
        KeyError: Nếu thiếu ``psnr``, ``ssim`` hoặc ``lpips``.
        ValueError: Nếu có camera chỉ test mà không có ``frames``, không biết vị trí camera
            trong manifest, hoặc không có ảnh test nào của camera đã train hoặc camera chỉ test.
    """

    def optional(name: str) -> Optional[float]:
        value = metrics.get(name)
        return value if value is not None and value >= 0 else None

    def stat(name: str) -> Optional[int]:
        value = (train_stats or {}).get(name)
        return int(value) if isinstance(value, (int, float)) and value >= 0 else None

    width, height = optional("render_width"), optional("render_height")
    # Metric theo camera: khoá ``per_camera/<chỉ số hoặc camera_id>/<metric>``.
    per_camera: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == PER_CAMERA_PREFIX:
            per_camera.setdefault(_camera_name(parts[1], camera_ids), {})[parts[2]] = value
    depth = {
        key[len(DEPTH_PREFIX) + 1 :]: value
        for key, value in metrics.items()
        if key.startswith(DEPTH_PREFIX + "/")
    }
    train_time = (train_stats or {}).get("train_time_s")
    eval_time = (train_stats or {}).get("eval_time_s")
    speedups = (train_stats or {}).get("speedups")
    if not (isinstance(speedups, list) and all(isinstance(x, str) for x in speedups)):
        speedups = []

    overall_values = {name: metrics[name] for name in ("psnr", "ssim", "lpips")}
    shifted = None
    if test_only_cameras:
        if frames is None:
            raise ValueError("split có camera chỉ test nhưng run không có frames.jsonl")
        if camera_ids is None:
            raise ValueError(
                "split có camera chỉ test nhưng không biết thứ tự camera của manifest"
            )
        test_only_indices = {camera_ids.index(camera) for camera in test_only_cameras}
        trained, shifted_frames = split_frames_by_role(frames, test_only_indices)
        if not trained:
            raise ValueError("frames.jsonl không có ảnh test nào của camera đã train")
        if not shifted_frames:
            raise ValueError(
                f"frames.jsonl không có ảnh test nào của camera chỉ test {list(test_only_cameras)}"
            )
        # R1 chỉ tính trên camera đã train; camera chỉ test tính riêng ở R2.
        overall_values = {
            name: value
            for name, value in aggregate_frames(trained).items()
            if name in overall_values and value is not None
        }
        shifted_values = aggregate_frames(shifted_frames)
        shifted = ShiftedMetrics(
            cameras=list(test_only_cameras),
            num_images=len(shifted_frames),
            psnr=shifted_values["psnr"],
            ssim=shifted_values["ssim"],
            lpips=shifted_values["lpips"],
            psnr_cc=shifted_values["psnr_cc"],
            lpips_cc=shifted_values["lpips_cc"],
        )

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
            psnr=overall_values["psnr"],
            ssim=overall_values["ssim"],
            lpips=overall_values["lpips"],
            human_psnr=optional("human_psnr"),
            vehicle_psnr=optional("vehicle_psnr"),
            dynamic_psnr=optional("masked_psnr"),
            occupied_psnr=optional("occupied_psnr"),
        ),
        per_camera={
            camera: Metrics(psnr=values["psnr"], ssim=values["ssim"], lpips=values["lpips"])
            for camera, values in per_camera.items()
            if {"psnr", "ssim", "lpips"} <= set(values)
        },
        train_time_s=float(train_time) if isinstance(train_time, (int, float)) else None,
        peak_vram_mb=stat("peak_vram_mb") or None,
        num_gaussians=stat("num_gaussians"),
        num_iters=stat("num_iters"),
        render_fps=optional("render_fps"),
        render_resolution=f"{int(width)}x{int(height)}" if width and height else None,
        eval_time_s=float(eval_time) if isinstance(eval_time, (int, float)) else None,
        speedups=speedups,
        depth=DepthMetrics(**depth) if depth else None,
        shifted=shifted,
    )


@register_stage("eval.standard")
class EvalStandardStage(Stage):
    """Đổi metric DriveStudio ghi khi train sang ``summary.json`` theo protocol chung."""

    version: ClassVar[str] = "0.6.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {
        "model": "model",
        "split": "split",
        "masks": "masks",
        "smpl": "smpl",
    }
    optional_inputs: ClassVar[frozenset[str]] = frozenset({"masks", "smpl"})
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
        frames = latest_frames_file(inputs["model"])
        checkpoint = inputs["model"] / FINAL_CHECKPOINT
        # Metric thiếu khoá theo camera (protocol < 0.3), thiếu file từng ảnh (protocol < 0.4) hoặc
        # file từng ảnh thiếu metric sau chỉnh màu (protocol < 0.5) được đo lại từ checkpoint bằng
        # code đo hiện tại.
        stale = (
            not has_per_camera(metrics) or frames is None or not frames_have_color_match(frames)
        )
        if stale and checkpoint.is_file():
            metrics, frames = self.reevaluate(ctx, inputs, checkpoint)
        (ctx.output_dir / REPORT_FILE).write_text(json.dumps(metrics, indent=2) + "\n")
        if frames is not None:
            shutil.copyfile(frames, ctx.output_dir / FRAMES_FILE)
        files.dump(
            make_summary(
                metrics,
                ctx.split.split_id,
                inputs["model"],
                read_train_stats(inputs["model"]),
                camera_ids=[camera.camera_id for camera in ctx.scene.manifest.cameras],
                frames=read_frames(frames) if frames is not None else None,
                test_only_cameras=ctx.split.test_only_cameras,
            ),
            ctx.output_dir / SUMMARY_FILE,
        )
        return {"report": ctx.output_dir}

    def reevaluate(
        self, ctx: StageContext, inputs: dict[str, Path], checkpoint: Path
    ) -> tuple[dict[str, float], Optional[Path]]:
        """Render lại tập test từ checkpoint bằng code đo hiện tại.

        Returns:
            ``(metric mới, file metric từng ảnh test)``; file là ``None`` nếu code đo không ghi.
        """
        extra = view_extras(inputs)
        view_root = ctx.output_dir / VIEW_DIR
        build_view(ctx.scene.root, view_root, exclude=(*MASK_ENTRIES, SMPL_ENTRY), extra=extra)
        log_dir = ctx.output_dir / REEVAL_DIR
        # eval.py chỉ tạo hai thư mục này cạnh checkpoint, không trong log_dir được chỉ định.
        for folder in ("videos_eval", "metrics_eval"):
            (log_dir / folder).mkdir(parents=True, exist_ok=True)
        env = {}
        if SMPL_ENTRY in extra:
            env[SMPL_MODEL_ENV] = str(smpl_model_for(ctx.output_dir))
        env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        ctx.progress(0.0, "đo lại tập test từ checkpoint bằng code đo hiện tại")
        run_logged(
            reeval_command(checkpoint, view_root, log_dir),
            ctx.output_dir / LOG_FILE,
            should_stop=ctx.should_stop,
            env=env,
        )
        return (
            read_drivestudio_metrics(log_dir, "metrics_eval"),
            latest_frames_file(log_dir, "metrics_eval"),
        )
