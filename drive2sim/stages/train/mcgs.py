"""Stage ``train.mcgs``: chạy MCGS-SLAM (``demo.py``) trên đầu vào của ``prep.mcgs``.

MCGS-SLAM chạy trong conda env riêng (PyTorch 2.1.2, CUDA 11.8; ``envs/install_mcgs.sh``) qua tiến
trình con, như Instant NuRec. Stage đọc thư mục ``mcgs_input`` của ``prep.mcgs`` (``calib.yml``,
``images/<camera>/``, ``mcgs_prep.json``), gọi ``demo.py`` với ``--stride 1`` (mặc định của nó là 3
nên sẽ bỏ ảnh) và ghi quỹ đạo ``traj_mcgs.txt``, bản đồ ``3dgs_final.ply`` vào thư mục stage.

Chế độ (``mode``): ``full`` bật cả ``--prgbd`` (độ sâu sau tinh chỉnh) và ``--jdsa`` (tối ưu chung
độ sâu và pose); ``prgbd`` chỉ bật ``--prgbd``; ``minimal`` không bật gì.

Stage chép ``mcgs_prep.json`` sang thư mục ra để ``eval.mcgs`` biết khung test có bị loại khỏi
đầu vào không và camera 0 là camera nào mà không cần đầu vào ``mcgs_input``. Chưa chạy thật
(ARCHITECTURE mục 7.1).

File này chứa:
- ``TrainMcgsParams``: tham số.
- ``MODE_FLAGS``: cờ của ``demo.py`` theo chế độ.
- ``build_command``: câu lệnh ``demo.py`` (tách riêng để test không cần MCGS-SLAM).
- ``ply_vertex_count``: số Gaussian trong PLY.
- ``TrainMcgsStage``: stage đăng ký với engine.
"""

import json
import shutil
import time
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict

from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.external import mcgs
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.prep.mcgs import CALIB_FILE, IMAGES_DIR, PREP_FILE

STATS_FILE = "train_stats.json"
MODE_FLAGS: dict[str, list[str]] = {
    "full": ["--prgbd", "--jdsa"],
    "prgbd": ["--prgbd"],
    "minimal": [],
}
# EGL để vẽ không cần màn hình; Agg để matplotlib không mở cửa sổ.
DEMO_ENV = {"PYOPENGL_PLATFORM": "egl", "MPLBACKEND": "Agg"}
HEADER_MAX_LINES = 200


class TrainMcgsParams(BaseModel):
    """Tham số của ``train.mcgs``.

    Attributes:
        mode: ``full``, ``prgbd`` hoặc ``minimal`` (xem ``MODE_FLAGS``).
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["full", "prgbd", "minimal"] = "full"


def build_command(input_dir: Path, output_dir: Path, mode: str) -> list[str]:
    """Câu lệnh ``demo.py`` của MCGS-SLAM cho một thư mục ``mcgs_input``.

    Mỗi phần tử của ``layout`` trong ``mcgs_prep.json`` là một đối số của ``--imagedir`` theo đúng
    thứ tự; camera đối tác stereo lặp hai lần thì thư mục của nó cũng xuất hiện hai lần.

    Args:
        input_dir: Thư mục ra của ``prep.mcgs``.
        output_dir: Thư mục ``demo.py`` ghi kết quả (``--output``).
        mode: Khoá của ``MODE_FLAGS``.

    Returns:
        Câu lệnh dạng danh sách; chạy với ``cwd=mcgs.mcgs_dir()``.
    """
    prep = json.loads((input_dir / PREP_FILE).read_text(encoding="utf-8"))
    directories = [str(input_dir / IMAGES_DIR / camera) for camera in prep["layout"]]
    return [
        str(mcgs.mcgs_python()), "demo.py",
        "--calib", str(input_dir / CALIB_FILE),
        "--imagedir", *directories,
        "--stride", "1",
        "--output", str(output_dir),
        *MODE_FLAGS[mode],
    ]  # fmt: skip


def ply_vertex_count(path: Path) -> Optional[int]:
    """Số đỉnh (Gaussian) ghi trong header của file PLY.

    Args:
        path: File PLY.

    Returns:
        Số ở dòng ``element vertex N``; ``None`` nếu không đọc được hoặc không phải PLY.
    """
    try:
        with path.open("rb") as stream:
            if stream.readline().decode("ascii", "replace").strip() != "ply":
                return None
            for _ in range(HEADER_MAX_LINES):
                line = stream.readline().decode("ascii", "replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[2])
                if line in ("end_header", ""):
                    break
    except (OSError, ValueError, IndexError):
        return None
    return None


@register_stage("train.mcgs")
class TrainMcgsStage(Stage):
    """Chạy MCGS-SLAM trên ảnh đã xuất, ghi quỹ đạo và bản đồ Gaussian."""

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    uses_lidar: ClassVar[bool] = False
    inputs: ClassVar[dict[str, str]] = {"data": "mcgs_input"}
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = TrainMcgsParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy ``demo.py``, chép ``mcgs_prep.json``, ghi ``train_stats.json``.

        Returns:
            ``{"model": <thư mục stage>}``.

        Raises:
            RuntimeError: Chưa cài MCGS-SLAM, hoặc ``demo.py`` xong mà không ghi quỹ đạo, PLY.
        """
        assert isinstance(params, TrainMcgsParams)
        python = mcgs.mcgs_python()
        if not python.is_file():
            raise RuntimeError(
                f"chưa cài MCGS-SLAM ({python} không có): chạy bash envs/install_mcgs.sh"
            )
        data = inputs["data"]
        ctx.progress(0.0, f"MCGS-SLAM chế độ {params.mode}")
        began = time.monotonic()
        run_logged(
            build_command(data, ctx.output_dir, params.mode),
            ctx.output_dir / LOG_FILE,
            cwd=mcgs.mcgs_dir(),
            should_stop=ctx.should_stop,
            env=DEMO_ENV,
        )
        elapsed = time.monotonic() - began
        missing = [
            name
            for name in (mcgs.OUTPUT_TRAJECTORY, mcgs.OUTPUT_GAUSSIANS)
            if not (ctx.output_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"demo.py xong nhưng không ghi {', '.join(missing)}; xem {LOG_FILE}"
            )
        shutil.copyfile(data / PREP_FILE, ctx.output_dir / PREP_FILE)
        stats = {
            "train_time_s": round(elapsed, 1),
            "num_gaussians": ply_vertex_count(ctx.output_dir / mcgs.OUTPUT_GAUSSIANS),
            "mode": params.mode,
        }
        (ctx.output_dir / STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n")
        ctx.progress(1.0, f"{stats['num_gaussians'] or 0:,} Gaussian")
        return {"model": ctx.output_dir}
