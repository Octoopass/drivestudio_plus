"""Stage ``train.drivestudio``: train một cảnh bằng trainer DriveStudio có sẵn.

Stage này không viết lại việc train. Nó chọn một file config method có sẵn trong
``drive2sim/configs/methods/`` theo bốn tham số (method, renderer, CamPose, FastGS), dựng câu lệnh
``python -m drive2sim.recon.tools.train`` với dữ liệu và split của run, rồi chạy câu lệnh đó trong
một tiến trình con (ARCHITECTURE mục 4.3, 4.4).

File này chứa:
- ``CONFIG_FILES``: bảng tổ hợp (method, renderer, CamPose, FastGS) → file config có sẵn.
- ``TrainDrivestudioParams``: tham số của stage; tổ hợp không có file config bị từ chối ngay khi
  engine kiểm tra tham số, trước khi run bắt đầu.
- ``build_command``: dựng câu lệnh train từ tham số, split và thư mục; tách riêng để test được
  trên máy không có GPU.
- ``TrainDrivestudioStage``: stage đăng ký với engine.

Vì sao chạy bằng tiến trình con: engine và CLI không phải nạp PyTorch; lỗi CUDA hay hết bộ nhớ
chỉ làm hỏng tiến trình con, engine vẫn ghi được lỗi vào ``run.json``.

Kết quả: DriveStudio ghi log, config đã gộp, checkpoint, metric và video vào thư mục của stage
(``d2s/stages/train.drivestudio/``). Nếu thư mục đã có ``checkpoint_resume.pth`` từ lần chạy bị
ngắt, stage truyền ``--resume_from`` để train tiếp thay vì train lại từ đầu.

Chưa chạy thật trên GPU: phần dựng lệnh có test, phần gọi DriveStudio sẽ kiểm tra trên Kaggle.
"""

import sys
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from drive2sim import paths
from drive2sim.core.schema import Manifest, Split, parse_strategy
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.scene_view import DRIVESTUDIO_DATASETS, VIEW_SCENE, build_view

Method = Literal["omnire", "streetgs", "pvg", "deformablegs"]
Renderer = Literal["3dgs", "3dgut", "3dgeer"]

# Tổ hợp (method, renderer, campose, fastgs) → file config có sẵn. Chỉ liệt kê tổ hợp đã có file;
# thêm tổ hợp mới thì thêm file config rồi thêm một dòng ở đây.
CONFIG_FILES: dict[tuple[str, str, bool, bool], str] = {
    ("omnire", "3dgs", True, False): "omnire.yaml",
    ("omnire", "3dgs", False, False): "omnire_nocampose.yaml",
    ("omnire", "3dgeer", False, False): "omnire_geer.yaml",
    ("omnire", "3dgs", False, True): "omnire_fastgs.yaml",
    ("streetgs", "3dgs", True, False): "streetgs.yaml",
    ("streetgs", "3dgs", False, False): "streetgs_nocampose.yaml",
    ("streetgs", "3dgut", False, False): "streetgs_ut.yaml",
    ("streetgs", "3dgeer", False, False): "streetgs_geer.yaml",
    ("streetgs", "3dgs", False, True): "streetgs_fastgs.yaml",
    ("streetgs", "3dgeer", False, True): "streetgs_geer_fastgs.yaml",
    ("pvg", "3dgs", True, False): "pvg.yaml",
    ("deformablegs", "3dgs", True, False): "deformablegs.yaml",
}
RESUME_CHECKPOINT = "checkpoint_resume.pth"
VIEW_DIR = "scene_view"
# Mục lấy từ artifact của prep.masks_segformer và prep.smpl, đè lên mục cùng tên của cảnh.
MASK_ENTRIES = ("sky_masks", "fine_dynamic_masks")
SMPL_ENTRY = "humanpose"
FINAL_CHECKPOINT = "checkpoint_final.pth"


class TrainDrivestudioParams(BaseModel):
    """Tham số của stage ``train.drivestudio``.

    Attributes:
        method: Method của DriveStudio: ``omnire``, ``streetgs``, ``pvg`` hoặc ``deformablegs``.
        renderer: ``3dgs``, ``3dgut`` hoặc ``3dgeer``.
        campose: Có tối ưu pose camera (node ``CamPose``) không.
        fastgs: Có dùng luật densify kiểu FastGS không.
        dataset: Config dataset của DriveStudio, dạng ``<thư mục>/<k>cams``, ví dụ
            ``nuscenes/3cams``. Bỏ trống thì chọn theo dataset gốc trong manifest và số camera
            ``k`` của split (``dataset_config_for``).
        num_iters: Số iteration; bỏ trống thì theo file config (thường 30000).
        seed: Seed ngẫu nhiên.
        overrides: Tham số DriveStudio khác cần đè, dạng ``{"khoá.con": giá trị}``, ví dụ
            ``{"trainer.render.near_plane": 0.2}``.
    """

    model_config = ConfigDict(extra="forbid")

    method: Method
    renderer: Renderer = "3dgs"
    campose: bool = False
    fastgs: bool = False
    dataset: Optional[str] = None
    num_iters: Optional[int] = None
    seed: int = 0
    overrides: dict[str, object] = {}

    @model_validator(mode="after")
    def _check_config_exists(self) -> "TrainDrivestudioParams":
        """Kiểm tra tổ hợp (method, renderer, campose, fastgs) có file config sẵn.

        Raises:
            ValueError: Nếu chưa có file config cho tổ hợp này; thông báo liệt kê các tổ hợp có.
        """
        if self.config_key not in CONFIG_FILES:
            available = ", ".join(
                f"{m}/{r}/campose={c}/fastgs={f}"
                for (m, r, c, f) in CONFIG_FILES
                if m == self.method
            )
            raise ValueError(
                f"chưa có config DriveStudio cho method={self.method}, renderer={self.renderer}, "
                f"campose={self.campose}, fastgs={self.fastgs}; các tổ hợp có: {available}"
            )
        return self

    @property
    def config_key(self) -> tuple[str, str, bool, bool]:
        """Trả về khoá tra trong ``CONFIG_FILES``."""
        return (self.method, self.renderer, self.campose, self.fastgs)

    @property
    def config_file(self) -> Path:
        """Trả về đường dẫn file config method ứng với tham số."""
        return paths.CONFIGS_DIR / "methods" / CONFIG_FILES[self.config_key]


def _format_value(value: object) -> str:
    """Viết một giá trị thành dạng OmegaConf đọc được trên dòng lệnh.

    Args:
        value: Giá trị Python, ví dụ ``True``, ``[0, 1]``, ``0.2``.

    Returns:
        Chuỗi, ví dụ ``"true"``, ``"[0,1]"``, ``"0.2"``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    return str(value)


def dataset_config_for(dataset: str, num_cameras: int) -> str:
    """Chọn config dataset có sẵn của DriveStudio cho một cảnh.

    Mỗi dataset có các config ``<k>cams.yaml`` trong ``drive2sim/configs/datasets/<thư mục>/``
    (ví dụ nuScenes có 1, 3, 6 camera). Hàm chọn config có đúng số camera của split; danh sách
    camera cụ thể vẫn được đặt lại theo split trong ``build_command``.

    Args:
        dataset: Dataset gốc trong manifest, ví dụ ``"nuscenes"``.
        num_cameras: Số camera của split.

    Returns:
        Tên config, ví dụ ``"nuscenes/3cams"``.

    Raises:
        ValueError: Nếu dataset chưa có loader DriveStudio, hoặc không có config đúng số camera;
            thông báo liệt kê các config có sẵn để người dùng đặt tham số ``dataset``.
    """
    if dataset not in DRIVESTUDIO_DATASETS:
        raise ValueError(f"dataset {dataset} chưa có loader DriveStudio")
    folder = DRIVESTUDIO_DATASETS[dataset]
    name = f"{folder}/{num_cameras}cams"
    if not (paths.CONFIGS_DIR / "datasets" / f"{name}.yaml").is_file():
        available = sorted(
            f"{folder}/{path.stem}"
            for path in (paths.CONFIGS_DIR / "datasets" / folder).glob("*.yaml")
        )
        raise ValueError(
            f"không có config {name}.yaml; các config có: {', '.join(available)}. "
            "Đặt tham số dataset của train.drivestudio để chọn một config"
        )
    return name


def build_command(
    params: TrainDrivestudioParams,
    split: Split,
    manifest: Manifest,
    view_root: Path,
    output_dir: Path,
    has_sky_masks: bool = True,
    has_smpl: bool = True,
) -> list[str]:
    """Dựng câu lệnh train của DriveStudio cho một run.

    Dịch split của tool sang tham số của DriveStudio: chiến lược ``every_nth_frame:N`` thành
    ``test_image_stride=N``; ``frame_range`` thành ``start_timestep`` và ``end_timestep``; danh
    sách camera thành chỉ số camera theo thứ tự trong manifest (CONTRACTS mục 5.4, loại
    ``scene``). Thư mục log của DriveStudio được đặt đúng bằng ``output_dir``. Dữ liệu đọc từ
    thư mục cảnh ảo ``<view_root>/000/`` (``build_view``), gồm thư mục cảnh cộng mask và SMPL
    từ cache. Thiếu mask hoặc SMPL thì tắt ``load_sky_mask`` hoặc ``load_smpl`` để DriveStudio
    không đi tìm file không có.

    Args:
        params: Tham số đã kiểm tra.
        split: Split của run.
        manifest: Manifest của cảnh, để đổi ``camera_id`` sang chỉ số.
        view_root: Thư mục chứa cảnh ảo, truyền làm ``data.data_root``.
        output_dir: Thư mục kết quả của stage.
        has_sky_masks: Cảnh ảo có ``sky_masks/`` không.
        has_smpl: Cảnh ảo có ``humanpose/smpl.pkl`` không.

    Returns:
        Câu lệnh dạng danh sách, dùng được cho ``subprocess.run``.

    Raises:
        ValueError: Nếu split không dùng chiến lược ``every_nth_frame``.
    """
    strategy_name, stride = parse_strategy(split.strategy)
    if strategy_name != "every_nth_frame":
        raise ValueError(
            f"train.drivestudio chỉ nhận split every_nth_frame, nhận {split.strategy}"
        )
    camera_ids = [camera.camera_id for camera in manifest.cameras]
    camera_indices = [camera_ids.index(camera_id) for camera_id in split.cameras]
    dataset = params.dataset or dataset_config_for(manifest.source.dataset, len(split.cameras))
    first, last = split.frame_range

    command = [
        sys.executable,
        "-m",
        "drive2sim.recon.tools.train",
        "--config_file",
        str(params.config_file),
        # DriveStudio ghi log vào <output_root>/<project>/<run_name>; tách output_dir làm ba.
        "--output_root",
        str(output_dir.parent.parent),
        "--project",
        output_dir.parent.name,
        "--run_name",
        output_dir.name,
    ]
    resume = output_dir / RESUME_CHECKPOINT
    if resume.is_file():
        command += ["--resume_from", str(resume)]
    options = {
        "dataset": dataset,
        "seed": params.seed,
        "data.data_root": str(view_root),
        "data.scene_idx": VIEW_SCENE,
        "data.start_timestep": first,
        # end_timestep trong config tính cả khung cuối; DriveStudio tự cộng 1
        # (driving_dataset.py, dòng ``self.end_timestep = end_timestep + 1``).
        "data.end_timestep": last,
        "data.pixel_source.cameras": camera_indices,
        "data.pixel_source.test_image_stride": stride,
    }
    if not has_sky_masks:
        options["data.pixel_source.load_sky_mask"] = False
    if not has_smpl:
        options["data.pixel_source.load_smpl"] = False
    if params.num_iters is not None:
        options["trainer.optim.num_iters"] = params.num_iters
    options.update(params.overrides)
    command += [f"{key}={_format_value(value)}" for key, value in options.items()]
    return command


@register_stage("train.drivestudio")
class TrainDrivestudioStage(Stage):
    """Train một cảnh bằng trainer DriveStudio, qua một tiến trình con.

    Đầu vào là cảnh (thư mục ``data/<scene_id>/``, định dạng chuẩn ở CONTRACTS mục 2.1), split
    của run, và tuỳ chọn mask (``prep.masks_segformer``) và SMPL (``prep.smpl``); đầu ra là
    thư mục log của DriveStudio, chứa ``checkpoint_final.pth``.
    """

    version: ClassVar[str] = "0.1.0"
    cacheable: ClassVar[bool] = False
    # Mọi config OmniRe, Street GS hiện có khởi tạo Gaussian từ LiDAR: nền từ điểm LiDAR
    # (model.Background.init.from_lidar), xe và người từ điểm LiDAR trong box. Cảnh không có
    # LiDAR thì train dừng ở bước khởi tạo, nên engine khoá approach ngay từ lúc chọn. Approach
    # hoặc preset không cần LiDAR (sau này) sẽ dùng stage hoặc tham số khác.
    uses_lidar: ClassVar[bool] = True
    inputs: ClassVar[dict[str, str]] = {
        "scene": "scene",
        "split": "split",
        "masks": "masks",
        "smpl": "smpl",
    }
    optional_inputs: ClassVar[frozenset[str]] = frozenset({"masks", "smpl"})
    outputs: ClassVar[dict[str, str]] = {"model": "model"}
    Params: ClassVar[type[BaseModel]] = TrainDrivestudioParams

    def run(
        self, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        """Chạy train.py, ghi toàn bộ stdout và stderr vào ``log.txt`` của stage.

        Args:
            ctx: Ngữ cảnh do engine tạo.
            inputs: ``masks`` và ``smpl`` nếu approach có stage tạo ra chúng; cảnh và split lấy
                từ ``ctx``.
            params: ``TrainDrivestudioParams`` đã kiểm tra.

        Returns:
            ``{"model": <thư mục stage>}``; thư mục chứa ``checkpoint_final.pth``.

        Raises:
            RuntimeError: Nếu train.py thoát với mã khác 0 hoặc không ghi ra checkpoint cuối;
                thông báo kèm vài dòng cuối của log.
        """
        assert isinstance(params, TrainDrivestudioParams)
        extra = {}
        if "masks" in inputs:
            extra.update(
                {
                    name: inputs["masks"] / name
                    for name in MASK_ENTRIES
                    if (inputs["masks"] / name).exists()
                }
            )
        if "smpl" in inputs:
            extra[SMPL_ENTRY] = inputs["smpl"] / SMPL_ENTRY
        view_root = ctx.output_dir / VIEW_DIR
        build_view(ctx.scene.root, view_root, exclude=(*MASK_ENTRIES, SMPL_ENTRY), extra=extra)
        command = build_command(
            params,
            ctx.split,
            ctx.scene.manifest,
            view_root,
            ctx.output_dir,
            has_sky_masks="sky_masks" in extra,
            has_smpl=SMPL_ENTRY in extra,
        )
        ctx.progress(0.0, f"train {params.method} bằng {params.config_file.name}")
        run_logged(command, ctx.output_dir / LOG_FILE, should_stop=ctx.should_stop)
        if not (ctx.output_dir / FINAL_CHECKPOINT).is_file():
            raise RuntimeError(f"train.py xong nhưng không có {FINAL_CHECKPOINT}")
        ctx.progress(1.0, "train xong")
        return {"model": ctx.output_dir}
