"""Stage ``train.drivestudio``: train một cảnh bằng trainer DriveStudio có sẵn.

Stage này không viết lại việc train. Nó chọn một file config method có sẵn trong
``drive2sim/configs/methods/`` theo năm tham số (method, renderer, CamPose, FastGS, Faster-GS),
dựng câu lệnh ``python -m drive2sim.recon.tools.train`` với dữ liệu và split của run, rồi chạy câu
lệnh đó trong một tiến trình con (ARCHITECTURE mục 4.3, 4.4).

File này chứa:
- ``CONFIG_FILES``: bảng tổ hợp (method, renderer, CamPose, FastGS, Faster-GS) → file config.
- ``TrainDrivestudioParams``: tham số của stage; tổ hợp không có file config bị từ chối ngay khi
  engine kiểm tra tham số, trước khi run bắt đầu.
- ``build_command``: dựng câu lệnh train từ tham số, split và thư mục; tách riêng để test được
  trên máy không có GPU.
- ``TrainDrivestudioStage``: stage đăng ký với engine.

Vì sao chạy bằng tiến trình con: engine và CLI không phải nạp PyTorch; lỗi CUDA hay hết bộ nhớ
chỉ làm hỏng tiến trình con, engine vẫn ghi được lỗi vào ``run.json``.

File config method có thể kế thừa file khác bằng ``extends`` (CONTRACTS mục 10.5). Stage gộp sẵn
(``drive2sim.core.config.load_config``), ghi bản đầy đủ vào ``method_config.yaml`` của stage và
đưa bản đó cho DriveStudio; config sai bị báo ngay khi engine kiểm tra tham số.

Kết quả: DriveStudio ghi log, config đã gộp, checkpoint, metric và video vào thư mục của stage
(``d2s/stages/train.drivestudio/``). Nếu thư mục đã có ``checkpoint_resume.pth`` từ lần chạy bị
ngắt, stage truyền ``--resume_from`` để train tiếp thay vì train lại từ đầu.

Chưa chạy thật trên GPU: phần dựng lệnh có test, phần gọi DriveStudio sẽ kiểm tra trên Kaggle.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import ClassVar, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drive2sim import paths
from drive2sim.core.config import ConfigError, load_config
from drive2sim.core.schema import CAMERA_TRACK, Manifest, Split, parse_strategy
from drive2sim.engine.registry import register_stage
from drive2sim.engine.stage import Stage, StageContext
from drive2sim.paths import SMPL_MODEL_ENV, SMPL_MODEL_NAME, find_smpl_model
from drive2sim.stages.commands import LOG_FILE, run_logged
from drive2sim.stages.scene_view import DRIVESTUDIO_DATASETS, VIEW_SCENE, build_view

# train.py ghi cùng mỗi checkpoint: thời gian train, VRAM, số Gaussian.
TRAIN_STATS_FILE = "train_stats.json"

Method = Literal["omnire", "streetgs", "pvg", "deformablegs"]
Renderer = Literal["3dgs", "3dgut", "3dgeer"]

# Tổ hợp (method, renderer, campose, fastgs, fastergs) → file config có sẵn. Chỉ liệt kê tổ hợp đã
# có file; thêm tổ hợp mới thì thêm file config (kế thừa bằng extends) rồi thêm một dòng ở đây.
CONFIG_FILES: dict[tuple[str, str, bool, bool, bool], str] = {
    ("omnire", "3dgs", True, False, False): "omnire.yaml",
    ("omnire", "3dgs", False, False, False): "omnire_nocampose.yaml",
    ("omnire", "3dgeer", False, False, False): "omnire_geer.yaml",
    ("omnire", "3dgs", False, True, False): "omnire_fastgs.yaml",
    ("omnire", "3dgs", True, False, True): "omnire_fastergs.yaml",
    ("omnire", "3dgs", False, False, True): "omnire_fastergs_nocampose.yaml",
    ("omnire", "3dgeer", False, False, True): "omnire_geer_fastergs.yaml",
    ("omnire", "3dgs", False, True, True): "omnire_fastgs_fastergs.yaml",
    ("omnire", "3dgeer", False, True, False): "omnire_geer_fastgs.yaml",
    ("omnire", "3dgeer", False, True, True): "omnire_geer_fastgs_fastergs.yaml",
    ("omnire", "3dgut", False, False, False): "omnire_ut.yaml",
    ("omnire", "3dgut", False, False, True): "omnire_ut_fastergs.yaml",
    ("omnire", "3dgut", False, True, False): "omnire_ut_fastgs.yaml",
    ("omnire", "3dgut", False, True, True): "omnire_ut_fastgs_fastergs.yaml",
    ("streetgs", "3dgs", True, False, False): "streetgs.yaml",
    ("streetgs", "3dgs", False, False, False): "streetgs_nocampose.yaml",
    ("streetgs", "3dgut", False, False, False): "streetgs_ut.yaml",
    ("streetgs", "3dgeer", False, False, False): "streetgs_geer.yaml",
    ("streetgs", "3dgs", False, True, False): "streetgs_fastgs.yaml",
    ("streetgs", "3dgeer", False, True, False): "streetgs_geer_fastgs.yaml",
    ("streetgs", "3dgs", True, False, True): "streetgs_fastergs.yaml",
    ("streetgs", "3dgs", False, False, True): "streetgs_fastergs_nocampose.yaml",
    ("streetgs", "3dgut", False, False, True): "streetgs_ut_fastergs.yaml",
    ("streetgs", "3dgeer", False, False, True): "streetgs_geer_fastergs.yaml",
    ("streetgs", "3dgs", False, True, True): "streetgs_fastgs_fastergs.yaml",
    ("streetgs", "3dgeer", False, True, True): "streetgs_geer_fastgs_fastergs.yaml",
    ("pvg", "3dgs", True, False, False): "pvg.yaml",
    ("deformablegs", "3dgs", True, False, False): "deformablegs.yaml",
}
# Bản OmniRe cho nhiều camera (hơn 3): 40k iteration, chọn ảnh theo sai số, ngưỡng densify thấp
# hơn; chỉ có cho tổ hợp gốc của paper (3DGS, CamPose, không FastGS, Faster-GS).
EXTENDED_CAM_KEY = ("omnire", "3dgs", True, False, False)
EXTENDED_CAM_FILE = "omnire_extended_cam.yaml"
RESUME_CHECKPOINT = "checkpoint_resume.pth"
# Trainer densify 3DGUT theo gradient tương đương pixel (ut_densify=screen)
SCREEN_GRAD_TRAINER = "drive2sim.recon.trainers.screen_grad.ScreenGradMultiTrainer"
VIEW_DIR = "scene_view"
# Mục lấy từ artifact của prep.masks_segformer và prep.smpl, đè lên mục cùng tên của cảnh.
MASK_ENTRIES = ("sky_masks", "fine_dynamic_masks")
SMPL_ENTRY = "humanpose"
FINAL_CHECKPOINT = "checkpoint_final.pth"
# Config method đã gộp extends, ghi trong thư mục stage và đưa cho DriveStudio.
METHOD_CONFIG = "method_config.yaml"
# Góc lệch trục lớn nhất (rad) của pixel camera mắt cá khi train bằng 3DGUT hoặc 3DGEER. Cả hai
# renderer của gsplat bỏ mọi Gaussian có tâm sau mặt phẳng camera (z <= near_plane:
# ProjectionUT3DGSFused.cu, IntersectGEERTile.cu), nên không vẽ được gì từ 90 độ trở ra; 3DGUT còn
# vẽ hỏng gần 90 độ vì chiếu điểm mẫu của Gaussian qua ống kính. Che từ 88 độ cho cả hai để chừa
# khoảng an toàn: run thử trên KITTI-360 0003 cho vùng 83..88 độ tốt hơn 1,2..2,4 dB so với che từ
# 83 độ, vùng trong không đổi; run 3DGEER che tới 96 độ có opacity 0 ở mọi pixel ngoài 90 độ
# (docs/datasets/kitti360.md mục 5.1).
RENDERER_MAX_ANGLE = 1.536
# Renderer vẽ được camera mắt cá (3DGS chỉ vẽ pinhole).
FISHEYE_RENDERERS = ("3dgut", "3dgeer")


class TrainDrivestudioParams(BaseModel):
    """Tham số của stage ``train.drivestudio``.

    Attributes:
        method: Method của DriveStudio: ``omnire``, ``streetgs``, ``pvg`` hoặc ``deformablegs``.
        renderer: ``3dgs``, ``3dgut`` hoặc ``3dgeer``.
        campose: Có tối ưu pose camera (node ``CamPose``) không.
        fastgs: Có dùng luật densify kiểu FastGS không.
        fastergs: Có bật phần Python của Faster-GS không (``docs/methods/fastergs.md``).
        dataset: Config dataset của DriveStudio, dạng ``<thư mục>/<k>cams``, ví dụ
            ``nuscenes/3cams``. Bỏ trống thì chọn theo dataset gốc trong manifest và số camera
            ``k`` của split (``dataset_config_for``).
        num_iters: Số iteration; bỏ trống thì theo file config (thường 30000).
        seed: Seed ngẫu nhiên.
        overrides: Tham số DriveStudio khác cần đè, dạng ``{"khoá.con": giá trị}``, ví dụ
            ``{"trainer.render.near_plane": 0.2}``.
        profile_at: Các bước bắt đầu đo thời gian GPU bằng torch.profiler
            (``drive2sim/recon/utils/step_profile.py``); rỗng là không đo. Chỉ dùng trong run đo
            riêng, vì profiler làm chậm các bước được đo.
        profile_steps: Số bước mỗi lần đo.
        ut_densify: Renderer 3DGUT densify theo gì: ``native`` là nhánh ``ut`` của Bosch
            (gradient theo tâm 3D, ngưỡng của 3DGS không hợp); ``screen`` đổi gradient đó sang
            gradient tương đương pixel qua Jacobian của phép chiếu
            (``drive2sim/recon/trainers/screen_grad.py``) để dùng lại ngưỡng của 3DGS. Renderer
            khác bỏ qua.
        refine_interval: Densify, prune mỗi bao nhiêu bước
            (``trainer.gaussian_ctrl_general_cfg.refine_interval``); bỏ trống thì theo config
            (3DGS, 3DGUT 100; 3DGEER 50, Bosch để 300). Dùng để so renderer với cùng nhịp densify.
        fisheye_max_angle: Góc lệch trục lớn nhất (rad) của pixel camera mắt cá được dùng khi
            train, đánh giá (``data.pixel_source.fisheye_max_angle``), thay giới hạn của mô hình
            ống kính và renderer; vùng đen dò từ ảnh vẫn bị che. Bỏ trống thì loader lấy góc nhỏ
            nhất giữa ``cameras[].max_angle`` của manifest (góc mô hình ống kính còn đúng,
            converter tính từ calib của từng camera), góc bắt đầu vùng đen và giới hạn của
            renderer (3DGUT, 3DGEER: ``RENDERER_MAX_ANGLE``). Hiện chỉ loader KITTI-360 đọc.
        densify: ``default`` là quy tắc của DriveStudio (theo gradient, có ngưỡng); ``mcmc`` là
            densify bằng MCMC cho nền, xe, vật biến dạng (``drive2sim/recon/densify/mcmc.py``,
            ``docs/methods/densify_mcmc.md``), không dùng cùng FastGS. Approach không còn cho
            chọn ``mcmc`` (kết quả kém hơn quy tắc gốc ở mọi run đã đo); giữ giá trị này để run cũ
            (``a022``, ``44d8``) đọc và resume được, và để thử lại bằng
            ``--set train.drivestudio.densify=mcmc``.
        densify_stats: Ghi phân bố của đại lượng densify ở mỗi lần refine vào
            ``densify_stats.jsonl`` (``drive2sim/recon/utils/densify_stats.py``), để chỉnh ngưỡng
            densify theo renderer (``docs/methods/densify_3dgut.md``). Chỉ đọc, không đổi kết quả.
        extended_cam: Dùng config OmniRe cho nhiều camera (``omnire_extended_cam.yaml``); chỉ
            với tổ hợp ``EXTENDED_CAM_KEY``.
        preload_device: Nơi giữ ảnh, mask, lidar đã nạp (``data.preload_device``). Mặc định
            ``cpu``: mỗi bước chỉ chuyển ảnh đang dùng lên GPU (train.py đã làm), nên tiết kiệm vài
            GB GPU mà kết quả như ``cuda``. OmniRe 3 camera trên T4 16 GB hết bộ nhớ ở bước ~8800
            khi để ``cuda`` (config gốc của DriveStudio).
    """

    model_config = ConfigDict(extra="forbid")

    method: Method
    renderer: Renderer = "3dgs"
    campose: bool = False
    fastgs: bool = False
    fastergs: bool = False
    dataset: Optional[str] = None
    num_iters: Optional[int] = Field(default=None, ge=1)
    seed: int = 0
    overrides: dict[str, object] = {}
    profile_at: list[int] = []
    profile_steps: int = Field(default=30, ge=1)
    ut_densify: Literal["native", "screen"] = "native"
    refine_interval: Optional[int] = Field(default=None, ge=1)
    fisheye_max_angle: Optional[float] = Field(default=None, gt=0, le=math.pi)
    densify: Literal["default", "mcmc"] = "default"
    densify_stats: bool = False
    preload_device: Literal["cpu", "cuda"] = "cpu"
    extended_cam: bool = False

    @field_validator("profile_at", mode="before")
    @classmethod
    def _one_step_is_a_list(cls, value: object) -> object:
        """``--set ...profile_at=2000`` (một bước) đọc ra số, không phải danh sách."""
        return [value] if isinstance(value, int) else value

    @model_validator(mode="after")
    def _check_config_exists(self) -> "TrainDrivestudioParams":
        """Kiểm tra tổ hợp (method, renderer, campose, fastgs, fastergs) có config đọc được.

        Raises:
            ValueError: Nếu chưa có file config cho tổ hợp này (thông báo liệt kê các tổ hợp có),
                hoặc file config và các tầng ``extends`` của nó không đọc được.
        """
        if self.extended_cam and self.config_key != EXTENDED_CAM_KEY:
            raise ValueError(
                "extended_cam chỉ có cho OmniRe, renderer 3dgs, bật CamPose, tắt FastGS và "
                "Faster-GS (omnire_extended_cam.yaml)"
            )
        if self.config_key not in CONFIG_FILES:
            available = ", ".join(
                f"{m}/{r}/campose={c}/fastgs={f}/fastergs={g}"
                for (m, r, c, f, g) in CONFIG_FILES
                if m == self.method
            )
            raise ValueError(
                f"chưa có config DriveStudio cho method={self.method}, renderer={self.renderer}, "
                f"campose={self.campose}, fastgs={self.fastgs}, fastergs={self.fastergs}; "
                f"các tổ hợp có: {available}"
            )
        try:
            load_config(self.config_file)
        except ConfigError as error:
            raise ValueError(f"config {self.config_file.name} không đọc được: {error}") from None
        return self

    @property
    def config_key(self) -> tuple[str, str, bool, bool, bool]:
        """Trả về khoá tra trong ``CONFIG_FILES``."""
        return (self.method, self.renderer, self.campose, self.fastgs, self.fastergs)

    @property
    def config_file(self) -> Path:
        """Trả về đường dẫn file config method ứng với tham số."""
        name = EXTENDED_CAM_FILE if self.extended_cam else CONFIG_FILES[self.config_key]
        return paths.CONFIGS_DIR / "methods" / name


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


def view_extras(inputs: dict[str, Path]) -> dict[str, Path]:
    """Mask và SMPL (từ cache) cần ghép vào cảnh ảo, theo đầu vào ``masks``, ``smpl`` của stage."""
    extra: dict[str, Path] = {}
    if "masks" in inputs:
        extra.update(
            {
                name: inputs["masks"] / name
                for name in MASK_ENTRIES
                if (inputs["masks"] / name).exists()
            }
        )
    # prep.smpl trả thư mục rỗng khi cảnh không có người (dataset chưa có bước dựng người).
    if "smpl" in inputs and (inputs["smpl"] / SMPL_ENTRY / "smpl.pkl").is_file():
        extra[SMPL_ENTRY] = inputs["smpl"] / SMPL_ENTRY
    return extra


def smpl_model_for(output_dir: Path) -> Path:
    """File mô hình SMPL cho trainer khi cảnh có người.

    Tìm trước khi chạy, để báo rõ ngay thay vì lỗi sau vài phút nạp dữ liệu. ``output_dir`` là
    ``<workdir>/runs/<cảnh>/<approach>/<run>/d2s/stages/<stage>``.

    Raises:
        RuntimeError: Nếu không thấy file.
    """
    workdir_root = output_dir.parents[6] if len(output_dir.parents) > 6 else None
    smpl_model = find_smpl_model(workdir_root)
    if smpl_model is None:
        raise RuntimeError(
            f"cảnh có người (SMPL) nhưng không thấy file mô hình {SMPL_MODEL_NAME}: "
            f"đặt ở <thư mục làm việc>/smpl_models/ hoặc trỏ biến {SMPL_MODEL_ENV} tới nó "
            "(tải ở https://smpl.is.tue.mpg.de); trên Kaggle gắn dataset chứa file "
            "(biến D2S_KAGGLE_SMPL_DATASET khi đẩy)"
        )
    return smpl_model


def build_command(
    params: TrainDrivestudioParams,
    split: Split,
    manifest: Manifest,
    view_root: Path,
    output_dir: Path,
    has_sky_masks: bool = True,
    has_smpl: bool = True,
    config_file: Optional[Path] = None,
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
        config_file: File config đưa cho DriveStudio; mặc định là ``params.config_file``. Stage
            truyền bản đã gộp ``extends`` (``write_method_config``).

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
        str(config_file or params.config_file),
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
        "data.preload_device": params.preload_device,
    }
    if not has_sky_masks:
        options["data.pixel_source.load_sky_mask"] = False
    if not has_smpl:
        options["data.pixel_source.load_smpl"] = False
    # Camera fisheye, F-theta: train trên ảnh gốc của cảnh. Engine đã khoá renderer không vẽ được
    # loại camera này (OptionSpec.camera_models); khử méo về pinhole sẽ là stage prep.undistort.
    # Ví dụ config physicalai/7cams đặt undistort: True, dòng này đè thành False.
    if any(CAMERA_TRACK[manifest.camera(c).model] == "fisheye" for c in split.cameras):
        options["data.pixel_source.undistort"] = False
    if params.num_iters is not None:
        options["trainer.optim.num_iters"] = params.num_iters
    if params.profile_at:
        options["logging.profile_at"] = list(params.profile_at)
        options["logging.profile_steps"] = params.profile_steps
    if params.renderer == "3dgut" and params.ut_densify == "screen":
        options["trainer.type"] = SCREEN_GRAD_TRAINER
    if params.refine_interval is not None:
        options["trainer.gaussian_ctrl_general_cfg.refine_interval"] = params.refine_interval
    if params.renderer in FISHEYE_RENDERERS:
        # 3DGUT, 3DGEER không vẽ được tia từ 90 độ trở ra (giới hạn của renderer, mọi dataset);
        # loader che theo góc nhỏ hơn giữa giới hạn này và cameras[].max_angle của manifest.
        options["data.pixel_source.fisheye_renderer_max_angle"] = RENDERER_MAX_ANGLE
    if params.fisheye_max_angle is not None:
        options["data.pixel_source.fisheye_max_angle"] = params.fisheye_max_angle
    if params.densify == "mcmc":
        # Densify bằng MCMC cho nền, xe, vật biến dạng (recon/densify/mcmc.py); ngân sách và hệ số
        # theo mặc định của module, đổi bằng overrides "trainer.mcmc.<khoá>".
        options["trainer.mcmc.enabled"] = True
    if params.densify_stats:
        options["trainer.densify_stats.enabled"] = True
    options.update(params.overrides)
    command += [f"{key}={_format_value(value)}" for key, value in options.items()]
    return command


def has_dynamic_masks(scene_root: Path, extra: dict[str, Path]) -> bool:
    """Cảnh có mask vật động không: ``dynamic_masks/`` của cảnh, hoặc mask tinh của SegFormer.

    Args:
        scene_root: Thư mục cảnh.
        extra: Mục ghép thêm vào cảnh ảo (``view_extras``).
    """
    return (scene_root / "dynamic_masks").is_dir() or "fine_dynamic_masks" in extra


def write_method_config(params: TrainDrivestudioParams, output_dir: Path) -> Path:
    """Gộp ``extends`` của config method rồi ghi bản đầy đủ vào thư mục stage.

    Args:
        params: Tham số của stage, để biết file config.
        output_dir: Thư mục của stage.

    Returns:
        Đường dẫn ``method_config.yaml`` vừa ghi.
    """
    path = output_dir / METHOD_CONFIG
    text = yaml.safe_dump(load_config(params.config_file), sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return path


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
        extra = view_extras(inputs)
        # Cảnh không có box 3D (PVG, DeformableGS trên KITTI-360 không box): không có box, mask
        # vật động; loader DriveStudio phải bỏ hai bước đọc đó thay vì lỗi thiếu file.
        overrides = dict(params.overrides)
        if not has_dynamic_masks(ctx.scene.root, extra):
            overrides.setdefault("data.pixel_source.load_dynamic_mask", False)
        if not (ctx.scene.root / "instances" / "instances_info.json").is_file():
            overrides.setdefault("data.pixel_source.load_objects", False)
        if overrides != params.overrides:
            params = params.model_copy(update={"overrides": overrides})
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
            config_file=write_method_config(params, ctx.output_dir),
        )
        # Số Gaussian đổi liên tục (densify, cull): segment mở rộng đỡ phân mảnh bộ nhớ GPU
        # (lần hết bộ nhớ trên Kaggle có 2,2 GB đã giữ mà không dùng được).
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
        }
        if SMPL_ENTRY in extra:
            env[SMPL_MODEL_ENV] = str(smpl_model_for(ctx.output_dir))
        ctx.progress(0.0, f"train {params.method} bằng {params.config_file.name}")
        run_logged(command, ctx.output_dir / LOG_FILE, should_stop=ctx.should_stop, env=env)
        if not (ctx.output_dir / FINAL_CHECKPOINT).is_file():
            raise RuntimeError(f"train.py xong nhưng không có {FINAL_CHECKPOINT}")
        # VRAM đỉnh do train.py đo trong tiến trình con (train_stats.json); engine không thấy.
        try:
            stats = json.loads((ctx.output_dir / TRAIN_STATS_FILE).read_text("utf-8"))
            ctx.report_peak_vram(int(stats.get("peak_vram_mb") or 0))
        except (OSError, ValueError, TypeError):
            pass
        ctx.progress(1.0, "train xong")
        return {"model": ctx.output_dir}
