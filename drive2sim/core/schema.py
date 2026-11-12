"""Schema của mọi file tool đọc và ghi (CONTRACTS mục 2–7).

Mỗi lớp ứng với một loại file. Mọi schema từ chối khoá lạ, để lỗi gõ sai trong file YAML viết
tay được báo ngay thay vì bị bỏ qua. Code viết cho Python 3.9 nên dùng ``Optional``, ``Union``,
``List``, ``Dict`` thay cho cú pháp ``X | Y``.
"""

import math
import re
from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from drive2sim.core import identifiers

# ---------------------------------------------------------------- kiểu dùng chung

CameraModel = Literal["opencv_pinhole", "opencv_fisheye", "ftheta", "ideal_pinhole"]
ShutterType = Literal[
    "GLOBAL",
    "ROLLING_TOP_TO_BOTTOM",
    "ROLLING_LEFT_TO_RIGHT",
    "ROLLING_BOTTOM_TO_TOP",
    "ROLLING_RIGHT_TO_LEFT",
]
Track = Literal["pinhole", "fisheye"]
Level = Literal["L1", "L2", "L3", "L4"]
Dataset = Literal[
    "waymo",
    "nuscenes",
    "nuplan",
    "argoverse2",
    "pandaset",
    "kitti",
    "physicalai",
    "carla",
    "colmap",
    "custom",
]
SplitStrategyName = Literal["every_nth_frame", "custom"]
# bool đứng đầu để giá trị true/false trong YAML giữ nguyên kiểu boolean.
OptionValue = Union[bool, int, float, str]
ArtifactType = Literal[
    "scene_subset",
    "masks",
    "smpl",
    "depth_prior",
    "points_init",
    "drivestudio_scene",
    "model",
    "eval_report",
    "videos",
    "export_ply",
    "preview",
]
RunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
JobKind = Literal["stage", "render"]
JobStatus = Literal["succeeded", "failed"]
# Mã lỗi dùng chung cho worker và API (CONTRACTS mục 9.1 và 11.3); thêm mã mới thì thêm vào đây.
ERROR_CODES: tuple[str, ...] = (
    "invalid_request",
    "not_found",
    "conflict",
    "blocked",
    "internal",
    "worker_unavailable",
    "env_missing",
    "worker_crashed",
    "cancelled",
)
Executor = Literal["local", "kaggle"]

# Phiên bản schema hiện tại của từng loại file.
SCHEMA_VERSIONS: dict[str, str] = {
    "manifest": "d2s-manifest/0.1",
    "split": "d2s-split/0.1",
    "report": "d2s-report/0.1",
    "approach": "d2s-approach/0.1",
    "artifact": "d2s-artifact/0.1",
    "run": "d2s-run/0.1",
    "summary": "d2s-summary/0.1",
    "env": "d2s-env/0.1",
    "preview": "d2s-preview/0.1",
    "job": "d2s-job/0.1",
    "result": "d2s-result/0.1",
}

OPTION_REFERENCE_RE = re.compile(r"^\$\{options\.([a-z][a-z0-9_]*)\}$")
EVERY_NTH_FRAME_RE = re.compile(r"^every_nth_frame:(\d+)$")


class StrictModel(BaseModel):
    """Lớp gốc của mọi schema: từ chối khoá không khai báo."""

    model_config = ConfigDict(extra="forbid")


class VersionedFile(StrictModel):
    """Lớp gốc của một file có trường ``schema_version``.

    Chấp nhận file cùng loại và cùng số major với ``SCHEMA_VERSIONS[FILE_KIND]``; số minor có thể
    khác, vì đổi minor không làm vỡ tương thích (CONTRACTS, phần đầu).
    """

    FILE_KIND: ClassVar[str] = ""
    schema_version: str

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        """Kiểm tra loại file và số major trong ``schema_version``."""
        kind, major, _ = identifiers.parse_schema_version(value)
        expected = SCHEMA_VERSIONS[cls.FILE_KIND]
        expected_kind, expected_major, _ = identifiers.parse_schema_version(expected)
        if kind != expected_kind or major != expected_major:
            raise ValueError(f"expected {expected} (same kind and major), got {value}")
        return value


class Transform(StrictModel):
    """Phép biến đổi cứng ghi trong YAML/JSON (CONTRACTS mục 1).

    Attributes:
        translation: Dịch chuyển ``[x, y, z]``, đơn vị mét.
        rotation_wxyz: Quaternion đơn vị theo thứ tự ``[w, x, y, z]``.
    """

    translation: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]

    @field_validator("rotation_wxyz")
    @classmethod
    def _validate_unit_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Kiểm tra quaternion có độ dài 1 (sai số cho phép 1e-3)."""
        norm = math.sqrt(sum(component * component for component in value))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"rotation_wxyz must be a unit quaternion, norm is {norm:.4f}")
        return value


# Kiểu chuỗi có kiểm tra định dạng, dùng lại cho mọi trường cùng loại định danh.
SceneId = Annotated[str, AfterValidator(identifiers.check_scene_id)]
StageId = Annotated[str, AfterValidator(identifiers.check_stage_id)]
CacheKey = Annotated[str, AfterValidator(identifiers.check_cache_key)]
Version = Annotated[str, AfterValidator(identifiers.check_version)]


# ---------------------------------------------------------------- manifest (CONTRACTS 2.2)


class ManifestSource(StrictModel):
    """Nguồn gốc của cảnh: dataset, chuỗi gốc, converter đã dùng."""

    dataset: Dataset
    sequence_id: str
    converter: str
    converter_version: str
    data_license: Optional[str] = None


class TimestampInterval(StrictModel):
    """Khoảng thời gian của bản ghi, đơn vị micro giây."""

    start: int = Field(ge=0)
    stop: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_order(self) -> "TimestampInterval":
        """Kiểm tra ``stop`` không trước ``start``."""
        if self.stop < self.start:
            raise ValueError("timestamp_interval_us.stop must not be before start")
        return self


class ManifestNCore(StrictModel):
    """Thông tin của kho NCore V4 chứa dữ liệu chuẩn."""

    version: Literal["v4"]
    sequence_id: str
    timestamp_interval_us: TimestampInterval


class ManifestCamera(StrictModel):
    """Một camera trong cảnh.

    Attributes:
        camera_id: Id của camera, giữ nguyên như trong nguồn.
        model: Mô hình ống kính.
        resolution: ``[width, height]`` tính bằng pixel.
        shutter: Loại shutter, giống NCore.
    """

    camera_id: str
    model: CameraModel
    resolution: tuple[int, int]
    shutter: ShutterType

    @field_validator("resolution")
    @classmethod
    def _validate_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Kiểm tra chiều rộng và chiều cao đều dương."""
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("resolution must be positive [width, height]")
        return value


class Timeline(StrictModel):
    """Dòng thời gian tham chiếu: khung thứ i là khung thứ i của ``reference_sensor``."""

    reference_sensor: str
    num_frames: int = Field(gt=0)


class Available(StrictModel):
    """Những gì cảnh có sẵn; engine dựa vào đây để khoá approach (CONTRACTS mục 4.2)."""

    lidar: bool
    cuboids: Literal["gt", "autolabel", "none"]
    camera_labels: list[str] = []
    egocar_mask: bool = False


class Manifest(VersionedFile):
    """File ``manifest.yaml`` đi kèm dữ liệu chuẩn của một cảnh (CONTRACTS mục 2.2)."""

    FILE_KIND: ClassVar[str] = "manifest"
    scene_id: SceneId
    source: ManifestSource
    ncore: ManifestNCore
    track: Track
    cameras: list[ManifestCamera] = Field(min_length=1)
    lidars: list[str] = []
    timeline: Timeline
    available: Available

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Manifest":
        """Kiểm tra các trường khớp nhau: id camera, cảm biến tham chiếu, cờ LiDAR, tiền tố."""
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera_id values must be unique")
        if self.timeline.reference_sensor not in camera_ids + self.lidars:
            raise ValueError(
                f"timeline.reference_sensor {self.timeline.reference_sensor!r} "
                "is not a camera or lidar"
            )
        if self.available.lidar != bool(self.lidars):
            raise ValueError("available.lidar must be true exactly when lidars is not empty")
        if not self.scene_id.startswith(self.source.dataset + "_"):
            raise ValueError("scene_id must start with source.dataset")
        return self

    def camera(self, camera_id: str) -> ManifestCamera:
        """Lấy camera theo id.

        Args:
            camera_id: Id của camera.

        Returns:
            Camera tương ứng.

        Raises:
            KeyError: Nếu cảnh không có camera này.
        """
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        raise KeyError(camera_id)


# ---------------------------------------------------------------- split (CONTRACTS 3.1)


def parse_strategy(strategy: str) -> tuple[str, Optional[int]]:
    """Tách chuỗi chiến lược split.

    Args:
        strategy: ``"every_nth_frame:N"`` với ``N >= 1``, hoặc ``"custom"``.

    Returns:
        Bộ ``(tên chiến lược, N)``; với ``custom`` thì ``N`` là ``None``.
        Ví dụ ``"every_nth_frame:8"`` cho ``("every_nth_frame", 8)``.

    Raises:
        ValueError: Nếu chuỗi sai định dạng.
    """
    if strategy == "custom":
        return "custom", None
    match = EVERY_NTH_FRAME_RE.match(strategy)
    if match and int(match.group(1)) >= 1:
        return "every_nth_frame", int(match.group(1))
    raise ValueError(
        f"invalid strategy {strategy!r}: expected 'every_nth_frame:N' (N >= 1) or 'custom'"
    )


class Split(VersionedFile):
    """File split train/test của một cảnh (CONTRACTS mục 3.1).

    Mỗi phần tử của ``train`` và ``test`` là cặp ``(chỉ số khung, camera_id)``.
    """

    FILE_KIND: ClassVar[str] = "split"
    split_id: str
    scene_id: SceneId
    strategy: str
    frame_range: tuple[int, int]
    cameras: list[str] = Field(min_length=1)
    train: list[tuple[int, str]]
    test: list[tuple[int, str]]

    @field_validator("split_id")
    @classmethod
    def _validate_split_id(cls, value: str) -> str:
        """Kiểm tra ``split_id`` dạng ``snake_case``."""
        return identifiers.check_snake_case(value, "split_id")

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: str) -> str:
        """Kiểm tra chuỗi chiến lược đọc được."""
        parse_strategy(value)
        return value

    @model_validator(mode="after")
    def _validate_pairs(self) -> "Split":
        """Kiểm tra các cặp nằm trong đoạn khung, dùng camera đã khai báo, không trùng nhau."""
        first, last = self.frame_range
        if first < 0 or last < first:
            raise ValueError("frame_range must be [first, last] with 0 <= first <= last")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras must be unique")
        allowed_cameras = set(self.cameras)
        for part_name, pairs in (("train", self.train), ("test", self.test)):
            if len(set(pairs)) != len(pairs):
                raise ValueError(f"{part_name} has duplicate pairs")
            for frame, camera_id in pairs:
                if not first <= frame <= last:
                    raise ValueError(
                        f"{part_name} pair {[frame, camera_id]} is outside frame_range"
                    )
                if camera_id not in allowed_cameras:
                    raise ValueError(
                        f"{part_name} pair {[frame, camera_id]} uses a camera "
                        "not listed in cameras"
                    )
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(f"pairs in both train and test: {sorted(overlap)[:5]}")
        return self

    @property
    def strategy_name(self) -> str:
        """Tên chiến lược, bỏ phần ``:N``; ví dụ ``every_nth_frame``."""
        return parse_strategy(self.strategy)[0]


# ---------------------------------------------------------------- báo cáo kiểm tra (CONTRACTS 3.2)


class Check(StrictModel):
    """Một mục kiểm tra trong báo cáo."""

    check_id: str
    severity: Literal["error", "warning", "info"]
    message: str
    camera_id: Optional[str] = None
    frame_index: Optional[int] = Field(default=None, ge=0)
    details: dict[str, Any] = {}


class ReportSummary(StrictModel):
    """Số lỗi và cảnh báo trong báo cáo."""

    errors: int = Field(ge=0)
    warnings: int = Field(ge=0)


class ValidationReport(VersionedFile):
    """File ``reports/validation.json`` của một cảnh (CONTRACTS mục 3.2)."""

    FILE_KIND: ClassVar[str] = "report"
    scene_id: SceneId
    created_utc: datetime
    summary: ReportSummary
    checks: list[Check] = []

    @model_validator(mode="after")
    def _validate_counts(self) -> "ValidationReport":
        """Kiểm tra phần tóm tắt khớp với số mục ``error`` và ``warning`` trong ``checks``."""
        errors = sum(check.severity == "error" for check in self.checks)
        warnings = sum(check.severity == "warning" for check in self.checks)
        if (errors, warnings) != (self.summary.errors, self.summary.warnings):
            raise ValueError(
                f"summary says {self.summary.errors} errors, {self.summary.warnings} warnings; "
                f"checks contain {errors} and {warnings}"
            )
        return self


# ---------------------------------------------------------------- approach (CONTRACTS 4.1)


class Requires(StrictModel):
    """Điều kiện dữ liệu của một approach (CONTRACTS mục 4.2)."""

    camera_models: list[CameraModel] = Field(min_length=1)
    cuboids: Literal["required", "optional", "none"]
    split_strategies: list[SplitStrategyName] = Field(min_length=1)


def contains_value(choices: list[object], value: object) -> bool:
    """Kiểm tra ``value`` có trong ``choices``, so sánh cả kiểu.

    Toán tử ``in`` của Python coi ``True == 1`` và ``False == 0``; hàm này thì không, để giá trị
    ``1`` không bị nhận nhầm là lựa chọn ``true`` của một ô bật/tắt.

    Args:
        choices: Danh sách lựa chọn.
        value: Giá trị cần tìm.

    Returns:
        ``True`` nếu có một lựa chọn cùng kiểu và bằng ``value``.
    """
    return any(type(choice) is type(value) and choice == value for choice in choices)


def option_reference(value: object) -> Optional[str]:
    """Lấy tên ô chọn nếu ``value`` là chỗ giữ chỗ ``${options.<tên>}``.

    Args:
        value: Giá trị bất kỳ trong file approach.

    Returns:
        Tên ô chọn, hoặc ``None`` nếu ``value`` không phải chỗ giữ chỗ.
    """
    if isinstance(value, str):
        match = OPTION_REFERENCE_RE.match(value)
        if match:
            return match.group(1)
    return None


class OptionSpec(StrictModel):
    """Một ô chọn: danh sách lựa chọn và giá trị mặc định.

    ``default`` có thể là ``"auto"``: engine chọn theo manifest (ARCHITECTURE mục 4.5).
    """

    choices: list[OptionValue] = Field(min_length=1)
    default: OptionValue

    @model_validator(mode="after")
    def _validate_default(self) -> "OptionSpec":
        """Kiểm tra lựa chọn không trùng và giá trị mặc định nằm trong danh sách hoặc là auto."""
        if len({repr(choice) for choice in self.choices}) != len(self.choices):
            raise ValueError("choices must be unique")
        if self.default != "auto" and not contains_value(self.choices, self.default):
            raise ValueError(f"default {self.default!r} is not one of the choices and not 'auto'")
        return self


class Constraint(StrictModel):
    """Tổ hợp lựa chọn không hợp lệ: khi các ô có giá trị ``when`` thì cấm ``forbid``."""

    when: dict[str, OptionValue] = Field(min_length=1)
    forbid: dict[str, OptionValue] = Field(min_length=1)
    reason: str


class StageEntry(StrictModel):
    """Một bước trong danh sách stage của approach.

    ``stage`` là ``stage_id`` hoặc chỗ giữ chỗ ``${options.<tên>}``; giá trị trong ``params``
    cũng có thể là chỗ giữ chỗ.
    """

    stage: str
    params: dict[str, Any] = {}


class Approach(VersionedFile):
    """File khai báo approach ``drive2sim/approaches/<approach_id>.yaml`` (CONTRACTS mục 4.1)."""

    FILE_KIND: ClassVar[str] = "approach"
    approach_id: str
    name: str
    version: Version
    codebase: str
    track: Track
    level: Level
    requires: Requires
    options: dict[str, OptionSpec] = {}
    constraints: list[Constraint] = []
    presets: dict[str, dict[str, OptionValue]] = {}
    stages: list[StageEntry] = Field(min_length=1)

    @field_validator("approach_id")
    @classmethod
    def _validate_approach_id(cls, value: str) -> str:
        """Kiểm tra ``approach_id`` dạng ``snake_case``."""
        return identifiers.check_snake_case(value, "approach_id")

    @field_validator("codebase")
    @classmethod
    def _validate_codebase(cls, value: str) -> str:
        """Kiểm tra codebase là ``drivestudio`` hoặc ``external:<tên>``."""
        if value != "drivestudio" and not re.match(r"^external:[a-z][a-z0-9_]*$", value):
            raise ValueError("codebase must be 'drivestudio' or 'external:<name>'")
        return value

    def _check_assignment(self, where: str, values: dict[str, Any]) -> None:
        """Kiểm tra mỗi cặp ô–giá trị trỏ tới ô có thật và lựa chọn có thật.

        Args:
            where: Vị trí trong file, dùng trong thông báo lỗi.
            values: Các cặp tên ô và giá trị.

        Raises:
            ValueError: Nếu ô không tồn tại hoặc giá trị không phải một lựa chọn của ô.
        """
        for option_name, value in values.items():
            if option_name not in self.options:
                raise ValueError(f"{where}: unknown option {option_name!r}")
            if not contains_value(self.options[option_name].choices, value):
                raise ValueError(f"{where}: {value!r} is not a choice of option {option_name!r}")

    def _check_stage_entry(self, index: int, entry: StageEntry) -> None:
        """Kiểm tra một bước: ``stage_id`` hợp lệ, chỗ giữ chỗ trỏ tới ô có thật.

        Nếu ``stage`` là chỗ giữ chỗ thì mọi lựa chọn của ô đó phải là ``stage_id``.

        Args:
            index: Vị trí của bước trong danh sách.
            entry: Bước cần kiểm tra.

        Raises:
            ValueError: Nếu bước không hợp lệ.
        """
        option_name = option_reference(entry.stage)
        if option_name is None:
            identifiers.check_stage_id(entry.stage)
        elif option_name not in self.options:
            raise ValueError(f"stages[{index}]: unknown option {option_name!r}")
        else:
            for choice in self.options[option_name].choices:
                if not isinstance(choice, str):
                    raise ValueError(
                        f"stages[{index}]: option {option_name!r} is used as a stage "
                        f"but has non-stage choice {choice!r}"
                    )
                identifiers.check_stage_id(choice)
        for param_name, value in entry.params.items():
            referenced = option_reference(value)
            if referenced is not None and referenced not in self.options:
                raise ValueError(
                    f"stages[{index}].params.{param_name}: unknown option {referenced!r}"
                )

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Approach":
        """Kiểm tra tên ô, preset, ràng buộc và danh sách stage khớp với nhau."""
        for option_name in self.options:
            identifiers.check_snake_case(option_name, "option name")
        for preset_name in self.presets:
            identifiers.check_snake_case(preset_name, "preset name")
        for index, constraint in enumerate(self.constraints):
            self._check_assignment(f"constraints[{index}].when", constraint.when)
            self._check_assignment(f"constraints[{index}].forbid", constraint.forbid)
        for preset_name, values in self.presets.items():
            self._check_assignment(f"presets.{preset_name}", values)
        for index, entry in enumerate(self.stages):
            self._check_stage_entry(index, entry)
        return self


# ---------------------------------------------------------------- artifact (CONTRACTS 5.3)


class Producer(StrictModel):
    """Stage đã sinh ra artifact."""

    stage_id: StageId
    version: Version
    code_commit: Optional[str] = None


class ArtifactRecord(VersionedFile):
    """File ``artifact.json`` mô tả một kết quả của stage (CONTRACTS mục 5.3).

    ``cache_key`` bỏ trống với kết quả nằm trong thư mục run (stage không cache).
    """

    FILE_KIND: ClassVar[str] = "artifact"
    artifact_type: ArtifactType
    cache_key: Optional[CacheKey] = None
    producer: Producer
    params: dict[str, Any] = {}
    inputs: list[str] = []
    created_utc: datetime
    files: list[str] = []
    meta: dict[str, Any] = {}


# ---------------------------------------------------------------- run (CONTRACTS 6.2–6.4)


class RunApproach(StrictModel):
    """Approach của một run và các lựa chọn đã quyết định cho mọi ô."""

    approach_id: str
    version: Version
    preset: Optional[str] = None
    options: dict[str, OptionValue] = {}


class StageRecord(StrictModel):
    """Trạng thái của một stage trong run."""

    stage_id: StageId
    status: RunStatus
    cache_hit: bool = False
    cache_key: Optional[CacheKey] = None
    duration_s: float = Field(default=0, ge=0)
    peak_vram_mb: int = Field(default=0, ge=0)


class RunRecord(VersionedFile):
    """File ``d2s/run.json`` trong thư mục run (CONTRACTS mục 6.2)."""

    FILE_KIND: ClassVar[str] = "run"
    run_id: str
    scene_id: SceneId
    split_id: str
    approach: RunApproach
    level: Level
    executor: Executor
    status: RunStatus
    lidar_usage: list[str] = []
    stages: list[StageRecord] = []

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        """Kiểm tra định dạng ``run_id``."""
        return identifiers.check_run_id(value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "RunRecord":
        """Kiểm tra ``run_id`` chứa ``approach_id`` và ``lidar_usage`` chỉ gồm stage của run."""
        # run_id = YYYYMMDD-HHMMSS-<approach_id>-<hex>; approach_id không chứa dấu "-".
        if self.run_id.split("-")[2] != self.approach.approach_id:
            raise ValueError("run_id must contain approach.approach_id")
        stage_ids = {stage.stage_id for stage in self.stages}
        for stage_id in self.lidar_usage:
            identifiers.check_stage_id(stage_id)
            if stage_id not in stage_ids:
                raise ValueError(f"lidar_usage lists {stage_id!r}, which is not among the stages")
        return self


class LpipsSetting(StrictModel):
    """Cấu hình LPIPS dùng khi đánh giá."""

    net: str
    package: str
    package_version: str


class EvalProtocol(StrictModel):
    """Protocol đánh giá (SPEC mục 9)."""

    version: str
    image_dtype: Literal["uint8"]
    resolution: str
    lpips: LpipsSetting


class Metrics(StrictModel):
    """Metric ảnh; các metric theo vùng chỉ có khi cảnh có box."""

    psnr: float
    ssim: float
    lpips: float
    human_psnr: Optional[float] = None
    vehicle_psnr: Optional[float] = None
    occupied_psnr: Optional[float] = None


class Summary(VersionedFile):
    """File ``d2s/summary.json``: tóm tắt metric của một run (CONTRACTS mục 6.3)."""

    FILE_KIND: ClassVar[str] = "summary"
    protocol: EvalProtocol
    split_id: str
    overall: Metrics
    per_camera: dict[str, Metrics] = {}
    train_time_s: float = Field(ge=0)
    peak_vram_mb: int = Field(ge=0)
    num_gaussians: int = Field(ge=0)


class EnvRecord(VersionedFile):
    """File ``d2s/env.json``: môi trường đã chạy run (CONTRACTS mục 6.4)."""

    FILE_KIND: ClassVar[str] = "env"
    code_commit: str
    python: str
    pytorch: Optional[str] = None
    cuda: Optional[str] = None
    gsplat: Optional[str] = None
    gpu: Optional[str] = None
    seed: Optional[int] = None
    executor: Executor


# ---------------------------------------------------------------- gói xem trước (CONTRACTS 7)


class Preview(VersionedFile):
    """File ``preview.json`` của gói xem trước cho Spark (CONTRACTS mục 7)."""

    FILE_KIND: ClassVar[str] = "preview"
    source_run: str
    sh_degree: int = Field(ge=0, le=3)
    rigid_actors: list[str] = []
    box_only_actors: list[str] = []
    omitted: list[str] = []


# ---------------------------------------------------------------- worker (CONTRACTS 9.1)


class ErrorInfo(StrictModel):
    """Lỗi do worker hoặc API báo về (CONTRACTS mục 9.1 và 11.3).

    Attributes:
        code: Mã lỗi ``snake_case`` trong ``ERROR_CODES``, để máy phân loại.
        message: Thông báo tiếng Việt cho người dùng.
        details: Thông tin thêm tuỳ loại lỗi.
    """

    code: str
    message: str
    details: dict[str, Any] = {}

    @field_validator("code")
    @classmethod
    def _validate_code(cls, value: str) -> str:
        """Mã lỗi phải nằm trong bảng ``ERROR_CODES``."""
        if value not in ERROR_CODES:
            raise ValueError(f"mã lỗi không có trong bảng: {value} (xem CONTRACTS mục 11.3)")
        return value


class JobSpec(VersionedFile):
    """File ``job.json`` engine giao cho ``d2s-worker`` (CONTRACTS mục 9.1).

    ``stage_id`` và ``params`` chỉ có khi ``kind`` là ``stage``; ``inputs`` ánh xạ tên đầu vào
    sang đường dẫn tương đối với ``D2S_WORKDIR``.
    """

    FILE_KIND: ClassVar[str] = "job"
    job_id: str
    kind: JobKind
    stage_id: Optional[StageId] = None
    params: dict[str, Any] = {}
    inputs: dict[str, str] = {}
    output_dir: str
    env: str

    @model_validator(mode="after")
    def _check_stage_fields(self) -> "JobSpec":
        """Việc loại ``stage`` phải có ``stage_id``; loại khác không được có."""
        if self.kind == "stage" and self.stage_id is None:
            raise ValueError("job loại stage phải có stage_id")
        if self.kind != "stage" and self.stage_id is not None:
            raise ValueError(f"job loại {self.kind} không có stage_id")
        return self


class JobResult(VersionedFile):
    """File ``result.json`` worker ghi khi xong việc (CONTRACTS mục 9.1).

    Thành công thì ``outputs`` ánh xạ tên đầu ra sang đường dẫn và ``error`` trống; thất bại
    thì ``error`` bắt buộc.
    """

    FILE_KIND: ClassVar[str] = "result"
    job_id: str
    status: JobStatus
    outputs: dict[str, str] = {}
    duration_s: float = Field(ge=0)
    peak_vram_mb: int = Field(ge=0)
    error: Optional[ErrorInfo] = None

    @model_validator(mode="after")
    def _check_error_matches_status(self) -> "JobResult":
        """``error`` có khi và chỉ khi ``status`` là ``failed``."""
        if self.status == "failed" and self.error is None:
            raise ValueError("kết quả failed phải có error")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("kết quả succeeded không được có error")
        return self


# Tra lớp schema theo loại file trong schema_version, dùng bởi files.load_any.
MODELS_BY_KIND: dict[str, type] = {
    "manifest": Manifest,
    "split": Split,
    "report": ValidationReport,
    "approach": Approach,
    "artifact": ArtifactRecord,
    "run": RunRecord,
    "summary": Summary,
    "env": EnvRecord,
    "preview": Preview,
    "job": JobSpec,
    "result": JobResult,
}
