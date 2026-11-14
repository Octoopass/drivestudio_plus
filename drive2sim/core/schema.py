"""Định nghĩa schema (cấu trúc và luật kiểm tra) của mọi file JSON/YAML mà tool đọc và ghi.

Mỗi loại file trong CONTRACTS có một lớp Pydantic ở đây; đọc file bằng lớp đó thì nội dung được
kiểm tra ngay: thiếu trường, sai kiểu, sai định dạng định danh hoặc các trường mâu thuẫn nhau đều
bị báo lỗi kèm vị trí. Mọi schema từ chối khoá lạ, để lỗi gõ sai trong file YAML viết tay được
báo ngay thay vì bị bỏ qua.

File này chứa, theo thứ tự:
- Kiểu dùng chung: mô hình camera, track, mức năng lực, loại artifact, trạng thái run, bảng mã
  lỗi ``ERROR_CODES``, phiên bản schema hiện tại ``SCHEMA_VERSIONS``.
- Lớp gốc: ``StrictModel`` (từ chối khoá lạ), ``VersionedFile`` (kiểm tra ``schema_version``),
  ``Transform``, ``ErrorInfo``.
- Schema theo loại file: ``Manifest`` (CONTRACTS 2.2), ``Split`` (3.1), ``ValidationReport``
  (3.2), ``Approach`` (4.1), ``ArtifactRecord`` (5.3), ``RunRecord`` (6.2), ``Summary`` (6.3),
  ``EnvRecord`` (6.4), ``Preview`` (7), ``JobSpec`` và ``JobResult`` (9.1).
- Hàm hỗ trợ: ``parse_strategy``, ``contains_value``, ``option_reference``.
- ``MODELS_BY_KIND``: tra lớp schema theo loại file, dùng bởi ``files.load_any``.

Đọc và ghi file đi qua ``drive2sim/core/files.py``. Thông báo lỗi trong file này viết tiếng Anh vì
là thông báo kỹ thuật; CLI và API bọc lại thành tiếng Việt (CONTRACTS mục 10.6). Test
``tests/drive2sim/core/test_contracts_examples.py`` kiểm tra mọi ví dụ trong CONTRACTS theo các
schema ở đây, nên sửa schema thì sửa cả ví dụ trong CONTRACTS.

Code viết cho Python 3.9 nên dùng ``Optional``, ``Union`` thay cho cú pháp ``X | Y``.
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
# Track của từng mô hình camera (SPEC mục 2): camera thường thuộc track pinhole, camera góc rộng có
# mô hình ống kính riêng thuộc track fisheye. Track của approach, preset suy ra từ bảng này.
CAMERA_TRACK: dict[str, str] = {
    "opencv_pinhole": "pinhole",
    "ideal_pinhole": "pinhole",
    "opencv_fisheye": "fisheye",
    "ftheta": "fisheye",
}
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
    "forbidden",
    "conflict",
    "blocked",
    "internal",
    "worker_unavailable",
    "env_missing",
    "worker_crashed",
    "cancelled",
    "stage_failed",
)
Executor = Literal["local", "kaggle"]

# Phiên bản schema hiện tại của từng loại file.
SCHEMA_VERSIONS: dict[str, str] = {
    "manifest": "d2s-manifest/1.0",
    "split": "d2s-split/0.1",
    "report": "d2s-report/0.1",
    "approach": "d2s-approach/0.3",
    "artifact": "d2s-artifact/0.2",
    "run": "d2s-run/0.3",
    "summary": "d2s-summary/0.3",
    "env": "d2s-env/0.2",
    "preview": "d2s-preview/0.1",
    "job": "d2s-job/0.1",
    "result": "d2s-result/0.1",
}

OPTION_REFERENCE_RE = re.compile(r"^\$\{options\.([a-z][a-z0-9_]*)\}$")
EVERY_NTH_FRAME_RE = re.compile(r"^every_nth_frame:(\d+)$")


class StrictModel(BaseModel):
    """Lớp gốc của mọi schema trong file này.

    Đặt ``extra="forbid"`` để từ chối khoá không khai báo: gõ sai ``camera_modles`` trong file
    approach sẽ bị báo lỗi thay vì bị bỏ qua lặng lẽ.
    """

    model_config = ConfigDict(extra="forbid")


class VersionedFile(StrictModel):
    """Lớp gốc của mọi schema ứng với một file có trường ``schema_version``.

    Lớp con đặt ``FILE_KIND`` (ví dụ ``"run"``). Khi đọc, file được chấp nhận nếu cùng loại và
    cùng số major với ``SCHEMA_VERSIONS[FILE_KIND]``; số minor có thể khác, vì đổi minor chỉ thêm
    trường tuỳ chọn, không làm vỡ tương thích (CONTRACTS, phần đầu). Ví dụ khi phiên bản hiện tại
    là ``d2s-run/0.2`` thì file ``d2s-run/0.1`` vẫn đọc được, file ``d2s-run/1.0`` thì không.

    Attributes:
        FILE_KIND: Loại file, khoá trong ``SCHEMA_VERSIONS``.
        schema_version: Chuỗi dạng ``d2s-<loại>/<major>.<minor>``.
    """

    FILE_KIND: ClassVar[str] = ""
    schema_version: str

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        """Kiểm tra ``schema_version`` đúng loại file và cùng số major với phiên bản hiện tại.

        Args:
            value: Giá trị ``schema_version`` trong file.

        Returns:
            Chính ``value`` nếu hợp lệ.

        Raises:
            ValueError: Nếu sai định dạng, khác loại file hoặc khác số major.
        """
        kind, major, _ = identifiers.parse_schema_version(value)
        expected = SCHEMA_VERSIONS[cls.FILE_KIND]
        expected_kind, expected_major, _ = identifiers.parse_schema_version(expected)
        if kind != expected_kind or major != expected_major:
            raise ValueError(f"expected {expected} (same kind and major), got {value}")
        return value


class Transform(StrictModel):
    """Một phép biến đổi cứng (xoay và dịch) ghi trong file YAML/JSON của tool.

    Dùng cho các trường ``T_<a>_<b>``, theo quy ước NCore: ``T_a_b`` đưa điểm từ hệ ``a`` sang hệ
    ``b`` (CONTRACTS mục 1).

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
        """Kiểm tra quaternion có độ dài 1, sai số cho phép 1e-3.

        Quaternion không có độ dài 1 không biểu diễn một phép xoay; thường là do chép nhầm thứ
        tự ``x, y, z, w`` của scipy.

        Args:
            value: Quaternion ``[w, x, y, z]``.

        Returns:
            Chính ``value`` nếu hợp lệ.

        Raises:
            ValueError: Nếu độ dài khác 1 quá sai số.
        """
        norm = math.sqrt(sum(component * component for component in value))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"rotation_wxyz must be a unit quaternion, norm is {norm:.4f}")
        return value


class ErrorInfo(StrictModel):
    """Một lỗi theo hình dạng chung của tool, dùng trong ``run.json``, ``result.json`` và API.

    Cùng một hình dạng ở mọi nơi để CLI, Studio và worker xử lý lỗi theo một cách (CONTRACTS mục
    9.1, 11.3).

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
        """Kiểm tra mã lỗi nằm trong bảng ``ERROR_CODES``, để không ai tự đặt mã mới ngoài bảng.

        Args:
            value: Mã lỗi.

        Returns:
            Chính ``value`` nếu hợp lệ.

        Raises:
            ValueError: Nếu mã không có trong bảng.
        """
        if value not in ERROR_CODES:
            raise ValueError(f"mã lỗi không có trong bảng: {value} (xem CONTRACTS mục 11.3)")
        return value


# Kiểu chuỗi có kiểm tra định dạng, dùng lại cho mọi trường cùng loại định danh.
SceneId = Annotated[str, AfterValidator(identifiers.check_scene_id)]
StageId = Annotated[str, AfterValidator(identifiers.check_stage_id)]
CacheKey = Annotated[str, AfterValidator(identifiers.check_cache_key)]
Version = Annotated[str, AfterValidator(identifiers.check_version)]


# ---------------------------------------------------------------- manifest (CONTRACTS 2.2)


class ManifestSource(StrictModel):
    """Nguồn gốc của một cảnh: lấy từ dataset nào, chuỗi nào, chuyển bằng converter nào.

    Attributes:
        dataset: Tên nguồn, ví dụ ``nuscenes``; cũng là tiền tố của ``scene_id``.
        sequence_id: Id của chuỗi trong nguồn, ví dụ ``scene-0061``.
        converter: Converter đã dùng để tạo thư mục chuẩn, ví dụ ``drivestudio.nuscenes`` (script
            tiền xử lý của DriveStudio cho nuScenes).
        converter_version: Phiên bản converter, dạng ``<phiên bản drive2sim>+<commit>``; nằm
            trong khoá cache của mọi stage nhận cảnh, nên ingest lại bằng code khác thì cache cũ
            không bị dùng nhầm.
        data_license: License của dữ liệu nguồn, nếu biết.
    """

    dataset: Dataset
    sequence_id: str
    converter: str
    converter_version: str
    data_license: Optional[str] = None


class ManifestCamera(StrictModel):
    """Một camera trong cảnh, như ghi trong manifest.

    Thông số chi tiết (intrinsics, hệ số méo, extrinsic) nằm trong kho NCore; manifest chỉ ghi
    những gì engine cần để khoá approach và Studio cần để hiện tổng quan.

    Attributes:
        camera_id: Id của camera, giữ nguyên như trong nguồn, ví dụ ``CAM_FRONT``.
        model: Mô hình ống kính: ``opencv_pinhole``, ``opencv_fisheye``, ``ftheta`` hoặc
            ``ideal_pinhole``.
        resolution: ``[width, height]`` tính bằng pixel.
        shutter: Loại shutter theo tên của NCore, ví dụ ``GLOBAL``.
    """

    camera_id: str
    model: CameraModel
    resolution: tuple[int, int]
    shutter: ShutterType

    @field_validator("resolution")
    @classmethod
    def _validate_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Kiểm tra chiều rộng và chiều cao đều dương.

        Raises:
            ValueError: Nếu một trong hai không dương.
        """
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("resolution must be positive [width, height]")
        return value


class Timeline(StrictModel):
    """Dòng thời gian tham chiếu của cảnh, dùng để đánh chỉ số khung.

    Khung thứ ``i`` của cảnh là khung thứ ``i`` của ``reference_sensor``; ảnh của camera khác được
    ghép với khung gần nhất về thời gian (CONTRACTS mục 2.2).

    Attributes:
        reference_sensor: Id của camera hoặc LiDAR làm mốc.
        num_frames: Số khung của cảnh, lớn hơn 0.
    """

    reference_sensor: str
    num_frames: int = Field(gt=0)


class Available(StrictModel):
    """Những gì cảnh có sẵn từ nguồn; engine dựa vào đây để khoá approach (CONTRACTS mục 4.2).

    Attributes:
        lidar: Cảnh có LiDAR không; phải khớp với danh sách ``lidars`` của manifest.
        cuboids: Nguồn của box 3D: ``gt`` (nhãn gốc), ``autolabel`` (sinh tự động) hoặc ``none``.
        camera_labels: Nhãn theo ảnh có sẵn trong nguồn, ví dụ ``["depth", "segmentation"]``.
        egocar_mask: Nguồn có mask thân xe không.
    """

    lidar: bool
    cuboids: Literal["gt", "autolabel", "none"]
    camera_labels: list[str] = []
    egocar_mask: bool = False


class Manifest(VersionedFile):
    """Schema của file ``manifest.yaml`` đi kèm dữ liệu chuẩn của một cảnh (CONTRACTS mục 2.2).

    Manifest ghi cảnh có gì và thiếu gì. Converter ghi file này lúc ingest; engine đọc nó để quyết
    định approach và lựa chọn nào chạy được; Studio đọc nó để hiện tổng quan cảnh.

    Attributes:
        scene_id: Định danh cảnh, ``<dataset>_<sequence_id>``.
        source: Nguồn gốc của cảnh.
        track: ``pinhole`` hoặc ``fisheye``.
        cameras: Các camera, ít nhất một, ``camera_id`` không trùng.
        lidars: Id các LiDAR; rỗng nếu cảnh không có LiDAR.
        timeline: Dòng thời gian tham chiếu.
        available: Những gì cảnh có sẵn.
    """

    FILE_KIND: ClassVar[str] = "manifest"
    scene_id: SceneId
    source: ManifestSource
    track: Track
    cameras: list[ManifestCamera] = Field(min_length=1)
    lidars: list[str] = []
    timeline: Timeline
    available: Available

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Manifest":
        """Kiểm tra các trường của manifest không mâu thuẫn nhau.

        Gồm: ``camera_id`` không trùng; ``reference_sensor`` là một camera hoặc LiDAR có thật;
        ``available.lidar`` đúng khi và chỉ khi ``lidars`` không rỗng; ``scene_id`` bắt đầu bằng
        tên dataset.

        Raises:
            ValueError: Nếu một điều kiện không thoả.
        """
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
        """Lấy thông tin một camera của cảnh theo id.

        Args:
            camera_id: Id của camera, ví dụ ``CAM_FRONT``.

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
    """Tách chuỗi chiến lược split thành tên chiến lược và tham số ``N``.

    Dùng khi kiểm tra file split và khi engine so chiến lược của split với danh sách chiến lược
    approach hỗ trợ (CONTRACTS mục 3.1, 4.2).

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
    """Schema của file split ``data/<scene_id>/splits/<split_id>.json`` (CONTRACTS mục 3.1).

    Split ghi ảnh nào dùng để train, ảnh nào dùng để test. Mọi approach đánh giá trên cùng split
    nên so sánh được với nhau. Tạo bằng ``drive2sim/core/split.py``.

    Attributes:
        split_id: Định danh split, ``snake_case``, ví dụ ``every8_allcams``.
        scene_id: Cảnh của split.
        strategy: Chiến lược, ``every_nth_frame:N`` hoặc ``custom``.
        frame_range: Khung đầu và khung cuối của đoạn dùng, tính cả hai đầu.
        cameras: Các camera dùng trong split, không trùng.
        train: Các cặp ``[chỉ số khung, camera_id]`` để train.
        test: Các cặp ``[chỉ số khung, camera_id]`` để test; không cặp nào nằm ở cả hai phần.
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
        """Kiểm tra ``split_id`` dạng ``snake_case``.

        Raises:
            ValueError: Nếu sai định dạng.
        """
        return identifiers.check_snake_case(value, "split_id")

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: str) -> str:
        """Kiểm tra chuỗi chiến lược đọc được bằng ``parse_strategy``.

        Raises:
            ValueError: Nếu sai định dạng.
        """
        parse_strategy(value)
        return value

    @model_validator(mode="after")
    def _validate_pairs(self) -> "Split":
        """Kiểm tra các cặp train/test hợp lệ với nhau và với đoạn khung.

        Gồm: ``frame_range`` đúng thứ tự và không âm; ``cameras`` không trùng; mỗi phần không có
        cặp trùng; mỗi cặp nằm trong ``frame_range`` và dùng camera có trong ``cameras``; không
        cặp nào vừa ở ``train`` vừa ở ``test`` (không đánh giá trên ảnh đã train, SPEC mục 6.4).

        Raises:
            ValueError: Nếu một điều kiện không thoả; thông báo nêu cặp sai.
        """
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
        """Trả về tên chiến lược, bỏ phần ``:N``, ví dụ ``"every_nth_frame"``."""
        return parse_strategy(self.strategy)[0]


# ---------------------------------------------------------------- báo cáo kiểm tra (CONTRACTS 3.2)


class Check(StrictModel):
    """Một mục trong báo cáo kiểm tra dữ liệu của cảnh.

    Attributes:
        check_id: Tên phép kiểm tra, dạng ``<nhóm>.<tên>``, ví dụ ``timestamps.camera_sync``.
        severity: ``error`` (không chạy được), ``warning`` hoặc ``info``.
        message: Mô tả vấn đề bằng tiếng Việt.
        camera_id: Camera liên quan, nếu có.
        frame_index: Khung liên quan, nếu có.
        details: Số liệu thêm, ví dụ ``{"max_offset_us": 6200}``.
    """

    check_id: str
    severity: Literal["error", "warning", "info"]
    message: str
    camera_id: Optional[str] = None
    frame_index: Optional[int] = Field(default=None, ge=0)
    details: dict[str, Any] = {}


class ReportSummary(StrictModel):
    """Tóm tắt của báo cáo kiểm tra.

    Attributes:
        errors: Số mục ``error``.
        warnings: Số mục ``warning``.
    """

    errors: int = Field(ge=0)
    warnings: int = Field(ge=0)


class ValidationReport(VersionedFile):
    """Schema của báo cáo kiểm tra dữ liệu ``data/<scene_id>/reports/validation.json``.

    Ingest ghi báo cáo này sau khi kiểm tra đồng bộ thời gian, calib, mô hình ống kính, ảnh hỏng
    hoặc thiếu (SPEC mục 5 bước 2, CONTRACTS mục 3.2); Studio hiện nó ở nhóm A.

    Attributes:
        scene_id: Cảnh được kiểm tra.
        created_utc: Thời điểm tạo báo cáo.
        summary: Số lỗi và cảnh báo; phải khớp với ``checks``.
        checks: Các mục kiểm tra.
    """

    FILE_KIND: ClassVar[str] = "report"
    scene_id: SceneId
    created_utc: datetime
    summary: ReportSummary
    checks: list[Check] = []

    @model_validator(mode="after")
    def _validate_counts(self) -> "ValidationReport":
        """Kiểm tra phần tóm tắt khớp với số mục ``error`` và ``warning`` trong ``checks``.

        Raises:
            ValueError: Nếu số đếm không khớp.
        """
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
    """Điều kiện dữ liệu mà một approach cần để chạy được (CONTRACTS mục 4.2).

    Attributes:
        camera_models: Các mô hình camera approach nhận trực tiếp, không cần khử méo, với ít
            nhất một tổ hợp lựa chọn. Lựa chọn nào chỉ nhận một phần thì ghi ở
            ``OptionSpec.camera_models``.
        cuboids: Approach cần box 3D không: ``required``, ``optional`` hoặc ``none``.
        split_strategies: Các chiến lược split codebase của approach hỗ trợ.
    """

    camera_models: list[CameraModel] = Field(min_length=1)
    cuboids: Literal["required", "optional", "none"]
    split_strategies: list[SplitStrategyName] = Field(min_length=1)


def contains_value(choices: list[object], value: object) -> bool:
    """Kiểm tra ``value`` có trong ``choices``, so sánh cả kiểu.

    Toán tử ``in`` của Python coi ``True == 1`` và ``False == 0``; hàm này thì không, để giá trị
    ``1`` không bị nhận nhầm là lựa chọn ``true`` của một mục cấu hình bật/tắt.

    Args:
        choices: Danh sách lựa chọn.
        value: Giá trị cần tìm.

    Returns:
        ``True`` nếu có một lựa chọn cùng kiểu và bằng ``value``.
    """
    return any(type(choice) is type(value) and choice == value for choice in choices)


def option_reference(value: object) -> Optional[str]:
    """Lấy tên mục cấu hình nếu ``value`` là chỗ giữ chỗ ``${options.<tên>}``.

    Args:
        value: Giá trị bất kỳ trong file approach.

    Returns:
        Tên mục cấu hình, hoặc ``None`` nếu ``value`` không phải chỗ giữ chỗ.
    """
    if isinstance(value, str):
        match = OPTION_REFERENCE_RE.match(value)
        if match:
            return match.group(1)
    return None


class OptionSpec(StrictModel):
    """Một mục cấu hình của approach: các lựa chọn thay thế được cho nhau và giá trị mặc định.

    Ví dụ mục cấu hình ``renderer`` có ``choices: [3dgs, 3dgut, 3dgeer]``. ``default`` có thể là
    ``"auto"``: engine lấy lựa chọn đầu tiên không bị khoá theo dữ liệu (CONTRACTS mục 4.2).

    Attributes:
        choices: Các lựa chọn, ít nhất một, không trùng; thứ tự là thứ tự ưu tiên của ``auto``.
        default: Giá trị mặc định: một lựa chọn trong ``choices``, hoặc ``"auto"``.
        label: Tên hiển thị của mục cấu hình trên Studio, ví dụ ``Renderer``; trống thì dùng tên
            mục cấu hình.
        help: Giải thích ngắn mục này làm gì, hiện trong icon (i) cạnh tên mục cấu hình.
        labels: Tên hiển thị của từng lựa chọn, cùng thứ tự với ``choices``; trống thì hiện giá
            trị (thêm từ ``d2s-approach/0.2``, cùng ``label``, ``help``).
        descriptions: Giải thích từng lựa chọn, cùng thứ tự với ``choices``; hiện trong icon (i).
        group: Nhóm của mục cấu hình trong ``Approach.groups``; Studio hiện các mục cấu hình theo
            nhóm.
        camera_models: Mô hình camera mỗi lựa chọn nhận, cùng thứ tự với ``choices``; trống thì
            mọi lựa chọn nhận đủ ``requires.camera_models``. Ví dụ renderer ``3dgs`` chỉ nhận
            ``opencv_pinhole``, còn ``3dgeer`` nhận thêm ``opencv_fisheye``, ``ftheta`` (thêm từ
            ``d2s-approach/0.3``).
    """

    choices: list[OptionValue] = Field(min_length=1)
    default: OptionValue
    label: Optional[str] = None
    help: Optional[str] = None
    labels: list[str] = []
    descriptions: list[str] = []
    group: Optional[str] = None
    camera_models: list[list[CameraModel]] = []

    @model_validator(mode="after")
    def _validate_default(self) -> "OptionSpec":
        """Kiểm tra lựa chọn không trùng và ``default`` nằm trong ``choices`` hoặc là ``auto``.

        Raises:
            ValueError: Nếu có lựa chọn trùng hoặc ``default`` không hợp lệ.
        """
        if len({repr(choice) for choice in self.choices}) != len(self.choices):
            raise ValueError("choices must be unique")
        if self.default != "auto" and not contains_value(self.choices, self.default):
            raise ValueError(f"default {self.default!r} is not one of the choices and not 'auto'")
        if self.labels and len(self.labels) != len(self.choices):
            raise ValueError("labels must have one entry per choice")
        if self.descriptions and len(self.descriptions) != len(self.choices):
            raise ValueError("descriptions must have one entry per choice")
        if self.camera_models and len(self.camera_models) != len(self.choices):
            raise ValueError("camera_models must have one entry per choice")
        if any(not models for models in self.camera_models):
            raise ValueError("camera_models: every choice needs at least one camera model")
        return self


class OptionGroup(StrictModel):
    """Một nhóm mục cấu hình hiện chung một mục trên Studio, ví dụ nhóm thuật toán phụ trợ.

    Attributes:
        label: Tên nhóm.
        help: Giải thích nhóm, hiện trong icon (i) cạnh tên nhóm.
    """

    label: str
    help: Optional[str] = None


class Constraint(StrictModel):
    """Một tổ hợp lựa chọn không hợp lệ trong approach.

    Khi mọi mục cấu hình trong ``when`` có đúng giá trị đó thì cấm mọi giá trị trong ``forbid``. Ví
    dụ ``when: {renderer: 3dgeer}, forbid: {campose: true}`` vì 3DGEER không tối ưu được pose
    camera.

    Attributes:
        when: Mục → giá trị làm ràng buộc có hiệu lực.
        forbid: Mục → giá trị bị cấm khi ``when`` khớp.
        reason: Lý do bằng tiếng Việt, hiện cho người dùng khi tổ hợp bị khoá.
    """

    when: dict[str, OptionValue] = Field(min_length=1)
    forbid: dict[str, OptionValue] = Field(min_length=1)
    reason: str


class StageEntry(StrictModel):
    """Một bước trong danh sách ``stages`` của approach.

    Attributes:
        stage: ``stage_id`` (ví dụ ``prep.select``) hoặc chỗ giữ chỗ ``${options.<mục>}``; chỗ giữ
            chỗ được thay bằng lựa chọn của mục cấu hình trước khi chạy.
        params: Tham số cho stage; giá trị cũng có thể là chỗ giữ chỗ.
    """

    stage: str
    params: dict[str, Any] = {}


class Approach(VersionedFile):
    """Schema của file khai báo approach ``drive2sim/approaches/<approach_id>.yaml``.

    Approach là một cách dựng cảnh: danh sách stage chạy lần lượt, có mục cấu hình, ràng buộc và
    preset (CONTRACTS mục 4.1). Schema kiểm tra cả sự nhất quán bên trong file: preset và ràng
    buộc chỉ dùng mục cấu hình và lựa chọn có thật, chỗ giữ chỗ trỏ tới mục cấu hình có thật.

    Attributes:
        approach_id: Định danh ``snake_case``, ví dụ ``omnire``.
        name: Tên hiển thị.
        version: Phiên bản của file approach.
        codebase: ``drivestudio`` hoặc ``external:<tên>``.
        level: Mức năng lực cao nhất đạt được, ``L1``–``L4``.
        requires: Điều kiện dữ liệu. Track (pinhole, fisheye) của approach và của từng preset suy
            ra từ mô hình camera nhận được (``CAMERA_TRACK``), không ghi riêng; file ``0.1``,
            ``0.2`` có trường ``track`` thì trường đó được bỏ qua.
        options: Tên mục → mục cấu hình.
        constraints: Các tổ hợp không hợp lệ.
        presets: Tên preset → giá trị của các mục cấu hình khác mặc định.
        preset_help: Tên preset → mô tả ngắn hiện trên Studio (thêm từ ``d2s-approach/0.2``).
        groups: Tên nhóm → nhóm mục cấu hình, theo thứ tự hiện trên Studio; chỉ để hiển thị.
        stages: Các bước theo thứ tự chạy, ít nhất một.
    """

    FILE_KIND: ClassVar[str] = "approach"
    approach_id: str
    name: str
    version: Version
    codebase: str
    level: Level
    requires: Requires
    options: dict[str, OptionSpec] = {}
    constraints: list[Constraint] = []
    presets: dict[str, dict[str, OptionValue]] = {}
    preset_help: dict[str, str] = {}
    groups: dict[str, OptionGroup] = {}
    stages: list[StageEntry] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _drop_old_track(cls, data: object) -> object:
        """Bỏ trường ``track`` của file ``0.1``, ``0.2``; từ ``0.3`` track suy ra từ camera."""
        old = ("d2s-approach/0.1", "d2s-approach/0.2")
        if isinstance(data, dict) and "track" in data and data.get("schema_version") in old:
            data = {key: value for key, value in data.items() if key != "track"}
        return data

    @field_validator("approach_id")
    @classmethod
    def _validate_approach_id(cls, value: str) -> str:
        """Kiểm tra ``approach_id`` dạng ``snake_case``.

        Raises:
            ValueError: Nếu sai định dạng.
        """
        return identifiers.check_snake_case(value, "approach_id")

    @field_validator("codebase")
    @classmethod
    def _validate_codebase(cls, value: str) -> str:
        """Kiểm tra codebase là ``drivestudio`` hoặc ``external:<tên>``.

        Raises:
            ValueError: Nếu không theo một trong hai dạng.
        """
        if value != "drivestudio" and not re.match(r"^external:[a-z][a-z0-9_]*$", value):
            raise ValueError("codebase must be 'drivestudio' or 'external:<name>'")
        return value

    def _check_assignment(self, where: str, values: dict[str, Any]) -> None:
        """Kiểm tra mỗi cặp mục–giá trị trỏ tới mục cấu hình có thật và lựa chọn có thật.

        Args:
            where: Vị trí trong file, dùng trong thông báo lỗi.
            values: Các cặp tên mục cấu hình và giá trị.

        Raises:
            ValueError: Nếu mục cấu hình không tồn tại hoặc giá trị không phải một lựa chọn của mục
                cấu hình.
        """
        for option_name, value in values.items():
            if option_name not in self.options:
                raise ValueError(f"{where}: unknown option {option_name!r}")
            if not contains_value(self.options[option_name].choices, value):
                raise ValueError(f"{where}: {value!r} is not a choice of option {option_name!r}")

    def _check_stage_entry(self, index: int, entry: StageEntry) -> None:
        """Kiểm tra một bước: ``stage_id`` hợp lệ, chỗ giữ chỗ trỏ tới mục cấu hình có thật.

        Nếu ``stage`` là chỗ giữ chỗ thì mọi lựa chọn của mục cấu hình đó phải là ``stage_id``.

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
        """Kiểm tra các phần của file approach nhất quán với nhau.

        Gồm: tên mục cấu hình và tên preset dạng ``snake_case``; ràng buộc và preset chỉ dùng mục
        cấu hình và lựa chọn có thật; mỗi bước trong ``stages`` hợp lệ (``_check_stage_entry``).

        Raises:
            ValueError: Nếu một phần không hợp lệ; thông báo nêu vị trí trong file.
        """
        for option_name in self.options:
            identifiers.check_snake_case(option_name, "option name")
        for preset_name in self.presets:
            identifiers.check_snake_case(preset_name, "preset name")
        for index, constraint in enumerate(self.constraints):
            self._check_assignment(f"constraints[{index}].when", constraint.when)
            self._check_assignment(f"constraints[{index}].forbid", constraint.forbid)
        for preset_name, values in self.presets.items():
            self._check_assignment(f"presets.{preset_name}", values)
        unknown = set(self.preset_help) - set(self.presets)
        if unknown:
            raise ValueError(f"preset_help: unknown preset {sorted(unknown)[0]!r}")
        for option_name, spec in self.options.items():
            if spec.group is not None and spec.group not in self.groups:
                raise ValueError(f"options.{option_name}.group: unknown group {spec.group!r}")
            for models in spec.camera_models:
                extra = set(models) - set(self.requires.camera_models)
                if extra:
                    raise ValueError(
                        f"options.{option_name}.camera_models: {sorted(extra)[0]!r} is not in "
                        "requires.camera_models"
                    )
        for index, entry in enumerate(self.stages):
            self._check_stage_entry(index, entry)
        return self


# ---------------------------------------------------------------- artifact (CONTRACTS 5.3)


class Producer(StrictModel):
    """Stage đã tạo ra một artifact, ghi trong ``artifact.json``.

    Attributes:
        stage_id: Stage đã tạo artifact.
        version: Phiên bản của stage lúc tạo.
        code_commit: Commit của code lúc tạo, nếu đọc được.
    """

    stage_id: StageId
    version: Version
    code_commit: Optional[str] = None


class ArtifactRecord(VersionedFile):
    """Schema của file ``artifact.json`` mô tả kết quả của một stage (CONTRACTS mục 5.3).

    Engine ghi file này vào thư mục kết quả của mỗi stage sau khi stage chạy xong. Thư mục cache
    có ``artifact.json`` nghĩa là kết quả đầy đủ, dùng lại được.

    Attributes:
        artifact_type: Loại artifact, theo bảng CONTRACTS mục 5.4.
        cache_key: Khoá cache; bỏ trống với kết quả nằm trong thư mục run (stage không cache).
        producer: Stage đã tạo artifact.
        params: Tham số đã gộp của stage.
        inputs: Khoá của các đầu vào, ví dụ ``["scene:nuscenes_scene-0061@19.9.0"]``.
        created_utc: Thời điểm tạo.
        files: Các file trong thư mục, đường dẫn tương đối với thư mục đó.
        meta: Thông tin thêm tuỳ stage, ví dụ tên model đã dùng.
        duration_s: Thời gian stage chạy để tạo ra kết quả này, giây; run sau dùng lại kết quả
            từ cache tính thời gian này vào tổng thời gian dựng cảnh (thêm từ
            ``d2s-artifact/0.2``).
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
    duration_s: Optional[float] = Field(default=None, ge=0)


# ---------------------------------------------------------------- run (CONTRACTS 6.2–6.4)


class RunApproach(StrictModel):
    """Approach của một run cùng các lựa chọn đã quyết định, ghi trong ``run.json``.

    Attributes:
        approach_id: Approach đã chạy.
        version: Phiên bản file approach lúc tạo run.
        preset: Preset đã chọn, nếu có.
        options: Giá trị cụ thể của mọi mục cấu hình, kể cả mục để ``auto``; Studio lọc và so sánh
            run theo trường này.
        quick: ``True`` nếu là run chạy thử nhanh (``--quick``): train ít iteration để kiểm tra
            approach chạy được từ đầu tới cuối; metric không so được với run đủ iteration
            (thêm từ ``d2s-run/0.3``).
    """

    approach_id: str
    version: Version
    preset: Optional[str] = None
    options: dict[str, OptionValue] = {}
    quick: bool = False


class StageRecord(StrictModel):
    """Trạng thái và kết quả của một stage trong ``run.json``.

    Attributes:
        stage_id: Stage.
        status: ``pending``, ``running``, ``succeeded``, ``failed`` hoặc ``cancelled``.
        cache_hit: ``True`` nếu dùng lại kết quả có sẵn trong cache.
        cache_key: Khoá cache với stage có cache; ``None`` với stage khác.
        duration_s: Thời gian máy thực sự chạy stage này trong run, giây, cộng dồn qua các lần
            chạy tiếp; ``0`` khi dùng lại kết quả cũ.
        reused_duration_s: Khi dùng lại kết quả từ cache: thời gian lần chạy gốc đã tạo ra kết
            quả đó (đọc từ ``artifact.json``); ``None`` nếu không dùng lại hoặc không biết.
        peak_vram_mb: VRAM cao nhất, megabyte, lấy lớn nhất qua các lần chạy; ``0`` khi dùng
            lại kết quả cũ hoặc không đo được.
        params: Tham số đã gộp, để chạy tiếp đúng như lúc tạo run (thêm từ ``d2s-run/0.2``).
        error: Lỗi nếu stage ``failed``.
    """

    stage_id: StageId
    status: RunStatus
    cache_hit: bool = False
    cache_key: Optional[CacheKey] = None
    duration_s: float = Field(default=0, ge=0)
    reused_duration_s: Optional[float] = Field(default=None, ge=0)
    peak_vram_mb: int = Field(default=0, ge=0)
    params: dict[str, Any] = {}
    error: Optional[ErrorInfo] = None


class RunRecord(VersionedFile):
    """Schema của file ``d2s/run.json`` trong thư mục run (CONTRACTS mục 6.2).

    Engine tạo file này khi tạo run và ghi lại sau mỗi lần một stage đổi trạng thái; Studio và
    ``d2s run status`` đọc nó để hiện tiến độ.

    Attributes:
        run_id: Định danh run, ``YYYYMMDD-HHMMSS-<approach_id>-<4 hex>``.
        scene_id: Cảnh của run.
        split_id: Split của run.
        approach: Approach và lựa chọn đã quyết định.
        level: Mức năng lực của approach.
        executor: Nơi chạy, ``local`` hoặc ``kaggle``.
        status: Trạng thái của cả run.
        created_utc: Thời điểm tạo run.
        lidar_usage: Các stage dùng LiDAR; rỗng nếu run không dùng LiDAR.
        stages: Trạng thái của từng stage theo thứ tự chạy.
    """

    FILE_KIND: ClassVar[str] = "run"
    run_id: str
    scene_id: SceneId
    split_id: str
    approach: RunApproach
    level: Level
    executor: Executor
    status: RunStatus
    created_utc: Optional[datetime] = None
    lidar_usage: list[str] = []
    stages: list[StageRecord] = []

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        """Kiểm tra định dạng ``run_id``.

        Raises:
            ValueError: Nếu sai định dạng.
        """
        return identifiers.check_run_id(value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "RunRecord":
        """Kiểm tra ``run_id`` chứa đúng ``approach_id`` và ``lidar_usage`` chỉ gồm stage của run.

        Raises:
            ValueError: Nếu một điều kiện không thoả.
        """
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
    """Cấu hình LPIPS đã dùng khi đánh giá; ghi rõ để so sánh được giữa các run (SPEC mục 9).

    Attributes:
        net: Backbone, ví dụ ``alex``.
        package: Gói Python đã dùng, ví dụ ``lpips``.
        package_version: Phiên bản gói.
    """

    net: str
    package: str
    package_version: str


class EvalProtocol(StrictModel):
    """Protocol đánh giá đã dùng cho một run (SPEC mục 9).

    Attributes:
        version: Phiên bản protocol.
        image_dtype: Kiểu ảnh khi đo; hiện chỉ ``uint8`` (ảnh 8 bit).
        resolution: Độ phân giải khi đo, ví dụ ``native``.
        lpips: Cấu hình LPIPS.
    """

    version: str
    image_dtype: Literal["uint8"]
    resolution: str
    lpips: LpipsSetting


class Metrics(StrictModel):
    """Các metric chất lượng ảnh trên tập test.

    Attributes:
        psnr: PSNR trên toàn ảnh, dB.
        ssim: SSIM.
        lpips: LPIPS.
        human_psnr: PSNR trong vùng người; chỉ có khi cảnh có box.
        vehicle_psnr: PSNR trong vùng xe; chỉ có khi cảnh có box.
        dynamic_psnr: PSNR trong vùng vật động (mọi người và xe, ``masked_psnr`` của
            DriveStudio); chỉ có khi cảnh có box (thêm từ ``d2s-summary/0.3``).
        occupied_psnr: PSNR trong vùng không phải bầu trời (``Non-Sky PSNR`` của DriveStudio);
            chỉ có khi cảnh có mask bầu trời.
    """

    psnr: float
    ssim: float
    lpips: float
    human_psnr: Optional[float] = None
    vehicle_psnr: Optional[float] = None
    dynamic_psnr: Optional[float] = None
    occupied_psnr: Optional[float] = None


class Summary(VersionedFile):
    """Schema của file ``summary.json``: tóm tắt kết quả đánh giá của một run.

    Stage ``eval.standard`` ghi file này vào thư mục của nó; Studio và ``d2s report`` đọc nó để
    lập bảng xếp hạng (CONTRACTS mục 6.3).

    Attributes:
        protocol: Protocol đánh giá.
        split_id: Split đã đánh giá.
        overall: Metric trên mọi camera.
        per_camera: ``camera_id`` → metric của camera đó.
        train_time_s: Thời gian train thực, giây: chỉ tính các bước train, cộng qua các lần chạy
            tiếp, không tính lúc dừng, nạp checkpoint, đánh giá hay lưu file; để trống nếu không
            biết.
        peak_vram_mb: VRAM cao nhất khi train, megabyte; để trống nếu không biết.
        num_gaussians: Số Gaussian của cả model cuối (mọi node); để trống nếu không đọc được.
        num_iters: Số iteration đã train (thêm từ ``d2s-summary/0.3``).
        render_fps: Tốc độ render khi đánh giá, khung mỗi giây, chỉ tính thời gian model vẽ ảnh
            (thêm từ ``d2s-summary/0.3``).
        render_resolution: Độ phân giải ảnh khi đo ``render_fps``, dạng ``<rộng>x<cao>``.
    """

    FILE_KIND: ClassVar[str] = "summary"
    protocol: EvalProtocol
    split_id: str
    overall: Metrics
    per_camera: dict[str, Metrics] = {}
    train_time_s: Optional[float] = Field(default=None, ge=0)
    peak_vram_mb: Optional[int] = Field(default=None, ge=0)
    num_gaussians: Optional[int] = Field(default=None, ge=0)
    num_iters: Optional[int] = Field(default=None, ge=0)
    render_fps: Optional[float] = Field(default=None, ge=0)
    render_resolution: Optional[str] = None


class Machine(StrictModel):
    """Cấu hình phần cứng của một máy đã chạy run (CONTRACTS mục 6.4).

    Thời gian, VRAM và FPS chỉ so được giữa các run chạy trên cùng loại máy, nên mỗi run ghi lại
    máy đã chạy nó. Trường nào không đọc được thì để trống.

    Attributes:
        cpu: Tên CPU, ví dụ ``Intel(R) Xeon(R) CPU @ 2.00GHz``.
        cpu_cores: Số luồng CPU.
        ram_gb: Dung lượng RAM, GB.
        gpu: Tên GPU, ví dụ ``Tesla T4``; ``None`` nếu không có GPU NVIDIA.
        gpu_count: Số GPU.
        gpu_memory_mb: VRAM của mỗi GPU, megabyte.
        gpu_driver: Phiên bản driver NVIDIA.
        os: Hệ điều hành.
        first_seen_utc: Lần đầu máy này chạy run.
    """

    cpu: Optional[str] = None
    cpu_cores: Optional[int] = None
    ram_gb: Optional[float] = None
    gpu: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_mb: Optional[int] = None
    gpu_driver: Optional[str] = None
    os: Optional[str] = None
    first_seen_utc: Optional[datetime] = None


class EnvRecord(VersionedFile):
    """Schema của file ``d2s/env.json``: môi trường đã chạy run, để chạy lại được (SPEC mục 11).

    Engine ghi file khi tạo run (chưa có máy nào) và cập nhật mỗi lần bắt đầu chạy hoặc chạy tiếp,
    trên đúng máy chạy: với run Kaggle là máy của notebook, không phải máy đã tạo run. Trường nào
    không xác định được thì bỏ trống (CONTRACTS mục 6.4).

    Attributes:
        code_commit: Commit của code, hoặc ``unknown``.
        python: Phiên bản Python.
        pytorch: Phiên bản PyTorch.
        cuda: Phiên bản CUDA mà PyTorch dùng.
        gsplat: Phiên bản gsplat.
        gpu: Tên GPU của máy chạy gần nhất.
        seed: Seed ngẫu nhiên.
        executor: Nơi chạy, ``local`` hoặc ``kaggle``.
        machines: Các máy đã chạy run, theo thứ tự; run chạy tiếp trên máy khác (ví dụ phiên
            Kaggle mới được cấp GPU khác) có nhiều máy (thêm từ ``d2s-env/0.2``).
    """

    FILE_KIND: ClassVar[str] = "env"
    code_commit: str
    python: str
    pytorch: Optional[str] = None
    cuda: Optional[str] = None
    gsplat: Optional[str] = None
    gpu: Optional[str] = None
    seed: Optional[int] = None
    executor: Executor
    machines: list[Machine] = []


# ---------------------------------------------------------------- gói xem trước (CONTRACTS 7)


class Preview(VersionedFile):
    """Schema của file ``preview.json`` trong gói xem trước cho Spark (CONTRACTS mục 7).

    Gói xem trước cho phép xem cảnh trong trình duyệt không cần GPU CUDA; file này ghi gói có gì
    và thiếu gì so với model đầy đủ.

    Attributes:
        source_run: Thư mục run tạo ra gói, tương đối với ``D2S_WORKDIR``.
        sh_degree: Bậc spherical harmonics giữ lại, 0–3.
        rigid_actors: ``track_id`` của các xe có Gaussian riêng.
        box_only_actors: ``track_id`` của actor không rigid, chỉ hiện bằng box.
        omitted: Những phần của model không có trong gói, ví dụ ``["sky"]``.
    """

    FILE_KIND: ClassVar[str] = "preview"
    source_run: str
    sh_degree: int = Field(ge=0, le=3)
    rigid_actors: list[str] = []
    box_only_actors: list[str] = []
    omitted: list[str] = []


# ---------------------------------------------------------------- worker (CONTRACTS 9.1)


class JobSpec(VersionedFile):
    """Schema của file ``job.json`` engine giao cho ``d2s-worker`` (CONTRACTS mục 9.1).

    Worker chạy trong môi trường riêng (codebase ngoài, hoặc tiến trình render của viewer); file
    này là cách duy nhất engine nói cho worker biết phải làm gì.

    Attributes:
        job_id: Định danh việc, thường là ``<run_id>/<stage_id>``.
        kind: ``stage`` (chạy một stage) hoặc ``render`` (giữ model và render cho viewer).
        stage_id: Stage cần chạy; chỉ có khi ``kind`` là ``stage``.
        params: Tham số đã gộp của stage.
        inputs: Tên đầu vào → đường dẫn tương đối với ``D2S_WORKDIR``.
        output_dir: Thư mục worker ghi kết quả, tương đối với ``D2S_WORKDIR``.
        env: Tên file môi trường trong ``envs/``, không có đuôi ``.yml``.
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
        """Kiểm tra việc loại ``stage`` có ``stage_id`` và việc loại khác không có.

        Raises:
            ValueError: Nếu không thoả.
        """
        if self.kind == "stage" and self.stage_id is None:
            raise ValueError("job loại stage phải có stage_id")
        if self.kind != "stage" and self.stage_id is not None:
            raise ValueError(f"job loại {self.kind} không có stage_id")
        return self


class JobResult(VersionedFile):
    """Schema của file ``result.json`` worker ghi khi xong việc, kể cả khi thất bại.

    Định dạng ở CONTRACTS mục 9.1.

    Attributes:
        job_id: Định danh việc, giống ``job.json``.
        status: ``succeeded`` hoặc ``failed``.
        outputs: Tên đầu ra → đường dẫn; rỗng khi thất bại.
        duration_s: Thời gian chạy, giây.
        peak_vram_mb: VRAM cao nhất, megabyte.
        error: Lỗi; có khi và chỉ khi ``status`` là ``failed``.
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
        """Kiểm tra ``error`` có khi và chỉ khi ``status`` là ``failed``.

        Raises:
            ValueError: Nếu không thoả.
        """
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
