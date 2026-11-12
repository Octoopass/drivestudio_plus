"""Schemas of the files written or read by the tool (docs/drive2sim/CONTRACTS.md).

Every model rejects unknown keys, so a typo in a hand-written YAML file is reported instead of ignored.
Written for Python 3.9: Optional/Union/List/Dict instead of the `X | Y` syntax.
"""
import math
import re
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drive2sim.core import ids

# ---------------------------------------------------------------- shared types

CameraModel = Literal["opencv_pinhole", "opencv_fisheye", "ftheta", "ideal_pinhole"]
ShutterType = Literal["GLOBAL", "ROLLING_TOP_TO_BOTTOM", "ROLLING_LEFT_TO_RIGHT",
                      "ROLLING_BOTTOM_TO_TOP", "ROLLING_RIGHT_TO_LEFT"]
Track = Literal["pinhole", "fisheye"]
Level = Literal["L1", "L2", "L3", "L4"]
Dataset = Literal["waymo", "nuscenes", "nuplan", "argoverse2", "pandaset",
                  "kitti", "physicalai", "carla", "colmap", "custom"]
SplitStrategyName = Literal["every_nth_frame", "custom"]
OptionValue = Union[bool, int, float, str]   # bool first so YAML true/false stay booleans
ArtifactType = Literal["scene_subset", "masks", "smpl", "depth_prior", "points_init", "drivestudio_scene",
                       "model", "eval_report", "videos", "export_ply", "preview"]
RunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]

SCHEMA_VERSIONS = {
    "manifest": "d2s-manifest/0.1",
    "split": "d2s-split/0.1",
    "report": "d2s-report/0.1",
    "approach": "d2s-approach/0.1",
    "artifact": "d2s-artifact/0.1",
    "run": "d2s-run/0.1",
    "summary": "d2s-summary/0.1",
    "env": "d2s-env/0.1",
    "preview": "d2s-preview/0.1",
}

OPTION_REF_RE = re.compile(r"^\$\{options\.([a-z][a-z0-9_]*)\}$")


class D2SModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Versioned(D2SModel):
    """A file with a schema_version field. Same kind and major version are accepted; minor may differ."""
    KIND: ClassVar[str] = ""
    schema_version: str

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        kind, major, _ = ids.parse_schema_version(v)
        want_kind, want_major, _ = ids.parse_schema_version(SCHEMA_VERSIONS[cls.KIND])
        if kind != want_kind or major != want_major:
            raise ValueError(f"expected {SCHEMA_VERSIONS[cls.KIND]} (same kind and major), got {v}")
        return v


class Transform(D2SModel):
    """Rigid transform as written in YAML/JSON (CONTRACTS section 1)."""
    translation: Tuple[float, float, float]
    rotation_wxyz: Tuple[float, float, float, float]

    @field_validator("rotation_wxyz")
    @classmethod
    def _unit_quaternion(cls, q):
        n = math.sqrt(sum(c * c for c in q))
        if abs(n - 1.0) > 1e-3:
            raise ValueError(f"rotation_wxyz must be a unit quaternion, norm is {n:.4f}")
        return q


# ---------------------------------------------------------------- manifest (CONTRACTS 2.2)

class ManifestSource(D2SModel):
    dataset: Dataset
    sequence_id: str
    converter: str
    converter_version: str
    data_license: Optional[str] = None


class TimestampInterval(D2SModel):
    start: int = Field(ge=0)
    stop: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.stop < self.start:
            raise ValueError("timestamp_interval_us.stop must not be before start")
        return self


class ManifestNCore(D2SModel):
    version: Literal["v4"]
    sequence_id: str
    timestamp_interval_us: TimestampInterval


class ManifestCamera(D2SModel):
    camera_id: str
    model: CameraModel
    resolution: Tuple[int, int]
    shutter: ShutterType

    @field_validator("resolution")
    @classmethod
    def _positive(cls, r):
        if r[0] <= 0 or r[1] <= 0:
            raise ValueError("resolution must be positive [width, height]")
        return r


class Timeline(D2SModel):
    reference_sensor: str
    num_frames: int = Field(gt=0)


class Available(D2SModel):
    lidar: bool
    cuboids: Literal["gt", "autolabel", "none"]
    camera_labels: List[str] = []
    egocar_mask: bool = False


class Manifest(Versioned):
    KIND: ClassVar[str] = "manifest"
    scene_id: str
    source: ManifestSource
    ncore: ManifestNCore
    track: Track
    cameras: List[ManifestCamera] = Field(min_length=1)
    lidars: List[str] = []
    timeline: Timeline
    available: Available

    @field_validator("scene_id")
    @classmethod
    def _scene_id(cls, v):
        return ids.check_scene_id(v)

    @model_validator(mode="after")
    def _consistent(self):
        cam_ids = [c.camera_id for c in self.cameras]
        if len(set(cam_ids)) != len(cam_ids):
            raise ValueError("camera_id values must be unique")
        if self.timeline.reference_sensor not in cam_ids + self.lidars:
            raise ValueError(f"timeline.reference_sensor {self.timeline.reference_sensor!r} is not a camera or lidar")
        if self.available.lidar != bool(self.lidars):
            raise ValueError("available.lidar must be true exactly when lidars is not empty")
        if not self.scene_id.startswith(self.source.dataset + "_"):
            raise ValueError("scene_id must start with source.dataset")
        return self

    def camera(self, camera_id: str) -> ManifestCamera:
        for c in self.cameras:
            if c.camera_id == camera_id:
                return c
        raise KeyError(camera_id)


# ---------------------------------------------------------------- split (CONTRACTS 3.1)

def parse_strategy(strategy: str) -> Tuple[str, Optional[int]]:
    """'every_nth_frame:8' -> ('every_nth_frame', 8); 'custom' -> ('custom', None)."""
    if strategy == "custom":
        return "custom", None
    m = re.match(r"^every_nth_frame:(\d+)$", strategy)
    if m and int(m.group(1)) >= 1:
        return "every_nth_frame", int(m.group(1))
    raise ValueError(f"invalid strategy {strategy!r}: expected 'every_nth_frame:N' (N >= 1) or 'custom'")


class Split(Versioned):
    KIND: ClassVar[str] = "split"
    split_id: str
    scene_id: str
    strategy: str
    frame_range: Tuple[int, int]
    cameras: List[str] = Field(min_length=1)
    train: List[Tuple[int, str]]
    test: List[Tuple[int, str]]

    @field_validator("split_id")
    @classmethod
    def _split_id(cls, v):
        return ids.check_snake(v, "split_id")

    @field_validator("scene_id")
    @classmethod
    def _scene_id(cls, v):
        return ids.check_scene_id(v)

    @field_validator("strategy")
    @classmethod
    def _strategy(cls, v):
        parse_strategy(v)
        return v

    @model_validator(mode="after")
    def _consistent(self):
        first, last = self.frame_range
        if first < 0 or last < first:
            raise ValueError("frame_range must be [first, last] with 0 <= first <= last")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras must be unique")
        cams = set(self.cameras)
        for name, pairs in (("train", self.train), ("test", self.test)):
            if len(set(pairs)) != len(pairs):
                raise ValueError(f"{name} has duplicate pairs")
            for f, c in pairs:
                if not first <= f <= last:
                    raise ValueError(f"{name} pair {[f, c]} is outside frame_range")
                if c not in cams:
                    raise ValueError(f"{name} pair {[f, c]} uses a camera not listed in cameras")
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(f"pairs in both train and test: {sorted(overlap)[:5]}")
        return self

    @property
    def strategy_name(self) -> str:
        return parse_strategy(self.strategy)[0]


# ---------------------------------------------------------------- validation report (CONTRACTS 3.2)

class Check(D2SModel):
    check_id: str
    severity: Literal["error", "warning", "info"]
    message: str
    camera_id: Optional[str] = None
    frame_index: Optional[int] = Field(default=None, ge=0)
    details: Dict[str, Any] = {}


class ReportSummary(D2SModel):
    errors: int = Field(ge=0)
    warnings: int = Field(ge=0)


class ValidationReport(Versioned):
    KIND: ClassVar[str] = "report"
    scene_id: str
    created_utc: datetime
    summary: ReportSummary
    checks: List[Check] = []

    @field_validator("scene_id")
    @classmethod
    def _scene_id(cls, v):
        return ids.check_scene_id(v)

    @model_validator(mode="after")
    def _counts(self):
        errors = sum(c.severity == "error" for c in self.checks)
        warnings = sum(c.severity == "warning" for c in self.checks)
        if (errors, warnings) != (self.summary.errors, self.summary.warnings):
            raise ValueError(f"summary says {self.summary.errors} errors, {self.summary.warnings} warnings; "
                             f"checks contain {errors} and {warnings}")
        return self


# ---------------------------------------------------------------- approach (CONTRACTS 4.1)

class Requires(D2SModel):
    camera_models: List[CameraModel] = Field(min_length=1)
    cuboids: Literal["required", "optional", "none"]
    split_strategies: List[SplitStrategyName] = Field(min_length=1)


class OptionSpec(D2SModel):
    choices: List[OptionValue] = Field(min_length=1)
    default: OptionValue

    @model_validator(mode="after")
    def _default_in_choices(self):
        if len(set(map(repr, self.choices))) != len(self.choices):
            raise ValueError("choices must be unique")
        if self.default != "auto" and not _contains(self.choices, self.default):
            raise ValueError(f"default {self.default!r} is not one of the choices and not 'auto'")
        return self


class Constraint(D2SModel):
    when: Dict[str, OptionValue] = Field(min_length=1)
    forbid: Dict[str, OptionValue] = Field(min_length=1)
    reason: str


class StageEntry(D2SModel):
    stage: str
    params: Dict[str, Any] = {}


def _contains(choices: List[Any], value: Any) -> bool:
    """Membership that does not treat True as 1 or False as 0."""
    return any(type(c) is type(value) and c == value for c in choices)


def option_ref(value: Any) -> Optional[str]:
    """Option name if value is a '${options.<name>}' placeholder, else None."""
    if isinstance(value, str):
        m = OPTION_REF_RE.match(value)
        if m:
            return m.group(1)
    return None


class Approach(Versioned):
    KIND: ClassVar[str] = "approach"
    approach_id: str
    name: str
    version: str
    codebase: str
    track: Track
    level: Level
    requires: Requires
    options: Dict[str, OptionSpec] = {}
    constraints: List[Constraint] = []
    presets: Dict[str, Dict[str, OptionValue]] = {}
    stages: List[StageEntry] = Field(min_length=1)

    @field_validator("approach_id")
    @classmethod
    def _approach_id(cls, v):
        return ids.check_snake(v, "approach_id")

    @field_validator("version")
    @classmethod
    def _version(cls, v):
        return ids.check_version(v)

    @field_validator("codebase")
    @classmethod
    def _codebase(cls, v):
        if v != "drivestudio" and not re.match(r"^external:[a-z][a-z0-9_]*$", v):
            raise ValueError("codebase must be 'drivestudio' or 'external:<name>'")
        return v

    @model_validator(mode="after")
    def _consistent(self):
        for name in self.options:
            ids.check_snake(name, "option name")
        for name in self.presets:
            ids.check_snake(name, "preset name")

        def check_assignment(where: str, values: Dict[str, Any]):
            for opt, val in values.items():
                if opt not in self.options:
                    raise ValueError(f"{where}: unknown option {opt!r}")
                if not _contains(self.options[opt].choices, val):
                    raise ValueError(f"{where}: {val!r} is not a choice of option {opt!r}")

        for i, c in enumerate(self.constraints):
            check_assignment(f"constraints[{i}].when", c.when)
            check_assignment(f"constraints[{i}].forbid", c.forbid)
        for name, values in self.presets.items():
            check_assignment(f"presets.{name}", values)

        for i, entry in enumerate(self.stages):
            ref = option_ref(entry.stage)
            if ref is not None:
                if ref not in self.options:
                    raise ValueError(f"stages[{i}]: unknown option {ref!r}")
                for choice in self.options[ref].choices:
                    if not isinstance(choice, str):
                        raise ValueError(f"stages[{i}]: option {ref!r} is used as a stage but has non-stage choice {choice!r}")
                    ids.check_stage_id(choice)
            else:
                ids.check_stage_id(entry.stage)
            for key, val in entry.params.items():
                r = option_ref(val)
                if r is not None and r not in self.options:
                    raise ValueError(f"stages[{i}].params.{key}: unknown option {r!r}")
        return self


# ---------------------------------------------------------------- artifact (CONTRACTS 5.3)

class Producer(D2SModel):
    stage_id: str
    version: str
    code_commit: Optional[str] = None

    @field_validator("stage_id")
    @classmethod
    def _stage_id(cls, v):
        return ids.check_stage_id(v)

    @field_validator("version")
    @classmethod
    def _version(cls, v):
        return ids.check_version(v)


class ArtifactRecord(Versioned):
    KIND: ClassVar[str] = "artifact"
    artifact_type: ArtifactType
    cache_key: Optional[str] = None      # None for results written inside a run folder
    producer: Producer
    params: Dict[str, Any] = {}
    inputs: List[str] = []
    created_utc: datetime
    files: List[str] = []
    meta: Dict[str, Any] = {}

    @field_validator("cache_key")
    @classmethod
    def _cache_key(cls, v):
        return v if v is None else ids.check_cache_key(v)


# ---------------------------------------------------------------- run (CONTRACTS 6.2-6.4)

class RunApproach(D2SModel):
    approach_id: str
    version: str
    preset: Optional[str] = None
    options: Dict[str, OptionValue] = {}


class StageRecord(D2SModel):
    stage_id: str
    status: RunStatus
    cache_hit: bool = False
    cache_key: Optional[str] = None
    duration_s: float = Field(default=0, ge=0)
    peak_vram_mb: int = Field(default=0, ge=0)

    @field_validator("stage_id")
    @classmethod
    def _stage_id(cls, v):
        return ids.check_stage_id(v)

    @field_validator("cache_key")
    @classmethod
    def _cache_key(cls, v):
        return v if v is None else ids.check_cache_key(v)


class RunRecord(Versioned):
    KIND: ClassVar[str] = "run"
    run_id: str
    scene_id: str
    split_id: str
    approach: RunApproach
    level: Level
    executor: Literal["local", "kaggle"]
    status: RunStatus
    lidar_usage: List[str] = []
    stages: List[StageRecord] = []

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, v):
        return ids.check_run_id(v)

    @field_validator("scene_id")
    @classmethod
    def _scene_id(cls, v):
        return ids.check_scene_id(v)

    @model_validator(mode="after")
    def _consistent(self):
        if not self.run_id.split("-")[2] == self.approach.approach_id:
            raise ValueError("run_id must contain approach.approach_id")
        stage_ids = {s.stage_id for s in self.stages}
        for sid in self.lidar_usage:
            ids.check_stage_id(sid)
            if sid not in stage_ids:
                raise ValueError(f"lidar_usage lists {sid!r}, which is not among the stages")
        return self


class Lpips(D2SModel):
    net: str
    package: str
    package_version: str


class EvalProtocol(D2SModel):
    version: str
    image_dtype: Literal["uint8"]
    resolution: str
    lpips: Lpips


class Metrics(D2SModel):
    psnr: float
    ssim: float
    lpips: float
    human_psnr: Optional[float] = None
    vehicle_psnr: Optional[float] = None
    occupied_psnr: Optional[float] = None


class Summary(Versioned):
    KIND: ClassVar[str] = "summary"
    protocol: EvalProtocol
    split_id: str
    overall: Metrics
    per_camera: Dict[str, Metrics] = {}
    train_time_s: float = Field(ge=0)
    peak_vram_mb: int = Field(ge=0)
    num_gaussians: int = Field(ge=0)


class EnvRecord(Versioned):
    KIND: ClassVar[str] = "env"
    code_commit: str
    python: str
    pytorch: Optional[str] = None
    cuda: Optional[str] = None
    gsplat: Optional[str] = None
    gpu: Optional[str] = None
    seed: Optional[int] = None
    executor: Literal["local", "kaggle"]


# ---------------------------------------------------------------- preview (CONTRACTS 7)

class Preview(Versioned):
    KIND: ClassVar[str] = "preview"
    source_run: str
    sh_degree: int = Field(ge=0, le=3)
    rigid_actors: List[str] = []
    box_only_actors: List[str] = []
    omitted: List[str] = []


MODELS_BY_KIND = {
    "manifest": Manifest, "split": Split, "report": ValidationReport, "approach": Approach,
    "artifact": ArtifactRecord, "run": RunRecord, "summary": Summary, "env": EnvRecord, "preview": Preview,
}
