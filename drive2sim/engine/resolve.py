"""Biến một approach cùng lựa chọn của người dùng thành danh sách stage cố định.

Đây là bước 1–2 của engine (ARCHITECTURE mục 4.5): gộp preset và lựa chọn vào các ô, chọn giá
trị ``auto`` theo manifest, kiểm tra điều kiện dữ liệu và ràng buộc (CONTRACTS mục 4.2), thay
``${options.<ô>}`` rồi nối đầu vào của từng stage với stage tạo ra nó. Studio và CLI cùng gọi
module này, nên chạy từ đâu cũng ra cùng một danh sách stage và cùng lý do bị khoá.

Thông báo trong ``ResolveError`` và ``BlockedError`` hiện thẳng cho người dùng nên viết tiếng Việt
(CONTRACTS mục 10.6).
"""

from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from drive2sim.core.schema import (
    Approach,
    Manifest,
    OptionValue,
    Split,
    contains_value,
    option_reference,
    parse_strategy,
)
from drive2sim.engine.registry import REGISTRY, StageRegistry
from drive2sim.engine.stage import SCENE_INPUT, SPLIT_INPUT

AUTO = "auto"
UNDISTORT_STAGE = "prep.undistort"
# NCore có mô hình pinhole lý tưởng; tool coi nó là pinhole OpenCV không méo (CONTRACTS 2.2).
CAMERA_MODEL_ALIASES: dict[str, str] = {"ideal_pinhole": "opencv_pinhole"}


class ResolveError(ValueError):
    """Yêu cầu không hợp lệ: preset, ô, giá trị hoặc tham số không tồn tại; approach sai.

    Ứng với mã lỗi ``invalid_request`` và mã thoát CLI ``2`` (CONTRACTS mục 10.4, 11.3).
    """


@dataclass(frozen=True)
class BlockReason:
    """Một lý do khiến approach hoặc một lựa chọn không chạy được trên cảnh này.

    Attributes:
        message: Lý do bằng tiếng Việt, hiện cho người dùng.
        option: Tên ô liên quan, nếu lý do gắn với một ô.
        choice: Lựa chọn bị khoá, nếu có.
    """

    message: str
    option: Optional[str] = None
    choice: Optional[OptionValue] = None


class BlockedError(Exception):
    """Approach không chạy được với dữ liệu hoặc tổ hợp lựa chọn này.

    Ứng với mã lỗi ``blocked`` và mã thoát CLI ``3`` (CONTRACTS mục 10.4, 11.3).

    Attributes:
        reasons: Mọi lý do tìm được, không chỉ lý do đầu tiên, để người dùng sửa một lần.
    """

    def __init__(self, reasons: list[BlockReason]) -> None:
        super().__init__("; ".join(reason.message for reason in reasons))
        self.reasons = reasons


@dataclass
class PlannedStage:
    """Một stage trong danh sách đã quyết định.

    Attributes:
        stage_id: Stage sẽ chạy.
        params: Tham số đã gộp và đã kiểm tra theo ``Params`` của stage.
        input_sources: Tên đầu vào → vị trí (chỉ số) của stage phía trước tạo ra artifact đó,
            hoặc ``"scene"`` / ``"split"``.
        cacheable: Kết quả vào ``cache/`` hay vào thư mục run.
        uses_lidar: Stage có dùng LiDAR không.
    """

    stage_id: str
    params: dict[str, object]
    input_sources: dict[str, object]
    cacheable: bool
    uses_lidar: bool


@dataclass
class Plan:
    """Kết quả của ``resolve``: đủ để ghi ``run.json`` và chạy từng stage.

    Attributes:
        approach_id: Approach được chạy.
        approach_version: Phiên bản của file approach.
        preset: Preset đã dùng, nếu có.
        options: Lựa chọn đã quyết định cho mọi ô, kể cả ô để ``auto``.
        stages: Các stage theo thứ tự chạy.
    """

    approach_id: str
    approach_version: str
    preset: Optional[str]
    options: dict[str, OptionValue]
    stages: list[PlannedStage] = field(default_factory=list)

    @property
    def lidar_usage(self) -> list[str]:
        """Các ``stage_id`` dùng LiDAR, ghi vào ``run.json`` (CONTRACTS mục 4.3)."""
        return [stage.stage_id for stage in self.stages if stage.uses_lidar]


def _choice_block(
    choice: OptionValue, manifest: Manifest, registry: StageRegistry
) -> Optional[str]:
    """Lý do một lựa chọn không dùng được với dữ liệu của cảnh, hoặc ``None`` nếu dùng được.

    Hiện chỉ có một điều kiện: lựa chọn là một stage cần LiDAR mà cảnh không có LiDAR.
    """
    if isinstance(choice, str) and choice in registry:
        if registry.get(choice).uses_lidar and not manifest.available.lidar:
            return f"{choice} cần LiDAR nhưng cảnh {manifest.scene_id} không có LiDAR"
    return None


def blocked_choices(
    approach: Approach, manifest: Manifest, registry: StageRegistry = REGISTRY
) -> dict[str, list[BlockReason]]:
    """Các lựa chọn bị khoá vì dữ liệu, theo từng ô; Studio dùng để làm mờ lựa chọn.

    Args:
        approach: File approach đã kiểm tra.
        manifest: Manifest của cảnh.
        registry: Bảng stage.

    Returns:
        Tên ô → danh sách lựa chọn bị khoá kèm lý do; ô không có lựa chọn bị khoá thì không có.
    """
    result: dict[str, list[BlockReason]] = {}
    for option_name, spec in approach.options.items():
        for choice in spec.choices:
            message = _choice_block(choice, manifest, registry)
            if message is not None:
                result.setdefault(option_name, []).append(
                    BlockReason(message, option_name, choice)
                )
    return result


def _check_assignment(approach: Approach, values: dict[str, OptionValue], where: str) -> None:
    """Kiểm tra mỗi cặp ô–giá trị do người dùng đưa trỏ tới ô và lựa chọn có thật.

    Raises:
        ResolveError: Nếu ô hoặc giá trị không có trong approach.
    """
    for option_name, value in values.items():
        spec = approach.options.get(option_name)
        if spec is None:
            known = ", ".join(approach.options) or "(không có ô nào)"
            raise ResolveError(
                f"{where}: approach {approach.approach_id} không có ô {option_name!r}; "
                f"các ô hiện có: {known}"
            )
        if not contains_value(spec.choices, value):
            choices = ", ".join(repr(choice) for choice in spec.choices)
            raise ResolveError(
                f"{where}: {value!r} không phải lựa chọn của ô {option_name!r}; "
                f"chọn một trong: {choices}"
            )


def _decide_options(
    approach: Approach,
    manifest: Manifest,
    preset: Optional[str],
    option_values: dict[str, OptionValue],
    registry: StageRegistry,
) -> tuple[dict[str, OptionValue], list[BlockReason]]:
    """Quyết định giá trị của mọi ô: mặc định → preset → lựa chọn của người dùng → ``auto``.

    ``auto`` lấy lựa chọn **đầu tiên** trong ``choices`` không bị khoá vì dữ liệu, nên thứ tự
    ``choices`` trong file approach là thứ tự ưu tiên (ví dụ ``[init.lidar, init.depth]``).

    Returns:
        Bộ ``(giá trị của mọi ô, lý do bị khoá của các lựa chọn đã quyết định)``.

    Raises:
        ResolveError: Nếu preset, ô hoặc giá trị không tồn tại.
    """
    chosen: dict[str, OptionValue] = {
        name: spec.default for name, spec in approach.options.items()
    }
    if preset is not None:
        if preset not in approach.presets:
            known = ", ".join(approach.presets) or "(không có preset nào)"
            raise ResolveError(
                f"approach {approach.approach_id} không có preset {preset!r}; "
                f"các preset hiện có: {known}"
            )
        chosen.update(approach.presets[preset])
    _check_assignment(approach, option_values, "lựa chọn của người dùng")
    chosen.update(option_values)

    reasons: list[BlockReason] = []
    for option_name, value in chosen.items():
        choices = approach.options[option_name].choices
        if value == AUTO and isinstance(value, str) and not contains_value(choices, AUTO):
            usable = [c for c in choices if _choice_block(c, manifest, registry) is None]
            if usable:
                chosen[option_name] = usable[0]
            else:
                reasons.append(
                    BlockReason(
                        f"ô {option_name!r}: mọi lựa chọn đều bị khoá với dữ liệu của cảnh",
                        option_name,
                    )
                )
            continue
        message = _choice_block(value, manifest, registry)
        if message is not None:
            reasons.append(BlockReason(f"ô {option_name!r}: {message}", option_name, value))
    return chosen, reasons


def _constraint_reasons(approach: Approach, chosen: dict[str, OptionValue]) -> list[BlockReason]:
    """Các ràng buộc bị vi phạm.

    Một ràng buộc bị vi phạm khi **mọi** cặp trong ``when`` và **mọi** cặp trong ``forbid``
    đều khớp với lựa chọn (so sánh cả kiểu, như ``contains_value``).
    """
    reasons = []
    for constraint in approach.constraints:
        pairs = {**constraint.when, **constraint.forbid}
        if all(contains_value([chosen.get(name)], value) for name, value in pairs.items()):
            combination = ", ".join(f"{name}={value!r}" for name, value in pairs.items())
            reasons.append(BlockReason(f"tổ hợp {combination} không hợp lệ: {constraint.reason}"))
    return reasons


def _data_reasons(
    approach: Approach, manifest: Manifest, split: Split, stage_ids: list[str]
) -> list[BlockReason]:
    """Các điều kiện dữ liệu ở cấp approach không thoả (CONTRACTS mục 4.2)."""
    reasons = []
    if approach.track != manifest.track:
        reasons.append(
            BlockReason(
                f"approach dành cho track {approach.track}, cảnh là track {manifest.track}"
            )
        )
    if approach.requires.cuboids == "required" and manifest.available.cuboids == "none":
        reasons.append(BlockReason("approach cần box 3D nhưng cảnh không có box"))
    strategy_name, _ = parse_strategy(split.strategy)
    if strategy_name not in approach.requires.split_strategies:
        allowed = ", ".join(approach.requires.split_strategies)
        reasons.append(
            BlockReason(f"split dùng chiến lược {strategy_name}, approach chỉ nhận: {allowed}")
        )
    if UNDISTORT_STAGE not in stage_ids:
        for camera_id in split.cameras:
            model = manifest.camera(camera_id).model
            model = CAMERA_MODEL_ALIASES.get(model, model)
            if model not in approach.requires.camera_models:
                allowed = ", ".join(approach.requires.camera_models)
                reasons.append(
                    BlockReason(
                        f"camera {camera_id} dùng mô hình {model}, approach chỉ nhận: {allowed}, "
                        f"và danh sách stage không có {UNDISTORT_STAGE}"
                    )
                )
    return reasons


def _substitute(value: object, chosen: dict[str, OptionValue]) -> object:
    """Thay ``${options.<ô>}`` bằng giá trị đã quyết định; giá trị khác giữ nguyên."""
    option_name = option_reference(value)
    return chosen[option_name] if option_name is not None else value


def _build_stages(
    approach: Approach,
    chosen: dict[str, OptionValue],
    stage_params: dict[str, dict[str, object]],
    registry: StageRegistry,
) -> list[PlannedStage]:
    """Thay chỗ giữ chỗ, gộp tham số, kiểm tra theo ``Params`` và nối đầu vào.

    Thứ tự gộp tham số: mặc định của stage → ``params`` trong file → ``stage_params`` của người
    dùng (CONTRACTS mục 4.1). Mỗi đầu vào artifact nhận artifact cùng loại gần nhất phía trước.

    Raises:
        ResolveError: Nếu stage chưa có, tham số sai, ``stage_params`` trỏ tới stage không có
            trong danh sách, hoặc thiếu stage tạo ra một đầu vào.
    """
    planned: list[PlannedStage] = []
    producers: dict[str, int] = {}  # loại artifact → vị trí stage gần nhất tạo ra nó
    for index, entry in enumerate(approach.stages):
        stage_id = str(_substitute(entry.stage, chosen))
        if stage_id not in registry:
            raise ResolveError(
                f"approach {approach.approach_id}, stages[{index}]: stage {stage_id} chưa có; "
                f"các stage hiện có: {', '.join(registry.stage_ids()) or '(chưa có stage nào)'}"
            )
        cls = registry.get(stage_id)
        raw_params = {name: _substitute(value, chosen) for name, value in entry.params.items()}
        raw_params.update(stage_params.get(stage_id, {}))
        try:
            params = cls.Params.model_validate(raw_params).model_dump(mode="json")
        except ValidationError as error:
            raise ResolveError(f"tham số của stage {stage_id} không hợp lệ:\n{error}") from None

        sources: dict[str, object] = {}
        for input_name, kind in cls.inputs.items():
            if kind in (SCENE_INPUT, SPLIT_INPUT):
                sources[input_name] = kind
            elif kind in producers:
                sources[input_name] = producers[kind]
            else:
                raise ResolveError(
                    f"stage {stage_id} cần artifact {kind} (đầu vào {input_name!r}) nhưng không "
                    "stage nào phía trước tạo ra nó; kiểm tra thứ tự stage trong approach"
                )
        for kind in cls.outputs.values():
            producers[kind] = index
        planned.append(PlannedStage(stage_id, params, sources, cls.cacheable, cls.uses_lidar))

    unknown = set(stage_params) - {stage.stage_id for stage in planned}
    if unknown:
        raise ResolveError(
            f"đặt tham số cho stage không có trong danh sách: {', '.join(sorted(unknown))}"
        )
    return planned


def resolve(
    approach: Approach,
    manifest: Manifest,
    split: Split,
    preset: Optional[str] = None,
    option_values: Optional[dict[str, OptionValue]] = None,
    stage_params: Optional[dict[str, dict[str, object]]] = None,
    registry: StageRegistry = REGISTRY,
) -> Plan:
    """Quyết định danh sách stage sẽ chạy, hoặc báo vì sao không chạy được.

    Args:
        approach: File approach đã kiểm tra.
        manifest: Manifest của cảnh.
        split: Split sẽ dùng; phải thuộc cùng cảnh.
        preset: Tên preset, nếu có.
        option_values: Lựa chọn của người dùng cho từng ô (Studio, hoặc ``--set <ô>=...``).
        stage_params: ``stage_id`` → tham số người dùng đặt cho stage đó
            (``--set <stage_id>.<tham số>=...``).
        registry: Bảng stage.

    Returns:
        Danh sách stage cố định cùng lựa chọn đã quyết định.

    Raises:
        ResolveError: Nếu yêu cầu hoặc file approach sai (mã thoát ``2``).
        BlockedError: Nếu dữ liệu hoặc tổ hợp lựa chọn không cho phép chạy (mã thoát ``3``); chứa
            mọi lý do tìm được.
    """
    if split.scene_id != manifest.scene_id:
        raise ResolveError(
            f"split {split.split_id} thuộc cảnh {split.scene_id}, không phải {manifest.scene_id}"
        )
    missing = [
        camera_id
        for camera_id in split.cameras
        if camera_id not in {camera.camera_id for camera in manifest.cameras}
    ]
    if missing:
        raise ResolveError(
            f"split {split.split_id} dùng camera không có trong cảnh: {', '.join(missing)}"
        )

    chosen, reasons = _decide_options(approach, manifest, preset, option_values or {}, registry)
    # Gom mọi lý do bị khoá trước khi dựng danh sách stage, để người dùng sửa một lần.
    stage_ids = [str(_substitute(entry.stage, chosen)) for entry in approach.stages]
    reasons += _constraint_reasons(approach, chosen)
    reasons += _data_reasons(approach, manifest, split, stage_ids)
    if reasons:
        raise BlockedError(reasons)
    stages = _build_stages(approach, chosen, stage_params or {}, registry)
    return Plan(approach.approach_id, approach.version, preset, chosen, stages)
