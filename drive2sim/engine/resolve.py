"""Biến một approach cùng lựa chọn của người dùng thành danh sách stage cố định để chạy.

File approach có các *mục cấu hình* (ví dụ mục cấu hình ``init`` chọn giữa ``init.lidar`` và
``init.depth``), *preset* và *ràng buộc* (CONTRACTS mục 4.1). Trước khi chạy, engine phải quyết
định giá trị cụ thể của mọi mục cấu hình, kiểm tra dữ liệu của cảnh có cho phép chạy không, rồi mới
có danh sách stage cố định. File này làm đúng việc đó (bước 1–2 của engine, ARCHITECTURE mục 4.5).

File này chứa:
- ``resolve``: hàm chính. Nhận approach, manifest, split và lựa chọn của người dùng; trả về
  ``Plan`` (danh sách stage sẽ chạy kèm tham số), hoặc ném lỗi kèm lý do.
- ``blocked_choices``: liệt kê lựa chọn nào của từng mục cấu hình bị khoá vì dữ liệu, để Studio
  làm mờ.
- ``wire_inputs``: nối đầu vào của từng stage với stage phía trước tạo ra nó.
- ``validate_params``: kiểm tra tham số của một stage theo khai báo ``Params`` của nó.
- ``Plan``, ``PlannedStage``: kết quả của ``resolve``.
- ``ResolveError``, ``BlockedError``, ``BlockReason``: hai loại lỗi và lý do bị khoá.

Studio, CLI (``d2s approach check``, ``d2s run start``) và ``run.py`` cùng gọi file này, nên
chạy từ đâu cũng ra cùng danh sách stage và cùng lý do bị khoá (CONTRACTS mục 4.2).

Thông báo trong ``ResolveError`` và ``BlockedError`` hiện thẳng cho người dùng nên viết tiếng
Việt (CONTRACTS mục 10.6).
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
    """Lỗi do yêu cầu hoặc file approach sai, không phải do dữ liệu của cảnh.

    Ví dụ: preset không tồn tại, mục cấu hình không tồn tại, giá trị không phải lựa chọn của mục
    cấu hình, tham số stage sai tên hoặc sai kiểu, stage chưa đăng ký, hoặc một stage cần đầu vào
    mà không stage nào phía trước tạo ra. Người dùng sửa lệnh hoặc sửa file approach là hết lỗi.

    Ứng với mã lỗi ``invalid_request`` và mã thoát CLI ``2`` (CONTRACTS mục 10.4, 11.3).
    """


@dataclass(frozen=True)
class BlockReason:
    """Một lý do khiến approach hoặc một lựa chọn không chạy được trên cảnh này.

    Studio hiện ``message`` cạnh lựa chọn bị làm mờ; CLI in ``message`` khi lệnh bị khoá.

    Attributes:
        message: Lý do bằng tiếng Việt, ví dụ ``"init.lidar cần LiDAR nhưng cảnh ... không có
            LiDAR"``.
        option: Tên mục cấu hình liên quan nếu lý do gắn với một mục cấu hình, ví dụ ``"init"``;
            ``None`` nếu lý do thuộc cả approach (ví dụ sai track).
        choice: Lựa chọn bị khoá nếu có, ví dụ ``"init.lidar"``.
    """

    message: str
    option: Optional[str] = None
    choice: Optional[OptionValue] = None


class BlockedError(Exception):
    """Lỗi do dữ liệu của cảnh hoặc tổ hợp lựa chọn không cho phép chạy approach.

    Ví dụ: chọn ``init.lidar`` mà cảnh không có LiDAR, approach cần box 3D mà cảnh không có, hay
    tổ hợp ``renderer=3dgeer`` với ``campose=true`` bị ràng buộc cấm. Lỗi chứa **mọi** lý do tìm
    được, không chỉ lý do đầu tiên, để người dùng sửa một lần.

    Ứng với mã lỗi ``blocked`` và mã thoát CLI ``3`` (CONTRACTS mục 10.4, 11.3).

    Attributes:
        reasons: Danh sách lý do bị khoá.
    """

    def __init__(self, reasons: list[BlockReason]) -> None:
        super().__init__("; ".join(reason.message for reason in reasons))
        self.reasons = reasons


@dataclass
class PlannedStage:
    """Một stage trong danh sách đã quyết định, kèm mọi thứ cần để chạy nó.

    Attributes:
        stage_id: Stage sẽ chạy, ví dụ ``"init.lidar"``.
        params: Tham số đã gộp, đã kiểm tra và đã điền giá trị mặc định, dạng JSON.
        input_sources: Tên đầu vào → nguồn của nó: một số nguyên là vị trí (chỉ số trong danh
            sách) của stage phía trước tạo ra artifact đó, hoặc ``"scene"`` / ``"split"``.
            Ví dụ ``{"points": 0, "split": "split"}``.
        cacheable: ``True`` nếu kết quả vào ``cache/``, ``False`` nếu vào thư mục run.
        uses_lidar: Stage có dùng LiDAR không.
    """

    stage_id: str
    params: dict[str, object]
    input_sources: dict[str, object]
    cacheable: bool
    uses_lidar: bool


QUICK_NUM_ITERS = 3000
"""Số iteration của run chạy thử nhanh (``--quick``): đủ để thấy approach chạy hết các stage và
loss giảm, ngắn hơn khoảng 10 lần so với 30000 iteration thường dùng."""


@dataclass
class Plan:
    """Kết quả của ``resolve``: đủ thông tin để ghi ``run.json`` và chạy từng stage.

    ``run.py`` nhận đối tượng này để tạo thư mục run (``create_run``); ``d2s approach check`` in
    nó ra để người dùng xem trước những gì sẽ chạy.

    Attributes:
        approach_id: Approach được chạy, ví dụ ``"omnire"``.
        approach_version: Phiên bản của file approach.
        level: Mức năng lực của approach, ``L1`` tới ``L4``.
        preset: Preset của cấu hình, hoặc ``None``; cũng là ``None`` khi người dùng chọn preset rồi
            chỉnh một mục cấu hình khác giá trị của preset (cấu hình không còn là preset đó).
        options: Giá trị cụ thể đã quyết định cho mọi mục cấu hình, kể cả mục để ``auto``, ví dụ
            ``{"init": "init.depth", "renderer": "3dgs"}``.
        stages: Các stage theo thứ tự chạy.
        quick: ``True`` nếu là run chạy thử nhanh (xem ``QUICK_NUM_ITERS``).
    """

    approach_id: str
    approach_version: str
    level: str
    preset: Optional[str]
    options: dict[str, OptionValue]
    stages: list[PlannedStage] = field(default_factory=list)
    quick: bool = False

    @property
    def lidar_usage(self) -> list[str]:
        """Liệt kê các stage dùng LiDAR, để ghi vào ``run.json`` (CONTRACTS mục 4.3).

        Returns:
            Danh sách ``stage_id`` theo thứ tự chạy; rỗng nếu run không dùng LiDAR.
        """
        return [stage.stage_id for stage in self.stages if stage.uses_lidar]


def _choice_block(
    choice: OptionValue, manifest: Manifest, registry: StageRegistry
) -> Optional[str]:
    """Kiểm tra một lựa chọn có dùng được với dữ liệu của cảnh không.

    Hiện chỉ có một điều kiện: lựa chọn là tên một stage có ``uses_lidar = True`` mà cảnh không
    có LiDAR. Lựa chọn không phải tên stage (ví dụ ``"3dgs"``, ``true``) luôn dùng được.

    Args:
        choice: Một lựa chọn của mục cấu hình.
        manifest: Manifest của cảnh.
        registry: Bảng stage, để biết stage có cần LiDAR không.

    Returns:
        Lý do bằng tiếng Việt nếu lựa chọn bị khoá, ``None`` nếu dùng được.
    """
    if isinstance(choice, str) and choice in registry:
        if registry.get(choice).uses_lidar and not manifest.available.lidar:
            return f"{choice} cần LiDAR nhưng cảnh {manifest.scene_id} không có LiDAR"
    return None


def blocked_choices(
    approach: Approach, manifest: Manifest, registry: StageRegistry = REGISTRY
) -> dict[str, list[BlockReason]]:
    """Liệt kê các lựa chọn bị khoá vì dữ liệu của cảnh, theo từng mục cấu hình.

    Studio gọi hàm này khi người dùng mở màn hình chọn approach, để làm mờ lựa chọn không chạy
    được và hiện lý do ngay bên cạnh, trước khi người dùng bấm chạy (SPEC mục 5 bước 4).

    Args:
        approach: File approach đã kiểm tra theo schema.
        manifest: Manifest của cảnh đang chọn.
        registry: Bảng stage; mặc định là bảng chung.

    Returns:
        Tên mục → danh sách lựa chọn bị khoá kèm lý do. Mục không có lựa chọn nào bị khoá thì không
        xuất hiện. Ví dụ với cảnh không có LiDAR:
        ``{"init": [BlockReason("init.lidar cần LiDAR ...", "init", "init.lidar")]}``.
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
    """Kiểm tra mỗi cặp mục–giá trị người dùng đưa vào đều trỏ tới mục và lựa chọn có thật.

    Args:
        approach: File approach.
        values: Tên mục → giá trị người dùng chọn.
        where: Mô tả nguồn của ``values``, đưa vào thông báo lỗi để người dùng biết sửa ở đâu.

    Raises:
        ResolveError: Nếu mục cấu hình không có trong approach, hoặc giá trị không phải lựa chọn
            của mục cấu hình; thông báo liệt kê các mục cấu hình hoặc lựa chọn hợp lệ.
    """
    for option_name, value in values.items():
        spec = approach.options.get(option_name)
        if spec is None:
            known = ", ".join(approach.options) or "(không có mục nào)"
            raise ResolveError(
                f"{where}: approach {approach.approach_id} không có mục cấu hình {option_name!r}; "
                f"các mục cấu hình hiện có: {known}"
            )
        if not contains_value(spec.choices, value):
            choices = ", ".join(repr(choice) for choice in spec.choices)
            raise ResolveError(
                f"{where}: {value!r} không phải lựa chọn của mục cấu hình {option_name!r}; "
                f"chọn một trong: {choices}"
            )


def _decide_options(
    approach: Approach,
    manifest: Manifest,
    preset: Optional[str],
    option_values: dict[str, OptionValue],
    registry: StageRegistry,
) -> tuple[dict[str, OptionValue], list[BlockReason]]:
    """Quyết định giá trị cụ thể cho mọi mục cấu hình.

    Giá trị được gộp theo thứ tự, cái sau đè cái trước: ``default`` của mục cấu hình → preset → lựa
    chọn của người dùng. Mục nào còn giá trị ``auto`` thì lấy lựa chọn **đầu tiên** trong
    ``choices`` không bị khoá vì dữ liệu. Vì vậy thứ tự ``choices`` trong file approach chính là
    thứ tự ưu tiên: với ``[init.lidar, init.depth]``, cảnh có LiDAR được ``init.lidar``, không có
    thì được ``init.depth``.

    Args:
        approach: File approach.
        manifest: Manifest của cảnh, để biết lựa chọn nào bị khoá.
        preset: Tên preset người dùng chọn, hoặc ``None``.
        option_values: Tên mục → giá trị người dùng chọn.
        registry: Bảng stage.

    Returns:
        Bộ hai phần: giá trị đã quyết định của mọi mục cấu hình, và danh sách lý do bị khoá của
        các giá trị đó (rỗng nếu mọi giá trị dùng được).

    Raises:
        ResolveError: Nếu preset, mục cấu hình hoặc giá trị không tồn tại.
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
                        f"mục cấu hình {option_name!r}: "
                        "mọi lựa chọn đều bị khoá với dữ liệu của cảnh",
                        option_name,
                    )
                )
            continue
        message = _choice_block(value, manifest, registry)
        if message is not None:
            reasons.append(
                BlockReason(f"mục cấu hình {option_name!r}: {message}", option_name, value)
            )
    return chosen, reasons


def _constraint_reasons(approach: Approach, chosen: dict[str, OptionValue]) -> list[BlockReason]:
    """Tìm các ràng buộc trong file approach bị tổ hợp lựa chọn hiện tại vi phạm.

    Một ràng buộc bị vi phạm khi **mọi** cặp trong ``when`` và **mọi** cặp trong ``forbid`` đều
    khớp với lựa chọn. So sánh cả kiểu dữ liệu, nên giá trị ``1`` không khớp ``true``. Ví dụ ràng
    buộc ``when: {renderer: 3dgeer}, forbid: {campose: true}`` chỉ bị vi phạm khi đồng thời
    ``renderer`` là ``3dgeer`` và ``campose`` là ``true``.

    Args:
        approach: File approach.
        chosen: Giá trị đã quyết định của mọi mục cấu hình.

    Returns:
        Lý do cho từng ràng buộc bị vi phạm, kèm ``reason`` ghi trong file approach.
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
    """Kiểm tra các điều kiện dữ liệu ở cấp approach (CONTRACTS mục 4.2).

    Gồm bốn điều kiện: track của approach khớp track của cảnh; approach cần box 3D thì cảnh phải
    có box; chiến lược của split nằm trong danh sách approach hỗ trợ; mô hình của mọi camera
    trong split nằm trong danh sách approach nhận, trừ khi danh sách stage có ``prep.undistort``
    để khử méo về pinhole trước.

    Args:
        approach: File approach.
        manifest: Manifest của cảnh.
        split: Split của run.
        stage_ids: Danh sách stage sau khi thay mục cấu hình, để biết có ``prep.undistort`` không.

    Returns:
        Lý do cho từng điều kiện không thoả; rỗng nếu thoả hết.
    """
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
    """Thay chỗ giữ chỗ ``${options.<mục>}`` bằng giá trị đã quyết định của mục cấu hình đó.

    Args:
        value: Một giá trị trong file approach, ví dụ ``"${options.renderer}"`` hoặc ``2``.
        chosen: Giá trị đã quyết định của mọi mục cấu hình.

    Returns:
        Giá trị của mục cấu hình nếu ``value`` là chỗ giữ chỗ, ví dụ ``"3dgs"``; nếu không thì trả
        lại nguyên ``value``.
    """
    option_name = option_reference(value)
    return chosen[option_name] if option_name is not None else value


def wire_inputs(
    stage_ids: list[str], registry: StageRegistry = REGISTRY
) -> list[dict[str, object]]:
    """Nối đầu vào của từng stage với stage phía trước tạo ra nó.

    Mỗi đầu vào kiểu artifact nhận artifact cùng loại do stage **gần nhất phía trước** tạo ra
    (CONTRACTS mục 4.1). Ví dụ ``train.drivestudio`` cần ``points_init`` thì nhận kết quả của
    stage ``init.*`` đứng trước nó. Đầu vào ``"scene"`` và ``"split"`` giữ nguyên tên. Đầu vào
    tuỳ chọn (``Stage.optional_inputs``) không có stage nào tạo ra thì không có trong kết quả.

    Gọi khi tạo run (trong ``resolve``) và khi chạy tiếp một run (trong ``run.py``), để hai lần
    nối luôn cho cùng kết quả.

    Args:
        stage_ids: Các stage theo thứ tự chạy; mọi stage phải đã đăng ký.
        registry: Bảng stage, để biết mỗi stage cần và tạo ra loại artifact nào.

    Returns:
        Một dict cho mỗi stage, cùng thứ tự với ``stage_ids``: tên đầu vào → vị trí của stage
        nguồn, hoặc ``"scene"`` / ``"split"``. Ví dụ ``[{"scene": "scene"}, {"points": 0}]``.

    Raises:
        ResolveError: Nếu một đầu vào bắt buộc không có stage nào phía trước tạo ra.
    """
    wiring: list[dict[str, object]] = []
    producers: dict[str, int] = {}  # loại artifact → vị trí stage gần nhất tạo ra nó
    for index, stage_id in enumerate(stage_ids):
        cls = registry.get(stage_id)
        sources: dict[str, object] = {}
        for input_name, kind in cls.inputs.items():
            if kind in (SCENE_INPUT, SPLIT_INPUT):
                sources[input_name] = kind
            elif kind in producers:
                sources[input_name] = producers[kind]
            elif input_name in cls.optional_inputs:
                continue  # đầu vào tuỳ chọn không có stage nào tạo ra: bỏ qua
            else:
                raise ResolveError(
                    f"stage {stage_id} cần artifact {kind} (đầu vào {input_name!r}) nhưng không "
                    "stage nào phía trước tạo ra nó; kiểm tra thứ tự stage trong approach"
                )
        for kind in cls.outputs.values():
            producers[kind] = index
        wiring.append(sources)
    return wiring


def _build_stages(
    approach: Approach,
    chosen: dict[str, OptionValue],
    stage_params: dict[str, dict[str, object]],
    registry: StageRegistry,
) -> list[PlannedStage]:
    """Dựng danh sách stage cố định từ file approach và giá trị đã quyết định của các mục cấu hình.

    Với từng bước trong ``stages`` của file approach: thay chỗ giữ chỗ trong tên stage và tham
    số, gộp tham số theo thứ tự mặc định của stage → ``params`` trong file → ``stage_params`` của
    người dùng (CONTRACTS mục 4.1), rồi kiểm tra theo ``Params`` của stage. Cuối cùng nối đầu vào
    bằng ``wire_inputs``.

    Args:
        approach: File approach.
        chosen: Giá trị đã quyết định của mọi mục cấu hình.
        stage_params: ``stage_id`` → tham số người dùng đặt riêng cho stage đó.
        registry: Bảng stage.

    Returns:
        Các stage theo thứ tự chạy.

    Raises:
        ResolveError: Nếu stage chưa đăng ký, tham số sai, ``stage_params`` đặt cho stage không
            có trong danh sách, hoặc một đầu vào không có stage nào tạo ra.
    """
    stage_ids: list[str] = []
    all_params: list[dict[str, object]] = []
    for index, entry in enumerate(approach.stages):
        stage_id = str(_substitute(entry.stage, chosen))
        if stage_id not in registry:
            raise ResolveError(
                f"approach {approach.approach_id}, stages[{index}]: stage {stage_id} chưa có; "
                f"các stage hiện có: {', '.join(registry.stage_ids()) or '(chưa có stage nào)'}"
            )
        raw_params = {name: _substitute(value, chosen) for name, value in entry.params.items()}
        raw_params.update(stage_params.get(stage_id, {}))
        stage_ids.append(stage_id)
        all_params.append(validate_params(stage_id, raw_params, registry))

    unknown = set(stage_params) - set(stage_ids)
    if unknown:
        raise ResolveError(
            f"đặt tham số cho stage không có trong danh sách: {', '.join(sorted(unknown))}"
        )
    wiring = wire_inputs(stage_ids, registry)
    return [
        PlannedStage(
            stage_id,
            params,
            sources,
            registry.get(stage_id).cacheable,
            registry.get(stage_id).uses_lidar,
        )
        for stage_id, params, sources in zip(stage_ids, all_params, wiring)
    ]


def validate_params(
    stage_id: str, raw_params: dict[str, object], registry: StageRegistry = REGISTRY
) -> dict[str, object]:
    """Kiểm tra tham số của một stage theo lớp ``Params`` và điền giá trị mặc định.

    Gọi khi dựng danh sách stage, và gọi lại trước khi chạy từng stage trong ``run.py``, vì
    code của stage có thể đã đổi từ lúc tạo run.

    Args:
        stage_id: Stage đã đăng ký.
        raw_params: Tham số đã gộp, chưa kiểm tra, ví dụ ``{"num_iters": 1000}``.
        registry: Bảng stage.

    Returns:
        Tham số đầy đủ dạng JSON, gồm cả các tham số lấy giá trị mặc định, ví dụ
        ``{"num_iters": 1000, "renderer": "3dgs"}``.

    Raises:
        ResolveError: Nếu tham số sai tên hoặc sai kiểu; thông báo kèm chi tiết của Pydantic.
    """
    try:
        model = registry.get(stage_id).Params.model_validate(raw_params)
    except ValidationError as error:
        raise ResolveError(f"tham số của stage {stage_id} không hợp lệ:\n{error}") from None
    return model.model_dump(mode="json")


def resolve(
    approach: Approach,
    manifest: Manifest,
    split: Split,
    preset: Optional[str] = None,
    option_values: Optional[dict[str, OptionValue]] = None,
    stage_params: Optional[dict[str, dict[str, object]]] = None,
    registry: StageRegistry = REGISTRY,
    quick: bool = False,
) -> Plan:
    """Quyết định danh sách stage sẽ chạy, hoặc báo rõ vì sao không chạy được.

    Các bước: kiểm tra split thuộc đúng cảnh; quyết định giá trị mọi mục cấu hình
    (``_decide_options``); gom mọi lý do bị khoá từ lựa chọn, ràng buộc và dữ liệu rồi báo một lần;
    cuối cùng dựng danh sách stage cố định (``_build_stages``).

    Args:
        approach: File approach đã kiểm tra theo schema.
        manifest: Manifest của cảnh.
        split: Split sẽ dùng; phải thuộc cùng cảnh với ``manifest``.
        preset: Tên preset, ví dụ ``"no_lidar"``, hoặc ``None``.
        option_values: Lựa chọn của người dùng cho từng mục cấu hình, lấy từ Studio hoặc từ
            ``--set <mục>=<giá trị>``, ví dụ ``{"renderer": "3dgeer"}``.
        stage_params: ``stage_id`` → tham số người dùng đặt cho stage đó, lấy từ
            ``--set <stage_id>.<tham số>=<giá trị>``, ví dụ
            ``{"train.drivestudio": {"num_iters": 10000}}``.
        registry: Bảng stage; mặc định là bảng chung.
        quick: Chạy thử nhanh: mọi stage có tham số ``num_iters`` train ``QUICK_NUM_ITERS``
            iteration, trừ khi người dùng tự đặt ``num_iters`` cho stage đó.

    Returns:
        ``Plan`` gồm giá trị của mọi mục cấu hình và danh sách stage cố định.

    Raises:
        ResolveError: Nếu yêu cầu hoặc file approach sai (mã thoát ``2``).
        BlockedError: Nếu dữ liệu hoặc tổ hợp lựa chọn không cho phép chạy (mã thoát ``3``);
            chứa mọi lý do tìm được.
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
    if preset is not None and option_values:
        # Người dùng chỉnh khác preset thì cấu hình không còn là preset đó: không ghi tên preset,
        # để run.json và bảng so sánh không gọi nhầm một cấu hình đã sửa bằng tên preset.
        as_preset, _ = _decide_options(approach, manifest, preset, {}, registry)
        if as_preset != chosen:
            preset = None
    # Gom mọi lý do bị khoá trước khi dựng danh sách stage, để người dùng sửa một lần.
    stage_ids = [str(_substitute(entry.stage, chosen)) for entry in approach.stages]
    reasons += _constraint_reasons(approach, chosen)
    reasons += _data_reasons(approach, manifest, split, stage_ids)
    if reasons:
        raise BlockedError(reasons)
    params = {stage_id: dict(values) for stage_id, values in (stage_params or {}).items()}
    if quick:
        _apply_quick(approach.approach_id, stage_ids, params, registry)
    stages = _build_stages(approach, chosen, params, registry)
    return Plan(
        approach.approach_id, approach.version, approach.level, preset, chosen, stages, quick
    )


def _apply_quick(
    approach_id: str,
    stage_ids: list[str],
    stage_params: dict[str, dict[str, object]],
    registry: StageRegistry,
) -> None:
    """Đặt ``num_iters = QUICK_NUM_ITERS`` cho mọi stage có tham số này (sửa ``stage_params``).

    Giá trị người dùng tự đặt bằng ``--set <stage_id>.num_iters=...`` được giữ nguyên.

    Raises:
        ResolveError: Nếu không stage nào của approach có tham số ``num_iters``.
    """
    targets = [
        stage_id
        for stage_id in stage_ids
        if stage_id in registry and "num_iters" in registry.get(stage_id).Params.model_fields
    ]
    if not targets:
        raise ResolveError(
            f"approach {approach_id} không có stage nào nhận num_iters, nên không chạy thử nhanh "
            "được"
        )
    for stage_id in targets:
        stage_params.setdefault(stage_id, {}).setdefault("num_iters", QUICK_NUM_ITERS)
