"""Đăng ký các stage theo tên và tra cứu lại khi engine cần chạy.

File này chứa:
- ``StageRegistry``: bảng tra từ ``stage_id`` (ví dụ ``init.depth``) sang lớp stage.
- ``register_stage``: decorator đặt trên lớp stage để đăng ký nó vào bảng mặc định ``REGISTRY``.
- ``load_builtin_stages``: import mọi file trong ``drive2sim/stages/`` để các decorator chạy.
- ``expected_class_name``: tính tên lớp đúng quy ước từ ``stage_id``.

Mỗi stage tự đăng ký ngay trong file của nó, nên thêm stage mới chỉ cần thêm một file vào
``drive2sim/stages/<nhóm>/``, không phải sửa engine (ARCHITECTURE mục 2.5). Khi đăng ký, khai
báo của stage được kiểm tra luôn (tên lớp, phiên bản, ``cacheable``, đầu vào, đầu ra, tham số),
để lỗi khai báo hiện ra ngay lúc import thay vì giữa chừng một run dài (CONTRACTS mục 5.1).

``resolve.py`` dùng bảng này để biết stage có tồn tại không và cần đầu vào gì; ``run.py`` dùng
nó để lấy lớp stage rồi gọi ``run``.
"""

import importlib
import pkgutil
import re
from collections.abc import Callable
from typing import TypeVar, get_args

from pydantic import BaseModel

from drive2sim.core import identifiers
from drive2sim.core.schema import ArtifactType
from drive2sim.engine.stage import SCENE_INPUT, SPLIT_INPUT, Stage

S = TypeVar("S", bound=type[Stage])

# Mọi loại artifact hợp lệ, lấy từ schema để chỉ khai báo ở một chỗ (CONTRACTS mục 5.4).
ARTIFACT_TYPES: frozenset[str] = frozenset(get_args(ArtifactType))
# Nhóm có kết quả dùng chung giữa các approach, nên được cache (CONTRACTS mục 5.2).
CACHEABLE_GROUPS: frozenset[str] = frozenset({"prep", "init"})
INPUT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def expected_class_name(stage_id: str) -> str:
    """Tính tên lớp đúng quy ước cho một ``stage_id``.

    Quy ước đặt tên lớp stage là ``<Nhóm><Tên>Stage`` viết ``PascalCase`` (CONTRACTS mục 10.2).
    Hàm này dùng để kiểm tra lúc đăng ký và để gợi ý tên đúng trong thông báo lỗi.

    Args:
        stage_id: Định danh stage dạng ``<nhóm>.<tên>``, ví dụ ``"prep.masks_segformer"``.

    Returns:
        Tên lớp, ví dụ ``"PrepMasksSegformerStage"``.
    """
    group, name = stage_id.split(".", 1)
    return group.capitalize() + "".join(part.capitalize() for part in name.split("_")) + "Stage"


def _check_declaration(stage_id: str, cls: type) -> None:
    """Kiểm tra khai báo của một lớp stage trước khi đăng ký.

    Kiểm tra lần lượt: lớp là con của ``Stage``; ``stage_id`` đúng dạng; tên lớp đúng quy ước;
    ``version`` đúng dạng ``MAJOR.MINOR.PATCH``; ``cacheable`` khớp quy tắc của nhóm; tên và loại
    của đầu vào, đầu ra hợp lệ; có đúng một đầu ra; ``Params`` là lớp Pydantic. Thông báo lỗi
    viết tiếng Anh vì dành cho người viết stage, không phải người dùng tool (CONTRACTS mục 10.6).

    Args:
        stage_id: ``stage_id`` sẽ gán cho lớp.
        cls: Lớp cần kiểm tra.

    Raises:
        TypeError: Nếu ``cls`` không phải lớp con của ``Stage``, hoặc ``Params`` không phải lớp
            Pydantic.
        ValueError: Nếu tên lớp, phiên bản, ``cacheable``, đầu vào hoặc đầu ra sai quy ước.
    """
    if not (isinstance(cls, type) and issubclass(cls, Stage)):
        raise TypeError(f"{cls!r} is not a subclass of Stage")
    identifiers.check_stage_id(stage_id)
    expected = expected_class_name(stage_id)
    if cls.__name__ != expected:
        raise ValueError(f"stage {stage_id}: class must be named {expected}, got {cls.__name__}")
    identifiers.check_version(cls.version)
    group = stage_id.split(".", 1)[0]
    if cls.cacheable != (group in CACHEABLE_GROUPS):
        raise ValueError(
            f"stage {stage_id}: cacheable must be {group in CACHEABLE_GROUPS} for group {group!r}"
        )
    for name, kind in cls.inputs.items():
        if not INPUT_NAME_RE.match(name):
            raise ValueError(f"stage {stage_id}: invalid input name {name!r}")
        if kind not in ARTIFACT_TYPES and kind not in (SCENE_INPUT, SPLIT_INPUT):
            raise ValueError(f"stage {stage_id}: input {name!r} has unknown type {kind!r}")
    # Mỗi stage đúng một đầu ra: thư mục kết quả của stage là một artifact, ``artifact.json``
    # có một ``artifact_type`` (CONTRACTS mục 5.1, 5.3).
    if len(cls.outputs) != 1:
        raise ValueError(f"stage {stage_id}: must declare exactly one output")
    for name, kind in cls.outputs.items():
        if not INPUT_NAME_RE.match(name):
            raise ValueError(f"stage {stage_id}: invalid output name {name!r}")
        if kind not in ARTIFACT_TYPES:
            raise ValueError(f"stage {stage_id}: output {name!r} has unknown type {kind!r}")
    if not (isinstance(cls.Params, type) and issubclass(cls.Params, BaseModel)):
        raise TypeError(f"stage {stage_id}: Params must be a pydantic BaseModel subclass")


class StageRegistry:
    """Bảng tra từ ``stage_id`` sang lớp stage.

    Engine dùng một bảng chung là ``REGISTRY`` ở cuối file; mọi stage trong
    ``drive2sim/stages/`` đăng ký vào bảng này. Test tạo bảng riêng bằng ``StageRegistry()`` để
    các stage giả của test không lẫn vào bảng chung và không ảnh hưởng nhau.
    """

    def __init__(self) -> None:
        self._stages: dict[str, type[Stage]] = {}

    def register(self, stage_id: str) -> Callable[[S], S]:
        """Tạo decorator đăng ký một lớp stage dưới tên ``stage_id``.

        Decorator kiểm tra khai báo của lớp, gán ``cls.stage_id = stage_id`` rồi thêm lớp vào
        bảng. Đăng ký lại đúng lớp đó lần nữa (ví dụ khi một module bị import hai lần) thì
        không sao; một lớp khác dùng trùng ``stage_id`` thì bị từ chối.

        Args:
            stage_id: Định danh dạng ``<nhóm>.<tên>``, ví dụ ``"init.depth"``.

        Returns:
            Decorator nhận lớp stage và trả lại chính lớp đó.

        Raises:
            ValueError: Nếu ``stage_id`` đã thuộc về lớp khác, hoặc khai báo của lớp sai.
            TypeError: Nếu lớp không phải con của ``Stage``.
        """

        def decorator(cls: S) -> S:
            _check_declaration(stage_id, cls)
            existing = self._stages.get(stage_id)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"stage {stage_id} already registered by "
                    f"{existing.__module__}.{existing.__name__}"
                )
            cls.stage_id = stage_id
            self._stages[stage_id] = cls
            return cls

        return decorator

    def get(self, stage_id: str) -> type[Stage]:
        """Lấy lớp stage đã đăng ký dưới ``stage_id``.

        Args:
            stage_id: Định danh stage, ví dụ ``"init.depth"``.

        Returns:
            Lớp stage.

        Raises:
            KeyError: Nếu chưa có stage này; thông báo liệt kê các stage đang có để dễ sửa.
        """
        try:
            return self._stages[stage_id]
        except KeyError:
            known = ", ".join(sorted(self._stages)) or "(none)"
            raise KeyError(f"unknown stage {stage_id!r}; registered: {known}") from None

    def __contains__(self, stage_id: object) -> bool:
        """Cho phép viết ``"init.depth" in registry`` để kiểm tra stage đã đăng ký chưa."""
        return stage_id in self._stages

    def stage_ids(self) -> list[str]:
        """Liệt kê mọi ``stage_id`` đã đăng ký.

        Returns:
            Danh sách ``stage_id`` sắp theo thứ tự chữ cái, ví dụ ``["init.depth", "train.x"]``.
        """
        return sorted(self._stages)


# Bảng chung của engine; mọi stage trong drive2sim/stages/ đăng ký vào đây.
REGISTRY = StageRegistry()


def register_stage(stage_id: str) -> Callable[[S], S]:
    """Đăng ký một stage vào bảng chung ``REGISTRY``; dùng làm decorator trong file của stage.

    Ví dụ::

        @register_stage("init.depth")
        class InitDepthStage(Stage):
            ...

    Args:
        stage_id: Định danh dạng ``<nhóm>.<tên>``.

    Returns:
        Decorator như ``StageRegistry.register``.
    """
    return REGISTRY.register(stage_id)


def load_builtin_stages(package: str = "drive2sim.stages") -> list[str]:
    """Import mọi module trong ``drive2sim.stages`` để các stage tự đăng ký vào ``REGISTRY``.

    Decorator ``register_stage`` chỉ chạy khi file của stage được import. CLI và Studio gọi hàm
    này một lần lúc khởi động, để mọi stage có sẵn trong bảng trước khi đọc file approach.

    Args:
        package: Tên package chứa stage; mặc định là ``drive2sim.stages``, đổi được khi test.

    Returns:
        Tên đầy đủ của các module đã import, ví dụ ``["drive2sim.stages.init", ...]``.
    """
    root = importlib.import_module(package)
    loaded = []
    for module in pkgutil.walk_packages(root.__path__, prefix=package + "."):
        importlib.import_module(module.name)
        loaded.append(module.name)
    return loaded
