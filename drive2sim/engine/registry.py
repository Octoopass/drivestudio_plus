"""Đăng ký và tra cứu stage theo ``stage_id`` (ARCHITECTURE mục 2.5, CONTRACTS mục 5.1).

Stage tự đăng ký bằng decorator ngay trong file của nó. Thêm stage mới chỉ cần thêm file vào
``drive2sim/stages/<nhóm>/``; engine không phải sửa.
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

ARTIFACT_TYPES: frozenset[str] = frozenset(get_args(ArtifactType))
# Nhóm có kết quả dùng chung giữa các approach, nên được cache (CONTRACTS mục 5.2).
CACHEABLE_GROUPS: frozenset[str] = frozenset({"prep", "init"})
INPUT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def expected_class_name(stage_id: str) -> str:
    """Tên lớp đúng quy ước cho một ``stage_id`` (CONTRACTS mục 10.2).

    Args:
        stage_id: Ví dụ ``"prep.masks_segformer"``.

    Returns:
        Ví dụ ``"PrepMasksSegformerStage"``.
    """
    group, name = stage_id.split(".", 1)
    return group.capitalize() + "".join(part.capitalize() for part in name.split("_")) + "Stage"


def _check_declaration(stage_id: str, cls: type) -> None:
    """Kiểm tra khai báo của một lớp stage trước khi đăng ký.

    Bắt lỗi ngay lúc import thay vì lúc chạy giữa chừng một run dài.

    Args:
        stage_id: ``stage_id`` sẽ gán cho lớp.
        cls: Lớp cần kiểm tra.

    Raises:
        TypeError: Nếu ``cls`` không phải lớp con của ``Stage`` hoặc ``Params`` sai kiểu.
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
    if not cls.outputs:
        raise ValueError(f"stage {stage_id}: must declare at least one output")
    for name, kind in cls.outputs.items():
        if not INPUT_NAME_RE.match(name):
            raise ValueError(f"stage {stage_id}: invalid output name {name!r}")
        if kind not in ARTIFACT_TYPES:
            raise ValueError(f"stage {stage_id}: output {name!r} has unknown type {kind!r}")
    if not (isinstance(cls.Params, type) and issubclass(cls.Params, BaseModel)):
        raise TypeError(f"stage {stage_id}: Params must be a pydantic BaseModel subclass")


class StageRegistry:
    """Bảng ``stage_id`` → lớp stage.

    Engine dùng bảng mặc định ``REGISTRY``; test tạo bảng riêng để không ảnh hưởng nhau.
    """

    def __init__(self) -> None:
        self._stages: dict[str, type[Stage]] = {}

    def register(self, stage_id: str) -> Callable[[S], S]:
        """Decorator đăng ký một lớp stage dưới ``stage_id``.

        Args:
            stage_id: ``<nhóm>.<tên>``.

        Returns:
            Decorator gán ``stage_id`` cho lớp, kiểm tra khai báo rồi đăng ký.

        Raises:
            ValueError: Nếu ``stage_id`` đã được đăng ký bởi lớp khác, hoặc khai báo sai.
            TypeError: Nếu lớp không phải ``Stage``.
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
        """Lấy lớp stage theo ``stage_id``.

        Raises:
            KeyError: Nếu chưa có stage này; thông báo liệt kê các stage đang có.
        """
        try:
            return self._stages[stage_id]
        except KeyError:
            known = ", ".join(sorted(self._stages)) or "(none)"
            raise KeyError(f"unknown stage {stage_id!r}; registered: {known}") from None

    def __contains__(self, stage_id: object) -> bool:
        return stage_id in self._stages

    def stage_ids(self) -> list[str]:
        """Mọi ``stage_id`` đã đăng ký, sắp theo thứ tự chữ cái."""
        return sorted(self._stages)


REGISTRY = StageRegistry()


def register_stage(stage_id: str) -> Callable[[S], S]:
    """Đăng ký một stage vào bảng mặc định; dùng làm decorator trong file của stage.

    Args:
        stage_id: ``<nhóm>.<tên>``.

    Returns:
        Decorator như ``StageRegistry.register``.
    """
    return REGISTRY.register(stage_id)


def load_builtin_stages(package: str = "drive2sim.stages") -> list[str]:
    """Import mọi module trong ``drive2sim.stages`` để các decorator ``register_stage`` chạy.

    Args:
        package: Package chứa stage; đổi được để test.

    Returns:
        Tên các module đã import.
    """
    root = importlib.import_module(package)
    loaded = []
    for module in pkgutil.walk_packages(root.__path__, prefix=package + "."):
        importlib.import_module(module.name)
        loaded.append(module.name)
    return loaded
