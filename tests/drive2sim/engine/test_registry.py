"""Kiểm tra việc đăng ký và tra cứu stage trong ``drive2sim/engine/registry.py``.

File này kiểm tra:
- stage khai báo đúng thì đăng ký được, tra lại được theo ``stage_id``;
- mỗi loại khai báo sai (tên lớp, ``cacheable``, phiên bản, loại đầu vào, số đầu ra, ``Params``,
  ``stage_id`` trùng) bị từ chối ngay lúc đăng ký, kèm thông báo chỉ ra chỗ sai;
- ``load_builtin_stages`` import được mọi package trong ``drive2sim/stages/``.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from drive2sim.engine.registry import StageRegistry, expected_class_name, load_builtin_stages
from drive2sim.engine.stage import Stage, StageContext


def make_stage(name: str, **attrs: object) -> type:
    """Tạo nhanh một lớp stage với thuộc tính cho trước."""

    def run(
        self: Stage, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
    ) -> dict[str, Path]:
        return {}

    base = {"version": "0.1.0", "outputs": {"points": "points_init"}, "run": run}
    return type(name, (Stage,), {**base, **attrs})


def test_expected_class_name() -> None:
    assert expected_class_name("prep.masks_segformer") == "PrepMasksSegformerStage"
    assert expected_class_name("export.ply_world") == "ExportPlyWorldStage"


def test_register_and_get() -> None:
    registry = StageRegistry()
    cls = make_stage("InitDepthStage", cacheable=True, inputs={"scene": "scene"})
    registry.register("init.depth")(cls)
    assert registry.get("init.depth") is cls
    assert cls.stage_id == "init.depth"
    assert "init.depth" in registry
    assert registry.stage_ids() == ["init.depth"]


def test_get_unknown_lists_registered() -> None:
    registry = StageRegistry()
    registry.register("init.depth")(make_stage("InitDepthStage", cacheable=True))
    with pytest.raises(KeyError, match="init.depth"):
        registry.get("init.slam")


def test_duplicate_stage_id_rejected() -> None:
    registry = StageRegistry()
    registry.register("init.depth")(make_stage("InitDepthStage", cacheable=True))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("init.depth")(make_stage("InitDepthStage", cacheable=True))


@pytest.mark.parametrize(
    "stage_id,name,attrs,match",
    [
        ("init.depth", "DepthStage", {"cacheable": True}, "must be named InitDepthStage"),
        ("init.depth", "InitDepthStage", {"cacheable": False}, "cacheable must be True"),
        ("train.x", "TrainXStage", {"cacheable": True}, "cacheable must be False"),
        ("init.depth", "InitDepthStage", {"cacheable": True, "version": "1"}, "version"),
        ("init.depth", "InitDepthStage", {"cacheable": True, "inputs": {"m": "mask"}}, "type"),
        ("init.depth", "InitDepthStage", {"cacheable": True, "outputs": {}}, "exactly one"),
        (
            "init.depth",
            "InitDepthStage",
            {"cacheable": True, "outputs": {"a": "points_init", "b": "masks"}},
            "exactly one",
        ),
        ("init.depth", "InitDepthStage", {"cacheable": True, "outputs": {"o": "split"}}, "type"),
        ("bad.depth", "BadDepthStage", {}, "stage_id"),
    ],
)
def test_bad_declaration_rejected(stage_id: str, name: str, attrs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StageRegistry().register(stage_id)(make_stage(name, **attrs))


def test_params_must_be_pydantic() -> None:
    cls = make_stage("InitDepthStage", cacheable=True, Params=dict)
    with pytest.raises(TypeError, match="Params"):
        StageRegistry().register("init.depth")(cls)


def test_non_stage_rejected() -> None:
    with pytest.raises(TypeError):
        StageRegistry().register("init.depth")(object)


def test_load_builtin_stages_imports_package() -> None:
    # drive2sim.stages hiện chỉ có các package khung rỗng; hàm phải chạy không lỗi.
    loaded = load_builtin_stages()
    assert "drive2sim.stages.init" in loaded
