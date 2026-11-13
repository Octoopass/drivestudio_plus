"""Kiểm tra bước quyết định danh sách stage trong ``drive2sim/engine/resolve.py``.

File này kiểm tra, trên một approach giả có mục cấu hình ``init``, ``renderer``, ``campose``:
- giá trị của mục cấu hình gộp đúng thứ tự mặc định → preset → lựa chọn của người dùng;
- ``auto`` chọn ``init.lidar`` khi cảnh có LiDAR và ``init.depth`` khi không có;
- tham số stage được gộp, kiểm tra, và đầu vào được nối với đúng stage phía trước;
- các trường hợp bị khoá (thiếu LiDAR, vi phạm ràng buộc, sai track, sai chiến lược split, mô
  hình camera không hỗ trợ) báo ``BlockedError`` kèm đủ mọi lý do;
- các yêu cầu sai (preset, mục cấu hình, giá trị, tham số, stage không tồn tại) báo
  ``ResolveError``.
"""

import copy
from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel, ConfigDict

from drive2sim.core.schema import Approach, Manifest, Split
from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.resolve import (
    QUICK_NUM_ITERS,
    BlockedError,
    Plan,
    ResolveError,
    blocked_choices,
    resolve,
    wire_inputs,
)
from drive2sim.engine.stage import Stage, StageContext
from tests.drive2sim.samples import MANIFEST, SPLIT

APPROACH = {
    "schema_version": "d2s-approach/0.1",
    "approach_id": "omnire",
    "name": "OmniRe",
    "version": "0.1.0",
    "codebase": "drivestudio",
    "track": "pinhole",
    "level": "L1",
    "requires": {
        "camera_models": ["opencv_pinhole"],
        "cuboids": "required",
        "split_strategies": ["every_nth_frame"],
    },
    "options": {
        "init": {"choices": ["init.lidar", "init.depth"], "default": "auto"},
        "renderer": {"choices": ["3dgs", "3dgeer"], "default": "3dgs"},
        "campose": {"choices": [False, True], "default": False},
    },
    "constraints": [
        {"when": {"renderer": "3dgeer"}, "forbid": {"campose": True}, "reason": "GEER"}
    ],
    "presets": {"no_lidar": {"init": "init.depth"}, "fisheye": {"renderer": "3dgeer"}},
    "stages": [
        {"stage": "${options.init}"},
        {
            "stage": "train.fake",
            "params": {"renderer": "${options.renderer}", "campose": "${options.campose}"},
        },
    ],
}


def _noop(
    self: Stage, ctx: StageContext, inputs: dict[str, Path], params: BaseModel
) -> dict[str, Path]:
    return {}


class TrainParams(BaseModel):
    """Tham số của stage giả ``train.fake``."""

    model_config = ConfigDict(extra="forbid")
    renderer: str = "3dgs"
    campose: bool = False
    num_iters: int = 30000


def make_registry() -> StageRegistry:
    registry = StageRegistry()
    stages = [
        (
            "init.lidar",
            "InitLidarStage",
            {"cacheable": True, "uses_lidar": True, "inputs": {"scene": "scene"}},
        ),
        ("init.depth", "InitDepthStage", {"cacheable": True, "inputs": {"scene": "scene"}}),
        (
            "train.fake",
            "TrainFakeStage",
            {
                "inputs": {"points": "points_init", "split": "split"},
                "outputs": {"model": "model"},
                "Params": TrainParams,
            },
        ),
        ("prep.undistort", "PrepUndistortStage", {"cacheable": True}),
    ]
    for stage_id, name, attrs in stages:
        body = {"version": "0.1.0", "outputs": {"points": "points_init"}, "run": _noop, **attrs}
        registry.register(stage_id)(type(name, (Stage,), body))
    return registry


def run_resolve(
    approach: Optional[dict] = None,
    manifest: Optional[dict] = None,
    split: Optional[dict] = None,
    **kwargs: object,
) -> Plan:
    return resolve(
        Approach.model_validate(approach or APPROACH),
        Manifest.model_validate(manifest or MANIFEST),
        Split.model_validate(split or SPLIT),
        registry=make_registry(),
        **kwargs,
    )


def no_lidar_manifest() -> dict:
    manifest = copy.deepcopy(MANIFEST)
    manifest["lidars"] = []
    manifest["available"]["lidar"] = False
    return manifest


def test_defaults_and_auto_with_lidar() -> None:
    plan = run_resolve()
    assert plan.options == {"init": "init.lidar", "renderer": "3dgs", "campose": False}
    assert [s.stage_id for s in plan.stages] == ["init.lidar", "train.fake"]
    assert plan.lidar_usage == ["init.lidar"]


def test_auto_falls_back_without_lidar() -> None:
    plan = run_resolve(manifest=no_lidar_manifest())
    assert plan.options["init"] == "init.depth"
    assert plan.lidar_usage == []


def test_params_merged_and_inputs_wired() -> None:
    plan = run_resolve(preset="fisheye", stage_params={"train.fake": {"num_iters": 100}})
    train = plan.stages[1]
    assert train.params == {"renderer": "3dgeer", "campose": False, "num_iters": 100}
    assert train.input_sources == {"points": 0, "split": "split"}


def test_preset_name_kept_only_when_configuration_matches() -> None:
    assert run_resolve(preset="fisheye").preset == "fisheye"
    # chỉnh lại đúng giá trị của preset: vẫn là preset đó
    same = run_resolve(preset="fisheye", option_values={"renderer": "3dgeer"})
    assert same.preset == "fisheye"
    # chỉnh khác preset: cấu hình không còn là "fisheye"
    changed = run_resolve(preset="fisheye", option_values={"renderer": "3dgs"})
    assert changed.preset is None and changed.options["renderer"] == "3dgs"


def test_quick_sets_num_iters_unless_user_set_it() -> None:
    plan = run_resolve(quick=True)
    assert plan.quick is True
    assert plan.stages[1].params["num_iters"] == QUICK_NUM_ITERS

    user = {"train.fake": {"num_iters": 500}}
    plan = run_resolve(quick=True, stage_params=user)
    assert plan.stages[1].params["num_iters"] == 500
    assert user == {"train.fake": {"num_iters": 500}}  # không sửa dict của người gọi
    assert run_resolve().quick is False


def test_quick_needs_a_stage_with_num_iters() -> None:
    approach = {**APPROACH, "stages": [{"stage": "${options.init}"}]}
    with pytest.raises(ResolveError, match="chạy thử nhanh"):
        run_resolve(approach=approach, quick=True)


def test_explicit_lidar_choice_blocked_without_lidar() -> None:
    with pytest.raises(BlockedError) as info:
        run_resolve(manifest=no_lidar_manifest(), option_values={"init": "init.lidar"})
    assert info.value.reasons[0].option == "init"
    assert "LiDAR" in info.value.reasons[0].message


def test_constraint_blocks() -> None:
    with pytest.raises(BlockedError, match="GEER"):
        run_resolve(preset="fisheye", option_values={"campose": True})


def test_all_reasons_reported_together() -> None:
    manifest = no_lidar_manifest()
    manifest["available"]["cuboids"] = "none"
    with pytest.raises(BlockedError) as info:
        run_resolve(manifest=manifest, option_values={"init": "init.lidar"})
    assert len(info.value.reasons) == 2


def test_camera_model_needs_undistort() -> None:
    split = {**SPLIT, "cameras": ["CAM_FISH"], "train": [[0, "CAM_FISH"]], "test": []}
    with pytest.raises(BlockedError, match="prep.undistort"):
        run_resolve(split=split)
    approach = copy.deepcopy(APPROACH)
    approach["stages"].insert(0, {"stage": "prep.undistort"})
    run_resolve(approach=approach, split=split)


def test_track_and_split_strategy_block() -> None:
    approach = copy.deepcopy(APPROACH)
    approach["track"] = "fisheye"
    approach["requires"]["split_strategies"] = ["custom"]
    with pytest.raises(BlockedError) as info:
        run_resolve(approach=approach)
    assert len(info.value.reasons) == 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"preset": "fast"}, "preset"),
        ({"option_values": {"color": "red"}}, "không có mục cấu hình"),
        ({"option_values": {"campose": 1}}, "không phải lựa chọn"),
        ({"stage_params": {"train.fake": {"bogus": 1}}}, "tham số"),
        ({"stage_params": {"eval.standard": {"x": 1}}}, "không có trong danh sách"),
    ],
)
def test_invalid_requests(kwargs: dict, match: str) -> None:
    with pytest.raises(ResolveError, match=match):
        run_resolve(**kwargs)


def test_unknown_stage_and_missing_input() -> None:
    approach = copy.deepcopy(APPROACH)
    approach["stages"].append({"stage": "eval.standard"})
    with pytest.raises(ResolveError, match="chưa có"):
        run_resolve(approach=approach)
    approach = copy.deepcopy(APPROACH)
    approach["stages"] = [approach["stages"][1]]
    with pytest.raises(ResolveError, match="points_init"):
        run_resolve(approach=approach)


def test_split_must_match_scene() -> None:
    with pytest.raises(ResolveError, match="camera"):
        run_resolve(split={**SPLIT, "cameras": ["CAM_X"], "train": [[0, "CAM_X"]], "test": []})


def test_blocked_choices_for_studio() -> None:
    result = blocked_choices(
        Approach.model_validate(APPROACH),
        Manifest.model_validate(no_lidar_manifest()),
        make_registry(),
    )
    assert list(result) == ["init"]
    assert result["init"][0].choice == "init.lidar"


def test_optional_input_skipped_when_not_produced() -> None:
    registry = make_registry()
    body = {
        "version": "0.1.0",
        "inputs": {"split": "split", "masks": "masks"},
        "optional_inputs": frozenset({"masks"}),
        "outputs": {"model": "model"},
        "run": _noop,
    }
    registry.register("train.opt")(type("TrainOptStage", (Stage,), body))
    wiring = wire_inputs(["train.opt"], registry)
    assert wiring == [{"split": "split"}]
