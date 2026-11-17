"""Kiểm tra box người và việc bỏ SMPL chỉ khi biết chắc split không có người."""

import ast
import json
from pathlib import Path

import pytest

from drive2sim import paths
from drive2sim.core.instances import HUMAN_CLASSES, human_frame_ranges
from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.registry import load_builtin_stages
from drive2sim.engine.resolve import BlockedError, resolve
from tests.drive2sim.samples import MANIFEST, SPLIT


def test_human_classes_match_loader_node_mapping() -> None:
    classes = set()
    for path in (paths.PACKAGE_DIR / "recon/data/loaders").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(value, ast.Attribute) and value.attr == "SMPLNodes":
                        classes.add(ast.literal_eval(key))
    assert classes == HUMAN_CLASSES


def test_ranges_and_corrupt_instances(tmp_path: Path) -> None:
    folder = tmp_path / "instances"
    folder.mkdir()
    info = folder / "instances_info.json"
    frames = folder / "frame_instances.json"
    assert human_frame_ranges(tmp_path) is None
    info.write_text(json.dumps({"0": {"class_name": "Pedestrian"}, "1": {"class_name": "Car"}}))
    frames.write_text(json.dumps({"1": [0], "2": ["0"], "3": [1], "5": [0]}))
    assert human_frame_ranges(tmp_path) == [(1, 2), (5, 5)]
    frames.write_text('{"1": 0}')
    assert human_frame_ranges(tmp_path) is None
    frames.write_text('{"1": [99]}')
    assert human_frame_ranges(tmp_path) is None
    frames.write_text("broken")
    assert human_frame_ranges(tmp_path) is None


@pytest.mark.parametrize(
    "human_frames,expected",
    [
        (None, "prep.smpl_release"),
        ([], "none"),
        ([(8, 9)], "prep.smpl_release"),
        ([(10, 12)], "none"),
    ],
)
def test_auto_smpl_uses_split_range(human_frames: object, expected: str) -> None:
    load_builtin_stages()
    manifest = Manifest.model_validate(MANIFEST)
    manifest.available.human_frames = human_frames
    split = Split.model_validate(SPLIT)
    approach = load_approach("omnire")
    plan = resolve(approach, manifest, split, quick=True)
    assert plan.options["smpl"] == expected
    assert (
        "smpl" in next(s for s in plan.stages if s.stage_id == "train.drivestudio").input_sources
    ) == (expected != "none")
    assert all(s.stage_id != "none" for s in plan.stages)
    if expected == "none":
        with pytest.raises(BlockedError, match="không có người"):
            resolve(approach, manifest, split, option_values={"smpl": "prep.smpl"})
