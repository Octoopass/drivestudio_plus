import copy

import pytest
from pydantic import ValidationError

from drive2sim.core.schema import Approach, Manifest, RunRecord, Split, ValidationReport

APPROACH = {
    "schema_version": "d2s-approach/0.1", "approach_id": "omnire", "name": "OmniRe", "version": "0.1.0",
    "codebase": "drivestudio", "track": "pinhole", "level": "L1",
    "requires": {"camera_models": ["opencv_pinhole"], "cuboids": "required", "split_strategies": ["every_nth_frame"]},
    "options": {
        "init": {"choices": ["init.lidar", "init.depth"], "default": "auto"},
        "renderer": {"choices": ["3dgs", "3dgut", "3dgeer"], "default": "3dgs"},
        "campose": {"choices": [False, True], "default": False},
    },
    "constraints": [{"when": {"renderer": "3dgeer"}, "forbid": {"campose": True}, "reason": "x"}],
    "presets": {"no_lidar": {"init": "init.depth"}},
    "stages": [{"stage": "${options.init}"},
               {"stage": "train.drivestudio", "params": {"renderer": "${options.renderer}"}}],
}


def mutate(base, fn):
    d = copy.deepcopy(base)
    fn(d)
    return d


def test_approach_ok():
    a = Approach.model_validate(APPROACH)
    assert a.options["campose"].choices == [False, True]


@pytest.mark.parametrize("fn", [
    lambda d: d.update(approach_id="OmniRe"),
    lambda d: d.update(codebase="nurec"),
    lambda d: d["options"]["renderer"].update(default="2dgs"),
    lambda d: d["presets"].update(bad={"renderer": "2dgs"}),
    lambda d: d["presets"].update(bad={"unknown": 1}),
    lambda d: d["constraints"][0]["forbid"].update(campose=1),           # 1 is not the boolean true
    lambda d: d["stages"].append({"stage": "${options.missing}"}),
    lambda d: d["stages"].append({"stage": "${options.renderer}"}),      # renderer choices are not stage ids
    lambda d: d["stages"].append({"stage": "train.drivestudio", "params": {"x": "${options.missing}"}}),
    lambda d: d["stages"].append({"stage": "drivestudio"}),
    lambda d: d.update(extra_key=1),
    lambda d: d.update(schema_version="d2s-approach/1.0"),
    lambda d: d.update(schema_version="d2s-run/0.1"),
])
def test_approach_rejects(fn):
    with pytest.raises(ValidationError):
        Approach.model_validate(mutate(APPROACH, fn))


MANIFEST = {
    "schema_version": "d2s-manifest/0.1", "scene_id": "nuscenes_scene-0061",
    "source": {"dataset": "nuscenes", "sequence_id": "scene-0061", "converter": "ncore.nuscenes", "converter_version": "19.9.0"},
    "ncore": {"version": "v4", "sequence_id": "scene-0061", "timestamp_interval_us": {"start": 1, "stop": 2}},
    "track": "pinhole",
    "cameras": [{"camera_id": "CAM_FRONT", "model": "opencv_pinhole", "resolution": [1600, 900], "shutter": "GLOBAL"}],
    "lidars": ["LIDAR_TOP"],
    "timeline": {"reference_sensor": "CAM_FRONT", "num_frames": 39},
    "available": {"lidar": True, "cuboids": "gt"},
}


def test_manifest_ok():
    assert Manifest.model_validate(MANIFEST).camera("CAM_FRONT").resolution == (1600, 900)


@pytest.mark.parametrize("fn", [
    lambda d: d["available"].update(lidar=False),                         # lidars listed but lidar false
    lambda d: d["timeline"].update(reference_sensor="CAM_BACK"),
    lambda d: d.update(scene_id="waymo_scene-0061"),                      # prefix differs from source.dataset
    lambda d: d["cameras"].append(dict(d["cameras"][0])),                 # duplicate camera_id
    lambda d: d["cameras"][0].update(model="fisheye"),
    lambda d: d["ncore"]["timestamp_interval_us"].update(start=5),
])
def test_manifest_rejects(fn):
    with pytest.raises(ValidationError):
        Manifest.model_validate(mutate(MANIFEST, fn))


SPLIT = {
    "schema_version": "d2s-split/0.1", "split_id": "every8_allcams", "scene_id": "nuscenes_scene-0061",
    "strategy": "every_nth_frame:8", "frame_range": [0, 10], "cameras": ["A", "B"],
    "train": [[0, "A"], [0, "B"]], "test": [[8, "A"], [8, "B"]],
}


@pytest.mark.parametrize("fn", [
    lambda d: d["test"].append([0, "A"]),          # in train and test
    lambda d: d["test"].append([11, "A"]),         # outside frame_range
    lambda d: d["test"].append([9, "C"]),          # unknown camera
    lambda d: d.update(strategy="every_nth_frame:0"),
    lambda d: d.update(strategy="random"),
])
def test_split_rejects(fn):
    Split.model_validate(SPLIT)
    with pytest.raises(ValidationError):
        Split.model_validate(mutate(SPLIT, fn))


def test_report_counts_must_match():
    base = {"schema_version": "d2s-report/0.1", "scene_id": "nuscenes_scene-0061",
            "created_utc": "2026-11-12T08:00:00Z", "summary": {"errors": 0, "warnings": 1},
            "checks": [{"check_id": "a", "severity": "warning", "message": "m"}]}
    ValidationReport.model_validate(base)
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(mutate(base, lambda d: d["summary"].update(warnings=0)))


def test_run_lidar_usage_must_be_a_stage():
    base = {"schema_version": "d2s-run/0.1", "run_id": "20261112-081000-omnire-a1b2",
            "scene_id": "nuscenes_scene-0061", "split_id": "every8_allcams",
            "approach": {"approach_id": "omnire", "version": "0.1.0"}, "level": "L1",
            "executor": "local", "status": "running", "lidar_usage": ["init.lidar"],
            "stages": [{"stage_id": "init.lidar", "status": "succeeded"}]}
    RunRecord.model_validate(base)
    with pytest.raises(ValidationError):
        RunRecord.model_validate(mutate(base, lambda d: d.update(lidar_usage=["train.drivestudio"])))
    with pytest.raises(ValidationError):
        RunRecord.model_validate(mutate(base, lambda d: d["approach"].update(approach_id="streetgs")))
