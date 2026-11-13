"""Test các schema trong ``drive2sim.core.schema``.

Dữ liệu đúng phải được nhận, dữ liệu sai phải bị từ chối.
"""

import copy
from typing import Callable

import pytest
from pydantic import ValidationError

from drive2sim.core.schema import (
    Approach,
    JobResult,
    JobSpec,
    Manifest,
    RunRecord,
    Split,
    ValidationReport,
)

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
        "renderer": {"choices": ["3dgs", "3dgut", "3dgeer"], "default": "3dgs"},
        "campose": {"choices": [False, True], "default": False},
    },
    "constraints": [{"when": {"renderer": "3dgeer"}, "forbid": {"campose": True}, "reason": "x"}],
    "presets": {"no_lidar": {"init": "init.depth"}},
    "stages": [
        {"stage": "${options.init}"},
        {"stage": "train.drivestudio", "params": {"renderer": "${options.renderer}"}},
    ],
}


def modified(base: dict, change: Callable[[dict], None]) -> dict:
    """Trả về bản sao của ``base`` đã áp ``change``; ``base`` giữ nguyên."""
    data = copy.deepcopy(base)
    change(data)
    return data


def test_approach_ok() -> None:
    approach = Approach.model_validate(APPROACH)
    assert approach.options["campose"].choices == [False, True]


def test_approach_display_text() -> None:
    def add_text(data: dict) -> None:
        data["options"]["campose"].update(
            label="CamPose", help="tự chỉnh pose", labels=["Tắt", "Bật"]
        )
        data["preset_help"] = {name: "mô tả" for name in data["presets"]}
        data["groups"] = {"speedup": {"label": "Thuật toán phụ trợ"}}
        data["options"]["campose"].update(group="speedup", descriptions=["tắt", "bật"])

    approach = Approach.model_validate(modified(APPROACH, add_text))
    assert approach.options["campose"].labels == ["Tắt", "Bật"]


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(approach_id="OmniRe"),
        lambda data: data.update(codebase="nurec"),
        lambda data: data["options"]["renderer"].update(default="2dgs"),
        lambda data: data["presets"].update(bad={"renderer": "2dgs"}),
        lambda data: data["presets"].update(bad={"unknown": 1}),
        lambda data: data["constraints"][0]["forbid"].update(
            campose=1
        ),  # 1 is not the boolean true
        lambda data: data["stages"].append({"stage": "${options.missing}"}),
        lambda data: data["stages"].append(
            {"stage": "${options.renderer}"}
        ),  # renderer choices are not stage ids
        lambda data: data["stages"].append(
            {"stage": "train.drivestudio", "params": {"x": "${options.missing}"}}
        ),
        lambda data: data["stages"].append({"stage": "drivestudio"}),
        lambda data: data.update(extra_key=1),
        lambda data: data.update(schema_version="d2s-approach/1.0"),
        lambda data: data.update(schema_version="d2s-run/0.1"),
        lambda data: data["options"]["campose"].update(labels=["chỉ một nhãn"]),
        lambda data: data.update(preset_help={"missing_preset": "mô tả"}),
        lambda data: data["options"]["campose"].update(descriptions=["chỉ một"]),
        lambda data: data["options"]["campose"].update(group="missing_group"),
    ],
)
def test_approach_rejects(change: Callable[[dict], None]) -> None:
    with pytest.raises(ValidationError):
        Approach.model_validate(modified(APPROACH, change))


MANIFEST = {
    "schema_version": "d2s-manifest/1.0",
    "scene_id": "nuscenes_scene-0061",
    "source": {
        "dataset": "nuscenes",
        "sequence_id": "scene-0061",
        "converter": "drivestudio.nuscenes",
        "converter_version": "0.1.0+test",
    },
    "track": "pinhole",
    "cameras": [
        {
            "camera_id": "CAM_FRONT",
            "model": "opencv_pinhole",
            "resolution": [1600, 900],
            "shutter": "GLOBAL",
        }
    ],
    "lidars": ["LIDAR_TOP"],
    "timeline": {"reference_sensor": "CAM_FRONT", "num_frames": 39},
    "available": {"lidar": True, "cuboids": "gt"},
}


def test_manifest_ok() -> None:
    assert Manifest.model_validate(MANIFEST).camera("CAM_FRONT").resolution == (1600, 900)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["available"].update(lidar=False),  # lidars listed but lidar false
        lambda data: data["timeline"].update(reference_sensor="CAM_BACK"),
        lambda data: data.update(
            scene_id="waymo_scene-0061"
        ),  # prefix differs from source.dataset
        lambda data: data["cameras"].append(dict(data["cameras"][0])),  # duplicate camera_id
        lambda data: data["cameras"][0].update(model="fisheye"),
    ],
)
def test_manifest_rejects(change: Callable[[dict], None]) -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(modified(MANIFEST, change))


SPLIT = {
    "schema_version": "d2s-split/0.1",
    "split_id": "every8_allcams",
    "scene_id": "nuscenes_scene-0061",
    "strategy": "every_nth_frame:8",
    "frame_range": [0, 10],
    "cameras": ["A", "B"],
    "train": [[0, "A"], [0, "B"]],
    "test": [[8, "A"], [8, "B"]],
}


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["test"].append([0, "A"]),  # in train and test
        lambda data: data["test"].append([11, "A"]),  # outside frame_range
        lambda data: data["test"].append([9, "C"]),  # unknown camera
        lambda data: data.update(strategy="every_nth_frame:0"),
        lambda data: data.update(strategy="random"),
    ],
)
def test_split_rejects(change: Callable[[dict], None]) -> None:
    Split.model_validate(SPLIT)
    with pytest.raises(ValidationError):
        Split.model_validate(modified(SPLIT, change))


def test_report_counts_must_match() -> None:
    base = {
        "schema_version": "d2s-report/0.1",
        "scene_id": "nuscenes_scene-0061",
        "created_utc": "2026-11-12T08:00:00Z",
        "summary": {"errors": 0, "warnings": 1},
        "checks": [{"check_id": "a", "severity": "warning", "message": "m"}],
    }
    ValidationReport.model_validate(base)
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(
            modified(base, lambda data: data["summary"].update(warnings=0))
        )


def test_run_lidar_usage_must_be_a_stage() -> None:
    base = {
        "schema_version": "d2s-run/0.1",
        "run_id": "20261112-081000-omnire-a1b2",
        "scene_id": "nuscenes_scene-0061",
        "split_id": "every8_allcams",
        "approach": {"approach_id": "omnire", "version": "0.1.0"},
        "level": "L1",
        "executor": "local",
        "status": "running",
        "lidar_usage": ["init.lidar"],
        "stages": [{"stage_id": "init.lidar", "status": "succeeded"}],
    }
    RunRecord.model_validate(base)
    with pytest.raises(ValidationError):
        RunRecord.model_validate(
            modified(base, lambda data: data.update(lidar_usage=["train.drivestudio"]))
        )
    with pytest.raises(ValidationError):
        RunRecord.model_validate(
            modified(base, lambda data: data["approach"].update(approach_id="streetgs"))
        )


JOB = {
    "schema_version": "d2s-job/0.1",
    "job_id": "20261112-081000-omnire-a1b2/train.external",
    "kind": "stage",
    "stage_id": "train.external",
    "params": {"codebase": "nurec"},
    "inputs": {"scene": "data/nuscenes_scene-0061"},
    "output_dir": "runs/x/d2s/stages/train.external",
    "env": "nurec",
}

RESULT_OK = {
    "schema_version": "d2s-result/0.1",
    "job_id": "20261112-081000-omnire-a1b2/train.external",
    "status": "succeeded",
    "outputs": {"model": "runs/x/d2s/stages/train.external/checkpoint_final.pth"},
    "duration_s": 4380,
    "peak_vram_mb": 7400,
}


def test_job_stage_requires_stage_id() -> None:
    JobSpec.model_validate(JOB)
    bad = copy.deepcopy(JOB)
    del bad["stage_id"]
    with pytest.raises(ValidationError):
        JobSpec.model_validate(bad)


def test_job_render_rejects_stage_id() -> None:
    bad = {**JOB, "kind": "render"}
    with pytest.raises(ValidationError):
        JobSpec.model_validate(bad)
    del bad["stage_id"]
    JobSpec.model_validate(bad)


def test_result_error_matches_status() -> None:
    JobResult.model_validate(RESULT_OK)
    with pytest.raises(ValidationError):
        JobResult.model_validate({**RESULT_OK, "status": "failed"})
    failed = {
        **RESULT_OK,
        "status": "failed",
        "outputs": {},
        "error": {"code": "env_missing", "message": "Thiếu môi trường"},
    }
    JobResult.model_validate(failed)
    with pytest.raises(ValidationError):
        JobResult.model_validate({**RESULT_OK, "error": failed["error"]})


def test_result_rejects_unknown_error_code() -> None:
    bad = {
        **RESULT_OK,
        "status": "failed",
        "outputs": {},
        "error": {"code": "something_else", "message": "x"},
    }
    with pytest.raises(ValidationError):
        JobResult.model_validate(bad)
