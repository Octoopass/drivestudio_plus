"""Test định dạng và hàm tạo định danh trong ``drive2sim.core.identifiers``."""

from datetime import datetime, timezone

import pytest

from drive2sim.core import identifiers


@pytest.mark.parametrize(
    "value", ["nuscenes_scene-0061", "waymo_10017090168044687777", "custom_mentor_01"]
)
def test_scene_id_ok(value: str) -> None:
    assert identifiers.check_scene_id(value) == value


@pytest.mark.parametrize("value", ["scene-0061", "NuScenes_scene", "nuscenes_", "foo_bar"])
def test_scene_id_bad(value: str) -> None:
    with pytest.raises(ValueError):
        identifiers.check_scene_id(value)


@pytest.mark.parametrize("value", ["prep.masks_segformer", "init.depth", "export.ply_world"])
def test_stage_id_ok(value: str) -> None:
    assert identifiers.check_stage_id(value) == value


@pytest.mark.parametrize(
    "value", ["masks.segformer_hf", "init", "train.DriveStudio", "prep.masks-segformer"]
)
def test_stage_id_bad(value: str) -> None:
    with pytest.raises(ValueError):
        identifiers.check_stage_id(value)


def test_run_id_format() -> None:
    run_id = identifiers.make_run_id(
        "omnire", now=datetime(2026, 11, 12, 8, 10, 0, tzinfo=timezone.utc), suffix="a1b2"
    )
    assert run_id == "20261112-081000-omnire-a1b2"


def test_cache_key_is_stable_and_order_independent() -> None:
    key_a = identifiers.make_cache_key(
        "nuscenes_scene-0061", "init.depth", "0.1.0", {"b": 1, "a": 2}, ["x", "y"]
    )
    key_b = identifiers.make_cache_key(
        "nuscenes_scene-0061", "init.depth", "0.1.0", {"a": 2, "b": 1}, ["y", "x"]
    )
    assert key_a == key_b
    assert identifiers.check_cache_key(key_a) == key_a
    assert key_a.startswith("nuscenes_scene-0061/init.depth@0.1.0-")


def test_cache_key_changes_with_params() -> None:
    key_a = identifiers.make_cache_key(
        "nuscenes_scene-0061", "init.depth", "0.1.0", {"model": "a"}, []
    )
    key_b = identifiers.make_cache_key(
        "nuscenes_scene-0061", "init.depth", "0.1.0", {"model": "b"}, []
    )
    assert key_a != key_b


def test_schema_version_parse() -> None:
    assert identifiers.parse_schema_version("d2s-run/0.1") == ("run", 0, 1)
    with pytest.raises(ValueError):
        identifiers.parse_schema_version("run/0.1")
