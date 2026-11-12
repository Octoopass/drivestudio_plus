from datetime import datetime, timezone

import pytest

from drive2sim.core import ids


@pytest.mark.parametrize("v", ["nuscenes_scene-0061", "waymo_10017090168044687777", "custom_mentor_01"])
def test_scene_id_ok(v):
    assert ids.check_scene_id(v) == v


@pytest.mark.parametrize("v", ["scene-0061", "NuScenes_scene", "nuscenes_", "foo_bar"])
def test_scene_id_bad(v):
    with pytest.raises(ValueError):
        ids.check_scene_id(v)


@pytest.mark.parametrize("v", ["prep.masks_segformer", "init.depth", "export.ply_world"])
def test_stage_id_ok(v):
    assert ids.check_stage_id(v) == v


@pytest.mark.parametrize("v", ["masks.segformer_hf", "init", "train.DriveStudio", "prep.masks-segformer"])
def test_stage_id_bad(v):
    with pytest.raises(ValueError):
        ids.check_stage_id(v)


def test_run_id_format():
    rid = ids.make_run_id("omnire", now=datetime(2026, 11, 12, 8, 10, 0, tzinfo=timezone.utc), suffix="a1b2")
    assert rid == "20261112-081000-omnire-a1b2"


def test_cache_key_is_stable_and_order_independent():
    a = ids.make_cache_key("nuscenes_scene-0061", "init.depth", "0.1.0", {"b": 1, "a": 2}, ["x", "y"])
    b = ids.make_cache_key("nuscenes_scene-0061", "init.depth", "0.1.0", {"a": 2, "b": 1}, ["y", "x"])
    assert a == b
    assert ids.check_cache_key(a) == a
    assert a.startswith("nuscenes_scene-0061/init.depth@0.1.0-")


def test_cache_key_changes_with_params():
    a = ids.make_cache_key("nuscenes_scene-0061", "init.depth", "0.1.0", {"model": "a"}, [])
    b = ids.make_cache_key("nuscenes_scene-0061", "init.depth", "0.1.0", {"model": "b"}, [])
    assert a != b


def test_schema_version_parse():
    assert ids.parse_schema_version("d2s-run/0.1") == ("run", 0, 1)
    with pytest.raises(ValueError):
        ids.parse_schema_version("run/0.1")
