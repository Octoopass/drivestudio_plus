"""Tên split tự đặt (engine/scenes.py, CONTRACTS mục 10.1)."""

import copy

from drive2sim.core.schema import Manifest
from drive2sim.engine.scenes import default_split_id
from tests.drive2sim.samples import MANIFEST


def mixed_manifest() -> Manifest:
    """Cảnh như KITTI-360: hai camera pinhole, hai camera mắt cá."""
    data = copy.deepcopy(MANIFEST)
    base = data["cameras"][0]
    data["cameras"] = [
        {**base, "camera_id": "image_00", "model": "opencv_pinhole"},
        {**base, "camera_id": "image_01", "model": "opencv_pinhole"},
        {**base, "camera_id": "image_02", "model": "opencv_fisheye"},
        {**base, "camera_id": "image_03", "model": "opencv_fisheye"},
    ]
    data["timeline"]["reference_sensor"] = "image_00"
    data["track"] = "fisheye"
    return Manifest.model_validate(data)


def pinhole_manifest() -> Manifest:
    """Cảnh như nuScenes: chỉ camera pinhole."""
    data = copy.deepcopy(MANIFEST)
    base = data["cameras"][0]
    data["cameras"] = [
        {**base, "camera_id": f"CAM_{i}", "model": "opencv_pinhole"} for i in range(3)
    ]
    data["timeline"]["reference_sensor"] = "CAM_0"
    data["track"] = "pinhole"
    return Manifest.model_validate(data)


def test_single_lens_scene_keeps_old_names() -> None:
    manifest = pinhole_manifest()
    every = [c.camera_id for c in manifest.cameras]
    assert default_split_id("every_nth_frame:8", every, manifest) == "every8_allcams"
    assert default_split_id("every_nth_frame:8", every[:1], manifest) == "every8_1cams"


def test_mixed_lens_scene_names_lens_type() -> None:
    manifest = mixed_manifest()
    name = default_split_id
    assert name("every_nth_frame:8", ["image_00", "image_01"], manifest) == "every8_pinhole2"
    assert name("every_nth_frame:4", ["image_03"], manifest) == "every4_fisheye1"
    all_four = ["image_00", "image_01", "image_02", "image_03"]
    assert name("every_nth_frame:8", all_four, manifest) == "every8_mixed4"
