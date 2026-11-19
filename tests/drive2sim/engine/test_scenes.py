"""Tên split tự đặt (engine/scenes.py, CONTRACTS mục 10.1)."""

import copy
import json
import math
from pathlib import Path
from typing import Optional

import pytest

from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.scenes import camera_lenses, default_split_id
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


def test_camera_lenses_reads_intrinsics_by_manifest_order(tmp_path: Path) -> None:
    manifest = mixed_manifest()
    scene = tmp_path / "scene"
    files.dump(manifest, scene / "manifest.yaml")
    width, height = manifest.cameras[0].resolution
    (scene / "intrinsics").mkdir()
    # image_00 (thứ tự 0): pinhole; image_02 (thứ tự 2): fisheye không méo; image_01, 03: thiếu.
    (scene / "intrinsics" / "0.txt").write_text(
        f"{width / 2} {width / 2} {width / 2} {height / 2}\n"
    )
    (scene / "intrinsics" / "2.txt").write_text(
        f"{width} {width} {width / 2} {height / 2} 0 0 0 0\n"
    )
    lenses = camera_lenses(scene)
    assert sorted(lenses) == ["image_00", "image_02"]
    assert lenses["image_00"]["model"] == "opencv_pinhole"
    assert lenses["image_00"]["fov_x_deg"] == pytest.approx(90.0)
    assert lenses["image_02"]["model"] == "opencv_fisheye"
    assert lenses["image_02"]["fov_x_deg"] == pytest.approx(57.2958, abs=1e-3)  # 2 · 0,5 rad


def fisheye_scene(tmp_path: Path, max_angle: Optional[float]) -> Path:
    """Cảnh có image_02 mắt cá; mép ảnh ở θ_d = 1,2 rad (không méo, f = width / 2,4)."""
    data = mixed_manifest().model_dump(mode="json")
    data["cameras"][2]["max_angle"] = max_angle
    manifest = Manifest.model_validate(data)
    scene = tmp_path / "scene"
    files.dump(manifest, scene / "manifest.yaml")
    width, height = manifest.cameras[2].resolution
    (scene / "intrinsics").mkdir()
    f = width / 2.4
    (scene / "intrinsics" / "2.txt").write_text(f"{f} {f} {width / 2} {height / 2} 0 0 0 0\n")
    return scene


def test_camera_lenses_uses_converter_limit_from_manifest(tmp_path: Path) -> None:
    lenses = camera_lenses(fisheye_scene(tmp_path, max_angle=1.0))
    assert lenses["image_02"]["fov_x_deg"] == pytest.approx(math.degrees(2.0), abs=1e-3)
    assert lenses["image_02"]["max_angle"] == 1.0


def test_camera_lenses_without_limit_has_no_hard_coded_angle(tmp_path: Path) -> None:
    lenses = camera_lenses(fisheye_scene(tmp_path, max_angle=None))
    assert lenses["image_02"]["fov_x_deg"] == pytest.approx(math.degrees(2.4), abs=1e-3)
    assert lenses["image_02"]["max_angle"] is None


def test_camera_lenses_reads_limit_of_old_scene_from_report(tmp_path: Path) -> None:
    # Cảnh ingest trước khi manifest có max_angle: converter đã ghi giới hạn vào báo cáo.
    scene = fisheye_scene(tmp_path, max_angle=None)
    report = {
        "checks": [
            {
                "check_id": "camera.fisheye_fov",
                "severity": "info",
                "message": "",
                "details": {"max_angle": 1.0, "cameras": ["image_02"]},
            }
        ]
    }
    (scene / "reports").mkdir()
    (scene / "reports" / "validation.json").write_text(json.dumps(report))
    assert camera_lenses(scene)["image_02"]["fov_x_deg"] == pytest.approx(
        math.degrees(2.0), abs=1e-3
    )


def test_camera_lenses_without_scene_data(tmp_path: Path) -> None:
    assert camera_lenses(tmp_path / "missing") == {}
