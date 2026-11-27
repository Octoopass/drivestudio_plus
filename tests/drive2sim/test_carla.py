"""Nối CARLA: file cảnh .usdz (bản đồ OpenDRIVE) và các phép đổi pose của server sensorsim."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from drive2sim.external import nurec
from drive2sim.sim.server import sensorsim


def _scene_tool() -> dict:
    """Các hàm thuần numpy của ncore_carla_scene.py (file cần ncore nên không import thẳng)."""
    source = (nurec.TOOLS_DIR / "ncore_carla_scene.py").read_text(encoding="utf-8")
    start = source.index("CAMERAS_FILE =")
    end = source.index("\n\n\ndef main")
    namespace: dict = {
        "np": np,
        "math": math,
        "escape": __import__("xml.sax.saxutils").sax.saxutils.escape,
    }
    exec("from __future__ import annotations\n" + source[start:end], namespace)  # noqa: S102
    return namespace


def test_quaternion_round_trip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = sensorsim.quat_to_matrix(*q)
        back = sensorsim.quat_to_matrix(*sensorsim.matrix_to_quat(r))
        assert np.allclose(r, back, atol=1e-9)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_opendrive_road_follows_path() -> None:
    tool = _scene_tool()
    theta = np.linspace(0, math.pi / 2, 200)
    path = np.stack([50 * np.sin(theta), 50 * (1 - np.cos(theta)), 0.02 * 50 * theta], 1)
    road = tool["resample_path"](path, 2.0)
    spacing = np.linalg.norm(np.diff(road[:, :2], axis=0), axis=1)
    assert np.allclose(spacing[3:-3], 2.0, atol=0.1) and spacing.min() > 0.3
    root = ET.fromstring(tool["opendrive"](road, "test", 3.5, 1.5))
    geometries = root.findall("./road/planView/geometry")
    assert len(geometries) == len(road) - 1
    total = sum(float(g.get("length")) for g in geometries)
    assert abs(float(root.find("road").get("length")) - total) < 1e-2
    assert abs(total - 50 * math.pi / 2) < 1.5  # một phần tư đường tròn bán kính 50 m
    lanes = sorted(int(lane.get("id")) for lane in root.iter("lane"))
    assert lanes == [-2, -1, 0, 1, 2]
    assert float(root.find("./road/lanes/laneOffset").get("a")) == 1.75  # xe giữa làn phải


def test_pinhole_written_as_ftheta_polynomial() -> None:
    tool = _scene_tool()
    # góc lớn nhất của camera trước KITTI-360 (f 552, góc ảnh cách tâm 750 px) khoảng atan(1,36)
    forward, backward = tool["angle_poly_from_radius"](lambda t: 500 * np.tan(t), math.atan(1.4))
    theta = np.linspace(0, math.atan(1.4), 50)
    radius = np.polynomial.polynomial.polyval(theta, forward)
    assert np.abs(radius - 500 * np.tan(theta)).max() < 2.0
    assert np.abs(np.polynomial.polynomial.polyval(radius, backward) - theta).max() < 5e-3


def test_carla_stage_commands(tmp_path: Path) -> None:
    import pytest
    from pydantic import ValidationError

    from drive2sim.engine.registry import REGISTRY, load_builtin_stages
    from drive2sim.stages.render import carla

    load_builtin_stages()
    cls = REGISTRY.get("render.carla")
    assert cls.inputs == {"model": "model", "clip": "ncore_clip", "split": "split"}
    params = carla.CarlaParams(ego="replay", max_seconds=12)
    command = carla.runner_command(tmp_path / "s.usdz", 5555, tmp_path, params)
    assert command[1:4] == ["-u", "-m", "drive2sim.sim.carla_nurec"]
    assert command[command.index("--render-port") + 1] == "5555"
    assert command[command.index("--ego") + 1] == "replay"
    assert command[command.index("--max-seconds") + 1] == "12.0"
    assert command[command.index("--resolution-ratio") + 1] == "0.5"
    assert command[command.index("--video-height") + 1] == "540"
    assert command[command.index("--video") + 1] == str(tmp_path / carla.VIDEO_FILE)
    whole = carla.runner_command(
        tmp_path / "s.usdz", 5555, tmp_path, carla.CarlaParams(max_seconds=0)
    )
    assert "--max-seconds" not in whole
    scene = carla.scene_command(tmp_path / "c.json", tmp_path / "s.usdz", ["a", "b"], params)
    assert scene[scene.index("--cameras") + 1] == "a,b" and "--scene-root" not in scene
    with_objects = carla.scene_command(
        tmp_path / "c.json", tmp_path / "s.usdz", ["a"], params, tmp_path / "scene"
    )
    assert with_objects[with_objects.index("--scene-root") + 1] == str(tmp_path / "scene")
    no_objects = carla.scene_command(
        tmp_path / "c.json",
        tmp_path / "s.usdz",
        ["a"],
        carla.CarlaParams(objects=False),
        tmp_path / "scene",
    )
    assert "--scene-root" not in no_objects
    with pytest.raises(ValidationError):
        carla.CarlaParams(ego="fly")
    with pytest.raises(ValidationError):
        carla.CarlaParams(resolution_ratio=2)
    port = carla.free_port()
    assert 0 < port < 65536 and not carla.port_open(port)
