"""Kiểm tra chỉ mục quan sát XSIM và câu lệnh xuất gói của prep.observations."""

import subprocess
import sys
from pathlib import Path

from drive2sim.core.schema import Manifest, Split
from drive2sim.stages.prep.observations import PrepObservationsParams, export_command
from drive2sim.stages.train.drivestudio import TrainDrivestudioParams, build_command
from tests.drive2sim.samples import MANIFEST, SPLIT


def test_train_index_uses_only_observed_camera_pairs() -> None:
    split = Split.model_validate(
        {
            **SPLIT,
            "strategy": "custom",
            "cameras": ["left", "right"],
            "train": [[8, "right"], [2, "left"], [3, "left"], [8, "left"]],
            "test": [[2, "right"], [6, "left"]],
        }
    )
    from drive2sim.external.xsim.observations import ObservationIndex

    train = ObservationIndex.from_split(split, {"right": 3, "left": 2}, train=True)
    assert train.lidar_frames == [2, 3, 8]
    assert train.cameras == {2: [2, 3, 8], 3: [8]}
    assert train.cameras_for_lidar == [[(2, 0)], [(2, 1)], [(2, 2), (3, 0)]]
    assert train.lidar_for_camera == {2: [0, 1, 2], 3: [2]}


def test_index_with_all_lidar_frames_links_only_matching_frames() -> None:
    split = Split.model_validate(
        {
            **SPLIT,
            "strategy": "custom",
            "cameras": ["left"],
            "train": [[2, "left"], [5, "left"]],
            "test": [[3, "left"]],
        }
    )
    from drive2sim.external.xsim.observations import ObservationIndex

    # LiDAR của khung test 3 vẫn dùng được, ảnh khung 5 không có sweep cùng khung.
    index = ObservationIndex.from_split(split, {"left": 0}, train=True, lidar_frames=[2, 3, 4])
    assert index.lidar_frames == [2, 3, 4]
    assert index.cameras_for_lidar == [[(0, 0)], [], []]
    assert index.lidar_for_camera == {0: [0, None]}


def test_export_command_is_the_train_command_of_the_same_data(tmp_path: Path) -> None:
    manifest, split = Manifest.model_validate(MANIFEST), Split.model_validate(SPLIT)
    view, out, config = tmp_path / "view", tmp_path / "out", tmp_path / "method.yaml"
    params = PrepObservationsParams(renderer="3dgut", downscale_when_loading=2)
    command = export_command(
        params, split, manifest, view, out, True, False, config,
        tmp_path / "index.json", tmp_path / "scene",
    )  # fmt: skip
    train = build_command(
        TrainDrivestudioParams(
            method="omnire",
            renderer="3dgut",
            overrides={"data.pixel_source.downscale_when_loading": [2]},
            eval_at=[],
        ),
        split,
        manifest,
        view,
        out,
        True,
        False,
        config,
    )
    data = [item for item in train if item.startswith(("data.", "dataset=", "seed="))]
    assert command[1:3] == ["-m", "drive2sim.recon.tools.export_observations"]
    assert [item for item in command if "=" in item] == data
    assert "data.pixel_source.fisheye_renderer_max_angle=1.536" in command
    for flag in ("--output_root", "--project", "--run_name"):
        assert flag not in command
    assert command[command.index("--out") + 1] == str(out)


def test_stages_load_without_numpy_or_torch() -> None:
    """CLI, Studio nạp mọi stage trên máy nhẹ: module stage không import thư viện nặng."""
    code = (
        "import sys\n"
        "for name in ('numpy', 'PIL', 'torch'): sys.modules[name] = None\n"
        "from drive2sim.engine.registry import load_builtin_stages\n"
        "load_builtin_stages()\n"
    )
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
