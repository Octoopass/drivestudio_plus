"""Test đọc và ghi file qua schema trong ``drive2sim.core.files``."""

from pathlib import Path

from drive2sim.core import files
from drive2sim.core.split import make_split


def test_roundtrip_json_and_yaml(tmp_path: Path) -> None:
    split = make_split("nuscenes_scene-0061", ["A"], (0, 9), "every_nth_frame:5", "every5_front")
    for name in ("split.json", "split.yaml"):
        path = files.dump(split, tmp_path / name)
        assert files.load(path, type(split)) == split
        assert files.load_any(path) == split
        assert not (tmp_path / (name + ".tmp")).exists()
