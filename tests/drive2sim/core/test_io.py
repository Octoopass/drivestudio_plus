from drive2sim.core import io
from drive2sim.core.split import make_split


def test_roundtrip_json_and_yaml(tmp_path):
    s = make_split("nuscenes_scene-0061", ["A"], (0, 9), "every_nth_frame:5", "every5_front")
    for name in ("split.json", "split.yaml"):
        path = io.dump(s, tmp_path / name)
        assert io.load(path, type(s)) == s
        assert io.load_any(path) == s
        assert not (tmp_path / (name + ".tmp")).exists()
