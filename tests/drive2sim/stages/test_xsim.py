"""Kiểm tra chỉ mục XSIM với camera chọn thưa và timestamp tuyệt đối lớn."""

import json
from pathlib import Path
from types import SimpleNamespace

from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.cache import input_key_for_split
from drive2sim.engine.stage import NoParams, Scene
from drive2sim.stages.prep.xsim import PrepXsimStage
from tests.drive2sim.samples import MANIFEST, SPLIT


def test_sparse_split_preserves_camera_index_and_time(tmp_path: Path) -> None:
    scene_id = "kitti360_example"
    manifest = Manifest.model_validate(
        {
            **MANIFEST,
            "scene_id": scene_id,
            "source": {**MANIFEST["source"], "dataset": "kitti360"},
        }
    )
    split = Split.model_validate(
        {
            **SPLIT,
            "scene_id": scene_id,
            "strategy": "custom",
            "frame_range": [2, 4],
            "cameras": ["CAM_FISH"],
            "train": [[2, "CAM_FISH"]],
            "test": [[4, "CAM_FISH"]],
        }
    )
    stamp = 1600000000000000
    (tmp_path / "timestamps.txt").write_text("\n".join(str(stamp + i * 17) for i in range(10)))
    (tmp_path / "frame_ids.txt").write_text("\n".join(str(20 + i * 2) for i in range(10)))
    ctx = SimpleNamespace(
        scene=Scene(scene_id, tmp_path, manifest), split=split, output_dir=tmp_path
    )
    PrepXsimStage().run(ctx, {}, NoParams())
    index = json.loads((tmp_path / "scene.json").read_text())
    assert index["camera_indices"] == {"CAM_FISH": 1}
    assert index["source_frames"] == [24, 26, 28]
    assert index["relative_time_us"] == [0, 17, 34]
    changed = split.model_copy(update={"test": [(3, "CAM_FISH")]})
    assert index["split_key"] != input_key_for_split(split.split_id, changed)
