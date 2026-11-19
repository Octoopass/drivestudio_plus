"""Dataset registry: detection, safe paths, local discovery and the DriveStudio preprocess job."""
import pytest

from fisheye_fusion.dataset_profiles import PROFILES, detect, detect_folder, safe_relative
from fisheye_fusion.pipeline import Discovery, PipelineManager


@pytest.mark.parametrize("paths,expected", [
    (["data_poses.zip", "data_2d_raw/x.tar"], "kitti360"),
    (["segment-123_with_camera_labels.tfrecord"], "waymo"),
    (["v1.0-mini/scene.json", "samples/CAM_FRONT/a.jpg"], "nuscenes"),
    (["2011_09_26/2011_09_26_drive_0001_sync/image_00/x.png", "2011_09_26/calib_cam_to_cam.txt"], "kitti"),
    (["maps/x.json", "nuplan-v1.1/splits/mini/a.db"], "nuplan"),
    (["clips/abc/pai_abc.json"], "physicalai"),
    (["train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/map.json"], "argoverse"),
    (["001/camera/front.jpg", "001/lidar/0.pkl"], "pandaset"),
    (["notes.docx"], None),
])
def test_detect(paths, expected):
    assert detect(paths) == expected


def test_detect_folder_does_not_claim_a_parent_folder():
    # MyDrive/datasets holds kitti360_full/data_poses.zip: only the inner folder is a dataset root
    assert detect_folder(["kitti360_full"], ["kitti360_full/data_poses.zip"]) is None
    assert detect_folder(["data_poses.zip", "data_2d_raw"]) == "kitti360"
    assert detect_folder(["train"], ["train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"]) == "argoverse"


@pytest.mark.parametrize("bad", ["../x", "a/../../b", "C:/x", ".ssh/id", "a\x00b", ""])
def test_safe_relative_rejects(bad):
    with pytest.raises(ValueError):
        safe_relative(bad)


def test_safe_relative_normalises():
    assert safe_relative("a" + chr(92) + "b/./c.bin") == "a/b/c.bin"
    assert safe_relative("/etc/passwd") == "etc/passwd"   # leading slash is dropped, path stays relative


def test_every_profile_except_kitti360_has_a_preprocess_preset():
    assert all("preprocess" in p for k, p in PROFILES.items() if k != "kitti360")


def test_local_datasets_lists_raw_and_processed(tmp_path):
    raw = tmp_path / "data" / "waymo" / "raw"
    raw.mkdir(parents=True)
    (raw / "segment-1.tfrecord").write_bytes(b"x")
    scene = tmp_path / "data" / "waymo" / "processed" / "training" / "000"
    (scene / "images").mkdir(parents=True)
    (scene / "sky_masks").mkdir()
    (tmp_path / "data" / "waymo_example_scenes.txt").write_text("id\n")
    disc = Discovery(tmp_path)
    local = disc.local_datasets()
    assert local["raw"][0]["dataset"] == "waymo" and local["raw"][0]["detected"]
    (proc,) = local["processed"]
    assert proc["data_root"] == "data/waymo/processed/training" and proc["scene"] == "000" and proc["sky_masks"]
    assert disc.split_files() == ["data/waymo_example_scenes.txt"]


def test_preprocess_builds_drivestudio_command(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    raw = tmp_path / "data" / "nuscenes" / "raw"
    (raw / "v1.0-mini").mkdir(parents=True)
    pm = PipelineManager(tmp_path)
    started = {}
    monkeypatch.setattr(pm, "_start", lambda kind, out, steps, env, info: started.update(kind=kind, steps=steps, info=info) or {})
    pm.preprocess({"dataset": "nuscenes", "scene_ids": "0, 3", "workers": 4, "sky_masks": True}, venv)
    (label, argv), (label2, masks) = started["steps"]
    assert label == "preprocess" and argv[argv.index("--dataset") + 1] == "nuscenes"
    assert argv[argv.index("--scene_ids") + 1:argv.index("--scene_ids") + 3] == ["0", "3"]
    assert "--interpolate_N" in argv and argv[argv.index("--split") + 1] == "v1.0-mini" and "pose" not in argv
    assert label2 == "sky masks" and masks[-2:] == ["0", "3"]
    assert started["info"]["data_root"] == "data/nuscenes/processed_10Hz/mini"
    for bad in ({"dataset": "kitti360"}, {"dataset": "waymo"}, {"dataset": "nuscenes", "split_file": "/etc/passwd"},
                {"dataset": "nuscenes", "split": "a b"}, {"dataset": "nuscenes", "rm": 1}):
        with pytest.raises(ValueError):
            pm.preprocess(bad, venv)
