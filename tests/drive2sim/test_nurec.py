"""Test approach Instant NuRec: môi trường, stage (câu lệnh, ánh xạ khung, summary), đổi hệ PLY."""

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.engine import env_install, remote
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.environment import environment_check
from drive2sim.engine.registry import StageRegistry, load_builtin_stages
from drive2sim.engine.workdir import Workdir
from drive2sim.external import nurec
from drive2sim.stages.eval.ncore import clip_test_indices, make_summary
from drive2sim.stages.prep import ncore as prep_ncore
from drive2sim.stages.train import nurec as train_nurec
from drive2sim.studio.api.app import create_app

load_builtin_stages()


@pytest.fixture
def nurec_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "instant-nurec"
    monkeypatch.setenv(nurec.NUREC_DIR_ENV, str(home))
    nurec.clear_cache()
    yield home
    nurec.clear_cache()


def test_approach_loads_with_its_stages() -> None:
    approach = load_approach("nurec")
    assert approach.codebase == "external:nurec"
    assert [s.stage for s in approach.stages] == [
        "prep.ncore", "train.nurec", "eval.ncore", "export.nurec_ply",
    ]  # fmt: skip


def test_environment_missing_says_how_to_install(nurec_home: Path) -> None:
    env = environment_check("external:nurec")
    assert env["env"] == "nurec" and env["level"] == "missing"
    ids = {c["id"]: c["level"] for c in env["checks"]}
    assert ids["nurec_source"] == ids["nurec_env"] == ids["nurec_checkpoint"] == "missing"
    assert "envs/install_nurec.sh" in env["message"]


def test_paths_follow_the_env_variable(nurec_home: Path) -> None:
    assert nurec.nurec_python() == nurec_home / ".venv" / "bin" / "python"
    assert nurec.checkpoint_path("pa-front").name == "instant_nurec_pa_front_1.1.0.pth"
    with pytest.raises(ValueError):
        nurec.checkpoint_path("nope")


def test_prep_ncore_only_for_kitti360_and_finds_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Source:
        dataset = "nuscenes"

    class Manifest:
        source = Source()

    assert "KITTI-360" in prep_ncore.PrepNcoreStage.unavailable_reason(Manifest())
    Source.dataset = "kitti360"
    assert prep_ncore.PrepNcoreStage.unavailable_reason(Manifest()) is None
    scene = tmp_path / "scene"
    scene.mkdir()
    monkeypatch.delenv(prep_ncore.RAW_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match="KITTI-360 gốc"):
        prep_ncore.raw_kitti360_dir(scene)
    raw = tmp_path / "kitti360"
    (raw / prep_ncore.ROSA_META).mkdir(parents=True)
    (raw / "data_poses.zip").write_bytes(b"")
    (scene / prep_ncore.RAW_SOURCE_FILE).write_text(json.dumps({"raw_dir": str(raw)}))
    assert prep_ncore.raw_kitti360_dir(scene) == raw
    command = prep_ncore.build_command(
        raw, tmp_path / "out", "2013_05_28_drive_0005_sync", 120, 90, ["image_00", "image_02"],
        prep_ncore.PrepNcoreParams(lidar=False), "seq",
    )  # fmt: skip
    joined = " ".join(command)
    assert "kitti360_to_ncore.py" in joined and "--start 120 --frames 90" in joined
    assert "--cameras image_00,image_02" in joined and command[-1] == "--no-lidar"


def test_train_nurec_command_and_environment(nurec_home: Path, tmp_path: Path) -> None:
    params = train_nurec.TrainNurecParams(
        model="pa-front", n_gaussians=200000, include_dynamic=True
    )
    sensors = {"image_00": "f", "image_02": "l", "image_03": "r"}
    assert train_nurec.context_cameras(sensors, "pa-multiview") == ["f", "l", "r"]
    assert train_nurec.context_cameras({"image_00": "f", "image_02": "l"}, "pa-multiview") == ["f"]
    assert train_nurec.context_cameras(sensors, "pa-front") == ["f"]
    for bad in ({"image_02": "l"}, sensors):
        with pytest.raises(ValueError):
            train_nurec.context_cameras(bad, "pq-front" if bad is sensors else "pa-front")
    command = train_nurec.build_command(params, tmp_path / "c.json", tmp_path / "out", ["f"])
    assert command[0] == str(nurec_home / ".venv" / "bin" / "python")
    assert "--merge" in command and "--include-dynamic" in command
    assert command[-2:] == ["--camera-id", "f"]
    assert command[command.index("--model") + 1] == "pa-front"
    with pytest.raises(ValueError):
        train_nurec.build_command(
            train_nurec.TrainNurecParams(model="pq-front", include_dynamic=True),
            tmp_path / "c.json",
            tmp_path / "out",
            [],
        )
    auto = train_nurec.TrainNurecParams()
    assert train_nurec.chunks_for(auto, 120) == 7 and train_nurec.chunks_for(auto, 10) == 1
    assert train_nurec.chunks_for(auto, 1000) == 8
    assert train_nurec.chunks_for(train_nurec.TrainNurecParams(max_chunks=2), 120) == 2
    env = train_nurec.nurec_env(tmp_path / "w.pth")
    assert env["INSTANT_NUREC_FULL_PT"].endswith("w.pth") and env["HF_HUB_OFFLINE"] == "1"


def test_test_frames_map_to_clip_indices() -> None:
    clip = {
        "frame_ids": [100, 101, 102, 104, 105],
        "scene_frame_ids": [100, 101, 102, 103, 104, 105],
        "scene_frames": [10, 15],
    }
    # khung 13 của cảnh (số gốc 103) không có trong clip nên bị bỏ
    assert clip_test_indices(clip, [11, 13, 15, 15, 99]) == [1, 4]


def test_summary_from_render_metrics() -> None:
    metrics = {
        "camera_front_left_00": {
            "psnr_mean": 25.0, "ssim_mean": 0.8, "lpips_mean": 0.2, "views": 4,
        },
        "_summary": {
            "gaussians": 1000, "render_fps": 30.0, "peak_vram_gib": 2.0,
            "psnr_mean": 24.0, "ssim_mean": 0.7, "lpips_mean": 0.3,
            "render_resolution": "1408x376",
        },
    }  # fmt: skip
    summary = make_summary(
        metrics, "every4", {"image_00": "camera_front_left_00"}, {"train_time_s": 12.5}
    )
    assert summary.overall.psnr == 24.0 and summary.per_camera["image_00"].lpips == 0.2
    assert summary.peak_vram_mb == 2048 and summary.train_time_s == 12.5
    assert summary.num_gaussians == 1000 and summary.render_resolution == "1408x376"


def test_remote_executors_refuse_external_codebases() -> None:
    with pytest.raises(ValueError, match="codebase ngoài"):
        remote.require_main_codebase("nurec", "Colab")
    remote.require_main_codebase("streetgs", "Kaggle")


def test_install_flow_knows_nurec_and_skips_python_version(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "envs").mkdir(parents=True)
    for name in ("install_nurec.sh", "install_main.sh"):
        (repo / "envs" / name).write_text("exit 0\n")
    flow = env_install.InstallFlow(
        tmp_path / "logs", repo=repo, gpus=lambda: [("GPU", 1, "1")], python_version=(3, 12)
    )
    assert flow.preflight("nurec") == repo / "envs" / "install_nurec.sh"
    with pytest.raises(ValueError, match="Python 3.10"):
        flow.preflight("main")


def test_notebook_route(tmp_path: Path) -> None:
    app = create_app(Workdir(tmp_path), registry=StageRegistry(), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/environment/nurec/notebook/colab")
        assert response.status_code == 200
        notebook = json.loads(response.content)
        assert notebook["cells"] and "instant-nurec" in json.dumps(notebook)
        assert client.get("/api/environment/nurec/notebook/other").status_code == 400


def test_ply_transform_moves_points_and_orientations(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    from drive2sim.external.nurec import ply

    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    names += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(2, dtype=[(n, "<f4") for n in names])
    data["x"], data["y"], data["z"] = [1, 0], [0, 2], [0, 0]
    half = math.sqrt(0.5)
    data["rot_0"], data["rot_3"] = [1, half], [0, half]  # đơn vị; xoay 90° quanh z
    header = ["ply", "format binary_little_endian 1.0", "element vertex 2"]
    header += [f"property float {n}" for n in names] + ["end_header"]
    path = tmp_path / "a.ply"
    ply.write_ply(path, header, data)
    read_header, read = ply.read_ply(path)
    assert read_header == header and np.array_equal(read, data)
    # xoay 90° quanh z rồi dịch (0, 0, 5)
    transform = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]], dtype=np.float64
    )
    moved = ply.transform_ply(read, transform)
    assert np.allclose(np.stack([moved["x"], moved["y"], moved["z"]], 1), [[0, 1, 5], [-2, 0, 5]])
    q = np.stack([moved[f"rot_{i}"] for i in range(4)], 1)
    assert np.allclose(q[0], [half, 0, 0, half], atol=1e-6)  # đơn vị thành 90° quanh z
    assert np.allclose(q[1], [0, 0, 0, 1], atol=1e-6)  # 90° + 90° = 180° quanh z
    with pytest.raises(ValueError):
        ply.transform_ply(read, np.diag([2.0, 1, 1, 1]))
    # rig FLU → camera OpenCV → Velodyne: trục tới của rig (x) thành trục tới của camera (z)
    cam0 = np.eye(4)
    assert np.allclose(ply.ncore_to_world(cam0) @ [1, 0, 0, 0], [0, 0, 1, 0])
