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
from drive2sim.stages.eval.ncore import clip_test_indices, make_summary, mean_coverage
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
        "prep.ncore", "train.nurec", "eval.ncore", "export.nurec_ply", "render.nurec_video",
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


def test_train_nurec_command_and_environment(
    nurec_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = train_nurec.nurec_env(tmp_path / "w.pth")
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
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
    assert summary.coverage is None


def test_mean_coverage() -> None:
    assert mean_coverage([{"coverage": 0.5}, {"coverage": 1.0}]) == 0.75
    assert mean_coverage([]) is None


def test_summary_has_coverage_and_drops_cameras_without_valid_pixels() -> None:
    nan = float("nan")
    metrics = {
        "_summary": {"psnr_mean": 20.0, "ssim_mean": 0.7, "lpips_mean": 0.3},
        "image_02": {"psnr_mean": nan, "ssim_mean": nan, "lpips_mean": nan},
        "image_03": {"psnr_mean": None, "ssim_mean": None, "lpips_mean": None},
    }
    summary = make_summary(
        metrics, "every8_x", {"image_02": "image_02", "image_03": "image_03"}, coverage=0.42
    )
    assert summary.coverage == 0.42 and summary.per_camera == {}
    assert "NaN" not in summary.model_dump_json()


def test_remote_executors_refuse_external_codebases() -> None:
    with pytest.raises(ValueError, match="codebase ngoài"):
        remote.require_remote_codebase("nurec", "Colab")
    assert remote.require_remote_codebase("streetgs", "Kaggle") is False
    assert remote.require_remote_codebase("xsim", "Kaggle") is True


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


def test_video_stage_command_and_wiring(tmp_path: Path) -> None:
    from drive2sim.engine.registry import REGISTRY, load_builtin_stages
    from drive2sim.stages.render.nurec_video import NurecVideoParams, build_render_command

    load_builtin_stages()
    cls = REGISTRY.get("render.nurec_video")
    assert cls.inputs == {"model": "model", "clip": "ncore_clip", "split": "split"}
    command = build_render_command(
        NurecVideoParams(), tmp_path / "s.ply", tmp_path / "v.npz", tmp_path / "f.mp4",
        ["camera_front_left_00", "camera_fisheye_left_02"],
    )  # fmt: skip
    assert command[2].endswith("ncore_render_video.py") and "--no-gt" not in command
    assert float(command[command.index("--fps") + 1]) == 30 and "--interp" in command
    assert float(command[command.index("--speed") + 1]) == 1  # mặc định: tốc độ gốc
    assert command[command.index("--cameras") + 1] == "camera_front_left_00,camera_fisheye_left_02"
    assert "--no-gt" in build_render_command(
        NurecVideoParams(with_gt=False, fps=5), tmp_path / "s.ply", tmp_path / "v.npz",
        tmp_path / "f.mp4", ["c"],
    )  # fmt: skip
    with pytest.raises(ValueError):
        NurecVideoParams(fps=0)


def test_nurec_runs_on_the_freest_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GPU dùng chung: NuRec chạy trên GPU còn trống nhiều nhất, trừ khi đã đặt sẵn."""
    import subprocess

    from drive2sim.engine import machine

    out = "0, 1200\n1, 25000\n2, 24000\n"
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""),
    )  # fmt: skip
    assert machine.freest_gpu() == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(train_nurec.nurec, "nurec_python", lambda: tmp_path / "bin" / "python")
    assert train_nurec.nurec_env(tmp_path / "w.pth")["CUDA_VISIBLE_DEVICES"] == "1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert "CUDA_VISIBLE_DEVICES" not in train_nurec.nurec_env(tmp_path / "w.pth")
    out = "0, 1200\n"
    assert machine.freest_gpu() is None  # một GPU: không cần chọn


def test_missing_raw_files_lists_absent_archives(tmp_path: Path) -> None:
    from drive2sim.stages.prep.ncore import missing_raw_files

    seq = "2013_05_28_drive_0003_sync"
    (tmp_path / "data_2d_raw").mkdir()
    (tmp_path / "data_2d_raw" / f"{seq}_image_00.tar").write_bytes(b"")
    assert missing_raw_files(tmp_path, seq, ["image_00", "image_02"], lidar=True) == [
        f"data_2d_raw/{seq}_image_02.tar",
        f"data_3d_raw/{seq}.tar",
    ]
    assert missing_raw_files(tmp_path, seq, ["image_00"], lidar=False) == []


def test_render_command_passes_dense_gt(tmp_path: Path) -> None:
    from drive2sim.stages.render.nurec_video import (
        NurecVideoParams,
        build_render_command,
        dense_gt_command,
    )

    args = (tmp_path / "s.ply", tmp_path / "v.npz", tmp_path / "f.mp4", ["cam"])
    dense = tmp_path / "gt_dense.npz"
    command = build_render_command(NurecVideoParams(), *args, dense)
    assert command[command.index("--gt-dense") + 1] == str(dense)
    assert "--gt-dense" not in build_render_command(NurecVideoParams(with_gt=False), *args, dense)
    assert "--gt-dense" not in build_render_command(NurecVideoParams(), *args)
    assert "--gt-flow" in command
    assert "--gt-flow" not in build_render_command(NurecVideoParams(gt_flow=False), *args)
    assert "--gt-flow" not in build_render_command(NurecVideoParams(interpolate=False), *args)
    assert command[command.index("--gt-flow-window") + 1] == "0.2"
    assert "--gt-flow" not in build_render_command(NurecVideoParams(gt_flow_window=0), *args)
    tool = dense_gt_command(tmp_path, "seq", tmp_path / "clip.json", dense, 360)
    assert tool[2].endswith("kitti360_dense_gt.py") and "--sequence" in tool


def test_dense_gt_between_skips_clip_frames() -> None:
    source = (nurec.TOOLS_DIR / "kitti360_dense_gt.py").read_text(encoding="utf-8")
    namespace: dict = {}
    exec(source[source.index("def between") : source.index("\n\n\ndef main")], namespace)  # noqa: S102
    assert namespace["between"]([102, 107, 112, 113]) == [103, 104, 105, 106, 108, 109, 110, 111]


def test_ply_export_stage_uses_nurec_ply_when_present() -> None:
    from drive2sim.engine.run import ply_export_stage

    assert ply_export_stage(["train.nurec", "export.nurec_ply"]) == "export.nurec_ply"
    assert ply_export_stage(["train.pvg", "export.ply_world"]) == "export.ply_world"
    assert ply_export_stage([]) == "export.ply_world"
