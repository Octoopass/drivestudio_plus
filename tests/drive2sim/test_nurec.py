"""Test approach Instant NuRec: môi trường, stage (câu lệnh, ánh xạ khung, summary), đổi hệ PLY."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from drive2sim.engine import env_install, remote
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.environment import environment_check
from drive2sim.engine.registry import StageRegistry, load_builtin_stages
from drive2sim.engine.workdir import Workdir
from drive2sim.external import nurec
from drive2sim.stages.eval.ncore import (
    clip_test_indices,
    clip_test_views,
    fisheye_max_angles,
    make_summary,
    mean_coverage,
)
from drive2sim.stages.prep import ncore as prep_ncore
from drive2sim.stages.train import nurec as train_nurec
from drive2sim.stages.train.drivestudio import RENDERER_MAX_ANGLE
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
        "prep.ncore", "prep.masks_segformer", "train.nurec", "train.nurec_refine", "eval.ncore",
        "export.nurec_ply", "render.nurec_video",
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
    drive_0006 = list(range(5697, 5697 + 3872 + 1, 5))  # 387,2 s: NuRec báo 29 đoạn
    assert train_nurec.chunks_for(auto, drive_0006) == 29
    assert train_nurec.chunks_for(auto, list(range(0, 50, 5))) == 1
    assert train_nurec.chunks_for(auto, list(range(0, 20000, 5))) == train_nurec.MAX_CHUNKS
    assert train_nurec.chunks_for(train_nurec.TrainNurecParams(max_chunks=2), drive_0006) == 2
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


def test_clip_test_views_follow_split_pairs() -> None:
    clip = {
        "frame_ids": [10, 12, 14],
        "scene_frame_ids": [10, 11, 12, 13, 14],
        "scene_frames": [0, 4],
    }
    sensors = {"image_00": "cam0", "image_02": "cam2"}
    views = clip_test_views(
        clip,
        [(2, "image_00"), (2, "image_02"), (4, "image_02"), (1, "image_00"), (2, "image_01")],
        sensors,
        {"image_00": 0, "image_01": 1, "image_02": 2},
    )
    # khung 1 (gốc 11) không có trong clip; image_01 không thuộc split
    assert views == {"cam0": {"1": "002_0"}, "cam2": {"1": "002_2", "2": "004_2"}}


def test_fisheye_max_angles_capped_by_renderer() -> None:
    camera = SimpleNamespace
    manifest = SimpleNamespace(
        cameras=[
            camera(camera_id="image_00", model="opencv_pinhole", max_angle=None),
            camera(camera_id="image_02", model="opencv_fisheye", max_angle=1.4),
            camera(camera_id="image_03", model="opencv_fisheye", max_angle=1.7),
        ]
    )
    sensors = {"image_00": "c0", "image_02": "c2", "image_03": "c3"}
    assert fisheye_max_angles(manifest, sensors) == {"c2": 1.4, "c3": RENDERER_MAX_ANGLE}


def test_summary_has_region_psnr() -> None:
    metrics = {
        "c0": {"psnr_mean": 20.0, "ssim_mean": 0.7, "lpips_mean": 0.3, "views": 2},
        "_summary": {
            "psnr_mean": 20.0, "ssim_mean": 0.7, "lpips_mean": 0.3,
            "occupied_psnr": 21.0, "vehicle_psnr": 15.0, "coverage": 0.8,
        },
    }  # fmt: skip
    overall = make_summary(metrics, "every8", {"image_00": "c0"}).overall
    assert overall.occupied_psnr == 21.0 and overall.vehicle_psnr == 15.0
    assert overall.human_psnr is None and overall.dynamic_psnr is None


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
    # XSIM bị tắt (approaches/xsim.yaml, disabled): không đẩy lên Kaggle, Colab nữa.
    with pytest.raises(ValueError, match="đã tắt"):
        remote.require_remote_codebase("xsim", "Kaggle")


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


def test_raise_open_file_limit_reaches_hard_limit() -> None:
    resource = pytest.importorskip("resource")
    from drive2sim.stages.commands import raise_open_file_limit

    raise_open_file_limit()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft == hard or soft >= 1048576 or soft == resource.RLIM_INFINITY


def test_live_log_progress_reads_last_tqdm_bar() -> None:
    from drive2sim.studio.api.routes.runs import log_progress

    text = (
        "x\nPredicting in chunks:  25%|██▌       | 2/8 [00:02<00:05,  1.01it/s]\n"
        "Predicting in chunks:  62%|██████▎   | 5/8 [00:04<00:02,  1.25it/s]\n"
    )
    assert log_progress(text) == {"label": "Predicting in chunks", "done": 5, "total": 8}
    assert log_progress("no bar here") is None


def test_gpu_render_not_supported_for_nurec_runs(tmp_path: Path) -> None:
    from drive2sim.engine.render_check import gpu_render_supported

    stages = tmp_path / "d2s" / "stages"
    (stages / "train.nurec").mkdir(parents=True)
    assert not gpu_render_supported(tmp_path)
    (stages / "train.pvg").mkdir()
    assert gpu_render_supported(tmp_path)


def test_nurec_parts_are_combined_not_overwritten(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    from drive2sim.external.nurec import ply

    source = (nurec.TOOLS_DIR / "nurec_predict.py").read_text(encoding="utf-8")
    namespace: dict = {"Path": Path}
    exec(source[source.index("PART_TAG =") : source.index("\n\n\ndef main")], namespace)  # noqa: S102
    sequence = tmp_path / "run" / "ply" / "clip_0_9"
    sequence.mkdir(parents=True)
    header = ["ply", "format binary_little_endian 1.0", "element vertex 2",
              "property float x", "end_header"]  # fmt: skip
    for part in range(2):
        points = np.array([(part,), (part + 10,)], [("x", "<f4")])
        ply.write_ply(sequence / "clip_0_9.ply", header, points)
        (sequence / "clip_0_9.sky.npz").write_bytes(b"s")
        namespace["rename_outputs"](tmp_path / "run" / "ply", part)
    names = sorted(p.name for p in sequence.iterdir())
    assert names == [f"clip_0_9_part0{i}.{ext}" for i in (0, 1) for ext in ("ply", "sky.npz")]
    parts = train_nurec._part_plies(tmp_path)
    assert [p.name for p in parts] == ["clip_0_9_part00.ply", "clip_0_9_part01.ply"]
    train_nurec.combine_parts(parts, tmp_path / "scene.ply")
    _, data = ply.read_ply(tmp_path / "scene.ply")
    assert sorted(data["x"].tolist()) == [0, 1, 10, 11]


def test_nurec_accepts_custom_gaussian_count() -> None:
    approach = load_approach("nurec")
    spec = approach.options["n_gaussians"]
    assert spec.accepts(1000000) and spec.accepts(3500000)
    assert not spec.accepts(50000) and not spec.accepts(9000000)
    assert not spec.accepts(1500000.5) and not spec.accepts(True) and not spec.accepts("2000000")
    assert train_nurec.TrainNurecParams(n_gaussians=3500000).n_gaussians == 3500000
    assert spec.accepts(0)  # Không giới hạn


def test_summary_preserves_scoring_protocol_and_coverage() -> None:
    metrics = {
        "_summary": {
            "psnr_mean": 20.0,
            "ssim_mean": 0.7,
            "lpips_mean": 0.3,
            "protocol_version": "0.5",
            "coverage": 0.8,
        }
    }
    summary = make_summary(metrics, "every8", {}, coverage=0.2)
    assert summary.protocol.version == "0.5"
    assert summary.coverage == 0.8


def test_shifted_metrics_exclude_held_out_camera_and_keep_color_matched_values() -> None:
    metrics = {
        "front": {"psnr_mean": 30.0, "ssim_mean": 0.9, "lpips_mean": 0.1, "views": 2},
        "right": {"psnr_mean": 10.0, "ssim_mean": 0.3, "lpips_mean": 0.7, "views": 3},
        "_summary": {
            "psnr_mean": 18.0,
            "ssim_mean": 0.54,
            "lpips_mean": 0.46,
            "dynamic_psnr": 999.0,
            "protocol_version": "0.6",
        },
    }
    per_view = [
        {"camera": "front", "dynamic_psnr": 26.0},
        {"camera": "right", "dynamic_psnr": 5.0, "psnr_cc": 24.0, "lpips_cc": 0.2},
        {"camera": "right", "dynamic_psnr": 5.0, "psnr_cc": None, "lpips_cc": None},
    ]
    result = make_summary(
        metrics,
        "every8",
        {"image_00": "front", "image_01": "right"},
        test_only_cameras=("image_01",),
        per_view=per_view,
    )
    assert result.overall.psnr == 30 and result.overall.dynamic_psnr == 26
    assert result.shifted.cameras == ["image_01"] and result.shifted.num_images == 3
    assert result.shifted.psnr == 10 and result.shifted.psnr_cc == 24
    assert result.shifted.lpips_cc == 0.2 and len(result.per_camera) == 2


def test_nurec_rejects_reference_camera_reserved_for_test() -> None:
    split = SimpleNamespace(test_only_cameras=["image_00"])
    assert "image_00" in train_nurec.TrainNurecStage.unavailable_reason(SimpleNamespace(), split)


def test_nurec_context_command_does_not_use_test_only_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "clip"
    source.mkdir()
    (source / "clip.json").write_text(
        json.dumps(
            {
                "clip": "sequence.json",
                "frame_ids": list(range(100)),
                "sensors": {"image_00": "front", "image_02": "left", "image_03": "right"},
            }
        )
    )
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(nurec, "checkpoint_path", lambda model: checkpoint)

    def stop_after_command(command: list[str], *args: object, **kwargs: object) -> None:
        cameras = command[command.index("--camera-id") + 1]
        assert "right" not in cameras and cameras == "front"
        raise RuntimeError("captured command")

    monkeypatch.setattr(train_nurec, "run_logged", stop_after_command)
    ctx = SimpleNamespace(
        output_dir=tmp_path,
        split=SimpleNamespace(test_only_cameras=["image_03"]),
        progress=lambda *args: None,
        should_stop=lambda: False,
    )
    with pytest.raises(RuntimeError, match="captured command"):
        train_nurec.TrainNurecStage().run(ctx, {"clip": source}, train_nurec.TrainNurecParams())


def test_refinement_uses_train_pairs_with_dense_held_out_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drive2sim.stages.train import nurec_refine

    source = tmp_path / "clip"
    model = tmp_path / "model"
    out = tmp_path / "out"
    for directory in (source, model, out):
        directory.mkdir()
    (model / "scene_nurec.ply").write_bytes(b"ply")
    (source / "clip.json").write_text(
        json.dumps(
            {
                "clip": "sequence.json",
                "frame_ids": [10, 11, 12],
                "scene_frame_ids": [10, 11, 12],
                "scene_frames": [0, 2],
                "sensors": {"image_00": "front", "image_01": "right"},
            }
        )
    )
    commands = []

    def fake_run(command: list[str], *args: object, **kwargs: object) -> None:
        commands.append(command)
        if "--views-json" in command:
            plan = json.loads(Path(command[command.index("--views-json") + 1]).read_text())
            assert plan == {"front": {"0": "000_0", "2": "002_0"}}
            assert command[command.index("--cameras") + 1] == "front"
            assert command[command.index("--frame-indices") + 1] == "0,2"
        else:
            (out / "refine.json").write_text(
                json.dumps({"refine_time_s": 1.0, "gaussians_after": 10})
            )

    monkeypatch.setattr(nurec_refine, "run_logged", fake_run)
    split = SimpleNamespace(
        cameras=["image_00", "image_01"],
        test_only_cameras=["image_01"],
        train=[(0, "image_00"), (2, "image_00")],
        test=[(1, "image_00"), (0, "image_01"), (1, "image_01"), (2, "image_01")],
    )
    ctx = SimpleNamespace(
        output_dir=out,
        split=split,
        scene=SimpleNamespace(
            manifest=SimpleNamespace(cameras=[SimpleNamespace(camera_id=c) for c in split.cameras])
        ),
        progress=lambda *args: None,
        report_peak_vram=lambda *args: None,
        should_stop=lambda: False,
    )
    nurec_refine.TrainNurecRefineStage().run(
        ctx, {"clip": source, "model": model}, nurec_refine.RefineParams(iters=1)
    )
    assert len(commands) == 2


def test_official_ncore_mask_overrides_estimation_and_validates_size(tmp_path: Path) -> None:
    import ast

    image = pytest.importorskip("PIL.Image")
    source = (nurec.TOOLS_DIR / "kitti360_to_ncore.py").read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "official_ego_mask"
    )
    namespace = {"Image": image, "Path": Path, "Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "mask_helper", "exec"), namespace)
    read = namespace["official_ego_mask"]
    assert read(None, "image_02", 32, 24) is None
    assert read(tmp_path, "image_02", 32, 24).getextrema() == (0, 0)
    image.new("L", (32, 24), 255).save(tmp_path / "image_02.png")
    assert read(tmp_path, "image_02", 32, 24).getextrema() == (255, 255)
    with pytest.raises(ValueError, match="kích thước"):
        read(tmp_path, "image_02", 64, 24)


def test_unlimited_gaussians_skips_voxelization(nurec_home: Path, tmp_path: Path) -> None:
    params = train_nurec.TrainNurecParams(n_gaussians=0)
    command = train_nurec.build_command(params, tmp_path / "c.json", tmp_path / "out", ["cam"])
    assert train_nurec.NO_VOXELIZE in command and "--n-gaussians" not in command
    assert "--merge" in command
    limited = train_nurec.build_command(
        train_nurec.TrainNurecParams(n_gaussians=200000), tmp_path / "c.json", tmp_path, ["cam"]
    )
    assert limited[limited.index("--n-gaussians") + 1] == "200000"
    assert train_nurec.NO_VOXELIZE not in limited
    with pytest.raises(ValueError):
        train_nurec.TrainNurecParams(n_gaussians=50000)


def test_refine_command_passes_gpus(tmp_path: Path) -> None:
    from drive2sim.stages.train import nurec_refine

    params = nurec_refine.RefineParams(iters=1000, gpus=0, overlap_m=30)
    command = nurec_refine.refine_command(
        params, tmp_path / "a.ply", tmp_path / "v.npz", tmp_path / "o"
    )
    assert command[command.index("--gpus") + 1] == "0"
    assert command[command.index("--overlap-m") + 1] == "30.0"
    assert nurec_refine.RefineParams().gpus == 1  # mặc định một GPU như trước
