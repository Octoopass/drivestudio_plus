"""Kiểm tra các stage của approach ``mcgs``: ``prep.mcgs``, ``train.mcgs`` và ``eval.mcgs``.

- ``prep.mcgs``: xuất ảnh pinhole không có khung test, loại camera chỉ test và camera không phải
  pinhole, bỏ khung thiếu ảnh ở mọi thư mục, ghi ``calib.yml`` và ``mcgs_prep.json``.
- ``train.mcgs``: dựng câu lệnh ``demo.py``, đọc số Gaussian trong PLY.
- ``eval.mcgs``: lấy ``held_out`` và camera tham chiếu từ ``mcgs_prep.json``, tách camera chỉ test.
Cảnh dựng giả nhỏ, không cần GPU.
"""

import json
from pathlib import Path
from typing import Optional

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
yaml = pytest.importorskip("yaml")

from drive2sim.core.schema import Manifest, Split, Summary  # noqa: E402
from drive2sim.engine.stage import Scene  # noqa: E402
from drive2sim.external import mcgs  # noqa: E402
from drive2sim.stages.eval import mcgs as eval_mcgs  # noqa: E402
from drive2sim.stages.prep import mcgs as prep_mcgs  # noqa: E402
from drive2sim.stages.train import mcgs as train_mcgs  # noqa: E402
from tests.drive2sim.samples import MANIFEST  # noqa: E402

FRAMES = 17
TEST_FRAMES = (8, 16)
SIZE = (24, 32)  # cao, rộng của ảnh giả
PINHOLE = [30.0, 30.0, 16.0, 12.0, 0, 0, 0, 0, 0]
FISHEYE = [20.0, 20.0, 16.0, 12.0, 0.01, -0.002, 0.0, 0.0]


def camera_entry(camera_id: str, model: str) -> dict:
    return {
        "camera_id": camera_id,
        "model": model,
        "resolution": [SIZE[1], SIZE[0]],
        "shutter": "GLOBAL",
    }


class Context:
    """Ngữ cảnh stage giả: thư mục ra, cảnh, split."""

    def __init__(self, output_dir: Path, scene: Scene, split: Split) -> None:
        self.output_dir, self.scene, self.split = output_dir, scene, split
        self.events: list[str] = []

    def progress(self, fraction: float, message: str = "") -> None:
        """Ghi lại thông điệp tiến độ."""
        self.events.append(message)

    def should_stop(self) -> bool:
        """Không bao giờ huỷ."""
        return False


def make_scene(
    root: Path, cameras: Optional[list[tuple[str, str]]] = None, test_only: tuple[str, ...] = ()
) -> tuple[Scene, Split]:
    """Cảnh giả 17 khung; mặc định hai camera pinhole ``image_00``, ``image_01`` cách nhau 0,59 m.

    Ảnh tên ``<khung:03d>_<chỉ số camera>.png`` như cảnh chuẩn; split ``every_nth_frame:8`` có
    khung test 8 và 16 cho mọi camera, camera chỉ test không có ảnh train.
    """
    cameras = cameras or [("image_00", "opencv_pinhole"), ("image_01", "opencv_pinhole")]
    for name in ("extrinsics", "intrinsics", "images"):
        (root / name).mkdir(parents=True)
    for index, (_, model) in enumerate(cameras):
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [0.59 * index, 0.0, 1.5]
        np.savetxt(root / "extrinsics" / f"{index}.txt", extrinsic)
        np.savetxt(
            root / "intrinsics" / f"{index}.txt", FISHEYE if "fisheye" in model else PINHOLE
        )
        for frame in range(FRAMES):
            image = np.full((*SIZE, 3), frame * 10 + index, dtype=np.uint8)
            cv2.imwrite(str(root / "images" / f"{frame:03d}_{index}.png"), image)
    manifest = Manifest.model_validate(
        dict(
            MANIFEST,
            scene_id="kitti360_0003",
            source=dict(MANIFEST["source"], dataset="kitti360"),
            cameras=[camera_entry(*camera) for camera in cameras],
            timeline={"reference_sensor": cameras[0][0], "num_frames": FRAMES},
        )
    )
    ids = [camera for camera, _ in cameras]
    trained = [camera for camera in ids if camera not in test_only]
    split = Split.model_validate(
        {
            "schema_version": "d2s-split/0.2",
            "split_id": "every8_x",
            "scene_id": "kitti360_0003",
            "strategy": "every_nth_frame:8",
            "frame_range": [0, FRAMES - 1],
            "cameras": ids,
            "test_only_cameras": list(test_only),
            "train": [[f, c] for f in range(FRAMES) if f not in TEST_FRAMES for c in trained],
            "test": [[f, c] for f in TEST_FRAMES for c in ids],
        }
    )
    return Scene("kitti360_0003", root, manifest), split


def run_prep(
    tmp_path: Path,
    rig: Optional[list[tuple[str, str]]] = None,
    test_only: tuple[str, ...] = (),
    **params: object,
) -> Path:
    """Chạy ``prep.mcgs`` trên cảnh giả ``rig``; ``params`` là tham số của stage."""
    scene, split = make_scene(tmp_path / "scene", rig, test_only)
    out = tmp_path / "out"
    out.mkdir()
    stage = prep_mcgs.PrepMcgsStage()
    result = stage.run(Context(out, scene, split), {}, prep_mcgs.PrepMcgsParams(**params))
    assert result == {"data": out}
    return out


def names_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def read_prep(out: Path) -> dict:
    return json.loads((out / "mcgs_prep.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- prep.mcgs


def test_held_out_frames_are_not_exported(tmp_path: Path) -> None:
    out = run_prep(tmp_path, held_out=True)
    names = names_in(out / "images" / "image_00")
    assert "000008.png" not in names and "000016.png" not in names and len(names) == 15
    assert names == names_in(out / "images" / "image_01")
    prep = read_prep(out)
    assert prep["held_out"] is True and prep["dropped_frames"] == []
    assert prep["frames"] == [f for f in range(FRAMES) if f not in TEST_FRAMES]


def test_held_out_false_keeps_test_frames(tmp_path: Path) -> None:
    out = run_prep(tmp_path, held_out=False)
    assert len(names_in(out / "images" / "image_00")) == FRAMES
    assert read_prep(out)["held_out"] is False


def test_exported_images_are_the_scene_images_named_by_frame(tmp_path: Path) -> None:
    out = run_prep(tmp_path)
    exported = cv2.imread(str(out / "images" / "image_01" / "000003.png"))
    assert exported is not None and int(exported[0, 0, 0]) == 3 * 10 + 1


def test_test_only_camera_is_left_out(tmp_path: Path) -> None:
    cameras = [
        ("image_00", "opencv_pinhole"),
        ("image_01", "opencv_pinhole"),
        ("image_02", "opencv_pinhole"),
    ]
    out = run_prep(tmp_path, cameras, test_only=("image_01",))
    prep = read_prep(out)
    assert prep["layout"] == ["image_00", "image_02", "image_02"] and prep["cameras"] == [
        "image_00",
        "image_02",
    ]
    assert names_in(out / "images") == ["image_00", "image_02"]


def test_test_only_camera_in_params_is_refused(tmp_path: Path) -> None:
    rig = [
        ("image_00", "opencv_pinhole"),
        ("image_01", "opencv_pinhole"),
        ("image_02", "opencv_pinhole"),
    ]
    with pytest.raises(ValueError, match="chỉ test"):
        run_prep(tmp_path, rig, test_only=("image_01",), cameras=["image_00", "image_01"])


def test_frame_missing_one_camera_is_dropped_everywhere(tmp_path: Path) -> None:
    scene, split = make_scene(tmp_path / "scene")
    (tmp_path / "scene" / "images" / "005_1.png").unlink()
    out = tmp_path / "out"
    out.mkdir()
    prep_mcgs.PrepMcgsStage().run(Context(out, scene, split), {}, prep_mcgs.PrepMcgsParams())
    assert read_prep(out)["dropped_frames"] == [5]
    for camera in ("image_00", "image_01"):
        assert not any(name.startswith("000005") for name in names_in(out / "images" / camera))
    assert len(names_in(out / "images" / "image_00")) == 14


def test_fisheye_camera_in_params_is_refused(tmp_path: Path) -> None:
    rig = [
        ("image_00", "opencv_pinhole"),
        ("image_01", "opencv_pinhole"),
        ("image_02", "opencv_fisheye"),
    ]
    with pytest.raises(ValueError, match="pinhole"):
        run_prep(tmp_path, rig, cameras=["image_00", "image_02"])


def test_fisheye_camera_of_the_split_is_skipped_by_default(tmp_path: Path) -> None:
    cameras = [
        ("image_00", "opencv_pinhole"),
        ("image_01", "opencv_pinhole"),
        ("image_02", "opencv_fisheye"),
    ]
    layout = read_prep(run_prep(tmp_path, cameras))["layout"]
    assert layout == ["image_00", "image_01", "image_01"]


def test_single_pinhole_camera_is_refused(tmp_path: Path) -> None:
    cameras = [("image_00", "opencv_pinhole"), ("image_02", "opencv_fisheye")]
    with pytest.raises(ValueError, match="ít nhất"):
        run_prep(tmp_path, cameras)


def test_calib_file_parses(tmp_path: Path) -> None:
    out = run_prep(tmp_path)
    calib = yaml.safe_load((out / "calib.yml").read_text(encoding="utf-8"))
    assert {"intrinsic", "camera", "baseline", "T_cami_cam0", "timescale"} <= set(calib)
    assert calib["camera"] == "pinhole" and calib["timescale"] == 1
    assert calib["baseline"] == pytest.approx([-0.59, 0, 0, 0, 0, 0, 1])
    # đối tác stereo lặp ở cuối (MCGS-SLAM chỉ chạy nhánh nhiều camera): hàng 1 = baseline
    assert calib["T_cami_cam0"][0] == [0, 0, 0, 0, 0, 0, 1]
    assert calib["T_cami_cam0"][1] == pytest.approx(calib["baseline"])
    assert calib["intrinsic"] == [[30.0, 30.0, 16.0, 12.0]] * 3
    assert read_prep(out)["reference_camera"] == "image_00"


def test_reference_and_partner_are_set_by_params(tmp_path: Path) -> None:
    cameras = [
        ("image_00", "opencv_pinhole"),
        ("image_01", "opencv_pinhole"),
        ("image_02", "opencv_pinhole"),
    ]
    out = run_prep(tmp_path, cameras, reference_camera="image_01", stereo_partner="image_02")
    prep = read_prep(out)
    assert prep["layout"] == ["image_01", "image_02", "image_00", "image_02"]
    assert prep["reference_camera"] == "image_01"
    calib = yaml.safe_load((out / "calib.yml").read_text(encoding="utf-8"))
    assert len(calib["intrinsic"]) == 4 and len(calib["T_cami_cam0"]) == 3


# ---------------------------------------------------------------- train.mcgs


def write_prep(directory: Path, layout: list[str]) -> None:
    """Thư mục ``mcgs_input`` giả: ``calib.yml`` và ``mcgs_prep.json`` đủ cho ``train.mcgs``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "calib.yml").write_text("camera: pinhole\n", encoding="utf-8")
    prep = {"layout": layout, "held_out": True, "reference_camera": layout[0]}
    (directory / "mcgs_prep.json").write_text(json.dumps(prep), encoding="utf-8")


def test_build_command_full_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("D2S_MCGS_DIR", str(tmp_path / "mcgs"))
    write_prep(tmp_path / "in", ["image_00", "image_01"])
    cmd = train_mcgs.build_command(tmp_path / "in", tmp_path / "out", "full")
    assert cmd[1:3] == ["demo.py", "--calib"]
    assert cmd[3] == str(tmp_path / "in" / "calib.yml")
    assert cmd[cmd.index("--stride") + 1] == "1"
    assert cmd[-2:] == ["--prgbd", "--jdsa"]
    assert cmd[cmd.index("--imagedir") + 1].endswith("images/image_00")
    assert cmd[cmd.index("--output") + 1] == str(tmp_path / "out")


def test_build_command_repeats_the_partner_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("D2S_MCGS_DIR", str(tmp_path / "mcgs"))
    write_prep(tmp_path / "in", ["front", "right", "left", "right"])
    cmd = train_mcgs.build_command(tmp_path / "in", tmp_path / "out", "prgbd")
    start = cmd.index("--imagedir") + 1
    assert [Path(p).name for p in cmd[start : start + 4]] == ["front", "right", "left", "right"]
    assert cmd[-1] == "--prgbd" and "--jdsa" not in cmd


@pytest.mark.parametrize(
    "mode,flags",
    [("full", ["--prgbd", "--jdsa"]), ("prgbd", ["--prgbd"]), ("minimal", [])],
)
def test_mode_flags(mode: str, flags: list[str]) -> None:
    assert train_mcgs.MODE_FLAGS[mode] == flags


def test_ply_vertex_count(tmp_path: Path) -> None:
    ply = tmp_path / "a.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 172300\nend_header\n")
    assert train_mcgs.ply_vertex_count(ply) == 172300
    other = tmp_path / "b.ply"
    other.write_bytes(b"not a ply file")
    assert train_mcgs.ply_vertex_count(other) is None
    assert train_mcgs.ply_vertex_count(tmp_path / "missing.ply") is None


def fake_install(root: Path) -> None:
    """Giả một bản cài MCGS-SLAM: chỉ cần ``env/bin/python`` tồn tại."""
    python = root / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")


def train_context(tmp_path: Path) -> tuple[Context, Path]:
    scene, split = make_scene(tmp_path / "scene")
    out = tmp_path / "out"
    out.mkdir()
    data = tmp_path / "in"
    write_prep(data, ["image_00", "image_01"])
    return Context(out, scene, split), data


def test_missing_env_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("D2S_MCGS_DIR", str(tmp_path / "nowhere"))
    ctx, data = train_context(tmp_path)
    with pytest.raises(RuntimeError, match="install_mcgs.sh"):
        train_mcgs.TrainMcgsStage().run(ctx, {"data": data}, train_mcgs.TrainMcgsParams())


def test_train_runs_demo_and_writes_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("D2S_MCGS_DIR", str(tmp_path / "mcgs"))
    fake_install(tmp_path / "mcgs")
    ctx, data = train_context(tmp_path)
    calls: list[dict] = []

    def fake_run(command: list[str], log: Path, **kwargs: object) -> None:
        calls.append({"command": command, **kwargs})
        (ctx.output_dir / mcgs.OUTPUT_TRAJECTORY).write_text("0 0 0 0 0 0 0 1\n")
        (ctx.output_dir / mcgs.OUTPUT_GAUSSIANS).write_bytes(
            b"ply\nelement vertex 42\nend_header\n"
        )

    monkeypatch.setattr(train_mcgs, "run_logged", fake_run)
    params = train_mcgs.TrainMcgsParams(mode="prgbd")
    result = train_mcgs.TrainMcgsStage().run(ctx, {"data": data}, params)
    assert result == {"model": ctx.output_dir}
    assert calls[0]["cwd"] == mcgs.mcgs_dir()
    env = calls[0]["env"]
    assert env["PYOPENGL_PLATFORM"] == "egl" and env["MPLBACKEND"] == "Agg"
    stats = json.loads((ctx.output_dir / "train_stats.json").read_text(encoding="utf-8"))
    assert stats["num_gaussians"] == 42 and stats["mode"] == "prgbd"
    assert stats["train_time_s"] >= 0
    copied = json.loads((ctx.output_dir / "mcgs_prep.json").read_text(encoding="utf-8"))
    assert copied["held_out"] is True and copied["layout"] == ["image_00", "image_01"]


# ---------------------------------------------------------------- eval.mcgs


def test_prep_settings_fill_defaults(tmp_path: Path) -> None:
    (tmp_path / "mcgs_prep.json").write_text(
        json.dumps({"held_out": True, "reference_camera": "front"})
    )
    settings = eval_mcgs.prep_settings(tmp_path)
    assert settings["held_out"] is True
    assert eval_mcgs.resolve_settings(eval_mcgs.EvalMcgsParams(), settings) == (True, "front")


def test_old_model_dir_keeps_old_defaults(tmp_path: Path) -> None:
    assert eval_mcgs.prep_settings(tmp_path) == {}
    assert eval_mcgs.resolve_settings(eval_mcgs.EvalMcgsParams(), {}) == (False, "image_00")


def test_explicit_params_override_prep_settings() -> None:
    params = eval_mcgs.EvalMcgsParams(held_out=False, reference_camera="rear")
    prep = {"held_out": True, "reference_camera": "front"}
    assert eval_mcgs.resolve_settings(params, prep) == (False, "rear")


def view_record(frame: int, camera_index: int, psnr: Optional[float]) -> dict:
    return {
        "frame_index": frame,
        "camera_index": camera_index,
        "psnr": psnr,
        "ssim": None if psnr is None else 0.8,
        "lpips": None if psnr is None else 0.2,
        "coverage": 0.5,
    }


def summarize_with(split: Split, records: list[dict]) -> "Summary":
    metrics = {
        "image_00": {"psnr_mean": 23.0, "ssim_mean": 0.8, "lpips_mean": 0.2, "views": 2},
        "image_01": {"psnr_mean": 20.0, "ssim_mean": 0.8, "lpips_mean": 0.2, "views": 1},
        "_summary": {"psnr_mean": 22.0, "ssim_mean": 0.8, "lpips_mean": 0.2},
    }
    return eval_mcgs.summarize(metrics, records, split, ["image_00", "image_01"], None, 0.5)


def test_shifted_block_for_test_only_camera(tmp_path: Path) -> None:
    _, split = make_scene(tmp_path / "scene", test_only=("image_01",))
    records = [view_record(8, 0, 24.0), view_record(16, 0, 22.0), view_record(8, 1, 20.0)]
    summary = summarize_with(split, records)
    # R1 chỉ tính trên camera đã train, R2 trên camera chỉ test
    assert summary.overall.psnr == pytest.approx(23.0)
    assert summary.shifted is not None and summary.shifted.cameras == ["image_01"]
    assert summary.shifted.psnr == pytest.approx(20.0) and summary.shifted.num_images == 1
    assert summary.shifted.psnr_cc is None
    assert summary.coverage == 0.5


def test_no_test_only_camera_keeps_overall_from_metrics(tmp_path: Path) -> None:
    _, split = make_scene(tmp_path / "scene")
    records = [view_record(8, 0, 24.0), view_record(8, 1, 20.0)]
    summary = summarize_with(split, records)
    assert summary.shifted is None and summary.overall.psnr == pytest.approx(22.0)


def test_views_without_valid_pixels_are_left_out_of_the_means(tmp_path: Path) -> None:
    _, split = make_scene(tmp_path / "scene", test_only=("image_01",))
    records = [
        view_record(8, 0, 24.0),
        view_record(16, 0, None),
        view_record(8, 1, None),
        view_record(16, 1, None),
    ]
    summary = summarize_with(split, records)
    assert summary.overall.psnr == pytest.approx(24.0)
    assert summary.shifted is None  # camera chỉ test không có ảnh nào chấm được


def test_summarize_needs_a_scored_trained_view(tmp_path: Path) -> None:
    _, split = make_scene(tmp_path / "scene", test_only=("image_01",))
    with pytest.raises(ValueError, match="camera đã train"):
        summarize_with(split, [view_record(8, 0, None), view_record(8, 1, 20.0)])


def test_frame_records_carry_coverage_and_null_psnr() -> None:
    views_info = [
        {"frame": 8, "camera_index": 0, "position": [1.0, 2.0, 3.0]},
        {"frame": 8, "camera_index": 1, "position": [1.0, 2.5, 3.0]},
    ]
    scored = [
        {"camera": "a", "psnr": 24.0, "ssim": 0.8, "lpips": 0.2, "coverage": 0.7},
        {"camera": "b", "psnr": None, "ssim": None, "lpips": None, "coverage": 0.0},
    ]
    rows = eval_mcgs.frame_records(views_info, scored)
    assert [r["coverage"] for r in rows] == [0.7, 0.0] and rows[1]["psnr"] is None
    assert rows[0]["frame_index"] == 8 and rows[0]["depth"] is None and rows[0]["time"] is None


def test_train_fails_when_demo_writes_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("D2S_MCGS_DIR", str(tmp_path / "mcgs"))
    fake_install(tmp_path / "mcgs")
    ctx, data = train_context(tmp_path)
    monkeypatch.setattr(train_mcgs, "run_logged", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match=mcgs.OUTPUT_GAUSSIANS):
        train_mcgs.TrainMcgsStage().run(ctx, {"data": data}, train_mcgs.TrainMcgsParams())
