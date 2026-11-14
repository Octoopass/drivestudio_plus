"""Kiểm tra tầng dữ liệu: đọc kích thước ảnh, nhận dạng nguồn, manifest, kiểm tra, ingest.

Không có dữ liệu nuScenes thật trên máy test, nên file này dựng:
- ảnh JPEG và PNG giả ở mức byte (chỉ phần đầu file chứa kích thước);
- một thư mục nuScenes gốc giả chỉ gồm các file JSON mà bước nhận dạng đọc;
- một thư mục cảnh DriveStudio giả theo đúng khung mà script nuScenes ghi ra.

Script tiền xử lý thật được thay bằng một hàm giả ghi ra thư mục cảnh đó, nên test kiểm tra được
mọi bước của ingest trừ chính việc chuyển dữ liệu của DriveStudio. Cuối file có một test chạy
trọn đường qua CLI: ingest, tạo split, rồi ``approach check omnire``.
"""

import json
import struct
from pathlib import Path

import pytest

from drive2sim.cli.main import main
from drive2sim.core import files
from drive2sim.core.schema import Manifest, ValidationReport
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest import drivestudio
from drive2sim.ingest.detect import inspect_source
from drive2sim.ingest.dispatch import IngestRequestError, IngestTarget, plan_ingest, run_ingest
from drive2sim.ingest.drivestudio import (
    SOURCES,
    IngestError,
    build_command,
    default_scene_id,
    find_scene_dir,
    ingest_drivestudio,
)
from drive2sim.ingest.images import read_image_size
from drive2sim.ingest.manifest import build_manifest
from drive2sim.ingest.validate import validate_scene
from drive2sim.paths import SCENE_LISTS_DIR

NUSCENES_CAMERAS = list(SOURCES["nuscenes"].cameras)


def write_png(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0" * 4)


def write_jpeg(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x11\0" * 3
    path.write_bytes(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9")


def make_raw_nuscenes(root: Path, scenes: tuple[str, ...] = ("scene-0061", "scene-0103")) -> Path:
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    (version / "scene.json").write_text(json.dumps([{"name": name} for name in scenes]))
    sensors = [{"channel": c, "modality": "camera"} for c in NUSCENES_CAMERAS]
    sensors.append({"channel": "LIDAR_TOP", "modality": "lidar"})
    (version / "sensor.json").write_text(json.dumps(sensors))
    (version / "sample_annotation.json").write_text(json.dumps([{"token": "a"}]))
    return root


def make_processed_scene(scene_dir: Path, frames: int = 3, cameras: int = 6) -> Path:
    for t in range(frames):
        for i in range(cameras):
            write_jpeg(scene_dir / "images" / f"{t:03d}_{i}.jpg", 1600, 900)
            (scene_dir / "extrinsics").mkdir(parents=True, exist_ok=True)
            (scene_dir / "extrinsics" / f"{t:03d}_{i}.txt").write_text("1 0 0 0")
        (scene_dir / "lidar").mkdir(parents=True, exist_ok=True)
        (scene_dir / "lidar" / f"{t:03d}.bin").write_bytes(b"\0" * 16)
    for i in range(cameras):
        (scene_dir / "intrinsics").mkdir(parents=True, exist_ok=True)
        (scene_dir / "intrinsics" / f"{i}.txt").write_text("1 1 1 1")
    (scene_dir / "instances").mkdir(parents=True, exist_ok=True)
    (scene_dir / "instances" / "instances_info.json").write_text(json.dumps({"0": {}}))
    (scene_dir / "instances" / "frame_instances.json").write_text(json.dumps({"0": [0]}))
    return scene_dir


def manifest_for(scene_dir: Path) -> Manifest:
    return build_manifest(
        scene_dir,
        "nuscenes_scene-0061",
        "nuscenes",
        "scene-0061",
        NUSCENES_CAMERAS,
        "drivestudio.nuscenes",
        ["LIDAR_TOP"],
    )


def test_read_image_size(tmp_path: Path) -> None:
    write_png(tmp_path / "a.png", 640, 480)
    write_jpeg(tmp_path / "b.jpg", 1600, 900)
    (tmp_path / "c.jpg").write_bytes(b"not a jpeg")
    assert read_image_size(tmp_path / "a.png") == (640, 480)
    assert read_image_size(tmp_path / "b.jpg") == (1600, 900)
    assert read_image_size(tmp_path / "c.jpg") is None
    assert read_image_size(tmp_path / "missing.png") is None


def test_inspect_nuscenes(tmp_path: Path) -> None:
    report = inspect_source(make_raw_nuscenes(tmp_path))
    assert report.dataset == "nuscenes"
    assert report.versions == {"v1.0-mini": ["scene-0061", "scene-0103"]}
    assert report.cameras == NUSCENES_CAMERAS
    assert report.has_lidar and report.has_cuboids and report.ingest_supported


WAYMO_SEGMENT = (SCENE_LISTS_DIR / "waymo_train_list.txt").read_text().split()[0]
ARGOVERSE_LOG = (SCENE_LISTS_DIR / "argoverse_train_list.txt").read_text().split()[0]


@pytest.mark.parametrize(
    "layout,dataset,versions",
    [
        (
            [f"training/{WAYMO_SEGMENT}.tfrecord", "training/not_listed.tfrecord"],
            "waymo",
            {"training": [WAYMO_SEGMENT]},
        ),
        (
            [f"{ARGOVERSE_LOG}/sensors/cameras/ring_front_center/0.jpg", "other/x.txt"],
            "argoverse2",
            {"": [ARGOVERSE_LOG]},
        ),
        (["001/camera/front_camera/00.jpg", "002/camera/x.jpg"], "pandaset", {"": ["001", "002"]}),
        (
            ["2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0.png"],
            "kitti",
            {"2011_09_26": ["2011_09_26_drive_0001_sync"]},
        ),
        (
            ["nuplan-v1.1/splits/mini/b.db", "nuplan-v1.1/splits/mini/a.db"],
            "nuplan",
            {"mini": ["a", "b"]},
        ),
        (
            ["clips/pai_c2.json", "clips/pai_c1.json", "clips/c1.zarr.itar"],
            "physicalai",
            {"": ["pai_c1", "pai_c2"]},
        ),
    ],
)
def test_inspect_drivestudio_datasets(
    tmp_path: Path, layout: list[str], dataset: str, versions: dict
) -> None:
    for name in layout:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    report = inspect_source(tmp_path)
    assert report.dataset == dataset
    assert report.versions == versions
    assert report.ingest_supported


@pytest.mark.parametrize(
    "layout,dataset",
    [
        (["sparse/0/cameras.bin"], "colmap"),
        (["clip/x.ncore4.zarr.itar"], "ncore"),
    ],
)
def test_inspect_signatures(tmp_path: Path, layout: list[str], dataset: str) -> None:
    for name in layout:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    report = inspect_source(tmp_path)
    assert report.dataset == dataset
    assert not report.ingest_supported
    assert "chưa hỗ trợ" in report.notes[0]


def test_inspect_unknown_lists_inventory(tmp_path: Path) -> None:
    for t in range(3):
        write_png(tmp_path / "cam_front" / f"{t}.png", 1920, 1080)
    (tmp_path / "poses.csv").write_text("t,x")
    report = inspect_source(tmp_path)
    assert report.dataset is None
    assert {"kind": "images", "path": "cam_front", "count": 3, "resolution": [1920, 1080]} in (
        report.inventory
    )
    assert {"kind": "files", "suffix": ".csv", "count": 1} in report.inventory
    with pytest.raises(FileNotFoundError):
        inspect_source(tmp_path / "nope")


def test_build_manifest(tmp_path: Path) -> None:
    manifest = manifest_for(make_processed_scene(tmp_path))
    assert [c.camera_id for c in manifest.cameras] == NUSCENES_CAMERAS
    assert manifest.cameras[0].resolution == (1600, 900)
    assert manifest.timeline.num_frames == 3
    assert manifest.lidars == ["LIDAR_TOP"] and manifest.available.lidar
    assert manifest.available.cuboids == "gt"
    assert manifest.source.converter_version.count("+") == 1
    assert manifest.track == "pinhole" and manifest.cameras[0].model == "opencv_pinhole"
    # PhysicalAI giữ ảnh F-theta gốc: cảnh thuộc track fisheye.
    fisheye = build_manifest(
        make_processed_scene(tmp_path / "f"),
        "physicalai_x",
        "physicalai",
        "x",
        NUSCENES_CAMERAS,
        "drivestudio.physicalai",
        camera_model=SOURCES["physicalai"].camera_model,
    )
    assert fisheye.track == "fisheye" and {c.model for c in fisheye.cameras} == {"ftheta"}
    with pytest.raises(ValueError, match="camera 0"):
        manifest_for(tmp_path / "empty")


def test_validate_clean_and_broken(tmp_path: Path) -> None:
    scene_dir = make_processed_scene(tmp_path)
    manifest = manifest_for(scene_dir)
    report = validate_scene(scene_dir, manifest)
    assert report.summary.errors == 0 and report.summary.warnings == 0
    assert [c.check_id for c in report.checks] == ["projection.not_run"]

    (scene_dir / "images" / "001_2.jpg").unlink()
    (scene_dir / "intrinsics" / "3.txt").unlink()
    (scene_dir / "lidar" / "002.bin").unlink()
    write_jpeg(scene_dir / "images" / "000_4.jpg", 800, 450)
    report = validate_scene(scene_dir, manifest)
    by_id = {(c.check_id, c.camera_id): c for c in report.checks}
    assert by_id[("images.missing", "CAM_FRONT_RIGHT")].details == {"count": 1, "frames": [1]}
    assert ("calib.intrinsics_missing", "CAM_BACK_LEFT") in by_id
    assert ("images.resolution", "CAM_BACK_RIGHT") in by_id
    assert ("lidar.missing", None) in by_id
    assert report.summary.errors == 2 and report.summary.warnings == 2
    ValidationReport.model_validate(report.model_dump())


def fake_preprocess(calls: list[list[str]], fail: bool = False):  # noqa: ANN201
    """Thay script tiền xử lý: ghi một cảnh giả vào chỗ giống script nuScenes thật."""

    def run(command: list[str], log_path: Path, cwd: Path) -> None:
        calls.append(command)
        log_path.write_text("ok\n")
        if fail:
            raise RuntimeError("lệnh thoát với mã 1")
        index = int(command[command.index("--scene_ids") + 1])
        target = command[command.index("--target_dir") + 1]
        make_processed_scene(cwd / target / "mini" / f"{index:03d}")

    return run


def test_default_scene_id_and_commands(tmp_path: Path) -> None:
    assert default_scene_id("kitti", "2011_09_26_drive_0001_sync") == (
        "kitti_2011_09_26_drive_0001_sync"
    )
    assert default_scene_id("pandaset", "001") == "pandaset_001"
    assert default_scene_id("nuscenes", "Scene.0061") == "nuscenes_scene-0061"

    argo = build_command(SOURCES["argoverse2"], tmp_path, "ignored", "3", 2)
    assert argo[argo.index("--dataset") + 1] == "argoverse"
    assert "--split" not in argo
    kitti = build_command(SOURCES["kitti"], tmp_path, "2011_09_26", "x", 4, ["--a", "1"])
    assert kitti[kitti.index("--split") + 1] == "2011_09_26"
    assert kitti[-2:] == ["--a", "1"]


def test_find_scene_dir(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="thấy 0"):
        find_scene_dir(tmp_path)
    (tmp_path / "log_a" / "images").mkdir(parents=True)
    assert find_scene_dir(tmp_path) == tmp_path / "log_a"
    (tmp_path / "log_b" / "images").mkdir(parents=True)
    with pytest.raises(IngestError, match="thấy 2"):
        find_scene_dir(tmp_path)


def test_plan_ingest(tmp_path: Path) -> None:
    report = inspect_source(make_raw_nuscenes(tmp_path))
    with pytest.raises(IngestRequestError, match="chọn một chuỗi; có 2"):
        plan_ingest(report)
    assert plan_ingest(report, sequence="scene-0103") == IngestTarget(
        "nuscenes", "v1.0-mini", "scene-0103"
    )
    with pytest.raises(IngestRequestError, match="không có phiên bản"):
        plan_ingest(report, version="v1.0-trainval")
    with pytest.raises(IngestRequestError, match="không có chuỗi"):
        plan_ingest(report, sequence="scene-9999")
    with pytest.raises(IngestRequestError, match="chưa hỗ trợ"):
        plan_ingest(report, source_type="colmap")
    with pytest.raises(IngestRequestError, match="không nhận ra"):
        plan_ingest(inspect_source(tmp_path / "v1.0-mini"))


def test_ingest_nuscenes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(drivestudio, "run_logged", fake_preprocess(calls))
    raw = make_raw_nuscenes(tmp_path / "raw")
    workdir = Workdir(tmp_path / "work")
    target = IngestTarget("nuscenes", "v1.0-mini", "scene-0103")
    scene_dir = run_ingest(raw, workdir, target)

    assert scene_dir == workdir.scene_dir("nuscenes_scene-0103")
    command = calls[0]
    assert command[command.index("--scene_ids") + 1] == "1"
    assert command[command.index("--target_dir") + 1] == "out"
    assert command[command.index("--split") + 1] == "v1.0-mini"
    assert command[command.index("--interpolate_N") + 1] == "4"
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    assert manifest.source.sequence_id == "scene-0103"
    assert manifest.source.data_license == "CC BY-NC-SA 4.0"
    report = files.load(scene_dir / "reports" / "validation.json", ValidationReport)
    assert report.summary.errors == 0
    assert (scene_dir / "reports" / "ingest_log.txt").is_file()
    assert not (workdir.data_dir / ".ingest" / "nuscenes_scene-0103").exists()

    with pytest.raises(IngestError, match="đã có"):
        run_ingest(raw, workdir, target)
    with pytest.raises(IngestError, match="không có scene-9999"):
        ingest_drivestudio(raw, workdir, "nuscenes", "scene-9999", "v1.0-mini")
    with pytest.raises(IngestError, match="chưa hỗ trợ"):
        ingest_drivestudio(raw, workdir, "colmap", "x")


def test_ingest_script_failure_keeps_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drivestudio, "run_logged", fake_preprocess([], fail=True))
    workdir = Workdir(tmp_path / "work")
    with pytest.raises(IngestError, match="script tiền xử lý lỗi"):
        ingest_drivestudio(tmp_path, workdir, "pandaset", "001")
    assert (workdir.data_dir / ".ingest" / "pandaset_001" / "ingest_log.txt").is_file()
    assert not workdir.scene_dir("pandaset_001").exists()


def test_cli_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(drivestudio, "run_logged", fake_preprocess([]))
    monkeypatch.setenv("D2S_WORKDIR", str(tmp_path / "work"))
    raw = make_raw_nuscenes(tmp_path / "raw")

    assert main(["source", "inspect", str(raw), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["dataset"] == "nuscenes"
    assert main(["scene", "ingest", str(raw), "--quiet"]) == 2  # nhiều cảnh, thiếu --sequence
    assert "chọn một chuỗi" in capsys.readouterr().err
    assert main(["scene", "ingest", str(raw), "--sequence", "scene-0061", "--quiet"]) == 0
    assert (
        main(
            [
                "split",
                "create",
                "nuscenes_scene-0061",
                "--strategy",
                "every_nth_frame:2",
                "--cameras",
                "CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT",
                "--quiet",
            ]
        )
        == 0
    )
    capsys.readouterr()
    code = main(
        [
            "approach",
            "check",
            "omnire",
            "--scene",
            "nuscenes_scene-0061",
            "--split",
            "every2_3cams",
            "--json",
            "--quiet",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [s["stage_id"] for s in data["stages"]][:3] == [
        "prep.masks_segformer",
        "prep.smpl",
        "train.drivestudio",
    ]
