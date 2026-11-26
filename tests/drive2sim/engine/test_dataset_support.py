"""Kiểm tra BezierGS và MCGS-SLAM hỗ trợ đủ các dataset mà tool ingest được.

- BezierGS (approach ``beziergs``, trainer DriveStudio): với mỗi dataset pinhole có box 3D và mọi
  số camera có config, engine phải xếp approach vào kế hoạch chạy được; thiếu box hoặc camera
  không phải pinhole thì phải bị khoá (không được chạy rồi mới lỗi).
- MCGS-SLAM (``eval.mcgs``): tập dataset ``external/mcgs/views.py`` đọc được pose phải đúng bằng
  các dataset pinhole của tool; thêm dataset pinhole vào ``ingest`` mà quên MCGS thì test này lỗi.

Chỉ kiểm bằng manifest giả (cấu hình và điều kiện), không train; run BezierGS thật mới có ở Waymo
(``docs/methods/beziergs.md``).
"""

from pathlib import Path

import pytest

from drive2sim import paths
from drive2sim.core.schema import Manifest, Split
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.registry import load_builtin_stages
from drive2sim.engine.resolve import BlockedError, resolve
from drive2sim.ingest.drivestudio import SOURCES
from drive2sim.stages.scene_view import DRIVESTUDIO_DATASETS
from drive2sim.stages.train.drivestudio import dataset_config_for

KITTI360_CAMERAS = (
    ("image_00", "opencv_pinhole"),
    ("image_01", "opencv_pinhole"),
    ("image_02", "opencv_fisheye"),
    ("image_03", "opencv_fisheye"),
)


def dataset_cameras() -> dict[str, list[tuple[str, str]]]:
    """Camera (id, mô hình) theo thứ tự chỉ số của mọi dataset mà tool ingest được."""
    cameras = {
        name: [(camera, source.camera_model) for camera in source.cameras]
        for name, source in SOURCES.items()
    }
    cameras["kitti360"] = list(KITTI360_CAMERAS)
    return cameras


def make_manifest(dataset: str, cameras: list[tuple[str, str]], cuboids: str = "gt") -> Manifest:
    return Manifest.model_validate(
        {
            "schema_version": "d2s-manifest/1.2",
            "scene_id": f"{dataset}_x",
            "source": {
                "dataset": dataset,
                "sequence_id": "x",
                "converter": "test",
                "converter_version": "0.1.0+test",
            },
            "track": "fisheye" if any(m != "opencv_pinhole" for _, m in cameras) else "pinhole",
            "cameras": [
                {"camera_id": c, "model": m, "resolution": [800, 600], "shutter": "GLOBAL"}
                for c, m in cameras
            ],
            "lidars": ["LIDAR"],
            "timeline": {"reference_sensor": cameras[0][0], "num_frames": 30},
            "available": {"lidar": True, "cuboids": cuboids},
        }
    )


def make_split(manifest: Manifest, cameras: list[str]) -> Split:
    return Split.model_validate(
        {
            "schema_version": "d2s-split/0.1",
            "split_id": "every4_x",
            "scene_id": manifest.scene_id,
            "strategy": "every_nth_frame:4",
            "frame_range": [0, 29],
            "cameras": cameras,
            "train": [[i, c] for i in range(30) if i % 4 for c in cameras],
            "test": [[i, c] for i in range(30) if i % 4 == 0 for c in cameras],
        }
    )


def config_counts(dataset: str) -> list[int]:
    folder = paths.CONFIGS_DIR / "datasets" / DRIVESTUDIO_DATASETS[dataset]
    return sorted(int(p.stem[:-4]) for p in folder.glob("*cams.yaml") if p.stem[:-4].isdigit())


def pinhole_datasets() -> list[str]:
    return sorted(
        name
        for name, cameras in dataset_cameras().items()
        if any(m == "opencv_pinhole" for _, m in cameras)
    )


@pytest.mark.parametrize("dataset", pinhole_datasets())
def test_beziergs_resolves_for_every_pinhole_config(dataset: str) -> None:
    load_builtin_stages()
    cameras = [(c, m) for c, m in dataset_cameras()[dataset] if m == "opencv_pinhole"]
    manifest = make_manifest(dataset, dataset_cameras()[dataset])
    checked = 0
    for count in config_counts(dataset):
        if count > len(cameras):
            continue  # KITTI-360 có 2 camera pinhole: config 4 camera cần cả mắt cá
        chosen = [c for c, _ in cameras[:count]]
        assert dataset_config_for(dataset, count)  # có file config dataset
        plan = resolve(load_approach("beziergs"), manifest, make_split(manifest, chosen))
        assert "train.drivestudio" in [stage.stage_id for stage in plan.stages]
        checked += 1
    assert checked >= 1


@pytest.mark.parametrize("dataset", pinhole_datasets())
def test_beziergs_is_blocked_without_boxes(dataset: str) -> None:
    load_builtin_stages()
    cameras = [(c, m) for c, m in dataset_cameras()[dataset] if m == "opencv_pinhole"]
    manifest = make_manifest(dataset, cameras, cuboids="none")
    with pytest.raises(BlockedError):
        resolve(load_approach("beziergs"), manifest, make_split(manifest, [cameras[0][0]]))


def test_beziergs_is_blocked_on_non_pinhole_cameras() -> None:
    load_builtin_stages()
    for dataset, cameras in (
        ("physicalai", dataset_cameras()["physicalai"]),  # F-theta
        ("kitti360", list(KITTI360_CAMERAS)),  # split có cả camera mắt cá
    ):
        manifest = make_manifest(dataset, cameras)
        with pytest.raises(BlockedError):
            resolve(
                load_approach("beziergs"), manifest, make_split(manifest, [c for c, _ in cameras])
            )


def test_mcgs_approach_resolves_on_pinhole_scene() -> None:
    load_builtin_stages()
    manifest = make_manifest("kitti360", dataset_cameras()["kitti360"])
    split = make_split(manifest, ["image_00", "image_01"])
    plan = resolve(load_approach("mcgs"), manifest, split)
    assert [stage.stage_id for stage in plan.stages] == ["prep.mcgs", "train.mcgs", "eval.mcgs"]
    assert plan.stages[1].params["mode"] == "full"
    assert plan.stages[1].input_sources == {"data": 0}
    assert not any(
        stage.uses_lidar for stage in plan.stages
    )  # cảnh có LiDAR nhưng SLAM không dùng
    minimal = resolve(load_approach("mcgs"), manifest, split, option_values={"mode": "minimal"})
    assert minimal.stages[1].params["mode"] == "minimal"


def test_mcgs_blocked_on_fisheye_only_split() -> None:
    load_builtin_stages()
    manifest = make_manifest("kitti360", dataset_cameras()["kitti360"])
    with pytest.raises(BlockedError):
        resolve(load_approach("mcgs"), manifest, make_split(manifest, ["image_02", "image_03"]))


def test_mcgs_blocked_when_split_has_no_stereo_pair() -> None:
    load_builtin_stages()
    manifest = make_manifest("kitti360", dataset_cameras()["kitti360"])
    with pytest.raises(BlockedError, match="2 camera"):
        resolve(load_approach("mcgs"), manifest, make_split(manifest, ["image_00"]))


def test_mcgs_refused_on_kaggle_and_colab() -> None:
    from drive2sim.engine.remote import require_main_codebase

    for where in ("Kaggle", "Colab"):
        with pytest.raises(ValueError, match="codebase ngoài"):
            require_main_codebase("mcgs", where)


def test_mcgs_supports_exactly_the_pinhole_datasets_of_the_tool() -> None:
    pytest.importorskip("numpy")
    from drive2sim.external.mcgs import views

    all_pinhole = {
        name
        for name, cameras in dataset_cameras().items()
        if all(m == "opencv_pinhole" for _, m in cameras)
    }
    # KITTI-360 có thêm hai camera mắt cá, nhưng hai camera pinhole của nó dùng được cho SLAM
    assert set(views.SUPPORTED_DATASETS) == all_pinhole | {"kitti360"}
    assert "physicalai" not in views.SUPPORTED_DATASETS
    for dataset in views.SUPPORTED_DATASETS:
        if dataset in views.DIRECT_POSE_DATASETS:
            continue
        module_name, class_name = views.LOADER_CAMERA_CLASSES[dataset]
        source = (
            Path(__file__).resolve().parents[3]
            / "drive2sim/recon/data/loaders"
            / f"{module_name}.py"
        ).read_text(encoding="utf-8")
        assert f"class {class_name}(" in source and "def get_camera2worlds" in source
