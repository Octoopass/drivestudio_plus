"""Test ``drive2sim.kaggle.scene_files``: tập file nhỏ của cảnh ingest trên Kaggle."""

import base64
import json
import os
from pathlib import Path

import pytest

from drive2sim.kaggle.scene_files import (
    LOG_LINE_BYTES,
    MAX_BYTES,
    light_scene_files,
    missing_view_files,
    read_log_lines,
    write_log_lines,
    write_scene_files,
)


def make_scene(root: Path) -> Path:
    scene = root / "kitti360_drive0003"
    for name, text in {
        "manifest.yaml": "schema_version: d2s-manifest/1.1\n",
        "frame_ids.txt": "167\n168\n",
        "reports/validation.json": "{}",
        "intrinsics/image_02.txt": "1 2 3 4 0 0 0 0\n",
        "extrinsics/image_02.txt": "1 0 0 0\n",
        "ego_pose/000.txt": "1 0 0 0\n",
        "ego_pose/001.txt": "1 0 0 0\n",
        "images/000_image_02.png": "nặng",
        "lidar/000.bin": "nặng",
        "dynamic_masks/all/000_image_02.png": "nặng",
        "instances/instances_info.json": "{}",
    }.items():
        path = scene / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")
    return scene


def test_light_files_keep_pose_calib_and_skip_sensor_data(tmp_path: Path) -> None:
    files = light_scene_files(make_scene(tmp_path))
    assert sorted(files) == [
        "ego_pose/000.txt",
        "ego_pose/001.txt",
        "extrinsics/image_02.txt",
        "frame_ids.txt",
        "intrinsics/image_02.txt",
        "manifest.yaml",
        "reports/validation.json",
    ]
    assert files["frame_ids.txt"] == "167\n168\n"


def test_light_files_refuse_oversized_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = make_scene(tmp_path)
    monkeypatch.setattr("drive2sim.kaggle.scene_files.MAX_BYTES", 10)
    with pytest.raises(ValueError, match="quá"):
        light_scene_files(scene)
    assert MAX_BYTES > 10


def test_missing_view_files(tmp_path: Path) -> None:
    scene = make_scene(tmp_path)
    assert missing_view_files(scene) == []
    bare = tmp_path / "bare"
    (bare / "lidar_pose").mkdir(parents=True)
    assert missing_view_files(bare) == ["intrinsics/", "extrinsics/"]
    (bare / "intrinsics").mkdir()
    (bare / "extrinsics").mkdir()
    assert missing_view_files(bare) == []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert missing_view_files(empty) == [
        "intrinsics/",
        "extrinsics/",
        "ego_pose/ hoặc lidar_pose/",
    ]


def test_write_scene_files_adds_missing_and_keeps_existing(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "manifest.yaml").write_text("local\n", "utf-8")
    written = write_scene_files(
        scene,
        {"manifest.yaml": "remote\n", "ego_pose/000.txt": "1\n", "intrinsics/a.txt": "2\n"},
    )
    assert sorted(written) == ["ego_pose/000.txt", "intrinsics/a.txt"]
    assert (scene / "manifest.yaml").read_text("utf-8") == "local\n"
    assert (scene / "ego_pose" / "000.txt").read_text("utf-8") == "1\n"


def test_write_scene_files_rejects_paths_outside_scene(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    with pytest.raises(ValueError, match="đường dẫn lạ"):
        write_scene_files(scene, {"../escape.txt": "x"})
    assert not (tmp_path / "escape.txt").exists()


def test_log_lines_survive_interleaved_output(tmp_path: Path) -> None:
    pose = "1 0 0 0\n" * 40
    payload = {"scene_id": "s", "files": {f"ego_pose/{i:03d}.txt": pose for i in range(300)}}
    path = tmp_path / "log"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT)
    try:
        write_log_lines("D2S_TEST", payload, fd=fd)
    finally:
        os.close(fd)
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    assert len(lines) > 3 and all(len(line.encode()) <= LOG_LINE_BYTES for line in lines)
    # Kaggle chèn stdout giữa các dòng, và trả log theo từng đoạn nối lại.
    stdout = "kitti360_drive0003 876 file\n"
    mixed = "pip ...\n" + "".join(
        line + (stdout if i == 1 else "") for i, line in enumerate(lines)
    )
    assert read_log_lines("D2S_TEST", mixed) == payload


def test_read_log_lines_reports_missing_chunk_and_reads_legacy() -> None:
    payload = {"a": 1}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    assert read_log_lines("D2S_TEST", f"x\nD2S_TEST {encoded}\n") == payload  # dạng một dòng cũ
    assert read_log_lines("D2S_TEST", "không có gì") is None
    with pytest.raises(ValueError, match="thiếu"):
        read_log_lines("D2S_TEST", "D2S_TEST 0/2 abcd\n")
