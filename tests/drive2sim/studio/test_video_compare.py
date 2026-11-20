"""Test ghép video hai run (``drive2sim/studio/api/video_compare.py`` và route)."""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api import preview, video_compare
from drive2sim.studio.api.app import create_app


def test_filter_and_labels() -> None:
    graph = video_compare._filter("hstack", ("run-a:1", "b"), with_text=True)
    assert "scale=-2:480" in graph and "hstack=inputs=2" in graph
    assert "text='run-a1'" in graph  # ":" bị bỏ, không thoát
    plain = video_compare._filter("vstack", ("x", "y"), with_text=False)
    assert "drawtext" not in plain and "scale=1280:-2" in plain
    assert video_compare._label("a'b;c$(rm)") == "abcrm"


def test_video_path_stays_inside_stage(tmp_path: Path) -> None:
    stages = tmp_path / "stages"
    (stages / "train.x" / "videos").mkdir(parents=True)
    video = stages / "train.x" / "videos" / "a.mp4"
    video.write_bytes(b"x")
    assert video_compare.video_path(stages, "train.x", "videos/a.mp4") == video.resolve()
    for stage, name in (
        ("train.x", "../../a.mp4"),
        ("train.x", "videos/missing.mp4"),
        ("..", "a.mp4"),
    ):
        with pytest.raises(LookupError):
            video_compare.video_path(stages, stage, name)


def test_same_video_refused(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError):
        video_compare.compare_video(tmp_path, video, "a", video, "a", "vstack")


def _tiny_video(ffmpeg: str, path: Path, size: str) -> None:
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=5",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ],
        check=True,
    )  # fmt: skip


def test_compose_real_videos_and_routes(tmp_path: Path) -> None:
    ffmpeg = preview._ffmpeg()
    if ffmpeg is None:
        pytest.skip("cần ffmpeg (imageio-ffmpeg)")
    workdir = Workdir(tmp_path / "work")
    paths = {}
    for run_id, size in (("run-a", "320x240"), ("run-b", "160x90")):
        folder = tmp_path / "videos" / run_id
        folder.mkdir(parents=True)
        paths[run_id] = folder / "v.mp4"
        _tiny_video(ffmpeg, paths[run_id], size)
    name = video_compare.compare_video(
        workdir.root, paths["run-a"], "run-a", paths["run-b"], "run-b", "hstack"
    )
    out = workdir.root / video_compare.COMPARE_DIR / name
    assert out.is_file() and out.stat().st_size > 0
    first = out.stat().st_mtime_ns
    assert (
        video_compare.compare_video(
            workdir.root, paths["run-a"], "run-a", paths["run-b"], "run-b", "hstack"
        )
        == name
        and out.stat().st_mtime_ns == first
    )  # dùng lại, không ghép lại
    app = create_app(workdir, registry=StageRegistry(), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get(f"/api/compare/videos/{name}").status_code == 200
        assert client.get("/api/compare/videos/../../x.mp4").status_code == 404
        assert client.get("/api/compare/videos/abc.mp4").status_code == 404
        missing = client.post(
            "/api/compare/videos",
            json={
                "a": {"run_id": "nope", "stage_id": "s", "name": "v.mp4"},
                "b": {"run_id": "nope2", "stage_id": "s", "name": "v.mp4"},
            },
        )
        assert missing.status_code == 404
