"""Test tải dữ liệu gốc lên qua trình duyệt (``drive2sim/studio/api/uploads.py`` và route)."""

import shutil
from collections import namedtuple
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.engine.registry import StageRegistry
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api import uploads
from drive2sim.studio.api.app import create_app


def test_safe_relative() -> None:
    assert uploads.safe_relative("a\\b/c.txt") == "a/b/c.txt"
    assert uploads.safe_relative("x/.DS_Store") is None
    for bad in ("/etc/passwd", "../x", "a/../../b", "C:/x", "a/b.part", "/".join("d" * 13)):
        with pytest.raises(uploads.UploadError):
            uploads.safe_relative(bad)


def test_plan_write_resume_and_delete(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    data = bytes(range(256)) * 40  # 10240 byte
    plan = uploads.plan_upload(
        workdir,
        "kitti",
        [{"path": "seq/a.bin", "size": len(data)}, {"path": "Thumbs.db", "size": 1}],
    )
    assert plan["files"][0] == {"path": "seq/a.bin", "size": 10240, "offset": 0, "done": False}
    assert plan["skipped"] == ["Thumbs.db"] and plan["path_abs"].endswith("uploads/kitti")
    first = uploads.write_chunk(workdir, "kitti", "seq/a.bin", 0, len(data), data[:4096])
    assert first == {"path": "seq/a.bin", "offset": 4096, "done": False}
    with pytest.raises(uploads.UploadOffsetError):
        uploads.write_chunk(workdir, "kitti", "seq/a.bin", 0, len(data), data[:4096])
    resumed = uploads.plan_upload(workdir, "kitti", [{"path": "seq/a.bin", "size": len(data)}])
    assert resumed["files"][0]["offset"] == 4096 and resumed["needed_bytes"] == 6144
    done = uploads.write_chunk(workdir, "kitti", "seq/a.bin", 4096, len(data), data[4096:])
    assert done["done"] and (tmp_path / "uploads" / "kitti" / "seq" / "a.bin").read_bytes() == data
    again = uploads.plan_upload(workdir, "kitti", [{"path": "seq/a.bin", "size": len(data)}])
    assert again["files"][0]["done"] and again["needed_bytes"] == 0
    listed = uploads.list_uploads(workdir)
    assert listed[0]["name"] == "kitti" and listed[0]["files"] == 1 and listed[0]["partial"] == 0
    assert uploads.delete_upload(workdir, "kitti") == {"removed": ["uploads/kitti"]}
    assert uploads.list_uploads(workdir) == []


def test_refuses_bad_name_and_full_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = Workdir(tmp_path)
    with pytest.raises(uploads.UploadError):
        uploads.plan_upload(workdir, "../x", [{"path": "a", "size": 1}])
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10, 10, 1024))
    with pytest.raises(uploads.UploadError, match="không đủ chỗ"):
        uploads.plan_upload(workdir, "big", [{"path": "a", "size": 2048}])


def test_routes(tmp_path: Path) -> None:
    app = create_app(Workdir(tmp_path / "work"), registry=StageRegistry(), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        plan = client.post(
            "/api/sources/uploads",
            json={"name": "nus", "files": [{"path": "v1/x.txt", "size": 6}]},
        )
        assert plan.status_code == 200 and plan.json()["files"][0]["offset"] == 0
        query = "path=v1/x.txt&offset=0&total=6"
        part = client.post(f"/api/sources/uploads/nus/chunk?{query}", content=b"abc")
        assert part.json() == {"path": "v1/x.txt", "offset": 3, "done": False}
        clash = client.post(f"/api/sources/uploads/nus/chunk?{query}", content=b"abc")
        assert clash.status_code == 409 and clash.json()["error"]["code"] == "conflict"
        rest = client.post(
            "/api/sources/uploads/nus/chunk?path=v1/x.txt&offset=3&total=6", content=b"def"
        )
        assert rest.json()["done"] is True
        listed = client.get("/api/sources/uploads").json()
        assert listed["total"] == 1 and listed["items"][0]["size_bytes"] == 6
        bad = client.post(
            "/api/sources/uploads", json={"name": "a/b", "files": [{"path": "x", "size": 1}]}
        )
        assert bad.status_code == 400
        inspected = client.post(
            "/api/sources/inspect", json={"path": listed["items"][0]["path_abs"]}
        )
        assert inspected.status_code == 200
        assert client.delete("/api/sources/uploads/nus").json() == {"removed": ["uploads/nus"]}
