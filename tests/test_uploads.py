"""Resumable KITTI-360 uploads from the user's computer."""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from fisheye_fusion.pipeline import Discovery
from fisheye_fusion.uploads import UploadManager, canonical, required

SEQ = "2013_05_28_drive_0003_sync"


@pytest.mark.parametrize("name,target", [
    ("data_poses.zip", "data_poses.zip"),
    ("D:/kitti/calibration.zip", "calibration.zip"),
    (f"C:\\x\\{SEQ}_image_02.tar", f"data_2d_raw/{SEQ}_image_02.tar"),
    (f"{SEQ}.tar", f"data_3d_raw/{SEQ}.tar"),
    ("ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl", "ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl"),
    ("calib_cam_to_pose.txt", "calibration/calib_cam_to_pose.txt"),
])
def test_canonical_places_known_files(name, target):
    assert canonical(name) == target


@pytest.mark.parametrize("name", ["../data_poses.zip.sh", "evil.tar", f"{SEQ}_image_05.tar", "data_poses.zip/..", ""])
def test_canonical_rejects_unknown_files(name):
    assert canonical(name) is None


def test_required_lists_every_file_for_a_sequence():
    files = required(SEQ)
    assert len(files) == 9 and f"data_3d_raw/{SEQ}.tar" in files
    with pytest.raises(ValueError):
        required("../x")


def test_chunked_upload_resumes_and_is_found_by_discovery(tmp_path):
    up = UploadManager(tmp_path)
    blob = b"0123456789" * 3
    target = f"kitti360/data_2d_raw/{SEQ}_image_00.tar"
    plan = up.plan([{"name": f"{SEQ}_image_00.tar", "size": len(blob)}, {"name": "notes.docx", "size": 3}])
    assert plan["skipped"] == ["notes.docx"] and plan["files"][0]["offset"] == 0
    assert up.write(target, 0, len(blob), blob[:10]) == {"target": target, "offset": 10, "done": False}
    # a reload asks again: the server remembers the partial file
    assert up.plan([{"name": f"{SEQ}_image_00.tar", "size": len(blob)}])["files"][0]["offset"] == 10
    # out-of-order chunk is not written, the client learns where to continue
    assert up.write(target, 20, len(blob), blob[20:])["mismatch"]
    assert up.write(target, 10, len(blob), blob[10:]) == {"target": target, "offset": 30, "done": True}
    assert (tmp_path / target).read_bytes() == blob
    assert up.plan([{"name": f"{SEQ}_image_00.tar", "size": len(blob)}])["files"][0]["done"]
    (tmp_path / "kitti360" / "data_poses.zip").write_bytes(b"x")
    (ds,) = Discovery(tmp_path).datasets()
    assert ds["path"] == "kitti360" and ds["sequences"][SEQ]["cameras"] == ["image_00"]
    status = up.status()
    assert status["files"][target.removeprefix("kitti360/")] == 30 and SEQ in status["sequences"]


def test_upload_rejects_unknown_targets_and_oversized_chunks(tmp_path):
    up = UploadManager(tmp_path)
    for bad in ("../x.tar", "kitti360/data_2d_raw/../../evil.tar", "kitti360/notes.txt", "data/waymo/raw/../../x",
                "data/unknown/raw/a.bin", "data/kitti360/raw/a.tar", "data/waymo/raw/a.tfrecord.part"):
        with pytest.raises(ValueError):
            up.write(bad, 0, 3, b"abc")
    with pytest.raises(ValueError):
        up.write("kitti360/data_poses.zip", 0, 2, b"abc")
    assert not (tmp_path / "kitti360").exists() or not any((tmp_path / "kitti360").rglob("*.tar"))


def test_cancel_removes_partial_file(tmp_path):
    up = UploadManager(tmp_path)
    up.write("kitti360/data_poses.zip", 0, 10, b"abc")
    up.cancel("kitti360/data_poses.zip")
    assert up.status()["partial"] == {}


def test_calibration_zip_is_extracted_after_upload(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("calibration/perspective.txt", "P")
    data = buf.getvalue()
    UploadManager(tmp_path).write("kitti360/calibration.zip", 0, len(data), data)
    assert (tmp_path / "kitti360" / "calibration" / "perspective.txt").read_text() == "P"


def test_upload_api_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("fisheye_fusion.server.WORKSPACE", tmp_path)
    from fisheye_fusion.server import create_app
    client = TestClient(create_app())
    plan = client.post("/api/upload/plan", json={"files": [{"name": "data_poses.zip", "size": 4}]}).json()
    assert plan["files"][0]["target"] == "kitti360/data_poses.zip" and plan["dataset"] == "kitti360"
    res = client.put("/api/upload/chunk", params={"target": "kitti360/data_poses.zip", "offset": 0, "total": 4}, content=b"abcd")
    assert res.json()["done"]
    assert client.get("/api/upload").json()["files"] == {"data_poses.zip": 4}
    bad = client.put("/api/upload/chunk", params={"target": "../../etc/passwd", "offset": 0, "total": 1}, content=b"x")
    assert bad.status_code == 400


def test_other_datasets_keep_their_folder_structure(tmp_path):
    up = UploadManager(tmp_path)
    plan = up.plan([{"name": "segment-1_with_camera_labels.tfrecord", "path": "segment-1_with_camera_labels.tfrecord", "size": 5},
                    {"name": ".DS_Store", "path": ".DS_Store", "size": 1}])
    assert plan["dataset"] == "waymo" and plan["files"][0]["target"] == "data/waymo/raw/segment-1_with_camera_labels.tfrecord"
    up.write(plan["files"][0]["target"], 0, 5, b"12345")
    assert (tmp_path / "data/waymo/raw/segment-1_with_camera_labels.tfrecord").read_bytes() == b"12345"
    nus = up.plan([{"name": "scene.json", "path": "v1.0-mini/scene.json", "size": 3},
                   {"name": "a.jpg", "path": "samples/CAM_FRONT/a.jpg", "size": 3}])
    assert nus["dataset"] == "nuscenes"
    assert [f["target"] for f in nus["files"]] == ["data/nuscenes/raw/v1.0-mini/scene.json",
                                                   "data/nuscenes/raw/samples/CAM_FRONT/a.jpg"]
    # an explicit choice wins over detection; unsafe relative paths are skipped
    forced = up.plan([{"name": "x.bin", "path": "clips/x.bin", "size": 2}, {"name": "y", "path": "../y", "size": 2}],
                     dataset="pandaset")
    assert [f["target"] for f in forced["files"]] == ["data/pandaset/raw/clips/x.bin"] and forced["skipped"] == ["y"]


def test_unrecognised_files_ask_for_a_dataset(tmp_path):
    with pytest.raises(ValueError, match="Loại dữ liệu"):
        UploadManager(tmp_path).plan([{"name": "notes.docx", "size": 3}])
