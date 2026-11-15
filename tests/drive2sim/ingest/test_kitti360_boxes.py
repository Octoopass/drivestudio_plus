"""Kiểm tra box 3D KITTI-360 (``drive2sim/ingest/kitti360_boxes.py``) và việc converter ghi chúng.

- đọc XML: tách kích thước khỏi phép quay, sửa ma trận lật, vật đứng yên có mặt nhiều khung, bỏ
  lớp không phải xe hay người;
- ``instances_info.json``, ``frame_instances.json`` theo định dạng script KITTI của DriveStudio;
- chiếu box lên ảnh pinhole và mắt cá; mặt nạ PNG đúng kích thước;
- converter có file box thì cảnh có ``cuboids: gt``, không có thì cảnh báo.
"""

import json
import math
import zipfile
from pathlib import Path

import pytest

from drive2sim.core import files
from drive2sim.core.schema import Manifest, ValidationReport
from drive2sim.engine.workdir import Workdir
from drive2sim.ingest import kitti360, kitti360_boxes
from drive2sim.ingest.detect import inspect_source
from drive2sim.ingest.dispatch import plan_ingest, run_ingest
from drive2sim.ingest.images import read_image_size, write_mask_png
from tests.drive2sim.ingest.test_kitti360 import SEQ, make_raw

IDENTITY = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def box_xml(objects: list[dict]) -> str:
    """File XML giống ``data_3d_bboxes/train/<chuỗi>.xml`` (chỉ các trường converter đọc)."""
    parts = ['<?xml version="1.0"?>\n<opencv_storage>']
    for number, item in enumerate(objects, 1):
        data = " ".join(f"{v:.9g}" for v in item["transform"])
        frames = (
            f"<start_frame>{item['start']}</start_frame><end_frame>{item['end']}</end_frame>"
            if item["timestamp"] < 0
            else "<start_frame>0</start_frame><end_frame>0</end_frame>"
        )
        parts.append(
            f"<object{number}><label>{item['label']}</label>"
            f"<semanticId>{item['semantic']}</semanticId><instanceId>{item['instance']}</instanceId>"
            f"<timestamp>{item['timestamp']}</timestamp><dynamic>{int(item['timestamp'] >= 0)}"
            f'</dynamic><transform type_id="opencv-matrix"><rows>4</rows><cols>4</cols><dt>f</dt>'
            f"<data>{data}</data></transform>{frames}</object{number}>"
        )
    return "\n".join(parts) + "\n</opencv_storage>\n"


def transform(yaw: float, size: tuple, center: tuple, flip_y: bool = False) -> list[float]:
    """Ma trận 4×4 (theo hàng) như trong XML: quay quanh z nhân kích thước, cộng tâm."""
    c, s = math.cos(yaw), math.sin(yaw)
    axes = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]  # trục x, y, z của box trong world
    if flip_y:
        axes[1] = [-v for v in axes[1]]
    rows = []
    for i in range(3):
        rows += [axes[j][i] * size[j] for j in range(3)] + [center[i]]
    return rows + [0.0, 0.0, 0.0, 1.0]


OBJECTS = [
    # xe chạy: một phần tử cho mỗi khung
    {"label": "car", "semantic": 26, "instance": 5, "timestamp": 9,
     "transform": transform(0.3, (4.5, 1.9, 1.6), (10.0, 2.0, 0.8))},
    {"label": "car", "semantic": 26, "instance": 5, "timestamp": 10,
     "transform": transform(0.3, (4.5, 1.9, 1.6), (12.0, 2.5, 0.8))},
    # xe tải đỗ, ma trận lật
    {"label": "truck", "semantic": 27, "instance": 1, "timestamp": -1, "start": 8, "end": 20,
     "transform": transform(1.0, (8.0, 2.5, 3.2), (30.0, -4.0, 1.6), flip_y=True)},
    # không phải xe hay người: bỏ
    {"label": "building", "semantic": 11, "instance": 2, "timestamp": -1, "start": 0, "end": 99,
     "transform": transform(0.0, (20.0, 10.0, 8.0), (50.0, 50.0, 4.0))},
]  # fmt: skip


def test_read_boxes_separates_size_and_fixes_flipped_matrix() -> None:
    boxes = kitti360_boxes.read_boxes(box_xml(OBJECTS))
    assert [(b.class_name, b.key, b.frames[:2]) for b in boxes] == [
        ("Car", (26, 5), (9,)),
        ("Car", (26, 5), (10,)),
        ("Truck", (27, 1), (8, 9)),
    ]
    assert boxes[2].frames == tuple(range(8, 21))
    assert boxes[0].size == pytest.approx([4.5, 1.9, 1.6])
    assert boxes[1].center == pytest.approx([12.0, 2.5, 0.8])
    truck = boxes[2].rotation
    assert kitti360_boxes._determinant(truck) == pytest.approx(1.0)
    assert [truck[i][0] for i in range(3)] == pytest.approx([math.cos(1.0), math.sin(1.0), 0.0])


def test_build_instances_maps_frames_and_world() -> None:
    boxes = kitti360_boxes.read_boxes(box_xml(OBJECTS))
    shift = [row[:] for row in IDENTITY]
    shift[0][3] = -10.0  # world của cảnh: world của chuỗi dời 10 m
    info, frame_instances = kitti360_boxes.build_instances(boxes, [8, 9, 10, 11], shift)
    assert sorted(frame_instances) == ["0", "1", "2", "3"]
    assert frame_instances == {"0": [1], "1": [0, 1], "2": [0, 1], "3": [1]}
    car = info["0"]
    assert car["class_name"] == "Car" and car["id"] == 0
    assert car["frame_annotations"]["frame_idx"] == [1, 2]
    assert car["frame_annotations"]["obj_to_world"][1][0][3] == pytest.approx(2.0)
    assert car["frame_annotations"]["box_size"][0] == pytest.approx([4.5, 1.9, 1.6])
    assert info["1"]["frame_annotations"]["frame_idx"] == [0, 1, 2, 3]
    json.dumps(info)  # ghi được ra JSON


def test_project_rect_pinhole_and_fisheye() -> None:
    pinhole = [500.0, 500.0, 320.0, 240.0, 0, 0, 0, 0, 0]
    ahead = [[x, y, 10.0] for x in (-1.0, 1.0) for y in (-0.5, 0.5)]
    assert kitti360_boxes.project_rect(ahead, "opencv_pinhole", pinhole, 640, 480) == (
        270,
        215,
        370,
        265,
    )
    behind = [[p[0], p[1], -p[2]] for p in ahead]
    assert kitti360_boxes.project_rect(behind, "opencv_pinhole", pinhole, 640, 480) is None
    wide = [[x, 0.0, 1.0] for x in (-5.0, 5.0)] + [[0.0, 1.0, 1.0]]
    assert kitti360_boxes.project_rect(wide, "opencv_pinhole", pinhole, 640, 480) is None

    fisheye = [300.0, 300.0, 700.0, 705.0, 0.0, 0.0, 0.0, 0.0]  # k = 0: r = fx·θ
    side = [[math.tan(0.5), 0.0, 1.0], [0.0, 0.2, 1.0]]
    rect = kitti360_boxes.project_rect(side, "opencv_fisheye", fisheye, 1400, 1400)
    assert rect == (700, 705, int(700 + 300 * 0.5), int(705 + 300 * math.atan(0.2)))
    # góc 90° (nằm ngang với camera) vẫn thấy với mắt cá, nhưng vượt góc đã khớp thì bỏ
    beside = [[1.0, 0.0, 0.0], [1.0, 0.1, 0.1]]
    assert kitti360_boxes.project_rect(beside, "opencv_fisheye", fisheye, 1400, 1400) is None


def test_write_mask_png(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    write_mask_png(path, 40, 30, [(5, 6, 10, 12), (-3, 28, 2, 40)])
    assert read_image_size(path) == (40, 30)
    import struct
    import zlib

    data = path.read_bytes()
    length = struct.unpack(">I", data[33:37])[0]
    raw = zlib.decompress(data[41 : 41 + length])
    rows = [raw[r * 41 + 1 : (r + 1) * 41] for r in range(30)]
    assert rows[6][5:10] == b"\xff" * 5 and rows[6][4] == 0 and rows[5][5] == 0
    assert rows[29][:2] == b"\xff\xff" and rows[29][2] == 0
    assert sum(row.count(255) for row in rows) == 5 * 6 + 2 * 2


def test_ingest_writes_instances_and_masks(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / "raw")
    with zipfile.ZipFile(raw / "data_3d_bboxes.zip", "w") as zipped:
        zipped.writestr(f"data_3d_bboxes/train/{SEQ}.xml", box_xml(OBJECTS))
    workdir = Workdir(tmp_path / "work")
    options = kitti360.Kitti360Options(max_gap=1)
    scene_dir = run_ingest(
        raw, workdir, plan_ingest(inspect_source(raw)), kitti360_options=options
    )
    manifest = files.load(scene_dir / "manifest.yaml", Manifest)
    assert manifest.available.cuboids == "gt"
    info = json.loads((scene_dir / "instances" / "instances_info.json").read_text())
    frame_instances = json.loads((scene_dir / "instances" / "frame_instances.json").read_text())
    assert {v["class_name"] for v in info.values()} == {"Car", "Truck"}
    assert len(frame_instances) == manifest.timeline.num_frames == 5
    for group in ("all", "human", "vehicle"):
        assert read_image_size(scene_dir / "dynamic_masks" / group / "000_2.png") == (1400, 1400)
        assert read_image_size(scene_dir / "dynamic_masks" / group / "004_0.png") == (1408, 376)
    report = files.load(scene_dir / "reports" / "validation.json", ValidationReport)
    checks = {check.check_id: check for check in report.checks}
    assert checks["boxes.loaded"].details["classes"] == {"Car": 1, "Truck": 1}


def test_ingest_without_boxes_warns(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / "raw")
    workdir = Workdir(tmp_path / "work")
    options = kitti360.Kitti360Options(max_gap=1)
    scene_dir = run_ingest(
        raw, workdir, plan_ingest(inspect_source(raw)), kitti360_options=options
    )
    assert not (scene_dir / "instances").exists()
    report = files.load(scene_dir / "reports" / "validation.json", ValidationReport)
    assert {c.check_id: c.severity for c in report.checks}["boxes.missing"] == "warning"
