"""Box 3D của KITTI-360 cho OmniRe: ``instances/`` và ``dynamic_masks/`` theo khung KITTI.

Converter KITTI-360 (``drive2sim/ingest/kitti360.py``) gọi file này để ghi box 3D chính thức
(``data_3d_bboxes.zip``, ``data_3d_bboxes/train/<chuỗi>.xml``) theo đúng định dạng mà script
KITTI của DriveStudio ghi (``drive2sim/recon/preprocess/kitti/process.py``, ``save_objects`` và
``save_dynamic_mask``), để loader KITTI đọc được và OmniRe dựng xe, người thành node riêng.

Quy ước của file XML (``kitti360Scripts``, ``helpers/annotation.py``):
- mỗi phần tử ``<objectN>`` có ``label``, ``semanticId``, ``instanceId``; vật được nhận diện bằng
  cặp (``semanticId``, ``instanceId``);
- ``transform`` (4×4, theo hàng) đưa khối lập phương đơn vị ``±0.5`` sang hệ world của chuỗi
  (cùng hệ với ``poses.txt``): 3 cột đầu là trục x, y, z của box nhân với kích thước;
- vật đứng yên có ``timestamp = -1`` và có mặt từ ``start_frame`` tới ``end_frame``; vật chuyển
  động có một phần tử cho mỗi khung, ``timestamp`` là số hiệu khung.

Trên dữ liệu chuỗi 0003 (kiểm tra 15/11/2026): mọi box xe có trục x dài nhất (chiều dài), z là
chiều cao; 2 trong 1144 box có ma trận lật (định thức âm), được sửa bằng cách đổi chiều trục y.

File này chứa ``CLASS_MAP``, ``Box``, ``box_xml_text``, ``read_boxes``, ``build_instances``,
``project_rect``, ``dynamic_mask_rects``.
"""

import math
import xml.etree.ElementTree as ElementTree
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from drive2sim.core.camera import MEI_FIT_MAX_ANGLE, fisheye_radius

Matrix = list[list[float]]

# Nhãn KITTI-360 → lớp của loader KITTI (``OBJECT_CLASS_NODE_MAPPING`` trong
# ``drive2sim/recon/data/loaders/kitti.py``). Loader không có lớp bus, rơ-moóc hay xe máy riêng:
# xe lớn vào Truck (node cứng), người đi xe vào Cyclist (node biến dạng) như KITTI.
CLASS_MAP = {
    "car": "Car",
    "truck": "Truck",
    "bus": "Truck",
    "trailer": "Truck",
    "caravan": "Truck",
    "train": "Tram",
    "person": "Pedestrian",
    "rider": "Cyclist",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}
# Nhóm mặt nạ như ``KITTI_*_DYNAMIC_CLASSES`` của script KITTI.
MASK_GROUPS = {
    "all": {"Car", "Van", "Truck", "Tram", "Pedestrian", "Person_sitting", "Cyclist"},
    "human": {"Pedestrian", "Person_sitting", "Cyclist"},
    "vehicle": {"Car", "Van", "Truck", "Tram"},
}
BOX_DIR = "data_3d_bboxes"


@dataclass(frozen=True)
class Box:
    """Một box trong file XML, đã tách phép quay khỏi kích thước.

    Attributes:
        key: (``semanticId``, ``instanceId``): cùng một vật qua các khung.
        class_name: Lớp của loader KITTI (``CLASS_MAP``).
        frames: Các số hiệu khung gốc mà box có mặt (một khung với vật chuyển động).
        rotation: Phép quay 3×3 box → world của chuỗi (trục x là chiều dài).
        center: Tâm box trong world của chuỗi.
        size: ``[dài, rộng, cao]`` (mét), thứ tự ``box_size`` của DriveStudio.
    """

    key: tuple[int, int]
    class_name: str
    frames: tuple[int, ...]
    rotation: Matrix
    center: list[float]
    size: list[float]


def box_xml_text(raw_dir: Path, sequence: str) -> Optional[str]:
    """Đọc ``data_3d_bboxes/train/<chuỗi>.xml`` (thư mục hoặc ``data_3d_bboxes.zip``).

    Returns:
        Nội dung file, hoặc ``None`` nếu nguồn không có box của chuỗi này.
    """
    member = f"{BOX_DIR}/train/{sequence}.xml"
    path = raw_dir / member
    if path.is_file():
        return path.read_text("utf-8")
    archive = raw_dir / f"{BOX_DIR}.zip"
    if archive.is_file():
        with zipfile.ZipFile(archive) as zipped:
            if member in zipped.namelist():
                return zipped.read(member).decode("utf-8")
    return None


def _text(element: ElementTree.Element, tag: str) -> str:
    child = element.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"box thiếu trường {tag}")
    return child.text.strip()


def _determinant(m: Matrix) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def read_boxes(text: str) -> list[Box]:
    """Đọc các box thuộc lớp trong ``CLASS_MAP`` từ nội dung file XML.

    Raises:
        ValueError: Nếu một box thiếu trường bắt buộc hoặc có kích thước bằng 0.
    """
    boxes = []
    for element in ElementTree.fromstring(text):
        label = element.find("label")
        if label is None or label.text not in CLASS_MAP:
            continue
        values = [float(v) for v in _text(element.find("transform"), "data").split()]
        m = [values[0:3], values[4:7], values[8:11]]
        size = [math.sqrt(sum(m[i][j] ** 2 for i in range(3))) for j in range(3)]
        if min(size) <= 0:
            raise ValueError(f"box {label.text} có kích thước bằng 0")
        rotation = [[m[i][j] / size[j] for j in range(3)] for i in range(3)]
        if _determinant(rotation) < 0:  # ma trận lật: đổi chiều trục y để thành phép quay
            for row in rotation:
                row[1] = -row[1]
        timestamp = int(_text(element, "timestamp"))
        if timestamp >= 0:
            frames: tuple[int, ...] = (timestamp,)
        else:
            start, end = int(_text(element, "start_frame")), int(_text(element, "end_frame"))
            frames = tuple(range(start, end + 1))
        boxes.append(
            Box(
                key=(int(_text(element, "semanticId")), int(_text(element, "instanceId"))),
                class_name=CLASS_MAP[label.text],
                frames=frames,
                rotation=rotation,
                center=[values[3], values[7], values[11]],
                size=size,
            )
        )
    return boxes


def _pose(rotation: Matrix, center: Sequence[float]) -> Matrix:
    return [[*rotation[i], center[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def build_instances(
    boxes: Sequence[Box], frames: Sequence[int], world_from_global: Matrix
) -> tuple[dict, dict]:
    """Dựng ``instances_info.json`` và ``frame_instances.json`` như script KITTI của DriveStudio.

    Args:
        boxes: Box đã đọc (``read_boxes``).
        frames: Số hiệu khung gốc của cảnh, theo thứ tự khung.
        world_from_global: World của chuỗi → world của cảnh (Velodyne ở khung đầu).

    Returns:
        ``(instances_info, frame_instances)``: id vật đánh lại từ 0 theo thứ tự xuất hiện;
        ``frame_idx`` là chỉ số khung trong cảnh; mọi khung đều có khoá trong ``frame_instances``.
    """
    position = {frame: index for index, frame in enumerate(frames)}
    info: dict[str, dict] = {}
    ids: dict[tuple[int, int], str] = {}
    frame_instances: dict[str, list[int]] = {str(index): [] for index in range(len(frames))}
    annotated: set[tuple[str, int]] = set()
    for box in boxes:
        visible = [position[frame] for frame in box.frames if frame in position]
        if not visible:
            continue
        if box.key not in ids:
            ids[box.key] = str(len(ids))
            info[ids[box.key]] = {
                "id": int(ids[box.key]),
                "class_name": box.class_name,
                "frame_annotations": {"frame_idx": [], "obj_to_world": [], "box_size": []},
            }
        object_id = ids[box.key]
        pose = _matmul(world_from_global, _pose(box.rotation, box.center))
        annotations = info[object_id]["frame_annotations"]
        for index in visible:
            if (object_id, index) in annotated:  # trùng khung (không có trong 0003): giữ box đầu
                continue
            annotated.add((object_id, index))
            annotations["frame_idx"].append(index)
            annotations["obj_to_world"].append(pose)
            annotations["box_size"].append(list(box.size))
            frame_instances[str(index)].append(int(object_id))
    for value in info.values():  # theo thứ tự khung, như script KITTI
        annotations = value["frame_annotations"]
        order = sorted(
            range(len(annotations["frame_idx"])), key=annotations["frame_idx"].__getitem__
        )
        for name in ("frame_idx", "obj_to_world", "box_size"):
            annotations[name] = [annotations[name][i] for i in order]
    return info, frame_instances


def _corners(pose: Matrix, size: Sequence[float]) -> list[list[float]]:
    half = [s / 2 for s in size]
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = (sx * half[0], sy * half[1], sz * half[2])
                corners.append(
                    [sum(pose[i][k] * local[k] for k in range(3)) + pose[i][3] for i in range(3)]
                )
    return corners


def project_rect(
    points: Sequence[Sequence[float]],
    model: str,
    intrinsics: Sequence[float],
    width: int,
    height: int,
) -> Optional[tuple[int, int, int, int]]:
    """Hình chữ nhật bao các điểm (hệ camera, trục OpenCV) chiếu lên ảnh.

    Như script KITTI của DriveStudio: chỉ nhận box khi **mọi** góc nằm trong ảnh. Với camera mắt
    cá, góc lệch trục quá ``MEI_FIT_MAX_ANGLE`` coi như không thấy (đa thức chỉ khớp tới đó).

    Args:
        points: Các góc box trong hệ camera.
        model: ``opencv_pinhole`` hoặc ``opencv_fisheye``.
        intrinsics: Nội dung ``intrinsics/<chỉ số>.txt`` (``fx fy cx cy ...``).
        width: Chiều rộng ảnh (pixel).
        height: Chiều cao ảnh (pixel).

    Returns:
        ``(u0, v0, u1, v1)`` (pixel, ``u1``, ``v1`` không tính), hoặc ``None``.
    """
    fx, fy, cx, cy = intrinsics[:4]
    us, vs = [], []
    for x, y, z in points:
        if model == "opencv_fisheye":
            rho = math.hypot(x, y)
            theta = math.atan2(rho, z)
            if theta > MEI_FIT_MAX_ANGLE:
                return None
            scale = fisheye_radius(theta, intrinsics[4:8]) / rho if rho > 0 else 0.0
            a, b = x * scale, y * scale
        else:
            if z <= 0:
                return None
            a, b = x / z, y / z
        u, v = fx * a + cx, fy * b + cy
        if not (0 <= u < width and 0 <= v < height):
            return None
        us.append(int(u))
        vs.append(int(v))
    if max(us) == min(us) or max(vs) == min(vs):
        return None
    return min(us), min(vs), max(us), max(vs)


def _inverse_rigid(m: Matrix) -> Matrix:
    rotation_t = [[m[j][i] for j in range(3)] for i in range(3)]
    translation = [-sum(rotation_t[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return [[*rotation_t[i], translation[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def dynamic_mask_rects(
    info: dict,
    frame: int,
    ego_pose: Matrix,
    cam_to_ego: Matrix,
    model: str,
    intrinsics: Sequence[float],
    width: int,
    height: int,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Hình chữ nhật của các vật thấy trên một ảnh, theo nhóm mặt nạ (``MASK_GROUPS``).

    Args:
        info: ``instances_info`` (``build_instances``).
        frame: Chỉ số khung trong cảnh.
        ego_pose: Velodyne → world của cảnh ở khung này.
        cam_to_ego: Camera → Velodyne.
        model: Mô hình camera.
        intrinsics: Intrinsics của camera.
        width: Chiều rộng ảnh.
        height: Chiều cao ảnh.
    """
    world_to_cam = _inverse_rigid(_matmul(ego_pose, cam_to_ego))
    rects: dict[str, list[tuple[int, int, int, int]]] = {group: [] for group in MASK_GROUPS}
    for value in info.values():
        annotations = value["frame_annotations"]
        if frame not in annotations["frame_idx"]:
            continue
        at = annotations["frame_idx"].index(frame)
        corners = _corners(
            _matmul(world_to_cam, annotations["obj_to_world"][at]), annotations["box_size"][at]
        )
        rect = project_rect(corners, model, intrinsics, width, height)
        if rect is None:
            continue
        for group, classes in MASK_GROUPS.items():
            if value["class_name"] in classes:
                rects[group].append(rect)
    return rects
