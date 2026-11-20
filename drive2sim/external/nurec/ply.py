"""Đọc, ghi và đổi hệ toạ độ PLY Gaussian của Instant NuRec (cần numpy).

PLY của Instant NuRec nằm trong hệ world của clip NCore; viewer và so sánh của drive2sim cần hệ
``world`` của cảnh (CONTRACTS mục 2), nên stage ``export.nurec_ply`` đổi hệ bằng các hàm ở đây.

File này chứa:
- ``read_ply``, ``write_ply``: PLY nhị phân little-endian một phần tử ``vertex``.
- ``transform_ply``: đổi vị trí và hướng Gaussian theo một biến đổi cứng.
- ``ncore_to_world``: ma trận hệ world của clip NCore → hệ ``world`` của cảnh KITTI-360.
"""

from pathlib import Path

import numpy as np

# camera OpenCV (x phải, y xuống, z tới) → rig FLU (x tới, y trái, z lên), như converter
R_CAM_TO_FLU = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)
PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    """Đọc PLY nhị phân little-endian một phần tử ``vertex``.

    Returns:
        ``(dòng header, mảng có cấu trúc theo thuộc tính)``.

    Raises:
        ValueError: Nếu không phải PLY nhị phân little-endian hay có thuộc tính lạ.
    """
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii", "replace").rstrip("\n")
            header.append(line)
            if line == "end_header":
                break
            if not line and len(header) > 200:
                raise ValueError(f"{path.name}: thiếu end_header")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path.name}: chỉ đọc được PLY nhị phân little-endian")
        count, props = 0, []
        for line in header:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
            elif parts[:1] == ["property"]:
                if parts[1] not in PLY_TYPES:
                    raise ValueError(f"{path.name}: kiểu thuộc tính lạ {parts[1]}")
                props.append((parts[2], PLY_TYPES[parts[1]]))
        data = np.frombuffer(stream.read(), dtype=np.dtype(props), count=count)
    return header, data


def write_ply(path: Path, header: list[str], data: np.ndarray) -> None:
    """Ghi lại PLY với header cũ (số phần tử cập nhật theo ``data``)."""
    lines = [
        f"element vertex {len(data)}" if line.startswith("element vertex") else line
        for line in header
    ]
    with path.open("wb") as stream:
        stream.write(("\n".join(lines) + "\n").encode("ascii"))
        stream.write(np.ascontiguousarray(data).tobytes())


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tích quaternion (w, x, y, z), ``a`` một quaternion, ``b`` mảng ``[N, 4]``."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _rotation_quat(rotation: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) của ma trận xoay 3×3."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quat = np.array(q, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def transform_ply(data: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Đổi hệ toạ độ Gaussian theo phép biến đổi cứng 4×4: vị trí và hướng (``rot_*``).

    Chỉ hợp với Gaussian không có SH bậc cao (``f_rest_*``), như PLY của Instant NuRec; tỉ lệ,
    độ mờ, màu giữ nguyên.

    Raises:
        ValueError: Nếu ``transform`` không phải biến đổi cứng hoặc PLY có ``f_rest_*``.
    """
    rotation, translation = transform[:3, :3], transform[:3, 3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("phép đổi hệ phải là biến đổi cứng (ma trận xoay trực chuẩn)")
    names = data.dtype.names or ()
    if any(name.startswith("f_rest_") for name in names):
        raise ValueError("PLY có SH bậc cao (f_rest_*): chưa hỗ trợ xoay SH")
    out = data.copy()
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    moved = xyz @ rotation.T + translation
    for i, axis in enumerate(("x", "y", "z")):
        out[axis] = moved[:, i]
    if all(f"rot_{i}" in names for i in range(4)):
        quats = np.stack([data[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        rotated = _quat_mul(_rotation_quat(rotation), quats)
        for i in range(4):
            out[f"rot_{i}"] = rotated[:, i]
    return out


def ncore_to_world(cam0_to_world: np.ndarray) -> np.ndarray:
    """Ma trận hệ world của clip NCore → hệ ``world`` của cảnh drive2sim (KITTI-360).

    World của clip là rig FLU của khung đầu, gốc ở camera ``image_00``
    (``tools/kitti360_to_ncore.py``); world của cảnh là Velodyne ở khung đầu
    (CONTRACTS mục 2). Hai khung đầu trùng nhau nên phép đổi là hằng: ``cam0 → Velodyne`` nhân
    ``rig → cam0``.

    Args:
        cam0_to_world: ``extrinsics/0.txt`` của cảnh (camera ``image_00`` → Velodyne, trục OpenCV).

    Returns:
        Ma trận 4×4.
    """
    return cam0_to_world @ np.linalg.inv(R_CAM_TO_FLU)
