"""Thứ tự thư mục ảnh và ``calib.yml`` mà MCGS-SLAM đọc, dựng từ rig camera của cảnh (cần numpy).

MCGS-SLAM nhận ảnh qua nhiều thư mục (mỗi thư mục một camera) và một file ``calib.yml``; cách nó
ghép thư mục và đọc calib suy ra từ mã nguồn @147eb3a (``streams.py``, ``depth_video.py``,
``options.py``), ghi ở ``docs/superpowers/specs/2026-11-25-mcgs-integration.md``:
- Hai thư mục đầu là một cặp stereo nối bằng ``baseline``; thư mục thứ ba trở đi là camera thêm,
  mỗi cái có một hàng ``T_cami_cam0`` (hàng 0 là đơn vị).
- Bảy số của một phép biến đổi theo lietorch ``SE3``: ``tx ty tz qx qy qz qw``, biến điểm từ hệ
  camera tham chiếu (thư mục 0) sang hệ camera đó.
- Khi có từ ba camera, mẫu Waymo của MCGS-SLAM lặp camera đối tác stereo ở cuối danh sách.

File này chứa:
- ``se3_vector``: ma trận 4×4 → bảy số SE3.
- ``relative_to_reference``: phép biến đổi từ camera tham chiếu sang một camera khác.
- ``rig_layout``: thứ tự thư mục ảnh.
- ``calib_document``: nội dung ``calib.yml``.
- ``check_cameras``: kiểm tra rig dùng được cho MCGS-SLAM.
- ``slam_frames``: các khung đưa vào SLAM.

``drive2sim/stages/prep/mcgs.py`` gọi file này khi xuất đầu vào cho SLAM. Xem ARCHITECTURE mục
2.19.
"""

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np

from drive2sim.external.mcgs.views import PINHOLE_MODELS

MIN_CAMERAS = 2  # MCGS-SLAM luôn chạy stereo: cần ít nhất một cặp camera
UNIT_TRANSFORM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # hàng 0 của T_cami_cam0


def se3_vector(transform: np.ndarray) -> list[float]:
    """Đổi ma trận biến đổi 4×4 thành bảy số ``[tx, ty, tz, qx, qy, qz, qw]`` của lietorch.

    Quaternion lấy từ ma trận quay bằng công thức Shepperd (chọn nhánh theo phần tử lớn nhất để
    không chia cho số gần 0) và đổi dấu sao cho ``qw >= 0``: ``q`` và ``-q`` là cùng một phép quay,
    nên chọn một dạng duy nhất.

    Args:
        transform: Ma trận ``(4, 4)``; ba cột đầu của ba hàng đầu là ma trận quay.

    Returns:
        Bảy số thực; ba số đầu là tịnh tiến.
    """
    rot = np.asarray(transform, dtype=np.float64)[:3, :3]
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx, qy, qz = (
            (rot[2, 1] - rot[1, 2]) / s,
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[1, 0] - rot[0, 1]) / s,
        )
    elif rot[0, 0] >= rot[1, 1] and rot[0, 0] >= rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw, qx = (rot[2, 1] - rot[1, 2]) / s, 0.25 * s
        qy, qz = (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] >= rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw, qy = (rot[0, 2] - rot[2, 0]) / s, 0.25 * s
        qx, qz = (rot[0, 1] + rot[1, 0]) / s, (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw, qz = (rot[1, 0] - rot[0, 1]) / s, 0.25 * s
        qx, qy = (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s
    quaternion = np.array([qx, qy, qz, qw])
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    translation = np.asarray(transform, dtype=np.float64)[:3, 3]
    return [float(v) for v in (*translation, *quaternion)]


def relative_to_reference(
    cam_to_ego: dict[str, np.ndarray], reference: str, camera: str
) -> np.ndarray:
    """Phép biến đổi đưa điểm từ hệ camera tham chiếu sang hệ của ``camera``.

    Bằng ``inv(cam_to_ego[camera]) @ cam_to_ego[reference]``: điểm ở hệ ``reference`` đi qua hệ
    ego rồi về hệ ``camera``. Đây là chiều ``T_cami_cam0`` của MCGS-SLAM (kiểm ở bước chạy thật,
    ghi trong kế hoạch Task 7).

    Args:
        cam_to_ego: ``camera_id`` → ma trận camera → ego ``(4, 4)`` (``extrinsics`` của cảnh).
        reference: Camera tham chiếu (camera 0 của SLAM).
        camera: Camera đích.

    Returns:
        Ma trận ``(4, 4)``.
    """
    return np.linalg.inv(cam_to_ego[camera]) @ cam_to_ego[reference]


def rig_layout(
    cameras: Sequence[str],
    cam_to_ego: dict[str, np.ndarray],
    reference: str,
    partner: Optional[str] = None,
) -> list[str]:
    """Xếp thứ tự thư mục ảnh đưa vào MCGS-SLAM.

    Hai camera: ``[reference, partner]``. Từ ba camera: ``[reference, partner, <các camera còn
    lại theo thứ tự cameras>..., partner]``, như mẫu Waymo của MCGS-SLAM (``calib/100613.yml``):
    thư mục thứ hai là camera đối tác stereo và nó lặp lại ở cuối, không phải lỗi gõ.

    Args:
        cameras: ``camera_id`` của các camera dùng cho SLAM, theo thứ tự ưu tiên.
        cam_to_ego: ``camera_id`` → ma trận camera → ego ``(4, 4)``.
        reference: Camera 0 của SLAM; phải nằm trong ``cameras``.
        partner: Camera đối tác stereo; ``None`` thì chọn camera khác ``reference`` có tâm gần
            ``reference`` nhất (hoà thì camera đứng trước trong ``cameras``).

    Returns:
        Danh sách ``camera_id`` theo thứ tự thư mục; một ``camera_id`` có thể xuất hiện hai lần.

    Raises:
        ValueError: Nếu ``reference`` hoặc ``partner`` không hợp lệ, hoặc có dưới hai camera.
    """
    cameras = list(cameras)
    if reference not in cameras:
        raise ValueError(f"reference_camera {reference} không nằm trong {', '.join(cameras)}")
    others = [camera for camera in cameras if camera != reference]
    if len(others) + 1 < MIN_CAMERAS:
        raise ValueError("MCGS-SLAM cần ít nhất hai camera (một cặp stereo)")
    if partner is None:
        origin = cam_to_ego[reference][:3, 3]
        # min giữ phần tử đầu khi hoà khoảng cách, nên thứ tự cameras phân xử
        partner = min(others, key=lambda c: float(np.linalg.norm(cam_to_ego[c][:3, 3] - origin)))
    elif partner not in others:
        raise ValueError(
            f"stereo_partner {partner} phải là một camera của rig khác reference_camera "
            f"({', '.join(others)})"
        )
    extra = [camera for camera in others if camera != partner]
    layout = [reference, partner]
    if extra:
        layout += [*extra, partner]
    return layout


def _intrinsic_row(values: Sequence[float]) -> list[float]:
    """Một hàng ``intrinsic``: ``fx fy cx cy``, thêm ``k1 k2 p1 p2 k3`` chỉ khi có hệ số khác 0.

    MCGS-SLAM chỉ gọi ``cv2.undistort`` khi hàng có hơn 4 cột, nên ảnh đã rectified (KITTI-360)
    giữ 4 cột để khỏi méo lại hai lần.
    """
    row = [float(v) for v in values]
    distortion = row[4:9]
    return row[:4] + distortion if any(distortion) else row[:4]


def calib_document(
    layout: list[str],
    intrinsics: dict[str, list[float]],
    cam_to_ego: dict[str, np.ndarray],
) -> dict:
    """Dựng nội dung ``calib.yml`` cho một thứ tự thư mục.

    Args:
        layout: Kết quả ``rig_layout``; ``layout[0]`` là camera tham chiếu.
        intrinsics: ``camera_id`` → ``[fx, fy, cx, cy]`` hoặc ``[fx, fy, cx, cy, k1, k2, p1, p2,
            k3]`` (``intrinsics/<chỉ số>.txt`` của cảnh).
        cam_to_ego: ``camera_id`` → ma trận camera → ego ``(4, 4)``.

    Returns:
        Từ điển có ``intrinsic`` (một hàng mỗi phần tử ``layout``), ``camera``, ``baseline``,
        ``T_cami_cam0``, ``timescale``; ghi ra YAML bằng ``yaml.safe_dump``.
    """
    reference = layout[0]

    def transform_to(camera: str) -> list[float]:
        return se3_vector(relative_to_reference(cam_to_ego, reference, camera))

    return {
        "intrinsic": [_intrinsic_row(intrinsics[camera]) for camera in layout],
        "camera": "pinhole",
        "baseline": transform_to(layout[1]),
        "T_cami_cam0": [list(UNIT_TRANSFORM)] + [transform_to(camera) for camera in layout[2:]],
        "timescale": 1,
    }


def check_cameras(models: dict[str, str]) -> None:
    """Kiểm tra rig dùng được cho MCGS-SLAM trước khi chạy GPU.

    Args:
        models: ``camera_id`` → mô hình camera trong manifest (``opencv_pinhole``, ...).

    Raises:
        ValueError: Nếu có camera không phải pinhole (SLAM không nhận fisheye) hoặc dưới hai
            camera (SLAM luôn chạy stereo).
    """
    unsupported = [f"{c} ({m})" for c, m in models.items() if m not in PINHOLE_MODELS]
    if unsupported:
        raise ValueError(
            "MCGS-SLAM chỉ nhận camera pinhole; bỏ camera "
            + ", ".join(unsupported)
            + " khỏi tham số cameras"
        )
    if len(models) < MIN_CAMERAS:
        raise ValueError(
            f"MCGS-SLAM cần ít nhất {MIN_CAMERAS} camera pinhole (một cặp stereo), "
            f"split chỉ có {len(models)}"
        )


def slam_frames(frame_range: tuple[int, int], test_frames: set[int], held_out: bool) -> list[int]:
    """Các khung đưa vào SLAM: cả đoạn của split, bỏ khung test nếu ``held_out``.

    Args:
        frame_range: Khung đầu và khung cuối của split, tính cả hai đầu.
        test_frames: Chỉ số khung có ảnh test của camera đã train.
        held_out: ``True`` thì khung test không vào SLAM, để điểm đo là điểm trên ảnh chưa thấy.

    Returns:
        Chỉ số khung tăng dần.
    """
    first, last = frame_range
    return [f for f in range(first, last + 1) if not (held_out and f in test_frames)]
