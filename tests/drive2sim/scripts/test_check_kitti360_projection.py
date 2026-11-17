"""Kiểm tra các hàm chiếu và tỉ số cạnh của ``drive2sim/scripts/check_kitti360_projection.py``.

Dữ liệu tổng hợp, chạy trên CPU, không cần dữ liệu KITTI-360 thật. Cần numpy và OpenCV (môi
trường ``drive2sim``); thiếu thì bỏ qua.
"""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from drive2sim.core.camera import MEI_FIT_MAX_ANGLE, fit_mei_to_fisheye  # noqa: E402
from drive2sim.scripts import check_kitti360_projection as check  # noqa: E402

# Tham số Mei cỡ camera mắt cá KITTI-360 (image_02), ảnh 1400 × 1400.
MEI = {
    "xi": 2.2134,
    "k1": 0.0168,
    "k2": 1.6549,
    "p1": 4.2e-4,
    "p2": 4.2e-4,
    "gamma1": 1336.32,
    "gamma2": 1335.79,
    "u0": 716.94,
    "v0": 705.76,
}
# Velodyne (x tới trước, y sang trái, z lên) → camera (x phải, y xuống, z tới trước).
VELO_TO_CAM = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def _directions(count: int, max_angle: float, seed: int = 0) -> "np.ndarray":
    """Tia đơn vị ngẫu nhiên lệch trục quang không quá ``max_angle``."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, max_angle, count)
    phi = rng.uniform(-np.pi, np.pi, count)
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1
    )


def test_mei_without_mirror_is_pinhole() -> None:
    """ξ = 0, không méo: Mei là pinhole với tiêu cự ``γ``."""
    mei = dict(MEI, xi=0.0, k1=0.0, k2=0.0, p1=0.0, p2=0.0)
    points = _directions(200, 1.0) * 7.0
    uv, valid = check.project_mei(points, mei)
    k = np.array([[mei["gamma1"], 0, mei["u0"]], [0, mei["gamma2"], mei["v0"]], [0, 0, 1.0]])
    expected, _ = check.project_pinhole(points, k)
    assert valid.all()
    np.testing.assert_allclose(uv, expected, atol=1e-9)


def test_opencv_fisheye_matches_cv2() -> None:
    """Công thức OpenCV fisheye viết bằng numpy cho cùng pixel với ``cv2.fisheye``."""
    intrinsics = [415.9, 415.7, 716.9, 705.8, -0.0096, 0.0094, -0.0018, -0.0004]
    points = _directions(300, 1.4) * 10.0
    uv, theta, valid = check.project_opencv_fisheye(points, intrinsics)
    fx, fy, cx, cy = intrinsics[:4]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    reference, _ = cv2.fisheye.projectPoints(
        points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, np.array(intrinsics[4:])
    )
    assert valid.all() and theta.max() <= 1.4
    np.testing.assert_allclose(uv, reference.reshape(-1, 2), atol=1e-6)


def test_fitted_fisheye_follows_mei_inside_fit_range() -> None:
    """OpenCV fisheye khớp từ Mei lệch Mei (bỏ méo tiếp tuyến) dưới 0,05 px trong miền khớp."""
    fx, fy, coeffs = fit_mei_to_fisheye(
        MEI["xi"], MEI["k1"], MEI["k2"], MEI["gamma1"], MEI["gamma2"]
    )
    points = _directions(500, MEI_FIT_MAX_ANGLE) * 15.0
    uv_mei, _ = check.project_mei(points, MEI, tangential=False)
    uv_fit, _, _ = check.project_opencv_fisheye(points, [fx, fy, MEI["u0"], MEI["v0"], *coeffs])
    assert np.abs(uv_mei - uv_fit).max() < 0.05


def test_yaw_rotation_shifts_pinhole_point_horizontally() -> None:
    """Xoay camera quanh trục y làm điểm trên trục quang dời ngang ``f·tan(góc)``."""
    k = np.array([[500.0, 0, 200], [0, 500.0, 100], [0, 0, 1.0]])
    point = np.array([[0.0, 0.0, 10.0]])
    uv, _ = check.project_pinhole(check.transform(check.yaw_rotation(3.0), point), k)
    assert uv[0, 0] - 200 == pytest.approx(500.0 * np.tan(np.radians(3.0)))
    assert uv[0, 1] == pytest.approx(100.0)


def test_depth_discontinuities_marks_near_side_only() -> None:
    """Trên một vòng quét, chỉ hai điểm ở viền vật phía trước được đánh dấu."""
    azimuth = np.radians(np.arange(-10.0, 10.0, 0.2))
    rng = np.where(np.abs(azimuth) < np.radians(3.0), 5.0, 20.0)
    points = np.stack([rng * np.cos(azimuth), rng * np.sin(azimuth), np.zeros_like(rng)], axis=1)
    edge = check.depth_discontinuities(points)
    near = np.flatnonzero(rng == 5.0)
    assert np.flatnonzero(edge).tolist() == [near[0], near[-1]]
    # Bước phương vị lớn (thiếu điểm phản xạ, đổi tia): không coi là kề nhau.
    gap = points[[0, -1]] * np.array([[1.0], [0.25]])
    assert not check.depth_discontinuities(gap).any()


def _pole_scene() -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Cột rộng 1 m cách 5 m trước một bức tường cách 20 m, quét như Velodyne.

    Returns:
        ``(điểm theo thứ tự quét trong hệ Velodyne, ma trận nội, ảnh)``; ảnh có cột sáng trên nền
        tối nhiễu, đúng hình học của camera đặt tại gốc.
    """
    rays = []
    for elevation in np.radians(np.linspace(-6.0, 4.0, 16)):
        for azimuth in np.radians(np.arange(-20.0, 20.0, 0.18)):
            rays.append(
                [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]
            )
    rays = np.array(rays)
    hits_pole = np.abs(rays[:, 1] / rays[:, 0] * 5.0) < 0.5
    depth_x = np.where(hits_pole, 5.0, 20.0)
    points = rays * (depth_x / rays[:, 0])[:, None]
    k = np.array([[500.0, 0, 200], [0, 500.0, 100], [0, 0, 1.0]])
    noise = np.random.default_rng(1).normal(0.0, 4.0, (200, 400))
    image = np.full((200, 400), 60.0) + noise
    image[:, 150:250] += 140.0  # cột chiếm |u − cx| < f · 0,5 / 5 = 50 px
    return points, k, np.clip(image, 0, 255).astype(np.uint8)


def test_edge_score_drops_when_camera_is_rotated() -> None:
    """Hình học đúng: tỉ số cạnh lớn hơn 1 rõ rệt; xoay lệch 3 độ thì tỉ số giảm mạnh."""
    points, k, image = _pole_scene()
    strength = check.edge_strength(image)
    is_edge = check.depth_discontinuities(points)
    assert is_edge.sum() == 32  # hai viền cột × 16 tia

    def ratio(degrees: float) -> float:
        cam = check.transform(check.yaw_rotation(degrees) @ VELO_TO_CAM, points)
        uv, valid = check.project_pinhole(cam, k)
        valid &= check.inside_image(uv, 400, 200)
        return check.edge_score(strength, uv[valid], is_edge[valid]).ratio

    aligned, rotated = ratio(0.0), ratio(3.0)
    assert aligned > 3.0
    assert rotated < 0.5 * aligned


def test_edge_score_adds_up_and_handles_no_edges() -> None:
    """Gộp hai kết quả bằng tổng; không có điểm mép thì tỉ số là ``nan``."""
    strength = np.ones((4, 4), np.float32)
    uv = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    none = check.edge_score(strength, uv, np.zeros(3, bool))
    assert np.isnan(none.ratio)
    some = check.edge_score(strength, uv, np.array([True, False, False]))
    total = none + some
    assert (total.edge_count, total.all_count) == (1, 6)
    assert total.ratio == pytest.approx(1.0)
