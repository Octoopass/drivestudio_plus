"""Toán của mô hình camera dùng chung: đổi mô hình Mei (omnidirectional) sang OpenCV fisheye.

KITTI-360 hiệu chỉnh hai camera mắt cá (``image_02``, ``image_03``) bằng mô hình Mei (mô hình
gương cầu, tham số ``xi`` cộng méo xuyên tâm ``k1``, ``k2`` và tiếp tuyến ``p1``, ``p2``). Tool chỉ
có ba mô hình camera chuẩn (ARCHITECTURE mục 2.15), và renderer 3DGUT của gsplat chỉ vẽ được
OpenCV fisheye (``theta_d = theta·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)``). File này khớp đa thức
OpenCV fisheye với Mei bằng bình phương tối thiểu, theo cách của teammate
(``tools/colab_cli_kitti360.py``, hàm ``mei_to_fisheye``, repo thaianh20021/fisheye-fusion-lab).

File này chứa:
- ``MEI_FIT_MAX_ANGLE``: góc tia lớn nhất (rad) mà phép khớp đảm bảo; tia lệch trục hơn góc này
  phải bị che khi train và đánh giá.
- ``mei_radius``: bán kính trên ảnh chuẩn hoá của một tia theo mô hình Mei (bỏ méo tiếp tuyến).
- ``fisheye_radius``: bán kính trên ảnh chuẩn hoá theo đa thức OpenCV fisheye.
- ``fit_mei_to_fisheye``: tìm ``fx, fy`` và ``k1..k4`` OpenCV fisheye gần nhất với một camera Mei.
- ``mei_fit_error_px``: sai số lớn nhất của phép khớp, tính bằng pixel.

Converter KITTI-360 (``drive2sim/ingest/kitti360.py``) gọi file này khi ghi intrinsics của camera
mắt cá; trainer ``drive2sim/recon/tools/train_gsplat_3dgut.py`` dùng ``fisheye_radius`` và
``MEI_FIT_MAX_ANGLE`` để dựng mask vùng ảnh không vẽ được. Viết bằng Python thuần (không numpy) để
chạy được trong môi trường nhẹ của CLI.

Giới hạn (ghi trong ``docs/datasets/kitti360.md``):
- Méo tiếp tuyến ``p1``, ``p2`` của Mei bị bỏ, vì OpenCV fisheye không có; với KITTI-360 hai hệ
  số này cỡ ``4e-4``, gây lệch dưới 1 pixel ở mép ảnh (đo ở ``docs/datasets/kitti360.md``).
- Camera Mei của KITTI-360 nhìn rộng hơn 180 độ; renderer UT của gsplat không vẽ được tia có
  ``theta >= pi/2``, và đa thức chỉ được khớp tới ``MEI_FIT_MAX_ANGLE``.
"""

import math
from collections.abc import Sequence

# Góc tia lớn nhất (rad, khoảng 83 độ) dùng khi khớp, theo teammate (np.linspace(0.02, 1.45, 256)).
# Vẫn dưới pi/2 là giới hạn của renderer UT trong gsplat.
MEI_FIT_MAX_ANGLE = 1.45
_FIT_MIN_ANGLE = 0.02
_FIT_SAMPLES = 256


def mei_radius(theta: float, xi: float, k1: float, k2: float) -> float:
    """Tính bán kính trên ảnh chuẩn hoá (đơn vị tiêu cự ``gamma``) của một tia theo mô hình Mei.

    Tia đơn vị hợp với trục quang góc ``theta`` được chiếu lên mặt cầu rồi lên mặt phẳng ảnh:
    ``u = sin θ / (cos θ + xi)``, sau đó méo xuyên tâm ``u·(1 + k1·u² + k2·u⁴)``. Pixel bằng
    ``gamma · bán kính`` (cộng tâm ảnh). Bỏ méo tiếp tuyến ``p1``, ``p2``.

    Args:
        theta: Góc giữa tia và trục quang, radian.
        xi: Tham số gương của Mei.
        k1: Hệ số méo xuyên tâm bậc 2 của Mei.
        k2: Hệ số méo xuyên tâm bậc 4 của Mei.

    Returns:
        Bán kính chuẩn hoá (nhân với ``gamma`` để ra pixel).
    """
    u = math.sin(theta) / (math.cos(theta) + xi)
    u2 = u * u
    return u * (1.0 + k1 * u2 + k2 * u2 * u2)


def fisheye_radius(theta: float, coeffs: Sequence[float]) -> float:
    """Tính bán kính trên ảnh chuẩn hoá theo đa thức OpenCV fisheye.

    ``theta_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)``; pixel bằng ``fx · theta_d`` theo hướng của
    tia (cộng tâm ảnh). Đây đúng là mô hình gsplat dùng với ``camera_model="fisheye"``.

    Args:
        theta: Góc giữa tia và trục quang, radian.
        coeffs: ``[k1, k2, k3, k4]``.

    Returns:
        ``theta_d``.
    """
    t2 = theta * theta
    k1, k2, k3, k4 = coeffs
    return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def _least_squares(rows: list[list[float]], target: list[float]) -> list[float]:
    """Giải bài toán bình phương tối thiểu ``min ‖A·x − b‖`` bằng Gram–Schmidt cải tiến.

    Dùng phân tích QR thay vì phương trình chuẩn ``AᵀA·x = Aᵀb`` vì các cột θ², θ⁴, θ⁶, θ⁸ gần
    phụ thuộc tuyến tính; QR giữ được độ chính xác tốt hơn.

    Args:
        rows: Ma trận ``A`` theo hàng, ``m × n`` với ``m >= n``.
        target: Vector ``b`` độ dài ``m``.

    Returns:
        Nghiệm ``x`` độ dài ``n``.
    """
    n = len(rows[0])
    columns = [[row[j] for row in rows] for j in range(n)]
    q: list[list[float]] = []
    r = [[0.0] * n for _ in range(n)]
    for j in range(n):
        v = list(columns[j])
        for i in range(j):
            r[i][j] = sum(a * b for a, b in zip(q[i], v))
            v = [a - r[i][j] * b for a, b in zip(v, q[i])]
        r[j][j] = math.sqrt(sum(a * a for a in v))
        q.append([a / r[j][j] for a in v])
    qtb = [sum(a * b for a, b in zip(q[j], target)) for j in range(n)]
    x = [0.0] * n
    for j in reversed(range(n)):
        x[j] = (qtb[j] - sum(r[j][k] * x[k] for k in range(j + 1, n))) / r[j][j]
    return x


def fit_mei_to_fisheye(
    xi: float, k1: float, k2: float, gamma1: float, gamma2: float
) -> tuple[float, float, list[float]]:
    """Tìm mô hình OpenCV fisheye gần nhất với một camera Mei.

    Gần trục quang, Mei cho bán kính ≈ ``θ / (1 + xi)``, nên tiêu cự fisheye là
    ``gamma / (1 + xi)``. Còn lại khớp ``k1..k4`` sao cho
    ``θ·(1 + k1·θ² + ...) = (1 + xi)·mei_radius(θ)`` trên 256 góc đều từ 0,02 tới
    ``MEI_FIT_MAX_ANGLE`` rad, đúng cách của teammate (``mei_to_fisheye``).

    Args:
        xi: Tham số gương của Mei.
        k1: Hệ số méo xuyên tâm bậc 2 của Mei.
        k2: Hệ số méo xuyên tâm bậc 4 của Mei.
        gamma1: Tiêu cự Mei theo trục u, pixel.
        gamma2: Tiêu cự Mei theo trục v, pixel.

    Returns:
        ``(fx, fy, [k1, k2, k3, k4])`` của OpenCV fisheye; tâm ảnh giữ nguyên ``u0``, ``v0``.
    """
    step = (MEI_FIT_MAX_ANGLE - _FIT_MIN_ANGLE) / (_FIT_SAMPLES - 1)
    rows, target = [], []
    for i in range(_FIT_SAMPLES):
        theta = _FIT_MIN_ANGLE + i * step
        t2 = theta * theta
        rows.append([t2, t2**2, t2**3, t2**4])
        target.append((1.0 + xi) * mei_radius(theta, xi, k1, k2) / theta - 1.0)
    coeffs = _least_squares(rows, target)
    return gamma1 / (1.0 + xi), gamma2 / (1.0 + xi), coeffs


def mei_fit_error_px(
    xi: float,
    k1: float,
    k2: float,
    gamma: float,
    coeffs: Sequence[float],
    max_angle: float = MEI_FIT_MAX_ANGLE,
    samples: int = 2000,
) -> float:
    """Tính sai số lớn nhất (pixel) giữa đa thức fisheye đã khớp và mô hình Mei.

    So bán kính pixel ``gamma·mei_radius(θ)`` với ``gamma/(1+xi)·fisheye_radius(θ)`` trên lưới dày
    từ 0 tới ``max_angle``. Không tính méo tiếp tuyến (đa thức không có).

    Args:
        xi: Tham số gương của Mei.
        k1: Hệ số méo xuyên tâm bậc 2 của Mei.
        k2: Hệ số méo xuyên tâm bậc 4 của Mei.
        gamma: Tiêu cự Mei, pixel.
        coeffs: ``[k1, k2, k3, k4]`` đã khớp.
        max_angle: Góc lớn nhất cần đo, radian.
        samples: Số góc đo.

    Returns:
        Sai số lớn nhất, pixel.
    """
    worst = 0.0
    for i in range(1, samples + 1):
        theta = max_angle * i / samples
        exact = gamma * mei_radius(theta, xi, k1, k2)
        fitted = gamma / (1.0 + xi) * fisheye_radius(theta, coeffs)
        worst = max(worst, abs(exact - fitted))
    return worst
