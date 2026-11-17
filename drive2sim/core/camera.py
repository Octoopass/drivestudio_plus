"""Đổi mô hình camera Mei (unified omnidirectional) sang OpenCV fisheye.

KITTI-360 hiệu chỉnh hai camera mắt cá (``image_02``, ``image_03``) bằng mô hình Mei (C. Mei và P.
Rives, "Single View Point Omnidirectional Camera Calibration from Planar Grids", ICRA 2007): một
tia đơn vị ``(sin θ, 0, cos θ)`` được chiếu qua mặt cầu đơn vị rồi lên mặt phẳng ảnh,
``m = sin θ / (cos θ + ξ)``, méo xuyên tâm ``m·(1 + k1·m² + k2·m⁴)`` (cộng méo tiếp tuyến ``p1``,
``p2``), rồi nhân tiêu cự ``γ1``, ``γ2``. Renderer 3DGUT, 3DGEER và loader của tool chỉ có OpenCV
fisheye: ``r = f·θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)``.

Cách khớp: cả hai mô hình là hàm của một góc ``θ``, đối xứng quanh trục quang. Với mỗi ``θ`` trong
miền khớp, đặt bán kính pixel bằng nhau theo trục u:

    γ1·r_Mei(θ) = a0·θ + a1·θ³ + a2·θ⁵ + a3·θ⁷ + a4·θ⁹,   với a0 = fx, a_i = fx·k_i.

Phương trình tuyến tính theo ``a0..a4``, nên giải trực tiếp bằng bình phương tối thiểu (phương
trình chuẩn, khử Gauss), không lặp. ``fx = a0``, ``k_i = a_i / a0``; tỉ lệ hai trục của Mei giữ
nguyên nên ``fy = fx·γ2/γ1``; tâm ảnh giữ ``u0``, ``v0``. Méo tiếp tuyến bị bỏ vì OpenCV fisheye
không có (KITTI-360: ``p1``, ``p2`` cỡ 4e-4).

Camera mắt cá KITTI-360 nhìn rộng hơn 180 độ, còn renderer UT không vẽ được tia từ π/2 trở ra, nên
chỉ khớp tới ``MEI_FIT_MAX_ANGLE`` (1,45 rad, khoảng 83°); pixel có tia lệch trục hơn góc này phải
bị che khi train và đánh giá.

Viết bằng Python thuần (không numpy) để chạy trong môi trường nhẹ của CLI.

File này chứa ``MEI_FIT_MAX_ANGLE``, ``mei_radius``, ``fisheye_radius``, ``fit_mei_to_fisheye``,
``mei_fit_error_px``.
"""

import math
from collections.abc import Sequence

MEI_FIT_MAX_ANGLE = 1.45
FIT_SAMPLES = 400  # số góc đều trên (0, MEI_FIT_MAX_ANGLE] dùng để khớp


def mei_radius(theta: float, xi: float, k1: float, k2: float) -> float:
    """Bán kính chuẩn hoá (nhân ``γ`` ra pixel) của tia lệch trục ``theta`` theo mô hình Mei.

    ``m = sin θ / (cos θ + ξ)`` rồi méo xuyên tâm ``m·(1 + k1·m² + k2·m⁴)``; bỏ méo tiếp tuyến.
    """
    m = math.sin(theta) / (math.cos(theta) + xi)
    m2 = m * m
    return m * (1.0 + k1 * m2 + k2 * m2 * m2)


def fisheye_radius(theta: float, coeffs: Sequence[float]) -> float:
    """``θ_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)`` của OpenCV fisheye (nhân ``f`` ra pixel)."""
    t2 = theta * theta
    poly, power = 1.0, 1.0
    for k in coeffs:
        power *= t2
        poly += k * power
    return theta * poly


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Giải hệ vuông bằng khử Gauss có chọn trụ theo cột."""
    n = len(rhs)
    a = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("hệ phương trình suy biến")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def fit_mei_to_fisheye(
    xi: float, k1: float, k2: float, gamma1: float, gamma2: float
) -> tuple[float, float, list[float]]:
    """OpenCV fisheye gần nhất với một camera Mei trên ``(0, MEI_FIT_MAX_ANGLE]``.

    Returns:
        ``(fx, fy, [k1, k2, k3, k4])``; tâm ảnh giữ nguyên ``u0``, ``v0`` của Mei.
    """
    powers = (1, 3, 5, 7, 9)
    normal = [[0.0] * 5 for _ in range(5)]
    rhs = [0.0] * 5
    for i in range(1, FIT_SAMPLES + 1):
        theta = MEI_FIT_MAX_ANGLE * i / FIT_SAMPLES
        row = [theta**p for p in powers]
        target = gamma1 * mei_radius(theta, xi, k1, k2)
        for r in range(5):
            rhs[r] += row[r] * target
            for c in range(5):
                normal[r][c] += row[r] * row[c]
    a = _solve(normal, rhs)
    fx = a[0]
    return fx, fx * gamma2 / gamma1, [value / fx for value in a[1:]]


def mei_fit_error_px(
    xi: float,
    k1: float,
    k2: float,
    gamma: float,
    fx: float,
    coeffs: Sequence[float],
    max_angle: float = MEI_FIT_MAX_ANGLE,
    samples: int = 2000,
) -> float:
    """Tính sai số bán kính lớn nhất (pixel) giữa Mei và fisheye trên ``[0, max_angle]``."""
    return max(
        abs(
            gamma * mei_radius(max_angle * i / samples, xi, k1, k2)
            - fx * fisheye_radius(max_angle * i / samples, coeffs)
        )
        for i in range(samples + 1)
    )
