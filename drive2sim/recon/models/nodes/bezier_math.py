"""Toán đường cong Bézier thuần torch, chép từ thuật toán của BezierGS.

Nguồn: BezierGS (ICCV 2025), LogosIntelligenceLab/BezierGS, giấy phép MIT, commit ``6df0a36``,
file ``scene/bz_gaussian_model.py``. Khác bản gốc: không gọi ``.cuda()``, dùng device của tensor
đầu vào nên chạy được trên CPU (test).

File này chứa:
- ``bernstein``, ``bernstein_derivative``: hệ số đa thức Bernstein và đạo hàm theo ``t``.
- ``fit_control_points``: fit control point của đường Bézier qua các điểm bằng bình phương tối
  thiểu.
- ``chord_length_params``: tham số hoá các điểm của một đường đi theo độ dài dây cung.
- ``fit_trajectory``: fit đường đi của một vật bằng đường Bézier, lặp chiếu lại tham số.
- ``fit_time_map``: fit đa thức Bernstein đổi thời gian thật sang tham số ``u`` của Bézier.

``BezierNodes`` (``drive2sim/recon/models/nodes/bezier.py``) dùng file này khi khởi tạo từ dữ
liệu instance và khi tính vị trí, vận tốc của từng Gaussian theo thời gian. Xem
``docs/superpowers/specs/2026-11-20-beziergs-design.md`` mục 1.
"""

import math

import torch
from torch import Tensor


def _binomial_coefficients(order: int, ts: Tensor) -> Tensor:
    """Tính ``C(order, k)`` với ``k = 0..order`` cùng dtype và device với ``ts``."""
    return torch.tensor(
        [math.comb(order, k) for k in range(order + 1)], dtype=ts.dtype, device=ts.device
    )


def bernstein(ts: Tensor, order: int) -> Tensor:
    """Tính hệ số Bernstein ``C(n,k)·t^k·(1−t)^(n−k)`` cho mọi ``k = 0..order``.

    Nhân ma trận này với control point cho ra điểm trên đường Bézier bậc ``order`` tại ``t``.

    Args:
        ts: Tham số ``t`` của các mẫu, shape ``[T, 1]``.
        order: Bậc của đường Bézier (3 nghĩa là 4 control point).

    Returns:
        Tensor shape ``[T, order + 1]``; mỗi hàng cộng lại bằng 1 khi ``t`` thuộc ``[0, 1]``.
    """
    ks = torch.arange(order + 1, dtype=ts.dtype, device=ts.device)
    return _binomial_coefficients(order, ts) * ts**ks * (1.0 - ts) ** (order - ks)


def bernstein_derivative(ts: Tensor, order: int) -> Tensor:
    """Tính đạo hàm theo ``t`` của từng hệ số Bernstein.

    Viết theo dạng của bản gốc (``get_d_bezier_coefficient``): hai số hạng
    ``k·t^(k−1)·(1−t)^(n−k)`` và ``(n−k)·t^k·(1−t)^(n−k−1)``, số hạng có số mũ âm đặt bằng 0.

    Args:
        ts: Tham số ``t`` của các mẫu, shape ``[T, 1]``.
        order: Bậc của đường Bézier.

    Returns:
        Tensor shape ``[T, order + 1]``.
    """
    ks = torch.arange(order + 1, dtype=ts.dtype, device=ts.device)
    coefs = _binomial_coefficients(order, ts)
    # k = 0 không có số hạng thứ nhất; k = n không có số hạng thứ hai (số mũ âm → 0).
    pow_k_minus_1 = torch.zeros(ts.shape[0], order + 1, dtype=ts.dtype, device=ts.device)
    pow_k_minus_1[:, 1:] = ts ** (ks[1:] - 1)
    pow_rest_minus_1 = torch.zeros_like(pow_k_minus_1)
    pow_rest_minus_1[:, :-1] = (1.0 - ts) ** (order - ks[:-1] - 1)
    term1 = coefs * ks * pow_k_minus_1 * (1.0 - ts) ** (order - ks)
    term2 = coefs * (order - ks) * ts**ks * pow_rest_minus_1
    return term1 - term2


def fit_control_points(points: Tensor, ts: Tensor, order: int) -> Tensor:
    """Fit control point Bézier qua các điểm đã biết tham số, bằng bình phương tối thiểu.

    Nghiệm là ``pinv(MᵀM)·Mᵀ·points`` với ``M = bernstein(ts)``; dùng ``pinv`` như bản gốc nên
    vẫn có nghiệm khi ít mẫu hơn control point.

    Args:
        points: Các điểm cần khớp, shape ``[N, T, 3]`` (N đường cong, T mẫu mỗi đường).
        ts: Tham số ``t`` của T mẫu, dùng chung cho mọi đường, shape ``[T, 1]``.
        order: Bậc của đường Bézier.

    Returns:
        Control point shape ``[N, order + 1, 3]``.
    """
    basis = bernstein(ts, order).to(dtype=points.dtype, device=points.device)  # [T, K]
    gram_pinv = torch.linalg.pinv(basis.T @ basis)  # [K, K]
    return (gram_pinv @ basis.T) @ points


def chord_length_params(trajectory: Tensor) -> Tensor:
    """Tham số hoá các điểm của đường đi theo độ dài dây cung, về đoạn ``[0, 1]``.

    Args:
        trajectory: Vị trí của vật ở từng khung, shape ``[T, 3]``.

    Returns:
        Tensor shape ``[T]``. Một khung cho ``[0]``; tổng độ dài dưới ``1e-12`` (vật đứng yên)
        cho ``linspace(0, 1, T)``.
    """
    num_frames = trajectory.shape[0]
    if num_frames == 1:
        return torch.zeros(1, dtype=trajectory.dtype, device=trajectory.device)
    steps = torch.norm(trajectory[1:] - trajectory[:-1], dim=1)
    arc = torch.cat([torch.zeros_like(steps[:1]), torch.cumsum(steps, dim=0)])
    total = arc[-1]
    eps = 1e-12
    if total < eps:
        return torch.linspace(0, 1, num_frames, dtype=trajectory.dtype, device=trajectory.device)
    return arc / (total + eps)


def fit_trajectory(
    trajectory: Tensor,
    order: int = 3,
    rounds: int = 100,
    num_samples: int = 10000,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Fit đường đi của một vật bằng đường Bézier, lặp chiếu lại tham số.

    Bắt đầu bằng tham số theo độ dài dây cung. Mỗi vòng lấy ``num_samples`` mẫu đều trên đường
    cong hiện tại, chiếu từng khung về mẫu gần nhất để có ``u`` mới, fit lại control point theo
    các ``u`` đó, và dừng khi control point đổi ít hơn ``tol`` (vòng lặp của
    ``create_from_ply_dict`` trong bản gốc).

    Args:
        trajectory: Vị trí tâm vật ở từng khung có box, shape ``[T, 3]``, đơn vị mét.
        order: Bậc Bézier.
        rounds: Số vòng lặp tối đa.
        num_samples: Số mẫu trên đường cong mỗi vòng khi chiếu các khung về ``u``.
        tol: Ngưỡng dừng: độ lệch lớn nhất của control point giữa hai vòng liên tiếp.

    Returns:
        ``(control_points [order + 1, 3], bezier_t [T])``; ``bezier_t`` là tham số ``u`` của
        từng khung ở vòng cuối.
    """
    bezier_t = chord_length_params(trajectory)
    control_points = fit_control_points(trajectory[None], bezier_t[:, None], order)[0]
    samples_t = torch.linspace(
        0, 1, num_samples, dtype=trajectory.dtype, device=trajectory.device
    )[:, None]
    basis_samples = bernstein(samples_t, order)  # [S, K]
    for _ in range(rounds):
        curve = basis_samples @ control_points  # [S, 3]
        # Dùng khoảng cách đúng (không dùng khai triển bình phương) để không mất độ chính xác
        # float32 khi toạ độ lớn.
        dist = torch.cdist(trajectory, curve, compute_mode="donot_use_mm_for_euclid_dist")
        bezier_t = samples_t[dist.argmin(dim=-1), 0]
        new_control_points = fit_control_points(trajectory[None], bezier_t[:, None], order)[0]
        change = torch.norm(new_control_points - control_points, dim=-1).max()
        control_points = new_control_points
        if change < tol:
            break
    return control_points, bezier_t


def fit_time_map(times: Tensor, bezier_t: Tensor, order: int = 3) -> Tensor:
    """Fit đa thức Bernstein đổi thời gian thật sang tham số ``u`` của Bézier.

    Mỗi vật có một đa thức riêng: ``u = bernstein(time) @ result``. Học tiếp trong lúc train
    (``time_control_points`` của ``BezierNodes``).

    Args:
        times: Thời gian chuẩn hoá của các khung có box, shape ``[T]``.
        bezier_t: Tham số ``u`` tương ứng từ ``fit_trajectory``, shape ``[T]``.
        order: Bậc của đa thức.

    Returns:
        Hệ số shape ``[order + 1, 1]``.
    """
    basis = bernstein(times[:, None], order)
    return torch.linalg.lstsq(basis, bezier_t[:, None]).solution
