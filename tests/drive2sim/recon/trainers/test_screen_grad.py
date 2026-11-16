"""Kiểm tra đổi gradient theo tâm 3D sang gradient tương đương pixel (``trainers/screen_grad.py``).

Dựng gradient 3D từ một gradient pixel biết trước qua ``g3d = Jᵀ g2d`` (``J`` tính bằng autograd,
độc lập với sai phân của module), rồi kiểm tra module tính ngược ra đúng ``g2d``: camera pinhole,
camera mắt cá OpenCV, camera đã xoay trong world; điểm sau camera pinhole cho 0.
"""

import math
from collections.abc import Iterator
from typing import Optional

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")  # geometry.py import pytorch3d

from drive2sim.recon.trainers.screen_grad import screen_equivalent_grad  # noqa: E402
from drive2sim.recon.utils.geometry import project_camera_model  # noqa: E402


@pytest.fixture(autouse=True)
def float64() -> Iterator[None]:
    """Tính bằng float64 trong file này, trả lại kiểu mặc định sau mỗi test."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def rotation_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def autograd_jacobian(
    points_world: "torch.Tensor",
    camtoworld: "torch.Tensor",
    intrinsics: "torch.Tensor",
    model: str,
    radial: Optional[list[float]] = None,
) -> "torch.Tensor":
    """∂pixel/∂world của từng điểm bằng autograd (mốc độc lập)."""
    world_to_cam = torch.linalg.inv(camtoworld)
    rows = []
    for point in points_world:
        point = point.clone().requires_grad_(True)

        def project(p: "torch.Tensor") -> "torch.Tensor":
            cam = world_to_cam[:3, :3] @ p + world_to_cam[:3, 3]
            pixel, _ = project_camera_model(cam[None], intrinsics, model, radial_coeffs=radial)
            return pixel[0]

        rows.append(torch.autograd.functional.jacobian(project, point))
    return torch.stack(rows)  # (N, 2, 3)


@pytest.mark.parametrize(
    ("model", "radial"),
    [("pinhole", None), ("fisheye", [-0.02, 0.003, -0.0004, 0.00002])],
)
def test_recovers_pixel_gradient(model: str, radial: Optional[list[float]]) -> None:
    generator = torch.Generator().manual_seed(0)
    camtoworld = torch.eye(4)
    camtoworld[:3, :3] = rotation_z(0.4)
    camtoworld[:3, 3] = torch.tensor([3.0, -2.0, 1.5])
    intrinsics = torch.tensor([[630.0, 0, 400.0], [0, 630.0, 225.0], [0, 0, 1.0]])
    # điểm trước camera (z camera từ 4 đến 40 m), lệch trục tới khoảng 45°
    cam_points = torch.rand(20, 3, generator=generator) * torch.tensor([20.0, 10.0, 36.0])
    cam_points += torch.tensor([-10.0, -5.0, 4.0])
    points = cam_points @ camtoworld[:3, :3].T + camtoworld[:3, 3]
    jac = autograd_jacobian(points, camtoworld, intrinsics, model, radial)
    grad_pixel = torch.randn(20, 2, generator=generator)
    grad_world = (jac.transpose(1, 2) @ grad_pixel[..., None])[..., 0]  # g3d = Jᵀ g2d
    recovered = screen_equivalent_grad(grad_world, points, camtoworld, intrinsics, model, radial)
    assert torch.allclose(recovered, grad_pixel, rtol=1e-4, atol=1e-6)


def test_points_behind_pinhole_camera_give_zero() -> None:
    intrinsics = torch.tensor([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    points = torch.tensor([[0.0, 0.0, -5.0], [1.0, 0.0, 10.0]])
    grads = torch.ones(2, 3)
    pixel = screen_equivalent_grad(grads, points, torch.eye(4), intrinsics, "pinhole")
    assert torch.equal(pixel[0], torch.zeros(2))
    assert pixel[1].abs().sum() > 0


def test_pixel_gradient_shrinks_with_distance_as_f_over_z() -> None:
    """Cùng gradient 3D: điểm xa hơn gấp đôi cho gradient pixel lớn gấp đôi (g2d ≈ g3d · z / f)."""
    intrinsics = torch.tensor([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    points = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]])
    grads = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pixel = screen_equivalent_grad(grads, points, torch.eye(4), intrinsics, "pinhole")
    assert pixel[0, 0] == pytest.approx(10.0 / 500.0, rel=1e-4)
    assert pixel[1, 0] == pytest.approx(2 * pixel[0, 0].item(), rel=1e-4)


def test_degenerate_jacobian_gives_zero_instead_of_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jacobian bằng 0 (gram suy biến) không làm hỏng cả batch: điểm đó cho 0, điểm khác vẫn đúng.

    Run 20261116-013741-omnire-90dd lỗi ``torch.linalg.solve: ... input matrix is singular``.
    """
    import drive2sim.recon.trainers.screen_grad as module

    real = module.projection_jacobian

    def with_zero_row(*args: object, **kwargs: object) -> tuple:
        jac, valid = real(*args, **kwargs)
        jac = jac.clone()
        jac[0, 1] = jac[0, 0]  # điểm đầu: hai hàng song song, J Jᵀ suy biến
        return jac, valid

    monkeypatch.setattr(module, "projection_jacobian", with_zero_row)
    # float32 như lúc train: phần giảm chấn tương đối nhỏ bị làm tròn mất
    f32 = torch.float32
    intrinsics = torch.tensor([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=f32)
    points = torch.tensor([[0.3, 0.1, 10.0], [0.0, 0.0, 10.0]], dtype=f32)
    grads = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=f32)
    pixel = screen_equivalent_grad(grads, points, torch.eye(4, dtype=f32), intrinsics, "pinhole")
    assert torch.equal(pixel[0], torch.zeros(2))
    assert pixel[1, 0] == pytest.approx(10.0 / 500.0, rel=1e-3)
