"""Densify của 3DGUT theo gradient tương đương pixel (docs/methods/densify_3dgut.md).

Renderer 3DGS densify theo gradient của loss theo tâm Gaussian trên ảnh (pixel), còn 3DGUT (nhánh
``ut`` của patch Bosch, ``trainers/base.py``) theo gradient theo tâm 3D (mét). Hai đại lượng nối
với nhau bằng Jacobian của phép chiếu ``J = ∂(pixel)/∂(tâm 3D)`` (2×3): theo quy tắc chuỗi,
``g3d = Jᵀ · g2d``. File này tính ngược ``g2d = (J Jᵀ)⁻¹ J · g3d`` (nghiệm bình phương tối thiểu)
với đúng camera của bước đó, nhân cùng hệ số ``W/2``, ``H/2`` như nhánh 3DGS, rồi để nhánh ``ut``
của Bosch chạy như cũ trên giá trị đã đổi. ``J`` chứa tiêu cự, khoảng cách và mô hình ống kính
(pinhole, mắt cá, F-Theta), nên ngưỡng ``densify_grad_thresh`` của 3DGS dùng lại được mà không
phụ thuộc dataset, độ phân giải hay loại camera.

Xấp xỉ: 3DGUT chiếu Gaussian bằng các điểm sigma (unscented transform) chứ không qua đúng một tâm
2D, nên ``g3d`` còn phần do hình dạng ảnh chiếu đổi theo vị trí; phép đổi giữ phần đi qua tâm.

Cách dùng: ``trainer.type: drive2sim.recon.trainers.screen_grad.ScreenGradMultiTrainer`` (stage:
``--set train.drivestudio.ut_densify=screen``). Không sửa code của Bosch: chỉ thay
``info["means3d"].grad`` trước khi gọi ``postprocess_per_train_step`` của lớp cha. Code densify
(``VanillaGaussians.after_train``) chỉ dùng độ lớn của vectơ gradient, nên đặt
``[g2d_x, g2d_y, 0]`` cho đúng độ lớn của gradient tương đương pixel.
"""

from typing import Optional, Sequence

import torch

from drive2sim.recon.trainers.scene_graph import MultiTrainer
from drive2sim.recon.utils.geometry import project_camera_model

# Bước sai phân trung tâm, tỉ lệ với khoảng cách từ camera tới điểm
_RELATIVE_STEP = 1e-3


def projection_jacobian(
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_model: str = "pinhole",
    radial_coeffs: Optional[Sequence[float]] = None,
    ftheta_parameters: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobian ``∂(pixel)/∂(điểm trong hệ camera)`` bằng sai phân trung tâm.

    Args:
        points_cam: ``(N, 3)`` điểm trong hệ camera (trục OpenCV).
        intrinsics: ``(3, 3)`` ma trận nội tại ở độ phân giải đang render.
        camera_model: ``pinhole``, ``fisheye`` hoặc ``ftheta``.
        radial_coeffs: Hệ số méo xuyên tâm (mắt cá: ``k1..k4``).
        ftheta_parameters: Tham số F-Theta (khi ``camera_model`` là ``ftheta``).

    Returns:
        ``(J, valid)``: ``J`` có dạng ``(N, 2, 3)``; ``valid`` là điểm mà mọi phép chiếu đều hợp lệ.
    """
    kwargs = {"radial_coeffs": radial_coeffs, "ftheta_parameters": ftheta_parameters}
    step = (points_cam.norm(dim=-1, keepdim=True) * _RELATIVE_STEP).clamp_min(1e-6)
    columns, valid = [], torch.ones(points_cam.shape[0], dtype=torch.bool, device=points_cam.device)
    for axis in range(3):
        offset = torch.zeros_like(points_cam)
        offset[:, axis] = step[:, 0]
        plus, valid_plus = project_camera_model(points_cam + offset, intrinsics, camera_model, **kwargs)
        minus, valid_minus = project_camera_model(points_cam - offset, intrinsics, camera_model, **kwargs)
        columns.append((plus - minus) / (2 * step))
        valid &= valid_plus & valid_minus
    return torch.stack(columns, dim=-1), valid


def screen_equivalent_grad(
    grad_world: torch.Tensor,
    means_world: torch.Tensor,
    camtoworld: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_model: str = "pinhole",
    radial_coeffs: Optional[Sequence[float]] = None,
    ftheta_parameters: Optional[dict] = None,
) -> torch.Tensor:
    """Gradient theo tâm trên ảnh (pixel) tương đương với gradient theo tâm 3D.

    Args:
        grad_world: ``(N, 3)`` gradient của loss theo tâm Gaussian trong hệ world.
        means_world: ``(N, 3)`` tâm Gaussian trong hệ world.
        camtoworld: ``(4, 4)`` camera → world của ảnh vừa render.
        intrinsics: ``(3, 3)`` ở độ phân giải đang render.
        camera_model: Mô hình ống kính của camera đó.
        radial_coeffs: Hệ số méo xuyên tâm.
        ftheta_parameters: Tham số F-Theta.

    Returns:
        ``(N, 2)`` gradient theo pixel; bằng 0 với điểm chiếu không hợp lệ.
    """
    world_to_cam = torch.linalg.inv(camtoworld)
    rotation = world_to_cam[:3, :3]
    points_cam = means_world @ rotation.T + world_to_cam[:3, 3]
    jac_cam, valid = projection_jacobian(
        points_cam, intrinsics, camera_model, radial_coeffs, ftheta_parameters
    )
    jac = jac_cam @ rotation  # ∂pixel/∂world = ∂pixel/∂cam · ∂cam/∂world
    gram = jac @ jac.transpose(1, 2)  # (N, 2, 2)
    eye = torch.eye(2, device=gram.device, dtype=gram.dtype)
    damping = 1e-12 * gram.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-30)[:, None, None]
    rhs = (jac @ grad_world.unsqueeze(-1)).squeeze(-1)  # J · g3d
    grad_pixel = torch.linalg.solve(gram + damping * eye, rhs.unsqueeze(-1)).squeeze(-1)
    grad_pixel = torch.nan_to_num(grad_pixel, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(valid[:, None], grad_pixel, torch.zeros_like(grad_pixel))


class ScreenGradMultiTrainer(MultiTrainer):
    """``MultiTrainer`` (OmniRe, Street GS) với densify của 3DGUT theo gradient tương đương pixel.

    Renderer khác 3DGUT chạy y như ``MultiTrainer``.
    """

    def postprocess_per_train_step(self, step: int) -> None:
        """Đổi ``info["means3d"].grad`` sang gradient tương đương pixel rồi gọi lớp cha."""
        if self.render_cfg.get("render_mode", "default") == "ut" and "means3d" in self.info:
            means = self.info["means3d"]
            if means.grad is not None and self._last_render_inputs is not None:
                cam = self._last_render_inputs[1]
                radial = cam.radial_coeffs
                if isinstance(radial, torch.Tensor):
                    radial = radial.flatten().tolist()
                with torch.no_grad():
                    pixel = screen_equivalent_grad(
                        means.grad,
                        means.detach(),
                        cam.camtoworlds,
                        cam.Ks,
                        cam.camera_model,
                        radial,
                        cam.ftheta_parameters,
                    )
                    # cùng hệ số như nhánh 3DGS của base.py: gradient theo pixel nhân W/2, H/2
                    # (tức theo toạ độ chuẩn hoá [-1, 1] như 3DGS gốc), nhân batch_size
                    scale = self.render_cfg.batch_size / 2.0
                    pixel[:, 0] *= self.info["width"] * scale
                    pixel[:, 1] *= self.info["height"] * scale
                    means.grad = torch.cat([pixel, torch.zeros_like(pixel[:, :1])], dim=-1)
        super().postprocess_per_train_step(step)
