"""Faster-GS, phần Python: các kỹ thuật tăng tốc không đụng rasterizer (docs/methods/fastergs.md).

- ``fused_adam``: ``torch.optim.Adam(fused=True)``. Đã kiểm trên GPU: qua 300 bước có densify và
  prune bằng ``dup_in_optim``, ``remove_from_optim``, tham số lệch tối đa 1,2e-7 (tương đối) so
  với Adam thường.
- ``fused_ssim``: gói ``fused-ssim`` (commit a7c48d6) thay ``pytorch_msssim.SSIM`` trong loss.
  Cùng cửa sổ Gaussian 11, sigma 1,5, C1, C2; dùng ``padding="valid"`` cho đúng cách
  ``pytorch_msssim`` tính. Đã kiểm trên GPU: giá trị lệch 6e-8, gradient lệch 1,3e-6 (tương đối);
  nhanh hơn khoảng 5 lần ở ảnh 900×1600.
- ``fewer_syncs``: kiểm NaN/Inf của mọi loss bằng một lần chờ GPU thay vì hai lần mỗi loss, và
  đọc mọi số để ghi log bằng một lần ``tolist()`` thay vì ``item()`` từng số. Kết quả không đổi.
  Không đụng hai lệnh ``torch.cuda.synchronize()`` dùng để đo ``train_time_s``.
- ``morton_order`` (thử nghiệm, mặc định tắt): mỗi ``morton_interval`` bước, sau khi model refine,
  sắp Gaussian của nền (``VanillaGaussians``, đúng lớp đó, không áp cho xe, người, PVG) theo mã
  Morton của tâm, cùng trạng thái Adam. Chỉ tương đương về thống kê với khi tắt.

Không có SelectiveAdam: ``SelectiveAdam`` của gsplat-geer bỏ cả bước hiệu chỉnh bias của Adam, còn
code công khai của Faster-GS không có bản bỏ qua Gaussian không nhìn thấy (docs/methods/fastergs.md).

Khi công tắc ``trainer.fastergs.enabled`` tắt, trainer chạy y như trước.

File này chứa ``FasterGS``, ``FusedSSIM``, ``morton_codes``, ``reorder_gaussians``.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from drive2sim.recon.speedups.fastergs_rules import active_techniques, morton_due, parse_config

logger = logging.getLogger()


class FusedSSIM(nn.Module):
    """SSIM bằng gói ``fused-ssim``, gọi giống ``pytorch_msssim.SSIM(data_range=1, size_average=True)``.

    ``fused_ssim`` chỉ tính gradient theo đối số đầu, nên ảnh cần gradient được đưa lên trước;
    giá trị SSIM đối xứng nên đổi chỗ không đổi kết quả.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            from fused_ssim import fused_ssim
        except ImportError as error:
            raise ImportError(
                "trainer.fastergs.fused_ssim cần gói fused-ssim; cài bằng envs/install_main.sh "
                "(bước 5) hoặc đặt trainer.fastergs.fused_ssim=false"
            ) from error
        self._fused_ssim = fused_ssim

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        assert not (img1.requires_grad and img2.requires_grad), "chỉ một ảnh được cần gradient"
        first, second = (img2, img1) if img2.requires_grad else (img1, img2)
        return self._fused_ssim(
            first.contiguous(), second.contiguous(), padding="valid", train=torch.is_grad_enabled()
        )


class FasterGS:
    """Bật các kỹ thuật phần Python của Faster-GS theo khối ``trainer.fastergs``.

    Args:
        cfg: Khối ``trainer.fastergs`` của config, hoặc ``None``.
    """

    def __init__(self, cfg: Optional[Any]) -> None:
        self.cfg = parse_config(cfg)
        self.active = active_techniques(self.cfg)
        if self.cfg.enabled:
            logger.info(f"Faster-GS: {', '.join(self.active) or 'không có kỹ thuật nào'}")

    def uses(self, name: str) -> bool:
        """Kỹ thuật ``name`` có đang bật không."""
        return name in self.active

    def adam_kwargs(self) -> Dict[str, Any]:
        """Tham số thêm cho ``torch.optim.Adam``."""
        return {"fused": True} if self.uses("fused_adam") else {}

    def make_ssim(self, default: nn.Module) -> nn.Module:
        """Hàm SSIM dùng trong loss: ``FusedSSIM`` khi bật, không thì ``default`` (pytorch_msssim)."""
        return FusedSSIM() if self.uses("fused_ssim") else default

    def check_finite(self, loss_dict: Dict[str, torch.Tensor], step: int) -> None:
        """Báo lỗi nếu một loss có NaN hoặc Inf, cùng thông báo như vòng train gốc.

        Với ``fewer_syncs``: một lần chờ GPU cho mọi loss; chỉ khi có lỗi mới xét từng loss.
        """
        if self.uses("fewer_syncs"):
            flags = torch.stack(
                [torch.stack((torch.isnan(v).any(), torch.isinf(v).any())) for v in loss_dict.values()]
            )
            if not bool(flags.any()):
                return
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")

    def to_floats(self, values: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Đổi các tensor một phần tử sang số Python để ghi log."""
        if not values:
            return {}
        tensors = [v.detach().reshape(()) for v in values.values()]
        if self.uses("fewer_syncs") and len({t.dtype for t in tensors}) == 1:
            # cùng kiểu dữ liệu thì gộp được mà không đổi giá trị
            return dict(zip(values.keys(), torch.stack(tensors).tolist()))
        return {k: v.item() for k, v in values.items()}

    def after_step(self, trainer: Any, step: int) -> None:
        """Gọi cuối mỗi bước, sau khi model refine và FastGS xoá mask: sắp lại theo Z-order."""
        if not morton_due(step, self.cfg):
            return
        from drive2sim.recon.models.gaussians.vanilla import VanillaGaussians

        for name in trainer.gaussian_classes:
            model = trainer.models[name]
            if type(model) is VanillaGaussians:  # đúng lớp nền; không áp cho lớp con (xe, người, PVG)
                with torch.no_grad():
                    order = torch.argsort(morton_codes(model._means.detach()))
                reorder_gaussians(model, order, trainer.optimizer)
                logger.info(f"Faster-GS: Z-order {name} ({model.num_points} Gaussian) @ step {step}")


# Tham số theo từng Gaussian của VanillaGaussians: tên nhóm optimizer (sau tiền tố lớp) → thuộc tính.
_PARAM_ATTRS = {
    "xyz": "_means",
    "sh_dc": "_features_dc",
    "sh_rest": "_features_rest",
    "opacity": "_opacities",
    "scaling": "_scales",
    "rotation": "_quats",
}
# Thống kê densify theo từng Gaussian; None sau mỗi lần refine.
_STAT_ATTRS = ("xys_grad_norm", "vis_counts", "max_2Dsize")


def _spread_bits(v: torch.Tensor) -> torch.Tensor:
    """Chèn hai bit 0 giữa các bit của số 21 bit (mã Morton 3 chiều 63 bit)."""
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def morton_codes(xyz: torch.Tensor) -> torch.Tensor:
    """Mã Morton (đường cong Z) 63 bit của các điểm, lưới 2^21 mỗi trục trong hộp bao của chúng."""
    lo = xyz.min(dim=0).values
    span = (xyz.max(dim=0).values - lo).clamp(min=1e-12)
    grid = ((xyz - lo) / span * (2**21 - 1)).round().long().clamp(0, 2**21 - 1)
    return _spread_bits(grid[:, 0]) | (_spread_bits(grid[:, 1]) << 1) | (_spread_bits(grid[:, 2]) << 2)


def reorder_gaussians(model: nn.Module, order: torch.Tensor, optimizer: Any) -> None:
    """Sắp lại mọi tham số, trạng thái Adam và thống kê densify của ``model`` theo ``order``.

    Theo cùng cách ``remove_from_optim`` thay tham số trong optimizer: tham số mới, trạng thái cũ
    được sắp lại (``step`` giữ nguyên).
    """
    groups = {group["name"]: group for group in optimizer.param_groups}
    for name, params in model.get_gaussian_param_groups().items():
        old = params[0]
        new = nn.Parameter(old.detach()[order])
        state = optimizer.state.pop(old, None)
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = state[key][order]
            optimizer.state[new] = state
        groups[name]["params"] = [new]
        setattr(model, _PARAM_ATTRS[name[len(model.class_prefix):]], new)
    for attr in _STAT_ATTRS:
        value = getattr(model, attr, None)
        if value is not None:
            setattr(model, attr, value[order])
