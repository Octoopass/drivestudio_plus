"""Densify bằng MCMC (Kheradmand và cộng sự, NeurIPS 2024) cho các node Gaussian của OmniRe.

Lý do và cách làm từ nguyên lý gốc: ``docs/methods/densify_mcmc.md``. Tóm tắt: thay quy tắc
"Gaussian có gradient vượt ngưỡng thì nhân hoặc tách" (ngưỡng có đơn vị, khác nhau giữa 3DGS,
3DGUT, 3DGEER) bằng một ngân sách cố định ``cap_max`` mỗi node:

- **dời**: Gaussian có độ đục ≤ ``min_opacity`` chuyển tới vị trí một Gaussian sống chọn ngẫu nhiên
  theo độ đục; độ đục và kích thước của các bản chia lại để ảnh không đổi (phương trình 9 của bài
  báo, ``relocation``); trạng thái Adam của chúng đặt về 0;
- **thêm**: mỗi lần thêm ``grow_rate`` (5 %) số hiện có, chọn theo độ đục, tới ``cap_max``;
- **nhiễu**: sau mỗi bước, cộng nhiễu vào vị trí theo hiệp phương sai của Gaussian, nhân hệ số
  ``sigmoid(100·((1−o)−0,995))``: chỉ đáng kể với Gaussian gần trong suốt (độ đục dưới khoảng
  0,005), Gaussian đậm gần như không bị đẩy (``noise_lr`` × learning rate vị trí);
- **điều chuẩn**: loss ``opacity_reg`` × độ đục trung bình và ``scale_reg`` × kích thước trung bình.

Port từ ``gsplat/strategy/mcmc.py`` và ``gsplat/strategy/ops.py`` (gsplat-geer 1.5.3): gsplat cần mỗi
tham số một optimizer, OmniRe dùng một optimizer chung chia nhóm theo tên, nên phần sửa optimizer
viết lại theo kiểu ``dup_in_optim`` của OmniRe. Phép chia lại độ đục, kích thước viết bằng PyTorch
(không cần kernel CUDA ``compute_relocation``), tính bằng float64.

Node người (``SMPLNodes``) không dùng: Gaussian của người gắn với đỉnh SMPL và không bao giờ được
densify. Node xe, vật biến dạng mang ``point_ids`` (Gaussian thuộc vật nào): khi dời hay thêm,
``point_ids`` chép theo Gaussian nguồn. Khi bật, node dùng MCMC bỏ qua toàn bộ quy tắc cũ
(``postprocess_per_train_step`` của node: densify theo gradient, cắt bớt, đặt lại độ đục).

File này chứa ``MCMC``, ``relocation``.
"""

import math
from typing import Any, Optional

import torch
from torch.nn import Parameter

from drive2sim.recon.densify.mcmc_defaults import DEFAULT_CLASSES, DEFAULTS, caps_for

# Tham số Gaussian của node (tên trong get_gaussian_param_groups bỏ tiền tố) → thuộc tính.
PARAMS = {
    "xyz": "_means",
    "sh_dc": "_features_dc",
    "sh_rest": "_features_rest",
    "opacity": "_opacities",
    "scaling": "_scales",
    "rotation": "_quats",
}
# Dữ liệu riêng mỗi Gaussian không phải tham số, chép theo khi dời hay thêm.
PER_POINT_BUFFERS = ("point_ids",)
N_MAX = 51  # số bản tối đa của một Gaussian trong một lần dời, như gsplat


def _comb_table(n_max: int) -> torch.Tensor:
    """``table[n, k] = C(n, k + 1)``, n từ 0 tới n_max."""
    table = torch.zeros(n_max + 1, n_max, dtype=torch.float64)
    for n in range(n_max + 1):
        for k in range(min(n, n_max)):
            table[n, k] = math.comb(n, k + 1)
    return table


_TABLE = _comb_table(N_MAX)


def relocation(
    opacities: torch.Tensor, scales: torch.Tensor, ratios: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Độ đục và kích thước mới khi một Gaussian được chia thành ``ratios`` bản.

    Phương trình 9 của bài báo, như kernel ``compute_relocation`` của gsplat:
    ``o' = 1 - (1 - o)^(1/n)`` và ``s' = o / Σ_{i=1..n} Σ_{k=0..i-1} C(i-1,k) (-1)^k o'^(k+1) /
    sqrt(k+1) · s``. Tổng theo ``i`` gộp lại thành ``C(n, k+1)`` (đẳng thức "cây gậy khúc côn
    cầu"), nên chỉ còn một tổng theo ``k``.

    Args:
        opacities: Độ đục (sau sigmoid), ``[N]``.
        scales: Kích thước (sau exp), ``[N, D]``.
        ratios: Số bản của mỗi Gaussian, ``[N]``; kẹp vào ``[1, N_MAX]``.
    """
    n = ratios.clamp(1, N_MAX).long()
    o = opacities.double()
    new_o = 1.0 - (1.0 - o) ** (1.0 / n.double())
    k = torch.arange(N_MAX, dtype=torch.float64, device=o.device)
    signs = (-1.0) ** k / torch.sqrt(k + 1.0)
    powers = new_o[:, None] ** (k + 1.0)[None, :]
    denom = (_TABLE.to(o.device)[n] * signs[None, :] * powers).sum(dim=1)
    new_s = (o / denom)[:, None] * scales.double()
    return new_o.to(opacities.dtype), new_s.to(scales.dtype)


def _sample(weights: torch.Tensor, n: int) -> torch.Tensor:
    """Chọn ``n`` chỉ số theo trọng số, có lặp lại."""
    if weights.numel() <= 2**24:
        return torch.multinomial(weights, n, replacement=True)
    probs = (weights / weights.sum()).cpu().numpy()  # torch.multinomial giới hạn 2^24 phần tử
    import numpy as np

    return torch.from_numpy(np.random.choice(len(probs), n, p=probs)).to(weights.device)


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Quaternion (w, x, y, z) đã chuẩn hoá → ma trận xoay ``[N, 3, 3]``."""
    w, x, y, z = quats.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quats.shape[:-1], 3, 3)  # fmt: skip


class MCMC:
    """Densify bằng MCMC cho một số node; tắt thì trainer chạy y như trước.

    Args:
        cfg: Khối ``trainer.mcmc`` của config, hoặc ``None``. Khoá: ``enabled``, ``classes``,
            ``cap_max`` (số, hoặc dict theo node), ``noise_lr``, ``min_opacity``,
            ``refine_start_iter``, ``refine_stop_iter``, ``refine_every``, ``grow_rate``,
            ``opacity_reg``, ``scale_reg``.
    """

    def __init__(self, cfg: Optional[Any]) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.classes = tuple(cfg.get("classes", DEFAULT_CLASSES))
        self.caps = caps_for(cfg.get("cap_max", None), self.classes)
        self.noise_lr = float(cfg.get("noise_lr", DEFAULTS["noise_lr"]))
        self.min_opacity = float(cfg.get("min_opacity", DEFAULTS["min_opacity"]))
        self.refine_start = int(cfg.get("refine_start_iter", DEFAULTS["refine_start_iter"]))
        self.refine_stop = int(cfg.get("refine_stop_iter", DEFAULTS["refine_stop_iter"]))
        self.refine_every = int(cfg.get("refine_every", DEFAULTS["refine_every"]))
        self.grow_rate = float(cfg.get("grow_rate", DEFAULTS["grow_rate"]))
        self.opacity_reg = float(cfg.get("opacity_reg", DEFAULTS["opacity_reg"]))
        self.scale_reg = float(cfg.get("scale_reg", DEFAULTS["scale_reg"]))

    def applies(self, class_name: str) -> bool:
        """Node này densify bằng MCMC không."""
        return self.enabled and class_name in self.classes

    # ------------------------------------------------------------------ một bước
    @torch.no_grad()
    def step(
        self, class_name: str, model: Any, step: int, optimizer: torch.optim.Optimizer
    ) -> None:
        """Gọi sau ``optimizer.step()``: dời, thêm (mỗi ``refine_every`` bước), rồi cộng nhiễu."""
        if (
            self.refine_start < step < self.refine_stop
            and step % self.refine_every == 0
            and model.num_points > 0
        ):
            dead = model.get_opacity.flatten() <= self.min_opacity
            if dead.any() and (~dead).any():
                self.relocate(model, optimizer, dead)
            target = min(
                self.caps.get(class_name, 1_000_000), int((1 + self.grow_rate) * model.num_points)
            )
            if target > model.num_points:
                self.add(model, optimizer, target - model.num_points)
        self.inject_noise(model, optimizer)

    def regularization(self, model: Any) -> dict[str, torch.Tensor]:
        """Loss điều chuẩn của MCMC cho một node."""
        losses = {}
        if self.opacity_reg > 0:
            losses["mcmc_opacity_reg"] = self.opacity_reg * model.get_opacity.abs().mean()
        if self.scale_reg > 0:
            losses["mcmc_scale_reg"] = self.scale_reg * model.get_scaling.abs().mean()
        return losses

    # ------------------------------------------------------------------ các phép
    def _groups(self, model: Any) -> dict[str, str]:
        """Tên nhóm tham số trong optimizer → thuộc tính của node."""
        return {model.class_prefix + key: attr for key, attr in PARAMS.items()}

    def _split_values(self, model: Any, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Độ đục (logit) và kích thước (log) mới của các Gaussian nguồn ``idx`` khi chia bản."""
        opac = model.get_opacity.flatten()
        scales = torch.exp(model._scales)
        ratios = torch.bincount(idx, minlength=model.num_points)[idx] + 1
        new_o, new_s = relocation(opac[idx], scales[idx], ratios)
        eps = torch.finfo(torch.float32).eps
        new_o = new_o.clamp(min=self.min_opacity, max=1.0 - eps)
        return torch.logit(new_o).reshape(-1, 1), torch.log(new_s)

    def relocate(self, model: Any, optimizer: torch.optim.Optimizer, dead: torch.Tensor) -> None:
        """Dời các Gaussian ``dead`` tới vị trí Gaussian sống chọn theo độ đục."""
        dead_idx = dead.nonzero(as_tuple=True)[0]
        alive_idx = (~dead).nonzero(as_tuple=True)[0]
        weights = model.get_opacity.flatten()[alive_idx]
        src = alive_idx[_sample(weights, len(dead_idx))]
        new_opac, new_scales = self._split_values(model, src)
        model._opacities.data[src] = new_opac
        model._scales.data[src] = new_scales
        for attr in PARAMS.values():
            getattr(model, attr).data[dead_idx] = getattr(model, attr).data[src]
        for name in PER_POINT_BUFFERS:
            buffer = getattr(model, name, None)
            if isinstance(buffer, torch.Tensor) and buffer.shape[0] == model.num_points:
                buffer[dead_idx] = buffer[src]
        self._zero_state(model, optimizer, torch.cat([src, dead_idx]))

    def add(self, model: Any, optimizer: torch.optim.Optimizer, n: int) -> None:
        """Thêm ``n`` Gaussian, bản sao của Gaussian chọn theo độ đục."""
        before = model.num_points
        src = _sample(model.get_opacity.flatten(), n)
        new_opac, new_scales = self._split_values(model, src)
        model._opacities.data[src] = new_opac
        model._scales.data[src] = new_scales
        groups = self._groups(model)
        for attr in PARAMS.values():
            old = getattr(model, attr)
            setattr(model, attr, Parameter(torch.cat([old.detach(), old.detach()[src]], dim=0)))
        for name in PER_POINT_BUFFERS:
            buffer = getattr(model, name, None)
            if isinstance(buffer, torch.Tensor) and buffer.shape[0] == before:
                setattr(model, name, torch.cat([buffer, buffer[src]], dim=0))
        # Thay tham số trong optimizer; trạng thái Adam của bản mới bằng 0, của nguồn đặt về 0.
        for group in optimizer.param_groups:
            attr = groups.get(group["name"])
            if attr is None:
                continue
            old_param = group["params"][0]
            new_param = getattr(model, attr)
            state = optimizer.state.pop(old_param, None)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        value = state[key]
                        value[src] = 0
                        state[key] = torch.cat([value, torch.zeros_like(value[: len(src)])], dim=0)
                optimizer.state[new_param] = state
            group["params"] = [new_param]

    def _zero_state(self, model: Any, optimizer: torch.optim.Optimizer, idx: torch.Tensor) -> None:
        groups = self._groups(model)
        for group in optimizer.param_groups:
            if group["name"] not in groups:
                continue
            state = optimizer.state.get(group["params"][0], {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][idx] = 0

    def inject_noise(self, model: Any, optimizer: torch.optim.Optimizer) -> None:
        """Cộng nhiễu vị trí theo hiệp phương sai, chỉ đáng kể khi độ đục dưới ~0,005 (như gsplat)."""
        if self.noise_lr <= 0 or model.num_points == 0:
            return
        lr = next(
            (g["lr"] for g in optimizer.param_groups if g["name"] == model.class_prefix + "xyz"),
            None,
        )
        if not lr:
            return
        opac = model.get_opacity.flatten()
        rot = _quat_to_rotmat(model.get_quats)
        scales = model.get_scaling
        covars = rot @ torch.diag_embed(scales**2) @ rot.transpose(-1, -2)
        gate = 1.0 / (1.0 + torch.exp(-100.0 * ((1.0 - opac) - 0.995)))
        noise = torch.randn_like(model._means) * gate[:, None] * lr * self.noise_lr
        model._means.data.add_(torch.einsum("bij,bj->bi", covars, noise))
