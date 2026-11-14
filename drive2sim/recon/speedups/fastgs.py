"""FastGS: densify và prune theo nhiều góc nhìn, theo tinh thần của FastGS (https://github.com/fastgs/FastGS).

Trước bước refine của model, render vài ảnh train, đánh dấu pixel sai nhiều, đo mỗi Gaussian góp
bao nhiêu vào các pixel đó; ra hai mask đặt lên model: Gaussian nào được densify
(``fastgs_densify_mask``), Gaussian nào bị prune (``fastgs_prune_mask``). Model áp mask trong
``refinement_after()`` (``models/gaussians/vanilla.py``). Chi tiết và khác biệt với paper:
``docs/methods/fastgs.md``.

Trainer chỉ gọi ba chỗ: ``before_step`` ở đầu bước, ``clear_masks`` sau khi model refine, và gán
``view_source`` (tập ảnh train) trong ``tools/train.py``. Thuật toán giữ nguyên bản teammate viết
trong ``trainers/base.py`` (commit 9586678); chỉ dời chỗ.

File này chứa lớp ``FastGS``.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from gsplat.rendering import rasterization

from drive2sim.recon.speedups.fastgs_rules import model_schedule, parse_config, sample_view_ids

logger = logging.getLogger()


class FastGS:
    """Chấm điểm Gaussian trên nhiều góc nhìn và đặt mask densify / prune lên model.

    Args:
        cfg: Khối ``trainer.fastgs`` của config, hoặc ``None``.
    """

    def __init__(self, cfg: Optional[Any]) -> None:
        self.cfg = parse_config(cfg)
        self.view_source = None  # tập ảnh train (SplitWrapper), do tools/train.py gán

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    @staticmethod
    def clear_masks(model: nn.Module) -> None:
        """Bỏ mask sau khi model đã refine: mask chỉ đúng với số Gaussian của bước này."""
        model.fastgs_densify_mask = None
        model.fastgs_prune_mask = None

    def models(self, trainer: Any) -> Dict[str, nn.Module]:
        """Các lớp Gaussian có hỗ trợ FastGS (``fastgs_supported``)."""
        return {
            name: trainer.models[name]
            for name in trainer.gaussian_classes.keys()
            if getattr(trainer.models[name], "fastgs_supported", False)
        }

    def schedule(self, trainer: Any, step: int) -> Tuple[List[str], List[str]]:
        """Các lớp densify, prune ở bước này (``model_schedule``)."""
        densify, prune = [], []
        for name, model in self.models(trainer).items():
            do_densify, do_prune = model_schedule(
                step, model.ctrl_cfg, model.num_train_images, self.cfg, trainer.num_iters
            )
            if do_densify:
                densify.append(name)
            if do_prune:
                prune.append(name)
        return densify, prune

    def contributions(
        self, trainer: Any, flagged: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tổng trọng số blend (alpha × độ trong suốt) của mỗi Gaussian trên các pixel bị đánh
        dấu của ảnh vừa render, và Gaussian đó có hiện trong ảnh không.

        Trọng số là gradient của tổng (pixel bị đánh dấu × ảnh) theo một "màu" riêng của mỗi
        Gaussian, nên chạy được với mọi render mode (default / ut / geer) mà không sửa rasterizer.
        """
        gs, cam, kwargs = trainer._last_render_inputs
        render_mode = cam.render_mode or trainer.render_cfg.get("render_mode", "default")
        camera_kwargs = trainer.camera_rasterization_kwargs(cam, render_mode)
        kwargs = dict(kwargs)
        kwargs["render_mode"] = "RGB"
        kwargs.pop("channel_chunk", None)
        opacities = gs.opacities.detach().reshape(-1)
        with torch.enable_grad():
            weight = torch.zeros(
                opacities.shape[0], 1, device=opacities.device, requires_grad=True
            )
            renders, _, info = rasterization(
                means=gs.means.detach(),
                quats=gs.quats.detach(),
                scales=gs.scales.detach(),
                opacities=opacities,
                colors=weight.repeat(1, 3),
                viewmats=torch.linalg.inv(cam.camtoworlds.detach())[None, ...],
                Ks=cam.Ks[None, ...],
                width=cam.W,
                height=cam.H,
                packed=False,
                absgrad=False,
                sparse_grad=False,
                rasterize_mode="antialiased" if trainer.render_cfg.antialiased else "classic",
                **camera_kwargs,
                **kwargs,
            )
            total = (renders[0][..., :3].mean(dim=-1) * flagged).sum()
            (contrib,) = torch.autograd.grad(total, weight)
        radii = info["radii"]
        if radii.ndim == 3:
            radii = radii.amax(dim=-1)
        # opacity bằng 0 với Gaussian của vật không có mặt ở khung này
        visible = (radii[0] > 0) & (opacities > 0)
        return contrib[:, 0].detach(), visible

    def before_step(self, trainer: Any, step: int) -> None:
        """Chấm điểm và đặt mask lên model, nếu bước này có model densify hoặc prune.

        Gọi ở đầu bước, trước lượt forward của chính bước đó, để mask khớp số Gaussian lúc model
        refine ở cuối bước. Hai khác biệt với paper, đều cần cho clip lái xe:
          * mỗi pixel bị đánh dấu tính theo trọng số blend của Gaussian (không phải 1), vì đếm
            chính xác cần rasterizer riêng của FastGS;
          * điểm chia trung bình trên số ảnh thật sự thấy Gaussian (không phải mọi ảnh được chọn),
            vì trong clip chạy thẳng phần lớn ảnh không thấy một Gaussian cho trước.
        """
        if not self.enabled or self.view_source is None:
            return
        cfg = self.cfg
        densify, prune = self.schedule(trainer, step)
        if not densify and not prune:
            return
        models = self.models(trainer)
        source = self.view_source
        num_views = min(int(cfg.num_views), len(source))
        offset = float(torch.rand(1))
        view_ids = sample_view_ids(len(source), num_views, offset)
        downscale = trainer._get_downscale_factor()
        device = trainer.device

        acc = {
            name: {
                k: torch.zeros(model.num_points, device=device)
                for k in ("weight", "views", "error")
            }
            for name, model in models.items()
        }
        with torch.no_grad():
            for view_id in view_ids:
                image_infos, cam_infos = source.get_image(view_id, downscale)
                image_infos = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in image_infos.items()
                }
                cam_infos = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in cam_infos.items()
                }
                outputs = trainer(image_infos, cam_infos)
                pred, gt = outputs["rgb"], image_infos["pixels"]
                if "egocar_masks" in image_infos:
                    valid = (1.0 - image_infos["egocar_masks"]).float()
                else:
                    valid = torch.ones_like(pred[..., 0])
                pred, gt = pred * valid[..., None], gt * valid[..., None]
                error = (pred - gt).abs().mean(dim=-1)
                lo, hi = error.min(), error.max()
                flagged = ((error - lo) / (hi - lo + 1e-8) > cfg.loss_thresh).float() * valid
                photometric = 0.8 * (pred - gt).abs().mean() + 0.2 * (
                    1 - trainer.ssim(gt.permute(2, 0, 1)[None, ...], pred.permute(2, 0, 1)[None, ...])
                )
                del outputs
                trainer.info = None  # giải phóng lần render đầu trước khi render lần hai
                contrib, visible = self.contributions(trainer, flagged)
                for name, model in models.items():
                    in_class = trainer.pts_labels == trainer.gaussian_classes[name]
                    rendered = model.filter_mask
                    if int(in_class.sum()) == 0 or int(in_class.sum()) != int(rendered.sum()):
                        continue  # lớp này không được render ở khung này
                    acc[name]["weight"][rendered] += contrib[in_class]
                    acc[name]["views"][rendered] += visible[in_class].float()
                    acc[name]["error"][rendered] += photometric * contrib[in_class]

        all_error = torch.cat([a["error"] for a in acc.values()])
        err_lo, err_hi = all_error.min(), all_error.max()
        for name, model in models.items():
            a = acc[name]
            seen = a["views"] > 0
            score = a["weight"] / a["views"].clamp(min=1)
            msg = (
                f"FastGS @ step {step} {name}: seen in sampled views "
                f"{int(seen.sum())}/{model.num_points}"
            )
            if name in densify:
                keep = score > cfg.score_thresh
                if cfg.densify_unseen:
                    keep = keep | ~seen
                model.fastgs_densify_mask = keep
                seen_scores = score[seen]
                # torch.quantile có giới hạn kích thước
                seen_scores = seen_scores[:: max(1, seen_scores.shape[0] // 1_000_000)]
                quantiles = (
                    torch.quantile(seen_scores, torch.tensor([0.5, 0.9, 0.99], device=device))
                    if seen.any()
                    else torch.zeros(3)
                )
                msg += (
                    f", may densify {int(keep.sum())}, score p50/p90/p99 "
                    f"{quantiles[0]:.2f}/{quantiles[1]:.2f}/{quantiles[2]:.2f}"
                )
            if name in prune:
                error_score = (a["error"] - err_lo) / (err_hi - err_lo + 1e-12)
                low_opacity = model.get_opacity.data.reshape(-1) < cfg.prune_min_opacity
                model.fastgs_prune_mask = low_opacity | (error_score > cfg.prune_score_thresh)
                msg += (
                    f", prune {int(model.fastgs_prune_mask.sum())} "
                    f"(low opacity {int(low_opacity.sum())}, "
                    f"high error {int((error_score > cfg.prune_score_thresh).sum())})"
                )
            logger.info(msg)
