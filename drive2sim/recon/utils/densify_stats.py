"""Ghi thống kê của đại lượng densify ở mỗi lần refine (docs/methods/densify_3dgut.md mục 4, 5).

Mỗi renderer densify theo một đại lượng khác nhau (3DGS: gradient trên ảnh, đơn vị pixel; 3DGUT:
gradient theo vị trí 3D; 3DGEER: ``geer_gradient``), nên ngưỡng ``densify_grad_thresh`` không dùng
chung được. Để chọn ngưỡng cho 3DGUT bằng số đo (khớp phân vị với 3DGS), cần biết phân bố của đại
lượng đó: file này ghi các phân vị và tỉ lệ Gaussian vượt ngưỡng của từng lớp Gaussian, ngay trước
mỗi lần refine, vào ``<log_dir>/densify_stats.jsonl``.

Chỉ đọc, không đổi việc densify; tắt mặc định (khối ``trainer.densify_stats.enabled``). Không sửa
nhánh tính gradient theo renderer của patch Bosch (``trainers/base.py``,
``postprocess_per_train_step``); model gọi ``record`` từ ``VanillaGaussians.postprocess_per_train_step``.
"""

import json
import os
from typing import Any, Optional

import torch

STATS_FILE = "densify_stats.jsonl"
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
# torch.quantile nhận tối đa 2^24 phần tử; lớp lớn hơn thì lấy mẫu đều
_MAX_SAMPLES = 1 << 24


class DensifyStats:
    """Thống kê densify, bật bằng ``trainer.densify_stats.enabled``.

    Args:
        cfg: Khối config ``trainer.densify_stats`` (hoặc ``None``: tắt).
    """

    def __init__(self, cfg: Optional[Any] = None) -> None:
        self.enabled = bool(cfg is not None and cfg.get("enabled", False))
        self.path: Optional[str] = None
        self.render_mode = "default"

    def attach(self, trainer: Any, log_dir: str) -> None:
        """Gắn vào mọi lớp Gaussian của trainer (gọi sau khi model được dựng hoặc nạp lại)."""
        if not self.enabled:
            return
        self.path = os.path.join(log_dir, STATS_FILE)
        self.render_mode = trainer.render_cfg.get("render_mode", "default")
        for model in trainer.models.values():
            if hasattr(model, "densify_stats"):
                model.densify_stats = self

    def record(self, model: Any, step: int) -> Optional[dict]:
        """Ghi một dòng cho một lớp Gaussian; trả dòng đã ghi (``None`` nếu chưa có số liệu)."""
        if self.path is None or model.xys_grad_norm is None or model.vis_counts is None:
            return None
        with torch.no_grad():
            seen = model.vis_counts > 0
            values = (model.xys_grad_norm[seen] / model.vis_counts[seen]).float().flatten()
            if values.numel() == 0:
                return None
            sample = values
            if sample.numel() > _MAX_SAMPLES:
                index = torch.linspace(0, sample.numel() - 1, _MAX_SAMPLES, device=sample.device)
                sample = sample.sort().values[index.long()]
            quantiles = torch.quantile(
                sample, torch.tensor(QUANTILES, device=sample.device, dtype=sample.dtype)
            )
            threshold = float(model.ctrl_cfg.densify_grad_thresh)
            row = {
                "step": int(step),
                "class": model.class_prefix,
                "render_mode": self.render_mode,
                "num_points": int(model.num_points),
                "num_seen": int(values.numel()),
                "threshold": threshold,
                "num_above": int((values > threshold).sum()),
                "frac_above": float((values > threshold).float().mean()),
                "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, quantiles.tolist())},
            }
        with open(self.path, "a") as stream:
            stream.write(json.dumps(row) + "\n")
        return row
