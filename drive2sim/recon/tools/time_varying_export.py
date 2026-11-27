"""Chọn Gaussian nền tĩnh khi xuất ``.ply`` từ model thay đổi theo thời gian (PVG).

``export_ply.py`` ghi tham số gốc của node ``Background``. Với PVG (Periodic Vibration Gaussian,
``models/gaussians/pvg.py``) mỗi Gaussian chỉ hiện quanh thời điểm đỉnh ``τ`` của nó: độ đục nhân
``exp(-0.5 (t - τ)² / β²)`` với ``β = exp(_betas)`` là độ dài đời, và tâm dao động
``v·sin(2π(t - τ)/T)·T/2π`` quanh ``_means``. Ở đúng ``t = τ`` tâm là ``_means`` và độ đục là độ
đục gốc, nên tham số gốc là ảnh chụp mỗi Gaussian ở đỉnh của chính nó. Gaussian của nền sống suốt
đoạn đường (``β`` lớn), Gaussian của xe chạy chỉ sống vài khung; ghi hết thì vệt xe chạy hiện cùng
lúc. Vì vậy ``.ply`` nền tĩnh của PVG = các Gaussian có độ dài đời ít nhất ``min_life_frames``
khung (``β / time_interval``, ``time_interval`` là bước thời gian của một khung trong config PVG).

Ngưỡng mặc định 20 khung (2 giây ở 10 Hz) là ước lượng, chưa chỉnh trên run thật: xe đỗ được giữ
(là nền tĩnh), xe chạy qua một chỗ trong ít khung bị bỏ. Xem phân bố trong ``scene.json``
(``time_varying.life_frames_quantiles``) để chỉnh.

File này chỉ dùng numpy và PyYAML để test được không cần PyTorch.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

PVG_TYPE = "PeriodicVibrationGaussians"
DEFAULT_TIME_INTERVAL = 0.02  # trainer.Background.ctrl.time_interval của configs/methods/pvg.yaml
DEFAULT_MIN_LIFE_FRAMES = 20.0


def is_pvg_state(state: dict) -> bool:
    """State dict của node có phải PVG không (có tham số thời gian ``_taus``, ``_betas``)."""
    return "_taus" in state and "_betas" in state


def pvg_time_interval(config_path: Path) -> Optional[float]:
    """``model.Background.ctrl.time_interval`` trong ``config.yaml`` của lần train.

    Args:
        config_path: ``config.yaml`` cạnh checkpoint (DriveStudio ghi khi train).

    Returns:
        Bước thời gian một khung; ``None`` nếu không có file hoặc không phải config PVG.
    """
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    background = (config.get("model") or {}).get("Background") or {}
    if PVG_TYPE not in str(background.get("type", "")):
        return None
    value = (background.get("ctrl") or {}).get("time_interval")
    return float(value) if isinstance(value, (int, float)) else None


def life_frames(betas_log: Any, time_interval: float) -> np.ndarray:
    """Độ dài đời của từng Gaussian PVG, tính bằng số khung: ``exp(_betas) / time_interval``."""
    return np.exp(np.asarray(betas_log, dtype=np.float64).reshape(-1)) / time_interval


def static_mask(
    betas_log: Any, time_interval: float, min_life_frames: float = DEFAULT_MIN_LIFE_FRAMES
) -> tuple[np.ndarray, dict]:
    """Mask Gaussian nền tĩnh của PVG và thống kê để ghi vào ``scene.json``.

    Args:
        betas_log: ``_betas`` của node (log của độ dài đời), shape ``[N, 1]`` hoặc ``[N]``.
        time_interval: Bước thời gian một khung trong config PVG.
        min_life_frames: Độ dài đời tối thiểu, khung; ``0`` giữ mọi Gaussian.

    Returns:
        ``(mask [N] bool, thống kê)``.
    """
    lives = life_frames(betas_log, time_interval)
    mask = lives >= min_life_frames
    quantiles = (
        np.quantile(lives, [0.1, 0.5, 0.9]).round(2).tolist() if lives.size else [0.0] * 3
    )
    stats = {
        "model": "pvg",
        "rule": "keep gaussians with life span exp(_betas)/time_interval >= min_life_frames",
        "time_interval": time_interval,
        "min_life_frames": min_life_frames,
        "kept": int(mask.sum()),
        "dropped": int((~mask).sum()),
        "life_frames_quantiles": {"p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2]},
    }
    return mask, stats
