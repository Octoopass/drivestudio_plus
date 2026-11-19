"""Giá trị mặc định của densify MCMC (``mcmc.py``), không cần PyTorch.

Tách khỏi ``mcmc.py`` để Studio (môi trường nhẹ, không có PyTorch) đọc đúng lịch MCMC của một run
từ cùng nguồn với trainer (``drive2sim/engine/densify_info.py``). Ý nghĩa từng khoá:
``docs/methods/densify_mcmc.md``.

File này chứa ``DEFAULT_CLASSES``, ``DEFAULT_CAPS``, ``DEFAULTS``, ``caps_for``.
"""

from typing import Any

DEFAULT_CLASSES = ("Background", "RigidNodes", "DeformableNodes")
# Ngân sách mặc định mỗi node. Nền: 1 triệu như mặc định của gsplat; nuScenes scene-0061 khởi tạo
# khoảng 0,66 triệu Gaussian nền (LiDAR và điểm ngẫu nhiên), nên ngân sách thấp hơn số đó thì MCMC
# chỉ dời mà không bao giờ thêm (MCMC không xoá Gaussian).
DEFAULT_CAPS = {"Background": 1_000_000, "RigidNodes": 50_000, "DeformableNodes": 10_000}
# Khoá số của khối ``trainer.mcmc`` và giá trị khi config không ghi.
DEFAULTS: dict[str, float] = {
    "noise_lr": 5e5,
    "min_opacity": 0.005,
    "refine_start_iter": 500,
    "refine_stop_iter": 25_000,
    "refine_every": 100,
    "grow_rate": 0.05,
    "opacity_reg": 0.01,
    "scale_reg": 0.01,
}


def caps_for(cap: Any, classes: tuple[str, ...]) -> dict[str, int]:
    """Ngân sách Gaussian của từng node từ ``cap_max`` của config.

    Args:
        cap: ``None`` (dùng ``DEFAULT_CAPS``), một số (mọi node trong ``classes``), hoặc dict theo
            node (đè lên ``DEFAULT_CAPS``).
        classes: Các node dùng MCMC.

    Returns:
        ``{node: ngân sách}``.
    """
    if isinstance(cap, (int, float)) and not isinstance(cap, bool):
        return {name: int(cap) for name in classes}
    caps = dict(DEFAULT_CAPS)
    if cap:
        caps.update({name: int(value) for name, value in dict(cap).items()})
    return caps
