"""Lưu và nạp checkpoint native giữa hai bước train đồng bộ của worker XSIM.

Checkpoint chứa trạng thái model, densification, optimizer, sampler và bộ sinh số
ngẫu nhiên. Worker truyền fingerprint của đầu vào và cấu hình để kiểm tra lần
chạy tiếp. Xem kế hoạch tích hợp mục A05.
"""

import os
import random
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from drive2sim.external.xsim.optimization import capture_optimization, restore_optimization

if TYPE_CHECKING:
    from xsim.data import Subset
    from xsim.modeling.nodes import NodeCollection


def save_checkpoint(
    path: Path, fingerprint: str, nodes: "NodeCollection", subset: "Subset"
) -> None:
    """Ghi trạng thái sau optimizer step và thay thế checkpoint cũ sau khi ghi đủ."""
    generator, keys, position, has_gaussian, cached = np.random.get_state()
    state = {
        "version": 1,
        "fingerprint": fingerprint,
        "weights": nodes.state_dict(),
        "buffers": {
            (name, key): getattr(module, key)
            for name, module in nodes.named_modules()
            for key in module._non_persistent_buffers_set
        },
        "optimization": capture_optimization(nodes),
        "pairs": subset.image_indices,
        "errors": subset.img_sampler.error_buffer,
        "python_rng": random.getstate(),
        "numpy_rng": (generator, keys.tolist(), position, has_gaussian, cached),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if nodes.step.is_cuda else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(
    path: Path, fingerprint: str, nodes: "NodeCollection", subset: "Subset"
) -> int:
    """Khôi phục trạng thái ở biên batch và trả về số bước đã hoàn thành.

    Args:
        path: Checkpoint do worker ghi ở lần train trước.
        fingerprint: Định danh dữ liệu, cấu hình và bản mã đang dùng.
        nodes: Cây node mới khởi tạo trên thiết bị đích.
        subset: Tập train với thứ tự camera và mẫu đã xác định.

    Raises:
        ValueError: Khi checkpoint không tương thích với pipeline hiện tại.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["fingerprint"] != fingerprint:
        raise ValueError("XSIM checkpoint configuration differs")
    if state["pairs"] != subset.image_indices:
        raise ValueError("XSIM checkpoint observations differ")
    errors = subset.img_sampler.error_buffer
    if errors.shape != state["errors"].shape:
        raise ValueError("XSIM checkpoint sampler dimensions differ")
    nodes.load_state_dict(state["weights"], strict=True)
    restore_optimization(nodes, state["optimization"])
    buffers = {
        (name, key): module
        for name, module in nodes.named_modules()
        for key in module._non_persistent_buffers_set
    }
    if buffers.keys() != state["buffers"].keys():
        raise ValueError("XSIM checkpoint buffer names differ")
    for pair, module in buffers.items():
        value = state["buffers"][pair]
        module.register_buffer(
            pair[1], value.to(nodes.step.device) if value is not None else None, persistent=False
        )
    errors.copy_(state["errors"])
    generator, keys, position, has_gaussian, cached = state["numpy_rng"]
    np.random.set_state(
        (generator, np.asarray(keys, dtype="uint32"), position, has_gaussian, cached)
    )
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if nodes.step.is_cuda:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("XSIM checkpoint CUDA generator count differs")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(nodes.step.item())
