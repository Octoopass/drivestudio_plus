"""Đo thời gian GPU của một cửa sổ bước train bằng torch.profiler (plan Faster-GS, bước D1).

Bật bằng ``logging.profile_at`` (danh sách bước bắt đầu đo) và ``logging.profile_steps`` (số bước
mỗi lần đo, mặc định 30). Mỗi lần đo ghi hai file vào thư mục log:
- ``profile_step<bước>.txt``: bảng kernel tốn thời gian GPU nhất (``key_averages``);
- ``profile_step<bước>.json``: tỉ lệ thời gian GPU theo nhóm: rasterizer (gsplat: chiếu, ghép
  tile, sort, vẽ, backward), optimizer, SSIM, còn lại. Con số này cho biết có đáng tối ưu kernel
  rasterizer không (``docs/methods/fastergs.md``).

Profiler làm chậm chính các bước đang đo, nên chỉ bật trong run đo riêng; không bật trong run dùng
để so ``train_time_s``. Mặc định tắt.

File này chứa ``StepProfiler``, ``categorize``.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

# Nhóm theo chữ có trong tên kernel CUDA, xét theo thứ tự; không khớp nhóm nào là "other".
CATEGORIES = (
    (
        "rasterizer",
        ("gsplat", "rasterize", "projection", "intersect", "isect", "radixsort", "radix_sort", "eval3d"),
    ),
    ("optimizer", ("adam", "multi_tensor_apply")),
    ("ssim", ("fusedssim", "fused_ssim", "ssim")),
)


def categorize(kernel_name: str) -> str:
    """Nhóm của một kernel theo tên (không phân biệt hoa thường)."""
    lowered = kernel_name.lower()
    for category, keys in CATEGORIES:
        if any(key in lowered for key in keys):
            return category
    return "other"


def _device_time_us(event) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, None)
        if value:
            return float(value)
    return 0.0


class StepProfiler:
    """Bật torch.profiler ở các bước ``profile_at``, mỗi lần ``steps`` bước, rồi ghi kết quả.

    Args:
        profile_at: Các bước bắt đầu đo; rỗng là tắt.
        steps: Số bước mỗi lần đo.
        log_dir: Thư mục ghi kết quả.
    """

    def __init__(self, profile_at: Iterable[int], steps: int, log_dir: str) -> None:
        self.profile_at = sorted(int(s) for s in profile_at)
        self.steps = int(steps)
        self.log_dir = log_dir
        self._prof: Optional[Any] = None  # torch.profiler.profile
        self._start = 0

    def begin_step(self, step: int) -> None:
        """Gọi đầu mỗi bước train."""
        if self._prof is None and step in self.profile_at:
            import torch  # trong hàm, để categorize test được ở nơi không có PyTorch

            self._prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            self._prof.__enter__()
            self._start = step

    def end_step(self, step: int) -> None:
        """Gọi cuối mỗi bước train, sau khi đã đồng bộ GPU."""
        if self._prof is None or step + 1 < self._start + self.steps:
            return
        prof, self._prof = self._prof, None
        prof.__exit__(None, None, None)
        self._write(prof, self._start, step + 1 - self._start)

    def _write(self, prof: Any, start: int, steps: int) -> None:
        import torch

        averages = prof.key_averages()
        table = averages.table(sort_by="self_cuda_time_total", row_limit=40)
        base = os.path.join(self.log_dir, f"profile_step{start}")
        with open(base + ".txt", "w") as f:
            f.write(f"steps {start}..{start + steps - 1}\n{table}\n")
        totals: Dict[str, float] = defaultdict(float)
        kernels: List[tuple] = []
        # Nhãn khoảng thời gian (record_function, ví dụ "Optimizer.step#Adam.step") hiện cả ở CPU
        # lẫn GPU và bao trùm các kernel bên trong; bỏ qua để không đếm hai lần. Kernel thật không
        # có bản trùng tên ở phía CPU.
        cpu_names = {
            e.key for e in averages if getattr(e, "device_type", None) == torch.autograd.DeviceType.CPU
        }
        for event in averages:
            if getattr(event, "device_type", None) != torch.autograd.DeviceType.CUDA:
                continue
            if event.key in cpu_names:
                continue
            time_us = _device_time_us(event)
            totals[categorize(event.key)] += time_us
            kernels.append((time_us, event.key))
        gpu_ms = sum(totals.values()) / 1000.0
        summary = {
            "steps": steps,
            "start_step": start,
            "gpu_ms_per_step": round(gpu_ms / max(steps, 1), 3),
            "share": {
                name: round(totals.get(name, 0.0) / 1000.0 / gpu_ms, 4) if gpu_ms else 0.0
                for name in [c for c, _ in CATEGORIES] + ["other"]
            },
            "top_kernels_ms_per_step": [
                {"kernel": key, "ms": round(t / 1000.0 / max(steps, 1), 3), "category": categorize(key)}
                for t, key in sorted(kernels, reverse=True)[:15]
            ],
        }
        with open(base + ".json", "w") as f:
            json.dump(summary, f, indent=2)
