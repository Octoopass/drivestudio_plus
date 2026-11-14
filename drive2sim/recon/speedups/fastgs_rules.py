"""Quy tắc của FastGS không cần PyTorch: cấu hình, lịch chạy, cách chọn ảnh để chấm điểm.

Tách khỏi ``fastgs.py`` để test được trên máy không có GPU. Hành vi giữ nguyên bản teammate viết
(commit 9586678, ``docs/methods/fastgs.md``).

File này chứa ``FASTGS_DEFAULTS``, ``parse_config``, ``model_schedule``, ``sample_view_ids``,
``sample_view_ids_by_camera``.
"""

from types import SimpleNamespace
from typing import Any, Mapping, Optional

# Giá trị mặc định của khối trainer.fastgs (docs/methods/fastgs.md).
FASTGS_DEFAULTS = dict(
    enabled=False,
    num_views=10,  # số ảnh train lấy mỗi lần chấm điểm, rải đều theo clip
    loss_thresh=0.1,  # pixel "sai nhiều" khi sai số L1 sau chuẩn hoá min-max vượt ngưỡng này
    score_thresh=1.0,  # chỉ densify khi số pixel sai (tính theo trọng số blend) mỗi ảnh thấy nó vượt ngưỡng
    densify_unseen=False,  # Gaussian không ảnh nào được chọn thấy có được densify chỉ theo gradient không
    prune=True,  # prune theo nhiều góc nhìn ở cuối quá trình train
    prune_from=15000,  # bước đầu tiên được prune
    prune_min_opacity=0.1,  # prune Gaussian có opacity dưới ngưỡng này...
    prune_score_thresh=0.9,  # ...hoặc điểm lỗi sau chuẩn hoá vượt ngưỡng này
)


def parse_config(cfg: Optional[Mapping[str, Any]]) -> SimpleNamespace:
    """Gộp khối ``trainer.fastgs`` với giá trị mặc định.

    Raises:
        AssertionError: Nếu có khoá lạ (giữ đúng cách trainer cũ báo lỗi).
    """
    values = dict(cfg) if cfg is not None else {}
    unknown = set(values) - set(FASTGS_DEFAULTS)
    assert not unknown, f"unknown trainer.fastgs keys: {sorted(unknown)}"
    return SimpleNamespace(**{**FASTGS_DEFAULTS, **values})


def model_schedule(
    step: int, ctrl: Any, num_train_images: int, cfg: SimpleNamespace, num_iters: int
) -> tuple[bool, bool]:
    """Ở bước ``step``, model có densify, có prune theo FastGS không.

    Chép đúng điều kiện trong ``refinement_after()`` của model, để chỉ chấm điểm khi model thật sự
    refine.

    Args:
        step: Bước train.
        ctrl: ``ctrl_cfg`` của model (``warmup_steps``, ``refine_interval``,
            ``reset_alpha_interval``, ``stop_split_at``).
        num_train_images: Số ảnh train của model.
        cfg: Cấu hình FastGS (``parse_config``).
        num_iters: Tổng số bước train.

    Returns:
        ``(densify, prune)``.
    """
    if step <= ctrl.warmup_steps or step % ctrl.refine_interval != 0:
        return False, False
    phase = step % ctrl.reset_alpha_interval
    if phase <= max(num_train_images, ctrl.refine_interval):
        return False, False  # ngay sau khi reset opacity, model không densify cũng không cull
    densify = step < ctrl.stop_split_at
    # prune ở lần refine cuối trước mỗi lần reset opacity, khi opacity đã hồi lại
    prune = (
        cfg.prune
        and cfg.prune_from < step < num_iters
        and phase == ctrl.reset_alpha_interval - ctrl.refine_interval
    )
    return densify, bool(prune)


def sample_view_ids(num_source: int, num_views: int, offset: float) -> list[int]:
    """Chọn chỉ số ảnh rải đều theo clip, lệch một đoạn ``offset`` (0..1) để mỗi lần mỗi khác.

    Args:
        num_source: Số ảnh train.
        num_views: Số ảnh muốn lấy (``cfg.num_views``).
        offset: Số ngẫu nhiên trong [0, 1).

    Returns:
        Các chỉ số khác nhau, tăng dần.
    """
    num_views = min(int(num_views), num_source)
    return sorted(
        {min(int((i + offset) * num_source / num_views), num_source - 1) for i in range(num_views)}
    )


def sample_view_ids_by_camera(cameras: list[int], num_views: int, offset: float) -> list[int]:
    """Chọn ảnh rải đều theo từng camera, để mỗi lần chấm điểm thấy mọi camera.

    DriveStudio xếp ảnh theo khung rồi tới camera (ảnh ``i`` là camera ``i % số camera``). Rải đều
    trên cả danh sách thì khi số khung train chia hết cho ``num_views``, bước nhảy là bội của số
    camera và cả lượt chỉ rơi vào một camera; Gaussian chỉ camera khác thấy bị coi là "không ảnh
    nào thấy" và không được densify ở lượt đó. Hàm này chia ``num_views`` đều cho các camera rồi rải
    đều trong ảnh của từng camera. Một camera thì trả đúng như ``sample_view_ids``.

    Args:
        cameras: Camera của từng ảnh train, theo thứ tự danh sách ảnh.
        num_views: Số ảnh muốn lấy.
        offset: Số ngẫu nhiên trong [0, 1).

    Returns:
        Chỉ số ảnh (trong danh sách train) khác nhau, tăng dần.
    """
    positions: dict[int, list[int]] = {}
    for index, camera in enumerate(cameras):
        positions.setdefault(camera, []).append(index)
    order = sorted(positions)
    if len(order) <= 1:
        return sample_view_ids(len(cameras), num_views, offset)
    num_views = min(int(num_views), len(cameras))
    start = int(offset * len(order)) % len(order)  # camera nhận phần dư thay đổi theo từng lượt
    order = order[start:] + order[:start]
    base, extra = divmod(num_views, len(order))
    chosen: list[int] = []
    for rank, camera in enumerate(order):
        count = min(base + (1 if rank < extra else 0), len(positions[camera]))
        if count:
            picks = sample_view_ids(len(positions[camera]), count, offset)
            chosen += [positions[camera][i] for i in picks]
    return sorted(set(chosen))
