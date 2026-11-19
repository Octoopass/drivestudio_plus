"""Lịch densify hiệu lực của một run, đọc từ config DriveStudio đã chạy.

Ô cấu hình của run chỉ ghi lựa chọn của approach (renderer, FastGS...), không cho biết densify bắt
đầu, lặp lại, dừng ở bước nào và ngưỡng bao nhiêu; các giá trị này khác nhau giữa renderer (3DGEER
densify mỗi 50 bước với ngưỡng 0,01, 3DGS mỗi 100 bước với 0,0005) và giữa các node (xe, vật biến
dạng ngừng tách ở bước 20 000, nền ở 15 000).

Nguồn là ``config.yaml`` mà DriveStudio ghi vào thư mục stage train lúc bắt đầu train, sau khi đã
gộp ``extends`` và override dòng lệnh (``refine_interval``, ``ut_densify``, ``overrides``...).
``method_config.yaml`` ghi trước khi áp override nên không dùng: ví dụ run ``a8a9`` đặt
``refine_interval=50`` mà ``method_config.yaml`` vẫn ghi 300.

File này chứa ``densify_info``, ``load_densify_info``.
"""

from pathlib import Path
from typing import Any, Optional

from drive2sim.core.config import ConfigError, load_config
from drive2sim.recon.densify.mcmc_defaults import DEFAULT_CLASSES, DEFAULTS, caps_for

D2S_DIR = "d2s"
TRAIN_STAGES = ("train.drivestudio",)
CONFIG_FILE = "config.yaml"
# Node có densify, theo thứ tự hiện. Người (SMPLNodes) không bao giờ densify.
DENSIFY_NODES = ("Background", "RigidNodes", "DeformableNodes")
# Khoá của quy tắc densify gốc (gaussian_ctrl_general_cfg, đè được bằng ctrl của từng node).
SCHEDULE_KEYS = (
    "warmup_steps",
    "refine_interval",
    "densify_grad_thresh",
    "densify_size_thresh",
    "cull_alpha_thresh",
    "reset_alpha_interval",
    "stop_split_at",
)
SCREEN_TRAINER = "ScreenGradMultiTrainer"


def _statistic(trainer: dict[str, Any]) -> str:
    """Đại lượng so với ``densify_grad_thresh``.

    ``2d_abs``, ``2d``: gradient tâm 2D trên ảnh (3DGS, có hay không ``absgrad``); ``3d``: gradient
    tâm 3D của nhánh ``ut`` (3DGUT gốc của Bosch); ``screen``: gradient đó đổi sang pixel
    (``ut_densify=screen``); ``geer``: ``geer_gradient`` của 3DGEER.
    """
    render = trainer.get("render") or {}
    mode = render.get("render_mode", "default")
    if mode == "ut":
        return "screen" if str(trainer.get("type", "")).endswith(SCREEN_TRAINER) else "3d"
    if mode == "geer":
        return "geer"
    return "2d_abs" if render.get("absgrad") else "2d"


def densify_info(config: dict[str, Any]) -> dict[str, Any]:
    """Tóm tắt cách densify của một config DriveStudio đã gộp.

    Args:
        config: Nội dung ``config.yaml`` của stage train.

    Returns:
        ``rule`` (``default``, ``fastgs``, ``mcmc``), ``renderer`` (``render_mode``),
        ``statistic`` (``_statistic``), ``nodes`` (``{node: {khoá trong SCHEDULE_KEYS: giá trị}}``
        cho node có trong config, đã gộp ``ctrl`` của node), thêm ``fastgs`` (khoá của khối
        ``trainer.fastgs`` trừ ``enabled``) hoặc ``mcmc`` (lịch, ``classes``, ``caps`` theo mặc
        định của ``mcmc_defaults``) khi bật.
    """
    trainer = config.get("trainer") or {}
    general = trainer.get("gaussian_ctrl_general_cfg") or {}
    models = config.get("model") or {}
    fastgs = trainer.get("fastgs") or {}
    mcmc = trainer.get("mcmc") or {}
    rule = "mcmc" if mcmc.get("enabled") else "fastgs" if fastgs.get("enabled") else "default"

    nodes: dict[str, dict[str, Any]] = {}
    for name in DENSIFY_NODES:
        if name not in models:
            continue
        ctrl = {**general, **((models[name] or {}).get("ctrl") or {})}
        nodes[name] = {key: ctrl[key] for key in SCHEDULE_KEYS if key in ctrl}

    info: dict[str, Any] = {
        "rule": rule,
        "renderer": (trainer.get("render") or {}).get("render_mode", "default"),
        "statistic": _statistic(trainer),
        "nodes": nodes,
    }
    if rule == "fastgs":
        info["fastgs"] = {key: value for key, value in fastgs.items() if key != "enabled"}
    if rule == "mcmc":
        classes = tuple(mcmc.get("classes", DEFAULT_CLASSES))
        caps = caps_for(mcmc.get("cap_max"), classes)
        used = [name for name in classes if name in nodes]
        info["mcmc"] = {
            **{key: mcmc.get(key, default) for key, default in DEFAULTS.items()},
            "classes": used,
            "caps": {name: caps[name] for name in used if name in caps},
        }
    return info


def load_densify_info(run_dir: Path) -> Optional[dict[str, Any]]:
    """Đọc ``densify_info`` của run từ ``config.yaml`` của stage train.

    Args:
        run_dir: Thư mục run.

    Returns:
        Kết quả của ``densify_info``; ``None`` khi chưa có ``config.yaml`` (train chưa bắt đầu, run
        Kaggle chưa tải về) hoặc file không đọc được.
    """
    for stage in TRAIN_STAGES:
        path = Path(run_dir) / D2S_DIR / "stages" / stage / CONFIG_FILE
        if not path.is_file():
            continue
        try:
            # Đọc như OmegaConf (``1e-4`` là số), giống cách DriveStudio đọc file này.
            config = load_config(path)
        except (OSError, ConfigError):
            return None
        return densify_info(config)
    return None
