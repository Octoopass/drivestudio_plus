"""Danh mục thư viện và môi trường Studio cần, kèm kiểm tra và cách cài từng mục (CONTRACTS 11.1).

Như bảng "Thư viện" của Fisheye Fusion Lab: mỗi mục nói dùng để làm gì, link tới nguồn, dung lượng,
thời gian cài ước lượng, trạng thái trên máy chạy Studio, và nút cài khi cài được không cần root.
Mục thuộc môi trường chính cài bằng các bước của ``envs/install_main.sh`` (``--from``), Instant
NuRec bằng ``envs/install_nurec.sh``; mục khác là một lệnh (pip vào Python đang chạy Studio, tải
weight, ``uv tool install``). Việc cài đi qua ``InstallFlow`` (mỗi lúc một việc, có log, dừng
được).

File này chứa:
- ``CATALOG``: danh mục (dữ liệu tĩnh).
- ``item_status``: trạng thái một mục trên máy này.
- ``install_plan``: cách cài một mục (``("env", tên, bước)`` hoặc ``("command", lệnh)``).
- ``catalog``: danh mục kèm trạng thái cho Studio.
"""

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

SEGFORMER = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
GROUPS = ("Môi trường", "Huấn luyện", "Dữ liệu", "Công cụ")
CATALOG: list[dict] = [
    {
        "id": "env-main", "group": "Môi trường", "name": "Môi trường chính (DriveStudio)",
        "desc": "PyTorch 2.4.1 + CUDA 12.1, requirements.txt, gsplat-geer, nvdiffrast, pytorch3d, "
        "smplx, 4D-Humans: mọi approach dòng DriveStudio chạy trên máy cần môi trường này.",
        "links": [["DriveStudio", "https://github.com/ziyc/drivestudio"]],
        "check": ("env", "drivestudio"), "install": ("env", "main", 1),
        "size": "~12 GB", "minutes": 60,
    },
    {
        "id": "env-nurec", "group": "Môi trường", "name": "Instant NuRec",
        "desc": "Mô hình feed-forward của NVIDIA (mã nguồn chốt commit, venv Python 3.11 riêng có "
        "nvidia-ncore, checkpoint pa-multiview) cho approach Instant NuRec.",
        "links": [
            ["GitHub", "https://github.com/NVIDIA/instant-nurec"],
            ["Hugging Face", "https://huggingface.co/nvidia/instant-nurec"],
        ],
        "check": ("env", "external:nurec"), "install": ("env", "nurec", 1),
        "size": "~7 GB", "minutes": 15,
    },
    {
        "id": "renderers", "group": "Huấn luyện",
        "name": "Renderer CUDA: gsplat-geer, nvdiffrast, pytorch3d",
        "desc": "Bắt buộc để train và chấm điểm (3DGS, 3DGUT, 3DGEER). Build từ mã nguồn với CUDA "
        "của máy (bước 4 tới 8 của install_main.sh).",
        "links": [
            ["3dgeer (gsplat-geer)", "https://github.com/boschresearch/3dgeer/tree/gsplat-geer"],
            ["pytorch3d", "https://github.com/facebookresearch/pytorch3d"],
            ["nvdiffrast", "https://github.com/NVlabs/nvdiffrast"],
        ],
        "check": ("import", ["gsplat", "nvdiffrast", "pytorch3d"]), "install": ("env", "main", 4),
        "requires": ["env-main"], "size": "~3 GB", "minutes": 40,
    },
    {
        "id": "ncore", "group": "Huấn luyện", "name": "NVIDIA NCore",
        "desc": "Đọc, ghi dữ liệu NCore: nguồn PhysicalAI và CARLA, xuất cho NuRec.",
        "links": [["nvidia-ncore", "https://pypi.org/project/nvidia-ncore/"]],
        "check": ("import", ["ncore"]), "install": ("pip", ["nvidia-ncore==19.9.0"]),
        "requires": ["env-main"], "size": "~50 MB", "minutes": 2,
    },
    {
        "id": "smpl-tools", "group": "Huấn luyện", "name": "4D-Humans, PHALP (dựng người)",
        "desc": "Cho stage prep.smpl (OmniRe tự tính tư thế người); bước 8 của install_main.sh. "
        "File mô hình SMPL_NEUTRAL.pkl tải riêng ở smpl.is.tue.mpg.de (cần tài khoản).",
        "links": [
            ["4D-Humans", "https://github.com/shubham-goel/4D-Humans"],
            ["PHALP", "https://github.com/brjathu/PHALP"],
        ],
        "check": ("import", ["hmr2", "phalp"]), "install": ("env", "main", 8),
        "requires": ["env-main"], "size": "~2 GB", "minutes": 15,
    },
    {
        "id": "segformer", "group": "Dữ liệu", "name": "SegFormer-B5 (mask bầu trời, vật động)",
        "desc": "Model của stage prep.masks_segformer; tải sẵn để bước tạo mask không phải chờ.",
        "links": [["Hugging Face", f"https://huggingface.co/{SEGFORMER}"]],
        "check": ("hf", SEGFORMER), "install": ("hf", SEGFORMER),
        "requires": ["env-main"], "size": "~340 MB", "minutes": 1,
    },
    {
        "id": "sdk-nuscenes", "group": "Dữ liệu", "name": "nuScenes devkit", "dataset": "nuscenes",
        "desc": "Đọc dữ liệu nuScenes khi ingest.",
        "links": [["GitHub", "https://github.com/nutonomy/nuscenes-devkit"]],
        "check": ("import", ["nuscenes"]), "install": ("pip", ["nuscenes-devkit"]),
        "size": "~30 MB", "minutes": 1,
    },
    {
        "id": "sdk-waymo", "group": "Dữ liệu", "name": "Waymo Open Dataset SDK",
        "dataset": "waymo",
        "desc": "Đọc file .tfrecord của Waymo (kèm TensorFlow 2.11; có thể đổi phiên bản numpy "
        "của môi trường).",
        "links": [["GitHub", "https://github.com/waymo-research/waymo-open-dataset"]],
        "check": ("import", ["waymo_open_dataset"]),
        "install": ("pip", ["waymo-open-dataset-tf-2-11-0==1.6.0"]),
        "size": "~1.5 GB", "minutes": 5,
    },
    {
        "id": "sdk-pandaset", "group": "Dữ liệu", "name": "PandaSet devkit", "dataset": "pandaset",
        "desc": "Đọc dữ liệu PandaSet.",
        "links": [["GitHub", "https://github.com/ziyc/pandaset-devkit"]],
        "check": ("import", ["pandaset"]),
        "install": ("pip", ["git+https://github.com/ziyc/pandaset-devkit.git#subdirectory=python"]),
        "size": "~20 MB", "minutes": 1,
    },
    {
        "id": "sdk-argoverse", "group": "Dữ liệu", "name": "Argoverse 2 API",
        "dataset": "argoverse",
        "desc": "Đọc dữ liệu Argoverse 2.",
        "links": [["GitHub", "https://github.com/ziyc/av2-api"]],
        "check": ("import", ["av2"]), "install": ("pip", ["git+https://github.com/ziyc/av2-api"]),
        "size": "~50 MB", "minutes": 3,
    },
    {
        "id": "sdk-kitti", "group": "Dữ liệu", "name": "pykitti", "dataset": "kitti",
        "desc": "Đọc dữ liệu KITTI (bản gốc; KITTI-360 không cần).",
        "links": [["GitHub", "https://github.com/utiasSTARS/pykitti"]],
        "check": ("import", ["pykitti"]), "install": ("pip", ["pykitti"]),
        "size": "~5 MB", "minutes": 1,
    },
    {
        "id": "sdk-nuplan", "group": "Dữ liệu", "name": "nuPlan devkit", "dataset": "nuplan",
        "desc": "Đọc dữ liệu nuPlan (nặng, kéo theo nhiều gói).",
        "links": [["GitHub", "https://github.com/motional/nuplan-devkit"]],
        "check": ("import", ["nuplan"]),
        "install": ("pip", ["git+https://github.com/motional/nuplan-devkit"]),
        "size": "~500 MB", "minutes": 6,
    },
    {
        "id": "sdk-physicalai", "group": "Dữ liệu", "name": "PhysicalAI (Hugging Face hub)",
        "dataset": "physicalai",
        "desc": "Tải và đọc bộ NVIDIA PhysicalAI-Autonomous-Vehicles (cùng NCore).",
        "links": [
            ["Dataset", "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles"]
        ],
        "check": ("import", ["huggingface_hub", "zarr"]),
        "install": ("pip", ["huggingface_hub>=0.34,<1.0", "zarr==2.17.1", "numcodecs==0.12.1"]),
        "size": "~20 MB", "minutes": 1,
    },
    {
        "id": "uv", "group": "Công cụ", "name": "uv (quản lý Python)",
        "desc": "Cài Instant NuRec và Colab CLI.",
        "links": [["Docs", "https://docs.astral.sh/uv/"]],
        "check": ("which", "uv"), "install": ("pip", ["uv"]),
        "size": "~40 MB", "minutes": 1,
    },
    {
        "id": "colab-cli", "group": "Công cụ", "name": "Google Colab CLI",
        "desc": "Phiên Colab, Google Drive, chạy run trên GPU Colab.",
        "links": [["PyPI", "https://pypi.org/project/google-colab-cli/"]],
        "check": ("which", "colab"),
        "install": ("uv-tool", ["google-colab-cli", "--with", "keyrings.alt"]),
        "requires": ["uv"], "size": "~60 MB", "minutes": 1,
    },
    {
        "id": "anthropic", "group": "Công cụ", "name": "Anthropic SDK (Trợ lý)",
        "desc": "Cho Trợ lý khi chọn nhà cung cấp Anthropic (Claude).",
        "links": [["GitHub", "https://github.com/anthropics/anthropic-sdk-python"]],
        "check": ("import", ["anthropic"]), "install": ("pip", ["anthropic"]),
        "size": "~10 MB", "minutes": 1,
    },
    {
        "id": "node", "group": "Công cụ", "name": "Node.js ≥ 18",
        "desc": "Chỉ cần khi sửa giao diện Studio (npm run build); bản build đã kèm repo. Cài cần "
        "quyền quản trị nên chỉ có hướng dẫn.",
        "links": [["Tải về", "https://nodejs.org/en/download"]],
        "check": ("which", "node"), "install": None,
        "size": "~100 MB", "minutes": 3,
    },
]  # fmt: skip
BY_ID = {item["id"]: item for item in CATALOG}

Plan = Union[tuple[str, str, int], tuple[str, list[str]]]


def _which(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for folder in (Path(sys.executable).parent, Path.home() / ".local" / "bin"):
        if (folder / name).is_file():
            return str(folder / name)
    return None


def _hf_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    found = try_to_load_from_cache(repo, "config.json")
    return isinstance(found, str)


def item_status(item: dict) -> dict:
    """Trạng thái một mục: ``{"status": "ok" | "partial" | "missing", "found"}``."""
    kind, value = item["check"]
    if kind == "env":
        from drive2sim.engine.environment import environment_check

        env = environment_check(value)
        status = {"ok": "ok", "warn": "partial"}.get(env["level"], "missing")
        missing = [c["id"] for c in env["checks"] if c["level"] != "ok"]
        return {"status": status, "found": ", ".join(missing) and "thiếu: " + ", ".join(missing)}
    if kind == "import":
        importlib.invalidate_caches()
        have = [name for name in value if importlib.util.find_spec(name) is not None]
        status = "ok" if len(have) == len(value) else "partial" if have else "missing"
        return {"status": status, "found": ", ".join(have) or None}
    if kind == "which":
        path = _which(value)
        return {"status": "ok" if path else "missing", "found": path}
    if kind == "hf":
        cached = _hf_cached(value)
        return {"status": "ok" if cached else "missing", "found": "đã tải" if cached else None}
    raise ValueError(f"kiểu kiểm tra lạ: {kind}")


def install_plan(item_id: str, python: str = sys.executable) -> Plan:
    """Cách cài một mục.

    Returns:
        ``("env", <môi trường>, <bước bắt đầu>)`` cho mục cài bằng script môi trường, hoặc
        ``("command", <lệnh>)``.

    Raises:
        ValueError: Mục lạ hoặc không cài được từ Studio.
    """
    item = BY_ID.get(item_id)
    if item is None:
        raise ValueError(f"không có mục {item_id}")
    plan = item["install"]
    if plan is None:
        raise ValueError(f"{item['name']} cần cài tay (quyền quản trị): {item['links'][0][1]}")
    kind = plan[0]
    if kind == "env":
        return ("env", plan[1], plan[2])
    if kind == "pip":
        return ("command", [python, "-m", "pip", "install", *plan[1]])
    if kind == "hf":
        code = (
            "from transformers import SegformerForSemanticSegmentation as M; "
            f"M.from_pretrained({plan[1]!r}); print('đã tải {plan[1]}')"
        )
        return ("command", [python, "-c", code])
    if kind == "uv-tool":
        uv = _which("uv")
        if not uv:
            raise ValueError("chưa có uv: cài mục uv trước")
        return ("command", [uv, "tool", "install", "--upgrade", *plan[1]])
    raise ValueError(f"kiểu cài lạ: {kind}")


def catalog() -> dict:
    """Danh mục kèm trạng thái cho Studio.

    Returns:
        ``{"groups": [...], "items": [{"id", "group", "name", "desc", "links", "size", "minutes",
        "requires", "dataset", "installable", "status", "found"}]}``.
    """
    items = []
    for item in CATALOG:
        public = {k: v for k, v in item.items() if k not in ("check", "install")}
        public.setdefault("requires", [])
        public["installable"] = item["install"] is not None
        public.update(item_status(item))
        items.append(public)
    return {"groups": list(GROUPS), "items": items}
