"""Catalog of the libraries and environments the Web UI needs, with a status check and an installer for each.

Every entry says what it is for, links to its source, and (when it can be installed without root) the job
steps that install it. Installs run as ordinary pipeline jobs (PipelineManager._start), so the UI shows
them with the same progress board; the heavy lifting is in tools/deps_setup.py.

    GET  /api/deps                  -> catalog with status and the running install job, if any
    POST /api/deps/{id}/install     -> start the install job
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
SEGFORMER = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

# id -> static description. "venv": which environment the check imports in ("ds", "eval", "nurec", None).
CATALOG = [
    # ---- training
    {"id": "ds-env", "group": "Huấn luyện", "name": "Môi trường DriveStudio+",
     "desc": "Python 3.10 venv cho train (omegaconf, kornia, lpips, open3d, viser, nerfview, smplx…).",
     "links": [["DriveStudio", "https://github.com/ziyc/drivestudio"], ["requirements", "requirements-drivestudio.txt"]],
     "venv": "ds", "imports": ["torch", "omegaconf", "kornia", "lpips", "open3d", "wandb", "viser", "nerfview", "cv2", "smplx"],
     "size": "~4 GB", "minutes": 8},
    {"id": "ds-cuda", "group": "Huấn luyện", "name": "Renderer CUDA: gsplat-geer, pytorch3d, nvdiffrast",
     "desc": "Bắt buộc để train (3DGS / 3DGUT / 3DGEER). Build từ mã nguồn với CUDA của máy; tự nâng PyTorch cho khớp phiên bản CUDA nếu cần.",
     "links": [["3dgeer (gsplat-geer)", "https://github.com/boschresearch/3dgeer/tree/gsplat-geer"],
               ["pytorch3d", "https://github.com/facebookresearch/pytorch3d"], ["nvdiffrast", "https://github.com/NVlabs/nvdiffrast"]],
     "venv": "ds", "imports": ["gsplat", "pytorch3d", "nvdiffrast"], "requires": ["ds-env"], "size": "~3 GB", "minutes": 40},
    {"id": "ds-ncore", "group": "Huấn luyện", "name": "NVIDIA NCore cho DriveStudio+",
     "desc": "Chuẩn bị dữ liệu KITTI-360 / PhysicalAI cho DriveStudio+ (bước 1) cần thư viện ncore trong môi trường train.",
     "links": [["nvidia-ncore", "https://pypi.org/project/nvidia-ncore/"]],
     "venv": "ds", "imports": ["ncore"], "requires": ["ds-env"], "size": "~50 MB", "minutes": 2},
    # ---- feed-forward + evaluation
    {"id": "nurec", "group": "Dựng nhanh & đánh giá", "name": "Instant NuRec",
     "desc": "Mô hình feed-forward của NVIDIA (mã nguồn, môi trường uv, checkpoint pa-multiview).",
     "links": [["GitHub", "https://github.com/NVIDIA/instant-nurec"], ["Hugging Face", "https://huggingface.co/nvidia/instant-nurec"]],
     "venv": "nurec", "imports": ["ncore", "torch"], "size": "~7 GB", "minutes": 15},
    {"id": "gsplat-eval", "group": "Dựng nhanh & đánh giá", "name": "gsplat 1.5.3 (chấm điểm PLY, 3DGUT)",
     "desc": "Venv riêng của fflab để render và tính PSNR cho kết quả Instant NuRec, và chạy backend 3DGUT.",
     "links": [["gsplat", "https://github.com/nerfstudio-project/gsplat"]],
     "venv": "eval", "imports": ["torch", "gsplat"], "size": "~5 GB", "minutes": 20},
    # ---- data
    {"id": "segformer", "group": "Dữ liệu", "name": "SegFormer-B5 (mask bầu trời)",
     "desc": "Model tạo sky mask khi chuẩn bị dữ liệu; tải sẵn để bước xử lý không phải chờ.",
     "links": [["Hugging Face", f"https://huggingface.co/{SEGFORMER}"]], "venv": None, "size": "~340 MB", "minutes": 1},
    {"id": "sdk-nuscenes", "group": "Dữ liệu", "name": "nuScenes devkit", "dataset": "nuscenes",
     "desc": "Đọc dữ liệu nuScenes khi preprocess.", "links": [["GitHub", "https://github.com/nutonomy/nuscenes-devkit"]],
     "venv": "ds", "imports": ["nuscenes"], "requires": ["ds-env"], "pip": ["nuscenes-devkit"], "size": "~30 MB", "minutes": 1},
    {"id": "sdk-waymo", "group": "Dữ liệu", "name": "Waymo Open Dataset SDK", "dataset": "waymo",
     "desc": "Đọc file .tfrecord của Waymo (kèm TensorFlow 2.11, có thể đổi phiên bản numpy trong venv train).",
     "links": [["GitHub", "https://github.com/waymo-research/waymo-open-dataset"]],
     "venv": "ds", "imports": ["waymo_open_dataset"], "requires": ["ds-env"], "pip": ["waymo-open-dataset-tf-2-11-0==1.6.0"],
     "size": "~1.5 GB", "minutes": 5},
    {"id": "sdk-pandaset", "group": "Dữ liệu", "name": "PandaSet devkit", "dataset": "pandaset",
     "desc": "Đọc dữ liệu PandaSet.", "links": [["GitHub", "https://github.com/ziyc/pandaset-devkit"]],
     "venv": "ds", "imports": ["pandaset"], "requires": ["ds-env"],
     "pip": ["git+https://github.com/ziyc/pandaset-devkit.git#subdirectory=python"], "size": "~20 MB", "minutes": 1},
    {"id": "sdk-argoverse", "group": "Dữ liệu", "name": "Argoverse 2 API", "dataset": "argoverse",
     "desc": "Đọc dữ liệu Argoverse 2.", "links": [["GitHub", "https://github.com/ziyc/av2-api"]],
     "venv": "ds", "imports": ["av2"], "requires": ["ds-env"], "pip": ["git+https://github.com/ziyc/av2-api"],
     "size": "~50 MB", "minutes": 3},
    {"id": "sdk-kitti", "group": "Dữ liệu", "name": "pykitti", "dataset": "kitti",
     "desc": "Đọc dữ liệu KITTI (bản gốc, không phải KITTI-360).", "links": [["GitHub", "https://github.com/utiasSTARS/pykitti"]],
     "venv": "ds", "imports": ["pykitti"], "requires": ["ds-env"], "pip": ["pykitti"], "size": "~5 MB", "minutes": 1},
    {"id": "sdk-nuplan", "group": "Dữ liệu", "name": "nuPlan devkit", "dataset": "nuplan",
     "desc": "Đọc dữ liệu nuPlan (nặng, kéo theo nhiều gói phụ thuộc).",
     "links": [["GitHub", "https://github.com/motional/nuplan-devkit"]],
     "venv": "ds", "imports": ["nuplan"], "requires": ["ds-env"], "pip": ["git+https://github.com/motional/nuplan-devkit"],
     "size": "~500 MB", "minutes": 6},
    {"id": "sdk-physicalai", "group": "Dữ liệu", "name": "PhysicalAI (Hugging Face hub)", "dataset": "physicalai",
     "desc": "Tải và đọc bộ NVIDIA PhysicalAI-Autonomous-Vehicles (cần thêm NCore ở mục Huấn luyện).",
     "links": [["Dataset", "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles"]],
     "venv": "ds", "imports": ["huggingface_hub", "zarr"], "requires": ["ds-env"],
     "pip": ["huggingface_hub>=0.34,<1.0", "zarr==2.17.1", "numcodecs==0.12.1"], "size": "~20 MB", "minutes": 1},
    # ---- tools
    {"id": "uv", "group": "Công cụ", "name": "uv (quản lý Python)",
     "desc": "Dùng để cài Instant NuRec và Colab CLI.", "links": [["Docs", "https://docs.astral.sh/uv/"]],
     "venv": None, "size": "~40 MB", "minutes": 1},
    {"id": "colab-cli", "group": "Công cụ", "name": "Google Colab CLI",
     "desc": "Kết nối phiên Colab, Google Drive và train trên GPU Colab.",
     "links": [["PyPI", "https://pypi.org/project/google-colab-cli/"]], "venv": None, "requires": ["uv"], "size": "~60 MB", "minutes": 1},
    {"id": "node", "group": "Công cụ", "name": "Node.js ≥ 18",
     "desc": "Chạy CLI fflab (train local, cài gsplat). Cần quyền quản trị để cài, nên chỉ có hướng dẫn.",
     "links": [["Tải về", "https://nodejs.org/en/download"]], "venv": None, "installable": False},
]
BY_ID = {d["id"]: d for d in CATALOG}


def _cuda_home() -> str | None:
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    found = sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True) or sorted(Path("/usr/local").glob("cuda*/bin/nvcc"))
    return str(found[0].parent.parent) if found else None


def nvcc_version() -> str | None:
    home = _cuda_home()
    if not home:
        return None
    try:
        out = subprocess.run([f"{home}/bin/nvcc", "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else None


class Deps:
    def __init__(self, workspace: Path, pipeline, ds_venv: Path):
        self.workspace = Path(workspace).resolve()
        self.pipeline = pipeline
        self.ds_venv = Path(ds_venv)

    # ---- environments
    def _python(self, venv: str | None) -> Path | None:
        if venv == "ds":
            p = self.ds_venv / "bin" / "python"
        elif venv == "eval":
            p = self.pipeline.discovery.gsplat_python
        elif venv == "nurec":
            p = self.pipeline.discovery.nurec_python
        else:
            return None
        return p if p and Path(p).is_file() else None

    def _probe(self, python: Path, modules: list[str]) -> dict:
        code = ("import importlib.util, json, sys\nres = {}\nfor m in %r:\n    res[m] = importlib.util.find_spec(m) is not None\n"
                "info = {'python': sys.version.split()[0]}\ntry:\n    import torch\n    info['torch'] = torch.__version__\n"
                "    info['torch_cuda'] = torch.version.cuda\nexcept Exception:\n    pass\nprint(json.dumps({'mods': res, 'info': info}))\n") % (modules,)
        try:
            r = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return {"mods": {m: False for m in modules}, "info": {}}

    def _uv(self) -> str | None:
        found = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        return found if Path(found).is_file() else None

    # ---- status
    def status(self) -> dict:
        probes: dict[str, dict] = {}
        wanted: dict[str, set] = {}
        for d in CATALOG:
            if d.get("venv"):
                wanted.setdefault(d["venv"], set()).update(d.get("imports", []))
        for venv, mods in wanted.items():
            py = self._python(venv)
            probes[venv] = self._probe(py, sorted(mods)) if py else None
        nvcc = nvcc_version()
        items = []
        for d in CATALOG:
            items.append({**{k: v for k, v in d.items() if k not in ("pip",)}, **self._check(d, probes, nvcc),
                          "installable": d.get("installable", True), "job": self.pipeline.active(f"deps-{d['id']}")})
        return {"items": items, "nvcc": nvcc, "ds_venv": str(self.ds_venv),
                "envs": {k: (v or {}).get("info") for k, v in probes.items()}}

    def _check(self, d: dict, probes: dict, nvcc: str | None) -> dict:
        i = d["id"]
        if i == "nurec":
            disc = self.pipeline.discovery
            ok = (disc.nurec_repo / "run_inference.py").is_file() and disc.nurec_python.is_file() and bool(disc.checkpoints())
            return {"ok": ok, "detail": ", ".join(disc.checkpoints()) if ok else "chưa cài đủ", "missing": []}
        if i == "segformer":
            hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
            ok = any((hub / f"models--{SEGFORMER.replace('/', '--')}" / "snapshots").glob("*/*.safetensors")) or \
                any((hub / f"models--{SEGFORMER.replace('/', '--')}" / "snapshots").glob("*/*.bin"))
            return {"ok": ok, "detail": "đã tải" if ok else "chưa tải", "missing": []}
        if i == "uv":
            uv = self._uv()
            return {"ok": bool(uv), "detail": uv or "không thấy", "missing": []}
        if i == "colab-cli":
            colab = self.pipeline.discovery.colab_bin
            return {"ok": bool(colab), "detail": colab or "chưa cài", "missing": []}
        if i == "node":
            node = shutil.which("node")
            ver = ""
            if node:
                try:
                    ver = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    pass
            ok = bool(re.match(r"v(1[89]|[2-9]\d)\.", ver))
            return {"ok": ok, "detail": ver or "chưa cài", "missing": []}
        probe = probes.get(d["venv"])
        if not probe:
            return {"ok": False, "detail": "chưa có môi trường", "missing": d.get("imports", [])}
        missing = [m for m in d.get("imports", []) if not probe["mods"].get(m)]
        detail = ""
        info = probe.get("info") or {}
        if i in ("ds-env", "gsplat-eval") and info.get("torch"):
            detail = f"Python {info.get('python')} · torch {info['torch']}"
        if i == "ds-cuda":
            tc = info.get("torch_cuda")
            if tc and nvcc and tc.split(".")[0] != nvcc.split(".")[0]:
                detail = f"torch dùng CUDA {tc}, máy có nvcc {nvcc}: sẽ nâng torch khi cài"
            elif not nvcc:
                detail = "không thấy CUDA toolkit (nvcc) trên máy"
        if missing:
            detail = (detail + " · " if detail else "") + "thiếu " + ", ".join(missing)
        return {"ok": not missing, "detail": detail or "đã cài", "missing": missing}

    # ---- install
    def install(self, dep_id: str) -> dict:
        d = BY_ID.get(dep_id)
        if not d:
            raise ValueError("Không có thư viện này")
        if not d.get("installable", True):
            raise ValueError(f"{d['name']} cần cài thủ công: {d['links'][0][1]}")
        if dep_id == "nurec":
            return self.pipeline.nurec_setup(["pa-multiview"])
        if d.get("requires"):
            current = {it["id"]: it for it in self.status()["items"]}
            for req in d["requires"]:
                if not current[req]["ok"]:
                    raise ValueError(f"Cài '{BY_ID[req]['name']}' trước")
        py, script = sys.executable, str(TOOLS / "deps_setup.py")
        ds = str(self.ds_venv)
        base = [py, "-u", script]
        if dep_id == "ds-env":
            steps = [("Tạo / cập nhật venv và cài thư viện Python", base + ["ds-env", "--venv", ds, "--repo", str(self.workspace),
                                                                          "--cuda-home", _cuda_home() or "", "--nvcc", nvcc_version() or "12.4"], 480)]
        elif dep_id == "ds-cuda":
            nvcc = nvcc_version()
            if not nvcc:
                raise ValueError("Không thấy CUDA toolkit (nvcc) trong /usr/local/cuda*; cần cài CUDA toolkit trước")
            common = ["--venv", ds, "--cuda-home", _cuda_home() or "", "--nvcc", nvcc]
            steps = [("Kiểm tra / nâng PyTorch cho khớp CUDA", base + ["torch-align", *common], 300),
                     ("Build gsplat-geer (3DGS / 3DGUT / 3DGEER)", base + ["gsplat-geer", *common], 900),
                     ("Build pytorch3d", base + ["pytorch3d", *common], 1500),
                     ("Build nvdiffrast", base + ["nvdiffrast", *common], 240)]
        elif dep_id == "ds-ncore":
            steps = [("Cài nvidia-ncore", base + ["pip", "--venv", ds, "--", "nvidia-ncore==19.8.0", "zarr==2.17.1", "numcodecs==0.12.1"], 120)]
        elif dep_id == "gsplat-eval":
            if not shutil.which("node"):
                raise ValueError("Cần Node.js để chạy fflab setup")
            nvcc = nvcc_version() or "12.4"
            idx = {"11": "cu118", "12": "cu124"}.get(nvcc.split(".")[0], "cu124")
            steps = [("fflab setup: venv + PyTorch + gsplat 1.5.3 (build CUDA)",
                      ["node", str(self.workspace / "fflab" / "bin" / "fflab.js"), "setup", "--backend", "3dgut-gsplat",
                       "--torch-index-url", f"https://download.pytorch.org/whl/{idx}"], 1200)]
        elif dep_id == "segformer":
            py_ds = self._python("ds") or Path(sys.executable)
            steps = [("Tải SegFormer-B5 từ Hugging Face", [str(py_ds), "-u", script, "hf-model", "--model", SEGFORMER], 90)]
        elif dep_id == "uv":
            steps = [("Cài uv vào ~/.local/bin", base + ["uv"], 30)]
        elif dep_id == "colab-cli":
            steps = [("uv tool install google-colab-cli", base + ["colab-cli"], 90)]
        elif d.get("pip"):
            steps = [(f"pip install {' '.join(d['pip'])}", base + ["pip", "--venv", ds, "--", *d["pip"]], d.get("minutes", 2) * 60)]
        else:
            raise ValueError("Chưa có cách cài tự động cho mục này")
        out = self.workspace / "outputs" / "setup" / f"{dep_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(Path.home() / ".local" / "bin"), env.get("PATH", "")])
        return self.pipeline._start(f"deps-{dep_id}", out, steps, env, {"dep": dep_id})
