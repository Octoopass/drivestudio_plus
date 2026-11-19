"""Installers behind the Web UI's "Thư viện & môi trường" panel (fisheye_fusion/deps.py). One subcommand per job step.

    ds-env       create/update the DriveStudio+ venv from requirements-drivestudio.txt (torch lines skipped)
    torch-align  make the venv's PyTorch use the same CUDA major version as the local nvcc (needed to build)
    gsplat-geer  build Bosch's gsplat-geer (3DGS / 3DGUT / 3DGEER) at the pinned commit
    pytorch3d    build pytorch3d (stable)          nvdiffrast   build nvdiffrast
    pip          pip install into a venv           hf-model     download a Hugging Face model into the cache
    uv           install uv into ~/.local/bin      colab-cli    uv tool install google-colab-cli

Each step is idempotent: what is already importable is skipped.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GEER_REPO = "https://github.com/boschresearch/3dgeer.git"
GEER_COMMIT = "f15cee2"
BUILD = Path(os.environ.get("FFLAB_BUILD_DIR", Path.home() / ".cache" / "fflab-build"))
TORCH = {"12": ("2.4.1", "0.19.1", "cu124"), "11": ("2.4.1", "0.19.1", "cu118")}
SKIP_REQ = re.compile(r"^\s*(torch|torchvision|torchaudio|xformers)\b|^\s*--(extra-)?index-url", re.I)


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd], **kw)


def vpy(venv: str) -> Path:
    return Path(venv) / "bin" / "python"


def can_import(python: Path, module: str) -> bool:
    return python.is_file() and subprocess.run([str(python), "-c", f"import {module}"], capture_output=True).returncode == 0


def torch_cuda(python: Path) -> str | None:
    r = subprocess.run([str(python), "-c", "import torch; print(torch.version.cuda or '')"], capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


def build_env(a) -> dict:
    env = dict(os.environ)
    if a.cuda_home:
        env["CUDA_HOME"] = a.cuda_home
        env["PATH"] = f"{a.cuda_home}/bin:{Path(a.venv) / 'bin'}:{env.get('PATH', '')}"
    try:
        cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
        if cc:
            env.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(sorted(set(cc))))
    except OSError:
        pass
    env.setdefault("MAX_JOBS", str(min(16, os.cpu_count() or 4)))
    return env


def pip(python: Path, *args, env=None):
    run([python, "-m", "pip", "install", "--progress-bar", "off", *args], env=env)


# ---------------------------------------------------------------- steps
def ds_env(a):
    py = vpy(a.venv)
    if not py.is_file():
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        if Path(uv).is_file():
            run([uv, "venv", "--seed", "--python", "3.10", a.venv])
        else:
            run(["python3", "-m", "venv", a.venv])
    pip(py, "--upgrade", "pip", "wheel", "setuptools<70", "ninja")
    if not can_import(py, "torch"):
        print("== Chưa có PyTorch: cài bản khớp CUDA của máy", flush=True)
        align(a, force=True)
    reqs = Path(a.repo) / "requirements-drivestudio.txt"
    lines = [l for l in reqs.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#") and not SKIP_REQ.search(l)]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(l.split("#")[0].strip() for l in lines) + "\n")
        req_file = f.name
    torch_ver = subprocess.run([str(py), "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"torch=={torch_ver}\n" if torch_ver else "")
        constraints = f.name
    if any(l.split("=")[0].strip().lower() == "chumpy" for l in lines) and not can_import(py, "chumpy"):
        # chumpy's setup.py imports pip, which pip's isolated build environment does not have
        pip(py, "--no-build-isolation", "chumpy")
    print(f"== Cài thư viện Python (giữ torch {torch_ver})", flush=True)
    pip(py, "-r", req_file, "-c", constraints)
    smplx = Path(a.repo) / "third_party" / "smplx"
    if (smplx / "setup.py").exists() and not can_import(py, "smplx"):
        pip(py, "--no-deps", str(smplx))
    print("== Môi trường DriveStudio+ sẵn sàng", flush=True)


def align(a, force=False):
    py = vpy(a.venv)
    major = (a.nvcc or "12").split(".")[0]
    cur = torch_cuda(py) if py.is_file() else None
    if cur and cur.split(".")[0] == major and not force:
        print(f"PyTorch đang dùng CUDA {cur}, khớp nvcc {a.nvcc}: giữ nguyên", flush=True)
        return
    torch, vision, tag = TORCH.get(major, TORCH["12"])
    print(f"== Cài torch {torch}+{tag} (đang là CUDA {cur or 'không có'}, nvcc {a.nvcc})", flush=True)
    pip(py, f"torch=={torch}", f"torchvision=={vision}", "numpy<2", "--index-url", f"https://download.pytorch.org/whl/{tag}",
        "--extra-index-url", "https://pypi.org/simple")
    run([py, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())"])


def gsplat_geer(a):
    py = vpy(a.venv)
    if can_import(py, "gsplat"):
        print("gsplat đã có", flush=True)
        return
    src = BUILD / "3dgeer"
    if not (src / "setup.py").exists():
        shutil.rmtree(src, ignore_errors=True)
        BUILD.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--branch", "gsplat-geer", GEER_REPO, src])
        run(["git", "-C", src, "checkout", GEER_COMMIT])
        run(["git", "-C", src, "submodule", "update", "--init", "--recursive"])
    glm = src / "gsplat" / "cuda" / "csrc" / "third_party" / "glm"
    if not (glm / "glm").exists():
        shutil.rmtree(glm, ignore_errors=True)
        run(["git", "clone", "--depth", "1", "--branch", "1.0.1", "https://github.com/g-truc/glm.git", glm])
    print("== Đang build gsplat-geer (biên dịch CUDA, 10-20 phút)", flush=True)
    pip(py, "--no-build-isolation", str(src), env=build_env(a))
    run([py, "-c", "import gsplat; print('gsplat', gsplat.__version__)"])


def build_git(a, module: str, url: str):
    py = vpy(a.venv)
    if can_import(py, module):
        print(f"{module} đã có", flush=True)
        return
    print(f"== Đang build {module} (biên dịch CUDA)", flush=True)
    pip(py, "--no-build-isolation", f"git+{url}", env=build_env(a))
    run([py, "-c", f"import {module}; print('{module} ok')"])


def hf_model(a):
    from huggingface_hub import HfApi, snapshot_download
    files = HfApi().list_repo_files(a.model)
    weights = [f for f in files if f.endswith(".safetensors")] or [f for f in files if f.endswith(".bin")]
    path = snapshot_download(a.model, allow_patterns=["*.json", *weights])
    print(f"Đã tải {a.model} vào {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--venv", default=str(Path.home() / "venvs" / "ds"))
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--cuda-home", default="")
    ap.add_argument("--nvcc", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("rest", nargs="*")
    # packages for "pip" come after "--"; argparse would reject them once the options are parsed
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    a.rest = [*a.rest, *extra]
    if a.cmd == "ds-env":
        ds_env(a)
    elif a.cmd == "torch-align":
        align(a)
    elif a.cmd == "gsplat-geer":
        gsplat_geer(a)
    elif a.cmd == "pytorch3d":
        build_git(a, "pytorch3d", "https://github.com/facebookresearch/pytorch3d.git@stable")
    elif a.cmd == "nvdiffrast":
        build_git(a, "nvdiffrast", "https://github.com/NVlabs/nvdiffrast")
    elif a.cmd == "pip":
        if not a.rest:
            raise SystemExit("pip: thiếu tên gói")
        pip(vpy(a.venv), *a.rest)
    elif a.cmd == "hf-model":
        hf_model(a)
    elif a.cmd == "uv":
        run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh"])
    elif a.cmd == "colab-cli":
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        run([uv, "tool", "install", "--upgrade", "google-colab-cli", "--with", "keyrings.alt"])
    else:
        raise SystemExit(f"lệnh không hợp lệ: {a.cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
