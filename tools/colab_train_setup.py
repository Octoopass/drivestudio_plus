"""Install the DriveStudio+ training dependencies on a Colab GPU runtime (run from the repository root).

Colab already ships torch, torchvision, opencv, scikit-image, matplotlib and friends, so this only adds what
tools/train.py imports on top. The three CUDA packages (gsplat-geer, pytorch3d, nvdiffrast) take 20-40 min
to build on a T4; their wheels are cached under --cache (a Google Drive folder) when it is reachable, and
reused by later runtimes with the same Python / torch / GPU architecture.

Lines starting with "==" are progress headings the Web UI shows; "!!" lines are warnings.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PIP = ["omegaconf==2.3.0", "timm", "pytorch_msssim", "torchmetrics", "kornia", "lpips", "nerfview==0.1.2",
       "splines", "viser==1.0.30", "pyquaternion", "plyfile", "imageio-ffmpeg", "open3d", "wandb", "joblib"]
GEER_REPO = "https://github.com/boschresearch/3dgeer.git"
GEER_COMMIT = "f15cee2"
WORK = Path("/content/fflab/build")


def say(msg: str):
    print(msg, flush=True)


def run(cmd, **kw):
    say("+ " + " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], **kw)


def pip(*args):
    run([sys.executable, "-m", "pip", "install", "-q", *args])


def has(module: str) -> bool:
    importlib.invalidate_caches()
    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 - a broken install counts as missing
        return False


def tag() -> str:
    import torch
    cc = "".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "cpu"
    return f"py{sys.version_info[0]}{sys.version_info[1]}-torch{torch.__version__.split('+')[0]}-{torch.version.cuda}-sm{cc}"


def gsplat_source() -> str:
    src = WORK / "3dgeer"
    if not (src / "setup.py").exists():
        shutil.rmtree(src, ignore_errors=True)
        run(["git", "clone", "--recursive", "--branch", "gsplat-geer", GEER_REPO, src])
        run(["git", "-C", src, "checkout", GEER_COMMIT])
        run(["git", "-C", src, "submodule", "update", "--init", "--recursive"])
    glm = src / "gsplat" / "cuda" / "csrc" / "third_party" / "glm"
    if not (glm / "glm").exists():
        shutil.rmtree(glm, ignore_errors=True)
        run(["git", "clone", "--depth", "1", "--branch", "1.0.1", "https://github.com/g-truc/glm.git", glm])
    return str(src)


BUILDS = [  # module to import, wheel name prefix, source
    ("gsplat", "gsplat", gsplat_source),
    ("pytorch3d", "pytorch3d", lambda: "git+https://github.com/facebookresearch/pytorch3d.git"),
    ("nvdiffrast", "nvdiffrast", lambda: "git+https://github.com/NVlabs/nvdiffrast"),
]


def build(module: str, prefix: str, source, cache: Path | None):
    if has(module):
        say(f"== {module}: đã có")
        return
    t = tag()
    cached = sorted((cache / t).glob(f"{prefix}-*.whl")) if cache and (cache / t).is_dir() else []
    if cached:
        say(f"== {module}: cài từ bản build trên Drive")
        pip("--no-deps", cached[-1])
        if has(module):
            return
        say(f"!! bản build trên Drive của {module} không dùng được, build lại")
    say(f"== {module}: đang build (lần đầu mất 10-30 phút)")
    out = WORK / "wheels"
    out.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", out, source()])
    wheels = sorted(out.glob(f"{prefix}-*.whl"), key=os.path.getmtime)
    if not wheels:
        raise SystemExit(f"không tạo được wheel cho {module}")
    pip("--no-deps", wheels[-1])
    if cache:
        try:
            (cache / t).mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheels[-1], cache / t / wheels[-1].name)
            say(f"== {module}: đã lưu bản build lên Drive")
        except OSError as exc:
            say(f"!! không lưu được lên Drive: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    ap.add_argument("--marker", default="")
    args = ap.parse_args()
    cache = Path(args.cache) if args.cache and Path(args.cache).parent.is_dir() else None
    say("== Kiểm tra GPU")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Phiên Colab không có GPU (torch.cuda.is_available() = False)")
    say(f"torch {torch.__version__} · CUDA {torch.version.cuda} · {torch.cuda.get_device_name(0)}")
    if cache:
        say(f"Bộ nhớ đệm bản build: {cache}")
    else:
        say("!! Drive chưa kết nối: các gói CUDA sẽ phải build lại ở phiên sau")
    os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 2))
    cc = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cc[0]}.{cc[1]}")
    say("== Cài thư viện Python")
    pip(*PIP)
    pip("ninja")
    if Path("third_party/smplx/setup.py").exists() and not has("smplx"):
        pip("--no-deps", "./third_party/smplx")
    WORK.mkdir(parents=True, exist_ok=True)
    for module, prefix, source in BUILDS:
        build(module, prefix, source, cache)
    say("== Kiểm tra import")
    missing = [m for m in ("omegaconf", "gsplat", "pytorch3d", "nvdiffrast", "open3d", "wandb", "kornia", "cv2") if not has(m)]
    if missing:
        raise SystemExit(f"Thiếu sau khi cài: {', '.join(missing)}")
    if args.marker:
        Path(args.marker).write_text(tag())
    if cache:
        try:
            (cache / ".has_wheels").write_text(tag())
        except OSError:
            pass
    say("== Môi trường sẵn sàng")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
