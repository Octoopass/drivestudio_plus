"""One-time Instant NuRec setup for the Web UI, in three resumable parts (each is one job step):

    source   clone NVIDIA/instant-nurec at the pinned commit into backends/instant-nurec
    env      `uv sync --frozen` there (its own .venv, Python 3.11; it ships nvidia-ncore)
    weights  download the released checkpoints from Hugging Face (nvidia/instant-nurec, pinned revision)
             into backends/instant-nurec/checkpoints/pth/, resuming partial files

Lines "@@PROGRESS <done> <total>" let the Web UI show a progress bar and the remaining time.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/NVIDIA/instant-nurec"
COMMIT = "f2ecafcdf28f2dc8f2cc17447a772571cde558be"
HF_REVISION = "73536a971c00ff9d505c93815c90ef31414c21da"
HF_BASE = f"https://huggingface.co/nvidia/instant-nurec/resolve/{HF_REVISION}/"
WEIGHTS = {  # model id -> (file in the HF repo, size in bytes)
    "pa-multiview": ("pth/instant_nurec_pa_multiview_1.1.0.pth", 730202407),
    "pa-front": ("pth/instant_nurec_pa_front_1.1.0.pth", 730200087),
    "pq-front": ("pth/instant_nurec_pq_road_1.0.0.pth", 1141125047),
}


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd], **kw)


def source(repo: Path):
    if (repo / ".git").exists():
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == COMMIT:
            print(f"Đã có mã nguồn ở commit {COMMIT[:10]}", flush=True)
            return
        raise SystemExit(f"{repo} đang ở commit khác ({head[:10]}); xoá thư mục đó rồi chạy lại")
    if repo.exists() and any(p.name != "checkpoints" for p in repo.iterdir()):
        raise SystemExit(f"{repo} đã tồn tại nhưng không phải repo git")
    repo.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "-q", repo])
    run(["git", "-C", repo, "remote", "add", "origin", REPO_URL])
    run(["git", "-C", repo, "fetch", "--depth", "1", "origin", COMMIT])
    run(["git", "-C", repo, "checkout", "-q", "--detach", COMMIT])
    print("@@PROGRESS 1 1", flush=True)


def env(repo: Path):
    if not (repo / "pyproject.toml").is_file():
        raise SystemExit("Chưa có mã nguồn Instant NuRec (chạy bước source trước)")
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).is_file():
        raise SystemExit("Không tìm thấy uv (cài: curl -LsSf https://astral.sh/uv/install.sh | sh)")
    e = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(repo / ".venv"))
    run([uv, "sync", "--frozen", "--python", "3.11", "--no-dev", "--directory", repo], env=e)
    run([repo / ".venv" / "bin" / "python", "-c", "import ncore, torch; print('ncore ok · torch', torch.__version__)"])


def weights(repo: Path, models: list[str]):
    todo = [(m, *WEIGHTS[m]) for m in models]
    total = sum(size for _, _, size in todo)
    done = 0
    for model, name, size in todo:
        dst = repo / "checkpoints" / name
        if dst.is_file() and dst.stat().st_size == size:
            print(f"{model}: đã có", flush=True)
            done += size
            print(f"@@PROGRESS {done} {total}", flush=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_suffix(dst.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        print(f"{model}: tải {name} ({size / 2**20:.0f} MB){' tiếp từ ' + str(have // 2**20) + ' MB' if have else ''}", flush=True)
        req = urllib.request.Request(HF_BASE + name, headers={"User-Agent": "fflab-setup", **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=60) as r, part.open("ab" if have and r.status == 206 else "wb") as f:
            if r.status != 206:
                have = 0
            got, last = have, 0.0
            while chunk := r.read(4 * 2**20):
                f.write(chunk)
                got += len(chunk)
                if time.time() - last > 2:
                    last = time.time()
                    print(f"@@PROGRESS {done + got} {total}", flush=True)
        if part.stat().st_size != size:
            raise SystemExit(f"{model}: tải thiếu ({part.stat().st_size}/{size} byte), chạy lại để tải tiếp")
        part.replace(dst)
        done += size
        print(f"@@PROGRESS {done} {total}", flush=True)
    print(f"Checkpoint sẵn sàng: {', '.join(models)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("source", "env", "weights"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--models", default="pa-multiview,pa-front,pq-front")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    if a.part == "source":
        source(repo)
    elif a.part == "env":
        env(repo)
    else:
        models = [m for m in a.models.split(",") if m]
        bad = set(models) - set(WEIGHTS)
        if bad:
            raise SystemExit(f"model không hợp lệ: {', '.join(sorted(bad))}")
        weights(repo, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
