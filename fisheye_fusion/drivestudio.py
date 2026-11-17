"""DriveStudio+ runs for the Web UI: list what fflab wrote under fflab-runs/, start new runs through fflab.

A run is the directory fflab creates for `fflab train --backend drivestudio --out fflab-runs/<name>`:

    fflab-runs/<name>/fflab-train.log                     train.py output (progress, evaluation)
    fflab-runs/<name>/drivestudio/<name>/images/*.png     DriveStudio's periodic renders
    fflab-runs/<name>/drivestudio/<name>/videos/*.mp4     test-set videos after training

Everything shown comes from those files, so runs started from the CLI appear too. Runs are started with
fflab itself (node fflab/bin/fflab.js), not by calling train.py directly, so the UI and the CLI cannot drift.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
RENDER_MODES = ("", "default", "ut", "geer")
STEP = re.compile(rb"\[\s*(\d+)/(\d+)\]\s+eta:\s*([^ ]+(?: days?, [^ ]+)?)\s+elapsed:\s*(\S+)")
METRIC = re.compile(rb"^\s*(Full Image|Non-Sky|Dynamic-Only|Human-Only|Vehicle-Only)\s+(PSNR|SSIM|LPIPS):\s*(-?[\d.]+)")
EVAL_HEAD = re.compile(rb"Eval over (\d+) images")
FILES = {".png", ".jpg", ".mp4", ".json", ".yaml", ".log"}
STALE_SECONDS = 600  # a log untouched this long with no final evaluation is reported as stopped


def _read_tail(path: Path, limit: int = 4 * 1024 * 1024) -> bytes:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit))
        return f.read()


def parse_log(data: bytes) -> dict:
    """Progress, the last evaluation block and the failure, if any, from a train.py log."""
    text = data.replace(b"\r", b"\n")
    steps = STEP.findall(text)
    out: dict = {"step": None, "total": None, "eta": None, "elapsed": None, "metrics": None, "eval_images": None, "error": None}
    if steps:
        s, t, eta, elapsed = steps[-1]
        out.update(step=int(s), total=int(t) - 1, eta=eta.decode(), elapsed=elapsed.decode())
    heads = list(EVAL_HEAD.finditer(text))
    if heads:
        last = heads[-1]
        metrics = {}
        for line in text[last.end():].split(b"\n")[:24]:
            m = METRIC.search(line.split(b"]")[-1])
            if m:
                value = float(m.group(3))
                if value >= 0:  # DriveStudio prints -1 for masks that do not exist
                    metrics[f"{m.group(1).decode()} {m.group(2).decode()}"] = value
        out["metrics"], out["eval_images"] = metrics or None, int(last.group(1))
    if b"Traceback (most recent call last)" in text:
        tail = text[text.rindex(b"Traceback (most recent call last)"):].strip().split(b"\n")
        out["error"] = tail[-1].decode(errors="replace")[:400]
    return out


class DriveStudioRuns:
    def __init__(self, workspace: Path, venv: Path | None = None):
        self.workspace = Path(workspace).resolve()
        self.runs_dir = self.workspace / "fflab-runs"
        default = Path(os.environ.get("FFLAB_DS_VENV", Path.home() / "venvs" / "ds"))
        self.venv = Path(venv) if venv else default
        self.procs: dict[str, subprocess.Popen] = {}

    # ---------------------------------------------------------------- discovery
    def options(self) -> dict:
        configs = sorted(p.relative_to(self.workspace).as_posix() for p in (self.workspace / "configs").glob("*.yaml")
                         if p.stem.startswith(("streetgs", "omnire", "pvg", "deformablegs")))
        datasets_root = self.workspace / "configs" / "datasets"
        datasets = sorted(p.relative_to(datasets_root).with_suffix("").as_posix() for p in datasets_root.glob("*/*.yaml"))
        data = self.workspace / "data"
        data_roots = sorted(p.parent.relative_to(self.workspace).as_posix() for p in data.glob("*/processed*/000")
                            if (p / "metadata.json").exists() or (p / "images").is_dir()) if data.exists() else []
        return {
            "methods": configs,
            "datasets": datasets,
            "data_roots": data_roots,
            "render_modes": [m for m in RENDER_MODES if m],
            "venv": str(self.venv),
            "venv_ok": (self.venv / "bin" / "python").exists(),
            "node_ok": shutil.which("node") is not None,
        }

    def _run_dir(self, name: str) -> Path:
        if not RUN_NAME.match(name):
            raise ValueError("invalid run name")
        path = (self.runs_dir / name).resolve()
        if not path.is_relative_to(self.runs_dir.resolve()):
            raise ValueError("invalid run name")
        return path

    def run(self, name: str) -> dict | None:
        root = self._run_dir(name)
        log = root / "fflab-train.log"
        out_dirs = sorted((root / "drivestudio").glob("*")) if (root / "drivestudio").exists() else []
        if not log.exists() and not out_dirs:
            return None
        info = parse_log(_read_tail(log)) if log.exists() else parse_log(b"")
        cfg = self._launch_cfg(log)
        proc = self.procs.get(name)
        alive = proc is not None and proc.poll() is None
        age = time.time() - log.stat().st_mtime if log.exists() else None
        finished = info["step"] is not None and info["total"] is not None and info["step"] >= info["total"] and info["metrics"]
        if info["error"]:
            state = "failed"
        elif finished and not alive:
            state = "succeeded"
        elif alive or (age is not None and age < STALE_SECONDS):
            state = "running"
        else:
            state = "stopped"
        out = out_dirs[-1] if out_dirs else None
        images = sorted(p.relative_to(root).as_posix() for p in (out / "images").glob("*.png")) if out else []
        videos = sorted(p.relative_to(root).as_posix() for p in (out / "videos").glob("*.mp4")) if out else []
        return {
            "name": name, "state": state, **info, **cfg,
            "updated": log.stat().st_mtime if log.exists() else None,
            "images": images, "videos": videos,
            "checkpoint": (out / "checkpoint_final.pth").exists() if out else False,
        }

    @staticmethod
    def _launch_cfg(log: Path) -> dict:
        """Method, dataset, renderer and step count from the train.py command line fflab logs first."""
        cfg = {"method": None, "dataset": None, "render_mode": None, "fastgs": None, "steps_cfg": None}
        if not log.exists():
            return cfg
        with log.open("rb") as f:
            head = f.read(8192).decode(errors="replace")
        m = re.search(r"--config_file (\S+)", head)
        if m:
            cfg["method"] = m.group(1)
        for key, pat in (("dataset", r"dataset=(\S+)"), ("render_mode", r"trainer\.render\.render_mode=(\S+)"),
                         ("fastgs", r"trainer\.fastgs\.enabled=(\S+)"), ("steps_cfg", r"trainer\.optim\.num_iters=(\d+)")):
            m = re.search(pat, head)
            if m:
                cfg[key] = m.group(1)
        return cfg

    def runs(self) -> list[dict]:
        if not self.runs_dir.exists():
            return []
        out = [r for p in sorted(self.runs_dir.iterdir()) if p.is_dir() and RUN_NAME.match(p.name)
               for r in [self.run(p.name)] if r and (r["method"] or r["images"])]
        return sorted(out, key=lambda r: r["updated"] or 0, reverse=True)

    def file(self, name: str, relative: str) -> Path:
        root = self._run_dir(name)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in FILES or not path.is_file():
            raise ValueError("file not available")
        return path

    # ---------------------------------------------------------------- launching
    def train(self, cfg: dict) -> dict:
        opts = self.options()
        name = cfg["name"]
        root = self._run_dir(name)
        if root.exists():
            raise ValueError(f"run {name} already exists")
        if cfg["method"] not in opts["methods"]:
            raise ValueError("unknown method config")
        if cfg["dataset"] not in opts["datasets"]:
            raise ValueError("unknown dataset config")
        if cfg.get("render_mode", "") not in RENDER_MODES:
            raise ValueError("render_mode must be default, ut or geer")
        data_root = cfg.get("data_root") or ""
        if data_root and data_root not in opts["data_roots"]:
            raise ValueError("unknown data_root")
        if not opts["venv_ok"]:
            raise ValueError(f"DriveStudio+ environment not found at {self.venv}")
        if not opts["node_ok"]:
            raise ValueError("node is not installed")
        argv = ["node", str(self.workspace / "fflab" / "bin" / "fflab.js"), "train", "--backend", "drivestudio",
                "--venv", str(self.venv), "--method", cfg["method"], "--dataset", cfg["dataset"],
                "--run-name", name, "--out", f"./fflab-runs/{name}", "--steps", str(int(cfg["steps"]))]
        if cfg.get("render_mode"):
            argv += ["--render-mode", cfg["render_mode"]]
        if cfg.get("fastgs") is not None:
            argv += ["--fastgs", "true" if cfg["fastgs"] else "false"]
        if data_root:
            argv += ["--overrides", f"data.data_root={data_root}"]
        env = dict(os.environ)
        if cfg.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(int(cfg["gpu"]))
        root.mkdir(parents=True)
        with (root / "ui-launch.log").open("wb") as log:
            self.procs[name] = subprocess.Popen(argv, cwd=self.workspace, env=env, stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
        return {"name": name, "argv": argv[1:]}

    def shutdown(self):
        """Runs started from the UI keep going when the server stops: training takes hours."""
