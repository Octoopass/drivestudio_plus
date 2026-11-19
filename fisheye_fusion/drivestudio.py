"""DriveStudio+ and fflab method runs for the Web UI: metrics, run comparison, video comparison, 3D viewer."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from .settings import display_path

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
RENDER_MODES = ("", "default", "ut", "geer")
STEP = re.compile(rb"\[\s*(\d+)/(\d+)\]\s+eta:\s*([^ ]+(?: days?, [^ ]+)?)\s+elapsed:\s*(\S+)")
METRIC = re.compile(
    rb"^\s*(Full Image|Non-Sky|Dynamic-Only|Human-Only|Vehicle-Only|Depth|Render|Peak VRAM)\s+"
    rb"(PSNR|SSIM|LPIPS|RMSE|AbsRel|Delta1|FPS|GiB):\s*(-?[\d.]+)"
)
EVAL_HEAD = re.compile(rb"Eval over (\d+) images")
FILES = {".png", ".jpg", ".mp4", ".json", ".yaml", ".log", ".ply"}
STALE_SECONDS = 600

JSON_METRIC_MAP = {
    "image_metrics/test/psnr": "Full Image PSNR",
    "image_metrics/test/ssim": "Full Image SSIM",
    "image_metrics/test/lpips": "Full Image LPIPS",
    "image_metrics/test/occupied_psnr": "Non-Sky PSNR",
    "image_metrics/test/occupied_ssim": "Non-Sky SSIM",
    "image_metrics/test/masked_psnr": "Dynamic-Only PSNR",
    "image_metrics/test/masked_ssim": "Dynamic-Only SSIM",
    "image_metrics/test/human_psnr": "Human-Only PSNR",
    "image_metrics/test/human_ssim": "Human-Only SSIM",
    "image_metrics/test/vehicle_psnr": "Vehicle-Only PSNR",
    "image_metrics/test/vehicle_ssim": "Vehicle-Only SSIM",
    "image_metrics/test/depth_rmse": "Depth RMSE",
    "image_metrics/test/depth_absrel": "Depth AbsRel",
    "image_metrics/test/depth_delta1": "Depth Delta1",
    "image_metrics/test/render_fps": "Render FPS",
    "image_metrics/test/peak_vram_gib": "Peak VRAM GiB",
}


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
        for line in text[last.end():].split(b"\n")[:36]:
            m = METRIC.search(line.split(b"]")[-1])
            if m:
                value = float(m.group(3))
                if value >= 0:
                    metrics[f"{m.group(1).decode()} {m.group(2).decode()}"] = value
        out["metrics"], out["eval_images"] = metrics or None, int(last.group(1))
    if b"Traceback (most recent call last)" in text:
        tail = text[text.rindex(b"Traceback (most recent call last)"):].strip().split(b"\n")
        out["error"] = tail[-1].decode(errors="replace")[:400]
    return out


def _fmt_seconds(sec: float | int | None) -> str | None:
    if sec is None:
        return None
    s = int(round(float(sec)))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class DriveStudioRuns:
    def __init__(self, workspace: Path, venv: Path | None = None):
        self.workspace = Path(workspace).resolve()
        self.runs_dir = self.workspace / "fflab-runs"
        default = Path(os.environ.get("FFLAB_DS_VENV", Path.home() / "venvs" / "ds"))
        self.venv = Path(venv) if venv else default
        self.procs: dict[str, subprocess.Popen] = {}

    def options(self) -> dict:
        configs = sorted(p.relative_to(self.workspace).as_posix() for p in (self.workspace / "configs").glob("*.yaml")
                         if p.stem.startswith(("streetgs", "omnire", "pvg", "deformablegs")))
        datasets_root = self.workspace / "configs" / "datasets"
        datasets = sorted(p.relative_to(datasets_root).with_suffix("").as_posix() for p in datasets_root.glob("*/*.yaml"))
        data = self.workspace / "data"
        data_roots = sorted({p.parent.relative_to(self.workspace).as_posix()
                             for p in [*data.glob("*/processed*/000"), *data.glob("*/processed*/*/000")]
                            if (p / "metadata.json").exists() or (p / "images").is_dir()}) if data.exists() else []
        return {
            "methods": configs,
            "datasets": datasets,
            "data_roots": data_roots,
            "render_modes": [m for m in RENDER_MODES if m],
            "venv": display_path(self.venv, self.workspace),
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

    def _standalone_3dgut(self, name: str, root: Path) -> dict | None:
        """Read a standalone 3dgut-gsplat run (`metrics.json`, `compare_*.png`, `fly.mp4` / `*.mp4`)."""
        mpath = root / "metrics.json"
        tlog = root / "train.log"
        vids = sorted(p.relative_to(root).as_posix() for p in root.glob("*.mp4"))
        imgs = sorted(p.relative_to(root).as_posix() for p in root.glob("*.png"))
        if not mpath.is_file() and not tlog.is_file() and not vids:
            return None
        metrics: dict[str, float] = {}
        step, total, elapsed, updated = None, None, None, None
        cfg_meta = {}
        if mpath.is_file():
            try:
                raw = json.loads(mpath.read_text(encoding="utf-8"))
                cfg_meta = raw.get("config") or {}
                per_cam = raw.get("psnr_per_camera") or {}
                vals = [float(v["masked"] if isinstance(v, dict) and "masked" in v else v)
                        for v in per_cam.values() if isinstance(v, (int, float, dict))]
                if vals:
                    metrics["Full Image PSNR"] = float(sum(vals) / len(vals))
                pin = [float(v["masked"] if isinstance(v, dict) and "masked" in v else v)
                       for k, v in per_cam.items() if k in ("image_00", "image_01") and isinstance(v, (int, float, dict))]
                fish = [float(v["masked"] if isinstance(v, dict) and "masked" in v else v)
                        for k, v in per_cam.items() if k in ("image_02", "image_03") and isinstance(v, (int, float, dict))]
                if pin:
                    metrics["Pinhole PSNR"] = float(sum(pin) / len(pin))
                if fish:
                    metrics["Fisheye PSNR"] = float(sum(fish) / len(fish))
                if raw.get("gaussians"):
                    metrics["Gaussians"] = float(raw["gaussians"])
                step = total = raw.get("steps")
                elapsed = _fmt_seconds(raw.get("train_seconds"))
                updated = mpath.stat().st_mtime
            except (OSError, ValueError, TypeError):
                pass
        if tlog.is_file():
            txt = _read_tail(tlog, 256_000).decode(errors="replace")
            vram = re.findall(r"peak_vram=([\d.]+)GiB", txt)
            if vram:
                metrics["Peak VRAM GiB"] = float(vram[-1])
            updated = max(updated or 0, tlog.stat().st_mtime)
        return {
            "name": name,
            "backend": "3dgut-gsplat",
            "state": "succeeded" if mpath.is_file() else "stopped",
            "step": step,
            "total": total,
            "eta": None,
            "elapsed": elapsed,
            "metrics": metrics or None,
            "eval_images": None,
            "error": None,
            "method": "3dgut-gsplat",
            "dataset": cfg_meta.get("sequence") or "kitti360",
            "render_mode": "ut",
            "fastgs": "selective-adam",
            "steps_cfg": str(total) if total else None,
            "updated": updated,
            "images": imgs,
            "videos": vids,
            "checkpoint": (root / "ckpt.pt").is_file(),
        }

    def run(self, name: str) -> dict | None:
        root = self._run_dir(name)
        if not root.is_dir():
            return None
        log = root / "fflab-train.log"
        out_dirs = sorted((root / "drivestudio").glob("*")) if (root / "drivestudio").exists() else []
        if not log.exists() and not out_dirs:
            return self._standalone_3dgut(name, root)
        info = parse_log(_read_tail(log)) if log.exists() else parse_log(b"")
        out = out_dirs[-1] if out_dirs else None
        if out and (out / "metrics").is_dir():
            jfiles = sorted((out / "metrics").glob("images_test_*.json"))
            if jfiles:
                try:
                    jdata = json.loads(jfiles[-1].read_text(encoding="utf-8"))
                    merged = dict(info["metrics"] or {})
                    for jk, label_k in JSON_METRIC_MAP.items():
                        if jk in jdata and float(jdata[jk]) >= 0:
                            merged[label_k] = float(jdata[jk])
                    info["metrics"] = merged or None
                except (OSError, ValueError, TypeError):
                    pass
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
        images = sorted(p.relative_to(root).as_posix() for p in (out / "images").glob("*.png")) if out and (out / "images").exists() else []
        videos = sorted(p.relative_to(root).as_posix() for p in out.rglob("*.mp4")) if out else []
        colab = None
        try:
            cj = json.loads((root / "colab.json").read_text(encoding="utf-8"))
            colab = {k: cj.get(k) for k in ("session", "gpu", "state", "stage")}
            if cj.get("state") == "running" and state == "stopped":
                state = "running"
            elif cj.get("state") in ("failed", "cancelled") and state == "running":
                state = "failed" if cj["state"] == "failed" else "stopped"
        except (OSError, ValueError):
            pass
        return {
            "name": name, "backend": "drivestudio", "state": state, **info, **cfg, "colab": colab,
            "updated": log.stat().st_mtime if log.exists() else (out.stat().st_mtime if out else None),
            "images": images, "videos": videos,
            "checkpoint": (out / "checkpoint_final.pth").exists() if out else False,
        }

    @staticmethod
    def _launch_cfg(log: Path) -> dict:
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
               for r in [self.run(p.name)] if r and (r["method"] or r["images"] or r["videos"])]
        return sorted(out, key=lambda r: r["updated"] or 0, reverse=True)

    def file(self, name: str, relative: str) -> Path:
        root = self._run_dir(name)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in FILES or not path.is_file():
            raise ValueError("file not available")
        return path

    # ---------------------------------------------------------------- method video catalog & comparison
    def videos(self) -> list[dict]:
        items: list[dict] = []
        for r in self.runs():
            method = (r.get("method") or "unknown").replace("configs/", "").replace(".yaml", "")
            psnr = (r.get("metrics") or {}).get("Full Image PSNR")
            for v in r.get("videos") or []:
                vid_id = f"run:{r['name']}:{v}"
                items.append({
                    "id": vid_id,
                    "run": r["name"],
                    "method": method,
                    "render_mode": r.get("render_mode") or "",
                    "psnr": psnr,
                    "label": f"{r['name']} ({method}{f' · {psnr:.2f} dB' if psnr is not None else ''}) — {Path(v).name}",
                    "url": f"/api/drivestudio/runs/{r['name']}/files/{v}",
                })
        eval_root = self.workspace / "outputs" / "eval"
        if eval_root.is_dir():
            for mp4 in sorted(eval_root.glob("*/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel = mp4.relative_to(self.workspace / "outputs").as_posix()
                job_file = mp4.parent / "job.json"
                ply_name = mp4.parent.name
                if job_file.is_file():
                    try:
                        ply_name = Path(json.loads(job_file.read_text()).get("ply", ply_name)).stem
                    except (OSError, ValueError):
                        pass
                items.append({
                    "id": f"out:{rel}",
                    "run": ply_name,
                    "method": "instant-nurec / ply",
                    "render_mode": "ut",
                    "psnr": None,
                    "label": f"Eval {ply_name} — {mp4.name}",
                    "url": f"/api/outputs/{rel}",
                })
        return items

    def _resolve_video(self, vid_id: str) -> tuple[Path, str]:
        if vid_id.startswith("run:"):
            parts = vid_id.split(":", 2)
            if len(parts) != 3:
                raise ValueError("invalid video id")
            _, name, rel = parts
            path = self.file(name, rel)
            if path.suffix.lower() != ".mp4":
                raise ValueError("video must be an mp4 file")
            r = self.run(name) or {}
            method = (r.get("method") or "").replace("configs/", "").replace(".yaml", "")
            psnr = (r.get("metrics") or {}).get("Full Image PSNR")
            tag = f"{name} [{method}]" + (f" {psnr:.2f}dB" if psnr is not None else "")
            return path, tag
        if vid_id.startswith("out:"):
            rel = vid_id[4:]
            if "\\" in rel or ":" in rel or ".." in Path(rel).parts:
                raise ValueError("invalid output video id")
            root = (self.workspace / "outputs").resolve()
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or path.suffix.lower() != ".mp4" or not path.is_file():
                raise ValueError("output video not found")
            return path, f"NuRec/Eval [{path.parent.name}]"
        raise ValueError("unknown video source")

    def compare_videos(self, video_a: str, video_b: str, layout: str = "vstack") -> dict:
        if video_a == video_b:
            raise ValueError("pick two different videos to compare")
        pa, la = self._resolve_video(video_a)
        pb, lb = self._resolve_video(video_b)
        key = hashlib.sha1(f"{pa}:{pa.stat().st_mtime_ns}|{pb}:{pb.stat().st_mtime_ns}|{layout}".encode()).hexdigest()[:12]
        out_rel = f"video_compare/compare_{key}_{layout}.mp4"
        out_path = self.workspace / "outputs" / out_rel
        if not out_path.is_file():
            from tools.video_compare import compose_videos
            compose_videos(pa, la, pb, lb, out_path, layout=layout)
        return {"url": f"/api/outputs/{out_rel}", "label_a": la, "label_b": lb, "layout": layout}

    # ---------------------------------------------------------------- launching
    def validate(self, cfg: dict, local: bool = True) -> dict:
        """Checks a train request; `local=False` skips the checks for this machine's environment (Colab runs)."""
        opts = self.options()
        self._run_dir(cfg["name"])
        if cfg["method"] not in opts["methods"]:
            raise ValueError("unknown method config")
        if cfg["dataset"] not in opts["datasets"]:
            raise ValueError("unknown dataset config")
        if cfg.get("render_mode", "") not in RENDER_MODES:
            raise ValueError("render_mode must be default, ut or geer")
        data_root = cfg.get("data_root") or ""
        if data_root and data_root not in opts["data_roots"]:
            raise ValueError("unknown data_root")
        if local and not opts["venv_ok"]:
            raise ValueError(f"DriveStudio+ environment not found at {self.venv}")
        if local and not opts["node_ok"]:
            raise ValueError("node is not installed")
        return opts

    def train(self, cfg: dict) -> dict:
        name = cfg["name"]
        root = self._run_dir(name)
        if root.exists():
            raise ValueError(f"run {name} already exists")
        self.validate(cfg)
        data_root = cfg.get("data_root") or ""
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

    # ---------------------------------------------------------------- interactive viewer (tools/viewer.py)
    VIEWER_PORT = int(os.environ.get("DRIVESTUDIO_VIEWER_PORT", "8080"))
    VIEWER_HOST = os.environ.get("DRIVESTUDIO_VIEWER_HOST", "127.0.0.1")

    def _checkpoint(self, name: str) -> Path:
        root = self._run_dir(name)
        cks = sorted(root.glob("drivestudio/*/checkpoint_final.pth")) or sorted(
            root.glob("drivestudio/*/checkpoint_*.pth"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
        if not cks:
            raise ValueError(f"run {name} has no checkpoint yet")
        return cks[-1]

    def viewer_status(self) -> dict:
        proc = getattr(self, "_viewer", None)
        alive = proc is not None and proc.poll() is None
        listening = False
        if alive:
            with socket.socket() as sock:
                sock.settimeout(0.3)
                listening = sock.connect_ex(("127.0.0.1", self.VIEWER_PORT)) == 0
        info = dict(getattr(self, "_viewer_info", {}) or {})
        state = "ready" if listening else "loading" if alive else ("failed" if proc is not None and proc.returncode not in (None, 0, -15) else "stopped")
        log = ""
        if info.get("log") and Path(info["log"]).exists():
            log = _read_tail(Path(info["log"]), 4000).decode(errors="replace")
        return {"state": state, "port": self.VIEWER_PORT, "host": self.VIEWER_HOST, "run": info.get("run"), "checkpoint": info.get("checkpoint"),
                "log": log if state in ("loading", "failed") else ""}

    def viewer_start(self, name: str, gpu: int | None = None) -> dict:
        ckpt = self._checkpoint(name)
        if not self.options()["venv_ok"]:
            raise ValueError(f"DriveStudio+ environment not found at {self.venv}")
        self.viewer_stop()
        env = dict(os.environ, PYTHONPATH=str(self.workspace), DRIVESTUDIO_VIEWER_HOST=self.VIEWER_HOST)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        log = self._run_dir(name) / "viewer.log"
        with log.open("wb") as f:
            self._viewer = subprocess.Popen(
                [str(self.venv / "bin" / "python"), "tools/viewer.py", "--ckpt", str(ckpt), "--port", str(self.VIEWER_PORT),
                 "--output_dir", str(self._run_dir(name) / "viewer_exports")],
                cwd=self.workspace, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                shell=False, start_new_session=True)
        self._viewer_info = {"run": name, "checkpoint": ckpt.relative_to(self.workspace).as_posix(), "log": str(log)}
        return self.viewer_status()

    def viewer_stop(self) -> dict:
        proc = getattr(self, "_viewer", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._viewer = None
        self._viewer_info = {}
        return {"state": "stopped", "port": self.VIEWER_PORT, "host": self.VIEWER_HOST, "run": None, "checkpoint": None, "log": ""}

    def shutdown(self):
        self.viewer_stop()
