"""DriveStudio+ training on a Colab GPU, driven from the Web UI through the `colab` CLI.

One background thread per run walks these stages and records them in fflab-runs/<run>/colab.json:

    session  - reuse the chosen Colab runtime or create one with the requested GPU;
    code     - upload the repository (tracked files, working-tree versions) as a tar, skipped if unchanged;
    data     - upload the processed scene (<data_root>/<scene>) under the same relative path;
    setup    - tools/colab_train_setup.py installs the dependencies once per runtime
               (gsplat-geer, pytorch3d and nvdiffrast wheels are cached on Drive when it is mounted);
    train    - tools/drivestudio_run.py runs detached; its log is copied into fflab-runs/<run>/fflab-train.log
               every few seconds, so the results table shows progress and metrics as for a local run;
    download - outputs come back into fflab-runs/<run>/ (intermediate checkpoints are skipped).

Long jobs never hold a `colab exec` open: they run detached with a log and an exit file that we poll.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path

REMOTE = "/content/fflab"
REPO = f"{REMOTE}/repo"
DRIVE = "/content/drive/MyDrive"
PART = 200 * 1024 * 1024
POLL = 10
STAGES = (("session", "Phiên GPU"), ("code", "Đẩy code"), ("data", "Đẩy dữ liệu"),
          ("setup", "Cài môi trường"), ("train", "Huấn luyện"), ("download", "Tải kết quả về"))
SKIP_DIRS = {".git", "fflab-runs", "outputs", "work_dirs", "__pycache__", "node_modules"}


def _scene_dir(workspace: Path, data_root: str, scene: int) -> Path:
    return workspace / data_root / f"{scene:03d}"


def dataset_data_root(workspace: Path, dataset: str) -> str:
    """data.data_root of configs/datasets/<dataset>.yaml (no YAML import needed for this one key)."""
    path = workspace / "configs" / "datasets" / f"{dataset}.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"^\s*data_root:\s*([^\s#]+)", text, re.M)
    return m.group(1).strip("'\"") if m else ""


class ColabTrainer:
    def __init__(self, workspace: Path, colab, drivestudio, settings=None):
        self.workspace = Path(workspace).resolve()
        self.runs_dir = self.workspace / "fflab-runs"
        self.colab = colab
        self.ds = drivestudio
        if settings is None:
            from .settings import Settings
            settings = Settings(workspace, data={})
        self.settings = settings
        self.cache = f"{DRIVE}/{settings.data['colab']['drive_cache']}"
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: set[str] = set()
        self._stop = threading.Event()
        self._resume()

    # ------------------------------------------------------------------ state file
    def _state_path(self, name: str) -> Path:
        return self.ds._run_dir(name) / "colab.json"

    def _load(self, name: str) -> dict | None:
        try:
            return json.loads(self._state_path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save(self, job: dict):
        job["updated"] = time.time()
        path = self._state_path(job["name"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        tmp.replace(path)

    def _stage(self, job: dict, key: str, state: str, note: str = "", frac: float | None = None, eta: float | None = None):
        prev = job["stages"].get(key) or {}
        now = time.time()
        cur = {"state": state, "note": note, "est": prev.get("est"), "started": prev.get("started"), "ended": None}
        if state == "busy":
            job["stage"] = key
            cur["started"] = cur["started"] or now
            cur["frac"], cur["eta"] = frac, eta
        else:
            cur["ended"] = now
        job["stages"][key] = cur
        self._save(job)

    def _note(self, job: dict, line: str):
        with (self.ds._run_dir(job["name"]) / "colab-ui.log").open("a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + line.rstrip() + "\n")

    # ------------------------------------------------------------------ public API
    def start(self, cfg: dict) -> dict:
        name = cfg["name"]
        root = self.ds._run_dir(name)
        if root.exists():
            raise ValueError(f"Đã có lần chạy tên {name}")
        self.ds.validate(cfg, local=False)
        data_root = cfg.get("data_root") or dataset_data_root(self.workspace, cfg["dataset"])
        if not data_root:
            raise ValueError("Không xác định được thư mục dữ liệu; chọn 'Dữ liệu đã chuẩn bị'")
        scene = int(cfg.get("scene") or 0)
        if not _scene_dir(self.workspace, data_root, scene).is_dir():
            raise ValueError(f"Chưa có dữ liệu đã xử lý ở {data_root}/{scene:03d} trên máy ảo (làm bước 1 trước)")
        session = self.colab._session(cfg["session"])
        gpu = cfg.get("colab_gpu") or self.settings.data["colab"]["default_gpu"]
        if not self.settings.gpu(gpu):
            raise ValueError(f"GPU Colab không hợp lệ: {gpu}")
        sec_per_step = float(self.settings.gpu(gpu).get("sec_per_step") or 0.4)
        root.mkdir(parents=True)
        job = {
            "name": name, "session": session, "gpu": gpu, "data_root": data_root, "scene": scene,
            "cfg": {k: cfg.get(k) for k in ("method", "dataset", "render_mode", "fastgs", "steps")},
            "with_ckpt": bool(cfg.get("with_ckpt", True)),
            "state": "running", "stage": "session", "error": None, "started": time.time(),
            "stages": {k: {"state": "todo", "note": "", "est": est} for (k, _), est in zip(STAGES, (
                30 if any(s["name"] == session for s in self._sessions_cache()) else 150,   # session
                30,                                                                       # code
                max(20, self._scene_bytes(data_root, scene) / (8 * 2**20)),               # data (~8 MB/s)
                2700,                                                                     # setup (first build)
                int(cfg.get("steps") or 30000) * sec_per_step + 120,                      # train + eval (rough)
                90))},                                                                    # download
            "remote": {}, "offsets": {"setup": 0, "train": 0},
        }
        self._save(job)
        self._spawn(name)
        return self.view(job)

    def _sessions_cache(self) -> list:
        try:
            return self.colab.sessions()["sessions"]
        except ValueError:
            return []

    def _scene_bytes(self, data_root: str, scene: int) -> int:
        return sum(p.stat().st_size for p in _scene_dir(self.workspace, data_root, scene).rglob("*") if p.is_file())

    def jobs(self) -> dict:
        items = []
        if self.runs_dir.exists():
            for p in self.runs_dir.glob("*/colab.json"):
                job = self._load(p.parent.name)
                if job:
                    items.append(self.view(job))
        items.sort(key=lambda j: j["started"] or 0, reverse=True)
        return {"jobs": items, "stages": [{"key": k, "label": label} for k, label in STAGES]}

    def status(self, name: str) -> dict:
        job = self._load(name)
        if not job:
            raise ValueError("Không có job Colab này")
        return self.view(job, log=True)

    def view(self, job: dict, log: bool = False) -> dict:
        out = {k: job.get(k) for k in ("name", "session", "gpu", "data_root", "scene", "cfg", "state", "stage",
                                         "error", "started", "updated", "stages")}
        out["now"] = time.time()
        if log:
            path = self.ds._run_dir(job["name"]) / "colab-ui.log"
            try:
                out["log"] = path.read_text(encoding="utf-8", errors="replace")[-6000:]
            except OSError:
                out["log"] = ""
        return out

    def cancel(self, name: str) -> dict:
        job = self._load(name)
        if not job:
            raise ValueError("Không có job Colab này")
        with self._lock:
            self._cancel.add(name)
        pid = job.get("remote", {}).get("pid")
        if pid and job["state"] == "running":
            try:
                self.colab.exec_py(job["session"], f"import os, signal\ntry:\n    os.killpg({int(pid)}, signal.SIGTERM)\nexcept OSError:\n    pass\nprint('@@KILLED')\n", timeout=60, attempts=1)
            except ValueError:
                pass
        if not (t := self._threads.get(name)) or not t.is_alive():
            job["state"], job["error"] = "cancelled", "Đã huỷ"
            self._save(job)
        return self.view(job)

    def shutdown(self):
        self._stop.set()

    # ------------------------------------------------------------------ worker
    def _resume(self):
        """After a server restart, re-attach to jobs that were still running."""
        if not self.runs_dir.exists():
            return
        for p in self.runs_dir.glob("*/colab.json"):
            job = self._load(p.parent.name)
            if job and job["state"] == "running":
                self._spawn(job["name"])

    def _spawn(self, name: str):
        t = threading.Thread(target=self._work, args=(name,), daemon=True, name=f"colab-train-{name}")
        self._threads[name] = t
        t.start()

    def _cancelled(self, name: str) -> bool:
        with self._lock:
            return name in self._cancel or self._stop.is_set()

    def _work(self, name: str):
        job = self._load(name)
        try:
            for key, label in STAGES:
                if job["stages"][key]["state"] == "done":
                    continue
                if self._cancelled(name):
                    raise InterruptedError
                self._stage(job, key, "busy")
                self._note(job, f"== {label}")
                note = getattr(self, f"_do_{key}")(job) or ""
                self._stage(job, key, "done", note)
            job["state"] = "succeeded"
            self._note(job, "Xong. Kết quả nằm trong bảng ở bước 3.")
        except InterruptedError:
            if self._stop.is_set() and name not in self._cancel:
                return                      # server shutting down: leave it "running" to resume later
            job["state"], job["error"] = "cancelled", "Đã huỷ"
            self._stage(job, job["stage"], "error", "đã huỷ")
        except Exception as exc:            # noqa: BLE001 - every failure is shown in the UI
            job["state"], job["error"] = "failed", str(exc)[:800]
            self._note(job, f"LỖI: {exc}")
            self._stage(job, job["stage"], "error", str(exc)[:200])
        self._save(job)

    # ---- stages
    def _do_session(self, job: dict) -> str:
        sessions = self.colab.sessions()["sessions"]
        found = next((s for s in sessions if s["name"] == job["session"]), None)
        if not found:
            self._note(job, f"Tạo phiên {job['session']} với GPU {job['gpu']} (có thể mất vài phút)")
            sessions = self.colab.create(job["session"], job["gpu"])["sessions"]
            found = next((s for s in sessions if s["name"] == job["session"]), None)
            if not found:
                raise ValueError("Colab không tạo được phiên GPU (hết hạn mức GPU miễn phí?)")
        hw = found.get("hardware") or ""
        if hw.upper() == "CPU" or not hw:
            raise ValueError(f"Phiên {job['session']} đang chạy CPU, không có GPU. Chọn phiên GPU hoặc tạo phiên mới.")
        out = self.colab.exec_py(job["session"], "import subprocess\nprint(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip() or 'NO_GPU')\n", timeout=60)
        gpu = out.strip().splitlines()[-1] if out.strip() else "?"
        if "NO_GPU" in gpu:
            raise ValueError("Phiên Colab không thấy GPU (nvidia-smi trống)")
        self._note(job, f"GPU: {gpu}")
        return gpu

    def _remote_marker(self, job: dict, path: str) -> str:
        out = self.colab.exec_py(job["session"], f"import os\np={path!r}\nprint('@@M', open(p).read().strip() if os.path.exists(p) else '')\n", timeout=60)
        for line in out.splitlines():
            if line.startswith("@@M"):
                return line[3:].strip()
        return ""

    def _push_tar(self, job: dict, tar_path: Path, dest: str, marker: str, digest: str, what: str):
        size = tar_path.stat().st_size
        parts = max(1, -(-size // PART))
        tmp = f"{REMOTE}/upload/{job['name']}-{what}"
        self.colab.exec_py(job["session"], f"import os, shutil\nshutil.rmtree({tmp!r}, ignore_errors=True)\nos.makedirs({tmp!r}, exist_ok=True)\nos.makedirs({dest!r}, exist_ok=True)\nprint('@@OK')\n", timeout=60)
        with tar_path.open("rb") as src, tempfile.TemporaryDirectory(prefix="fflab-part-") as tmpdir:
            for i in range(parts):
                if self._cancelled(job["name"]):
                    raise InterruptedError
                part = Path(tmpdir) / f"part{i:04d}"
                part.write_bytes(src.read(PART))
                self._stage(job, what, "busy", f"{min(size, i * PART) / 2**20:.0f}/{size / 2**20:.0f} MB", frac=i / parts)
                for attempt in range(3):
                    try:
                        self.colab.upload(job["session"], str(part), f"{tmp}/part{i:04d}")
                        break
                    except ValueError:
                        if attempt == 2:
                            raise
                        time.sleep(5 * (attempt + 1))
                part.unlink()
        code = (f"import glob, os, shutil, subprocess\nparts = sorted(glob.glob({tmp!r} + '/part*'))\n"
                f"assert len(parts) == {parts}, f'thiếu phần: {{len(parts)}}/{parts}'\n"
                f"cat = subprocess.run('cat ' + ' '.join(parts) + ' | tar -xf - -C ' + {shlex.quote(dest)!r}, shell=True, capture_output=True, text=True)\n"
                f"assert cat.returncode == 0, cat.stderr[-400:]\nshutil.rmtree({tmp!r}, ignore_errors=True)\n"
                f"open({marker!r}, 'w').write({digest!r})\nprint('@@OK')\n")
        out = self.colab.exec_py(job["session"], code, timeout=900, attempts=1)
        if "@@OK" not in out:
            raise ValueError(f"Giải nén trên Colab lỗi: {out.strip()[-400:]}")

    def _code_files(self) -> list[Path]:
        try:
            r = subprocess.run(["git", "ls-files", "-z"], cwd=self.workspace, capture_output=True, timeout=60, check=True)
            names = [n for n in r.stdout.decode(errors="replace").split("\0") if n]
        except (OSError, subprocess.SubprocessError):
            names = [p.relative_to(self.workspace).as_posix() for p in self.workspace.rglob("*") if p.is_file()]
        out = []
        for n in names:
            parts = n.split("/")
            if SKIP_DIRS.intersection(parts):
                continue
            p = self.workspace / n
            if p.is_file():
                out.append(p)
        return out

    def _do_code(self, job: dict) -> str:
        files = self._code_files()
        h = hashlib.sha256()
        for p in files:
            st = p.stat()
            h.update(f"{p.relative_to(self.workspace).as_posix()}:{st.st_size}:{int(st.st_mtime)}\n".encode())
        digest = h.hexdigest()[:20]
        marker = f"{REPO}/.fflab_code"
        if self._remote_marker(job, marker) == digest:
            return "không đổi, bỏ qua"
        with tempfile.TemporaryDirectory(prefix="fflab-code-") as tmpdir:
            tar_path = Path(tmpdir) / "code.tar"
            with tarfile.open(tar_path, "w") as tar:
                for p in files:
                    tar.add(p, arcname=p.relative_to(self.workspace).as_posix(), recursive=False)
            self._note(job, f"Code: {len(files)} file, {tar_path.stat().st_size / 2**20:.1f} MB")
            self._push_tar(job, tar_path, REPO, marker, digest, "code")
        return f"{len(files)} file"

    def _do_data(self, job: dict) -> str:
        src = _scene_dir(self.workspace, job["data_root"], job["scene"])
        files = sorted(p for p in src.rglob("*") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        digest = hashlib.sha256(f"{len(files)}:{total}:{max((int(p.stat().st_mtime) for p in files), default=0)}".encode()).hexdigest()[:20]
        dest = f"{REPO}/{job['data_root']}"
        marker = f"{dest}/{job['scene']:03d}/.fflab_data"
        if self._remote_marker(job, marker) == digest:
            return f"đã có trên Colab ({total / 2**30:.2f} GB)"
        self._note(job, f"Dữ liệu: {src.relative_to(self.workspace)} · {len(files)} file · {total / 2**30:.2f} GB")
        with tempfile.TemporaryDirectory(prefix="fflab-data-", dir=self.workspace / "fflab-runs") as tmpdir:
            tar_path = Path(tmpdir) / "scene.tar"
            with tarfile.open(tar_path, "w") as tar:
                tar.add(src, arcname=src.name)
            self._push_tar(job, tar_path, dest, marker, digest, "data")
        return f"{total / 2**30:.2f} GB"

    def _launch(self, job: dict, key: str, argv: list[str], env: dict, log: str) -> dict:
        exit_file = f"{REMOTE}/jobs/{job['name']}-{key}.exit"
        cfg = json.dumps({"argv": argv, "env": env, "log": log, "exit": exit_file, "cwd": REPO})
        code = f"""import json, os, shlex, subprocess, sys, pathlib
cfg = json.loads({cfg!r})
pathlib.Path(cfg['log']).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(cfg['exit']).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(cfg['exit']).unlink(missing_ok=True)
argv = [sys.executable if a == '{{python}}' else a for a in cfg['argv']]
env = dict(os.environ); env.update(cfg['env']); env['PYTHONUNBUFFERED'] = '1'
cmd = f"cd {{shlex.quote(cfg['cwd'])}} && {{shlex.join(argv)}} >> {{shlex.quote(cfg['log'])}} 2>&1; echo $? > {{shlex.quote(cfg['exit'])}}"
p = subprocess.Popen(['bash', '-c', cmd], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
print('@@PID', p.pid)
"""
        out = self.colab.exec_py(job["session"], code, timeout=60)
        m = re.search(r"@@PID (\d+)", out)
        if not m:
            raise ValueError(f"Không khởi động được {key} trên Colab: {out.strip()[-300:]}")
        job["remote"].update({f"{key}_log": log, f"{key}_exit": exit_file, "pid": int(m.group(1))})
        self._save(job)
        return job["remote"]

    def _follow(self, job: dict, key: str, local: Path, note=None) -> int:
        """Copy the remote log into `local` until the exit file appears; returns the exit code."""
        log, exit_file = job["remote"][f"{key}_log"], job["remote"][f"{key}_exit"]
        failures = 0
        while True:
            if self._cancelled(job["name"]):
                raise InterruptedError
            off = job["offsets"].get(key, 0)
            code = f"""import os
log, exitf, off, chunk = {log!r}, {exit_file!r}, {off}, 262144
data = b''
if os.path.exists(log):
    with open(log, 'rb') as f:
        f.seek(off); data = f.read(chunk)
done = os.path.exists(exitf)
more = len(data) == chunk
if not (done and not more):
    data = data[:data.rfind(b'\\n') + 1]
import base64
print('@@OFF', off + len(data)); print('@@B64', base64.b64encode(data).decode())
if done and not more:
    print('@@EXIT', open(exitf).read().strip() or '1')
"""
            try:
                out = self.colab.exec_py(job["session"], code, timeout=90, attempts=1)
                failures = 0
            except ValueError as exc:
                failures += 1
                if failures >= 30:
                    raise ValueError(f"Mất liên lạc với phiên Colab ({exc}); job có thể vẫn chạy, bấm Huỷ rồi kiểm tra trên Colab")
                time.sleep(POLL * 2)
                continue
            data, new_off, code_exit = b"", None, None
            for line in out.splitlines():
                if line.startswith("@@OFF "):
                    new_off = int(line[6:])
                elif line.startswith("@@B64 "):
                    data = base64.b64decode(line[6:])
                elif line.startswith("@@EXIT "):
                    code_exit = int(re.sub(r"\D", "", line[7:]) or 1)
            if data:
                with local.open("ab") as f:
                    f.write(data)
                if note:
                    note(data)
            if new_off is not None:
                job["offsets"][key] = new_off
                self._save(job)
            if code_exit is not None:
                return code_exit
            if len(data) < 262144:
                time.sleep(POLL)

    def _do_setup(self, job: dict) -> str:
        if self._remote_marker(job, f"{REMOTE}/.setup_ok"):
            return "đã cài trên phiên này"
        cached = self._remote_marker(job, f"{self.cache}/.has_wheels") if "setup_log" not in job["remote"] else ""
        if cached:
            job["stages"]["setup"]["est"] = 420
            self._save(job)
        if "setup_log" not in job["remote"]:
            self._launch(job, "setup", ["{python}", "tools/colab_train_setup.py", "--cache", self.cache, "--marker", f"{REMOTE}/.setup_ok"],
                         {}, f"{REMOTE}/jobs/{job['name']}-setup.log")
        local = self.ds._run_dir(job["name"]) / "colab-setup.log"

        def note(data: bytes):
            lines = [l for l in data.decode(errors="replace").splitlines() if l.startswith(("==", "!!"))]
            for l in lines:
                self._note(job, l)
            if lines:
                self._stage(job, "setup", "busy", lines[-1].lstrip("=! ")[:120])
        code = self._follow(job, "setup", local, note)
        if code != 0:
            tail = local.read_text(errors="replace")[-600:] if local.exists() else ""
            raise ValueError(f"Cài môi trường trên Colab lỗi (exit {code}). Cuối log:\n{tail}")
        return "xong"

    def _do_train(self, job: dict) -> str:
        cfg, name = job["cfg"], job["name"]
        run_root = f"{REPO}/fflab-runs/{name}"
        if "train_log" not in job["remote"]:
            env = {"DRIVESTUDIO_CONFIG": cfg["method"], "DRIVESTUDIO_DATASET": cfg["dataset"],
                   "DRIVESTUDIO_OUT": f"./fflab-runs/{name}", "DRIVESTUDIO_RUN_NAME": name,
                   "DRIVESTUDIO_SCENE_IDX": str(job["scene"]), "DRIVESTUDIO_STEPS": str(int(cfg["steps"])),
                   "DRIVESTUDIO_OVERRIDES": f"data.data_root={job['data_root']}", "WANDB_MODE": "disabled"}
            if cfg.get("render_mode"):
                env["DRIVESTUDIO_RENDER_MODE"] = cfg["render_mode"]
            if cfg.get("fastgs") is not None:
                env["DRIVESTUDIO_FASTGS"] = "true" if cfg["fastgs"] else "false"
            self._launch(job, "train", ["{python}", "tools/drivestudio_run.py"], env, f"{run_root}/colab-stdout.log")
        local = self.ds._run_dir(name) / "fflab-train.log"
        step = re.compile(rb"\[\s*(\d+)/(\d+)\]\s+eta:\s*(?:(\d+) days?, )?(\d+):(\d+):(\d+)")

        def note(data: bytes):
            found = step.findall(data)
            if found:
                s, t, d, h, m, sec = found[-1]
                eta = int(d or 0) * 86400 + int(h) * 3600 + int(m) * 60 + int(sec) + 120
                self._stage(job, "train", "busy", f"bước {int(s)}/{int(t) - 1}", frac=int(s) / max(1, int(t) - 1), eta=eta)
        code = self._follow(job, "train", local, note)
        if code != 0:
            raise ValueError(f"Train trên Colab lỗi (exit {code}); xem log ở bảng kết quả")
        return "xong"

    def _do_download(self, job: dict) -> str:
        name = job["name"]
        base = f"{REPO}/fflab-runs/{name}"
        out = self.colab.exec_py(job["session"], f"""import json, os
base = {base!r}; files = []
for d, _, names in os.walk(base):
    for n in names:
        p = os.path.join(d, n); files.append([os.path.relpath(p, base), os.path.getsize(p)])
print('@@FILES ' + json.dumps(files))
""", timeout=120)
        line = next((l for l in out.splitlines() if l.startswith("@@FILES ")), None)
        files = json.loads(line[8:]) if line else []
        keep = [(rel, size) for rel, size in files
                if rel not in ("fflab-train.log", "colab-stdout.log")
                and not re.search(r"checkpoint_\d+\.pth$", rel)
                and (job.get("with_ckpt", True) or not rel.endswith(".pth"))]
        local_root = self.ds._run_dir(name)
        total = sum(s for _, s in keep)
        done = 0
        for rel, size in keep:
            if self._cancelled(name):
                raise InterruptedError
            dst = (local_root / rel).resolve()
            if not dst.is_relative_to(local_root):
                continue
            if dst.exists() and dst.stat().st_size == size:
                done += size
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.colab.download(job["session"], f"{base}/{rel}", str(dst))
            done += size
            self._stage(job, "download", "busy", f"{done / 2**20:.0f}/{total / 2**20:.0f} MB", frac=done / total if total else None)
        return f"{len(keep)} file · {total / 2**20:.0f} MB"
