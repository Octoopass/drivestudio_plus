"""Guided Google Drive access through the `colab` CLI, so the web UI needs no hand-typed paths.

Steps the UI walks through:
1. sessions() / create()            - pick or start a Colab runtime;
2. drive_status() / mount() / ...   - mount Drive; when Google asks for consent the CLI prints a URL,
                                      which the UI shows as a button, then confirm_mount() continues;
3. browse() / find()                - list Drive folders and detect KITTI-360 roots and their sequences.
The download itself stays in PipelineManager.sync_drive (tools/colab_sync_drive.py).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .dataset_profiles import PROFILES, detect_folder

SESSION = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SESSION_LINE = re.compile(r"^\[([A-Za-z0-9_.-]{1,64})\]\s+(\S+)(.*)$")
URL = re.compile(r"https://\S+")
GPUS = ("T4", "L4", "A100", "H100", "G4")

# Runs on the Colab VM. Lists one Drive folder and says whether it is (or holds) a KITTI-360 root.
KITTI_FN = r'''
SEQ = re.compile(r"(2013_05_28_drive_\d{4}_sync)(?:_image_0[0-3])?\.tar$")
def kitti(p):
    try:
        names = set(os.listdir(p))
    except OSError:
        return None
    if "data_poses.zip" not in names and "data_2d_raw" not in names:
        return None
    seqs = {}
    for sub in ("data_2d_raw", "data_3d_raw"):
        try:
            for f in os.listdir(os.path.join(p, sub)):
                m = SEQ.match(f)
                if m:
                    s = seqs.setdefault(m.group(1), {"cameras": 0, "lidar": False, "bytes": 0})
                    if sub == "data_3d_raw": s["lidar"] = True
                    else: s["cameras"] += 1
                    try: s["bytes"] += os.path.getsize(os.path.join(p, sub, f))
                    except OSError: pass
        except OSError:
            pass
    common = {"data_poses.zip": "data_poses.zip" in names,
              "calibration": "calibration.zip" in names or os.path.isdir(os.path.join(p, "calibration"))}
    common["metadata"] = os.path.isfile(os.path.join(p, "ImageSets-rosa-6class", "kitti360_rosa_infos_val.pkl"))
    return {"sequences": seqs, "common": common}
'''

BROWSE = r'''
import json, os, re
base = __BASE__; path = __PATH__
__KITTI_FN__
out = {"mounted": os.path.isdir(os.path.join(base, "MyDrive")), "path": path, "exists": os.path.isdir(path)}
if out["exists"]:
    dirs, files, sample = [], [], []
    for e in sorted(os.scandir(path), key=lambda e: e.name.lower()):
        if e.name.startswith("."):
            continue
        if e.is_dir():
            dirs.append(e.name)
        else:
            try: files.append([e.name, e.stat().st_size])
            except OSError: files.append([e.name, 0])
    for d in dirs[:10]:
        try: sample += [d + "/" + n for n in sorted(os.listdir(os.path.join(path, d)))[:30]]
        except OSError: pass
    out.update(dirs=dirs[:300], files=files[:300], file_count=len(files), sample=sample, kitti=kitti(path))
print("FFLAB_JSON:" + json.dumps(out))
'''

# Walks MyDrive a few levels down and returns each folder's first names; detection runs on our side.
FIND = r'''
import json, os
base = __BASE__; depth = __DEPTH__
root = os.path.join(base, "MyDrive")
seen = {}
SKIP = {"data_2d_raw", "data_3d_raw", "calibration", "ImageSets-rosa-6class", "node_modules", ".git",
        "samples", "sweeps", "sensor_blobs", "camera", "lidar"}
def walk(p, d):
    if len(seen) >= 400:
        return
    try:
        names = sorted(n for n in os.listdir(p) if not n.startswith("."))
    except OSError:
        return
    seen[p] = names[:80]
    if d >= depth:
        return
    for n in names[:60]:
        if n in SKIP:
            continue
        q = os.path.join(p, n)
        if os.path.isdir(q):
            walk(q, d + 1)
walk(root, 0)
print("FFLAB_JSON:" + json.dumps({"mounted": os.path.isdir(root), "dirs": seen}))
'''


class ColabHelper:
    drive_root = "/content/drive"

    def __init__(self, colab_bin):
        self._bin = colab_bin            # callable -> path or None (PATH may change after install)
        self._lock = threading.Lock()
        self._mounts: dict[str, dict] = {}

    # ---- plumbing
    @property
    def bin(self) -> str:
        found = self._bin() if callable(self._bin) else self._bin
        if not found:
            raise ValueError("Chưa cài Colab CLI trên máy ảo (google-colab-cli)")
        return found

    @staticmethod
    def _env():
        env = dict(os.environ)
        env.setdefault("PYTHON_KEYRING_BACKEND", "keyrings.alt.file.PlaintextKeyring")
        return env

    def _run(self, args, stdin=None, timeout=60) -> str:
        try:
            r = subprocess.run([self.bin, *args], input=stdin, capture_output=True, text=True,
                               timeout=timeout, env=self._env(), stdin=None if stdin is not None else subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"colab {args[0]} quá thời gian ({timeout}s)") from exc
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode:
            raise ValueError(f"colab {args[0]} lỗi: {out.strip()[-600:]}")
        return out

    def _session(self, name) -> str:
        name = str(name or "").strip()
        if not SESSION.fullmatch(name):
            raise ValueError("Tên phiên Colab chỉ gồm chữ, số, _ . - (tối đa 64 ký tự)")
        return name

    def _drive_path(self, path) -> str:
        path = str(path or "").strip().rstrip("/") or f"{self.drive_root}/MyDrive"
        if not (path == self.drive_root or path.startswith(self.drive_root + "/")) or ".." in path.split("/") or "\\" in path:
            raise ValueError(f"Đường dẫn phải nằm trong {self.drive_root}/")
        return path

    def _remote(self, session, template, **values) -> dict:
        code = template.replace("__KITTI_FN__", KITTI_FN)
        for key, value in values.items():
            code = code.replace(f"__{key.upper()}__", repr(value))
        out = self._run(["exec", "-s", session], stdin=code, timeout=120)
        for line in reversed(out.splitlines()):
            if line.startswith("FFLAB_JSON:"):
                return json.loads(line[len("FFLAB_JSON:"):])
        raise ValueError(f"Phiên Colab trả về kết quả lạ: {out.strip()[-400:]}")

    # ---- 1. sessions
    def sessions(self) -> dict:
        out = self._run(["sessions"], timeout=45)
        items = []
        for line in out.splitlines():
            m = SESSION_LINE.match(line.strip())
            if m:
                extra = dict(part.strip().split(": ", 1) for part in m.group(3).split("|") if ": " in part)
                items.append({"name": m.group(1), "endpoint": m.group(2), "hardware": extra.get("Hardware", "")})
        return {"sessions": items}

    def create(self, name, gpu=None) -> dict:
        name = self._session(name)
        args = ["new", "-s", name]
        if gpu:
            if gpu not in GPUS:
                raise ValueError("GPU không hợp lệ")
            args += ["--gpu", gpu]
        if any(s["name"] == name for s in self.sessions()["sessions"]):
            raise ValueError(f"Đã có phiên '{name}'")
        self._run(args, timeout=300)
        return self.sessions()

    def url(self, name) -> dict:
        out = self._run(["url", "-s", self._session(name)], timeout=45)
        match = URL.search(out)
        if not match:
            raise ValueError("Không lấy được link mở phiên Colab")
        return {"url": match.group(0)}

    # ---- 2. Drive mount
    def drive_status(self, name) -> dict:
        name = self._session(name)
        data = self._remote(name, BROWSE, base=self.drive_root, path=f"{self.drive_root}/MyDrive")
        return {"session": name, "mounted": bool(data.get("mounted")), "mount": self._mount_view(name)}

    def mount(self, name) -> dict:
        name = self._session(name)
        with self._lock:
            job = self._mounts.get(name)
            if job and job["process"].poll() is None:
                return self._mount_view(name)
            process = subprocess.Popen(
                [self.bin, "drivemount", "-s", name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=self._env(), start_new_session=True)
            job = {"process": process, "log": "", "consent_url": None, "confirmed": False, "started": time.time()}
            self._mounts[name] = job
        threading.Thread(target=self._read_mount, args=(job,), daemon=True).start()
        for _ in range(40):                       # give the CLI a moment to print a consent URL
            time.sleep(0.25)
            if job["consent_url"] or job["process"].poll() is not None:
                break
        return self._mount_view(name)

    def _read_mount(self, job):
        for line in job["process"].stdout:
            with self._lock:
                job["log"] = (job["log"] + line)[-6000:]
                if not job["consent_url"]:
                    match = URL.search(line)
                    if match and "Please visit" in job["log"]:
                        job["consent_url"] = match.group(0)

    def mount_status(self, name) -> dict:
        return self._mount_view(self._session(name)) or {"state": "idle", "consent_url": None, "log": ""}

    def confirm_mount(self, name) -> dict:
        name = self._session(name)
        with self._lock:
            job = self._mounts.get(name)
            if not job or job["process"].poll() is not None:
                raise ValueError("Không có lần kết nối Drive nào đang chờ")
            try:
                job["process"].stdin.write("\n")
                job["process"].stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            job["confirmed"] = True
        return self._mount_view(name)

    def _mount_view(self, name) -> dict | None:
        with self._lock:
            job = self._mounts.get(name)
            if not job:
                return None
            code = job["process"].poll()
            state = "running" if code is None else "succeeded" if code == 0 else "failed"
            if state == "running" and job["consent_url"] and not job["confirmed"]:
                state = "needs_consent"
            return {"state": state, "consent_url": job["consent_url"], "log": job["log"][-1500:]}

    def shutdown(self):
        with self._lock:
            for job in self._mounts.values():
                if job["process"].poll() is None:
                    job["process"].kill()

    # ---- 3. browse Drive
    def browse(self, name, path=None) -> dict:
        name = self._session(name)
        data = self._remote(name, BROWSE, base=self.drive_root, path=self._drive_path(path))
        if data.get("exists"):
            names = data.get("dirs", []) + [f[0] for f in data.get("files", [])]
            data["dataset"] = "kitti360" if data.get("kitti") else detect_folder(names, data.get("sample", []))
        return data

    def find(self, name, depth=3) -> dict:
        """Folders on Drive that look like a known dataset root (outermost match only)."""
        name = self._session(name)
        data = self._remote(name, FIND, base=self.drive_root, depth=max(1, min(int(depth), 4)))
        dirs = data.get("dirs", {})
        roots = []
        for path in sorted(dirs):
            if any(path.startswith(r["path"] + "/") for r in roots):
                continue
            children = [f"{c}/{n}" for c in dirs[path] for n in dirs.get(f"{path}/{c}", [])[:30]]
            dataset = detect_folder(dirs[path], children)
            if dataset and path != f"{self.drive_root}/MyDrive":
                roots.append({"path": path, "dataset": dataset, "label": PROFILES[dataset]["label"]})
        for r in roots:
            if r["dataset"] == "kitti360":
                try:
                    r["kitti"] = self._remote(name, BROWSE, base=self.drive_root, path=r["path"]).get("kitti")
                except ValueError:
                    r["kitti"] = None
        return {"mounted": data.get("mounted", False), "roots": roots[:30]}


def default_colab_bin():
    found = shutil.which("colab")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "colab"
    return str(local) if local.is_file() else None
