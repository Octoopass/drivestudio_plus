"""KITTI-360 -> NCore -> Instant NuRec -> evaluation: environment discovery and background jobs.

The heavy lifting runs in subprocesses with the interpreter that has the right packages:
- conversion / projection check / view export: the Instant NuRec venv (it ships `nvidia-ncore`);
- PLY rendering for evaluation: the fflab venv (gsplat 1.5.3, `fflab setup`).
All paths given by clients are relative to the workspace, like the reconstruction runner.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import subprocess
import threading
import uuid
from pathlib import Path

from .dataset_profiles import PROFILES, detect, detect_folder
from .reconstruction import _stop, _tail

TOOLS = Path(__file__).resolve().parent.parent / "tools"
KITTI_CAMERAS = {
    "image_00": "camera_front_left_00",
    "image_01": "camera_front_right_01",
    "image_02": "camera_fisheye_left_02",
    "image_03": "camera_fisheye_right_03",
}
SEQUENCE = re.compile(r"2013_05_28_drive_\d{4}_sync")
MODEL_FILES = {"pa-front": "pa_front", "pa-multiview": "pa_multiview", "pq-front": "pq_road"}
ARTIFACTS = {".png", ".json", ".ply", ".mp4"}
# rectify120: what Instant NuRec takes; ftheta / opencv keep the full fisheye (3DGUT, gsplat)
FISHEYE_MODELS = ("rectify120", "ftheta", "opencv")


# ------------------------------------------------------------------ discovery
def _gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines() if out.returncode == 0 else []:
        i, name, used, total, util = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "name": name, "memory_used_mib": int(used),
                     "memory_total_mib": int(total), "utilization": int(util)})
    return gpus


def _cuda_home() -> str | None:
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    candidates = sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True)
    return str(candidates[0].parent.parent) if candidates else None


class Discovery:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def rel(self, path: Path) -> str:
        # abspath, not resolve: a venv's python is a symlink to an interpreter outside the workspace
        return Path(os.path.abspath(path)).relative_to(self.workspace).as_posix()

    @property
    def nurec_repo(self) -> Path:
        return self.workspace / "backends" / "instant-nurec"

    @property
    def nurec_python(self) -> Path:
        return self.nurec_repo / ".venv" / "bin" / "python"

    @property
    def colab_bin(self) -> str | None:
        found = shutil.which("colab")
        if found:
            return found
        local = Path.home() / ".local" / "bin" / "colab"
        return str(local) if local.is_file() else None

    @property
    def gsplat_python(self) -> Path | None:
        venv = os.environ.get("FFLAB_VENV") or str(Path.home() / ".fflab" / "venv")
        python = Path(venv) / "bin" / "python"
        return python if python.is_file() else None

    def checkpoints(self) -> dict[str, str]:
        found = {}
        for model, stem in MODEL_FILES.items():
            hits = sorted((self.nurec_repo / "checkpoints" / "pth").glob(f"instant_nurec_{stem}_*.pth"))
            if hits:
                found[model] = self.rel(hits[-1])
        return found

    def datasets(self) -> list[dict]:
        """KITTI-360 archive roots (folders with data_poses.zip) up to two levels below the workspace."""
        roots = []
        for poses in sorted(self.workspace.glob("data_poses.zip")) + sorted(self.workspace.glob("*/data_poses.zip")) \
                + sorted(self.workspace.glob("*/*/data_poses.zip")):
            root = poses.parent
            if "backends" in root.relative_to(self.workspace).parts:
                continue
            sequences = {}
            for tar in sorted((root / "data_2d_raw").glob("*_image_0[0-3].tar")):
                seq, cam = tar.name[: -len("_image_00.tar")], tar.name[-len("image_00.tar"):-4]
                if SEQUENCE.fullmatch(seq):
                    sequences.setdefault(seq, {"cameras": [], "lidar": (root / "data_3d_raw" / f"{seq}.tar").is_file()})
                    sequences[seq]["cameras"].append(cam)
            has_meta = all((root / f"ImageSets-rosa-6class/kitti360_rosa_infos_{s}.pkl").is_file() for s in ("train", "val"))
            roots.append({"path": self.rel(root), "sequences": sequences, "calibration_pickles": has_meta})
        return roots

    def clips(self) -> list[dict]:
        clips = []
        for path in sorted((self.workspace / "outputs" / "ncore").glob("*/*.json")):
            try:
                meta = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if meta.get("version") != "v4":
                continue
            cams = [c for s in meta.get("component_stores", []) for c in s.get("components", {}).get("cameras", {})]
            gm = meta.get("generic_meta_data", {})
            clips.append({"path": self.rel(path), "sequence_id": meta.get("sequence_id"), "cameras": cams,
                          "frames": len(gm.get("frame_ids", [])), "source": gm.get("source_sequence", "")})
        return clips

    def reconstructions(self) -> list[dict]:
        out = []
        root = self.workspace / "outputs" / "reconstruction"
        for ply in sorted(root.glob("*/*/ply/*/*.ply"), key=lambda p: p.stat().st_mtime, reverse=True):
            out.append({"path": self.rel(ply), "job_id": ply.relative_to(root).parts[0],
                        "size_mib": round(ply.stat().st_size / 2**20, 1)})
        return out

    def evaluations(self) -> list[dict]:
        out = []
        for metrics in sorted((self.workspace / "outputs" / "eval").glob("*/metrics.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                scores = json.loads(metrics.read_text())
                info = json.loads((metrics.parent / "job.json").read_text()) if (metrics.parent / "job.json").is_file() else {}
            except (OSError, ValueError):
                continue
            out.append({"path": self.rel(metrics.parent), "metrics": scores, **info,
                        "images": sorted(p.name for p in metrics.parent.glob("*.png")),
                "videos": sorted(p.name for p in metrics.parent.glob("*.mp4"))})
        return out

    def output_file(self, relative: str) -> Path:
        """A .png/.json/.ply under outputs/, for showing or downloading earlier results (read-only)."""
        if (not isinstance(relative, str) or "\\" in relative or ":" in relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise ValueError("invalid output path")
        root = (self.workspace / "outputs").resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {".png", ".json", ".ply", ".mp4"} or not path.is_file():
            raise ValueError("unknown output file")
        return path

    def split_files(self) -> list[str]:
        """Scene lists shipped with DriveStudio (data/*.txt), usable as --split_file."""
        return sorted(self.rel(p) for p in (self.workspace / "data").glob("*.txt"))

    def local_datasets(self) -> dict:
        """Raw folders of every non-KITTI-360 profile and every preprocessed scene ready for training."""
        raw = []
        for key, prof in PROFILES.items():
            if key == "kitti360":
                continue
            path = self.workspace / prof["raw"]
            if not path.is_dir():
                continue
            names = sorted(p.name for p in path.iterdir() if not p.name.startswith("."))
            sample = []
            for sub in names[:20]:
                if (path / sub).is_dir():
                    sample += [f"{sub}/{n}" for n in sorted(os.listdir(path / sub))[:50]]
            raw.append({"dataset": key, "label": prof["label"], "path": prof["raw"], "entries": len(names),
                        "preview": names[:8], "detected": detect_folder(names[:200], sample) == key,
                        "partial": sum(1 for _ in path.rglob("*.part")) if len(names) < 5000 else 0})
        processed = []
        data = self.workspace / "data"
        if data.is_dir():
            for scene in sorted(list(data.glob("*/processed*/[0-9][0-9][0-9]")) + list(data.glob("*/processed*/*/[0-9][0-9][0-9]"))):
                if not (scene / "images").is_dir():
                    continue
                parts = scene.relative_to(data).parts
                processed.append({"dataset": parts[0], "label": PROFILES.get(parts[0], {}).get("label", parts[0]),
                                  "data_root": self.rel(scene.parent), "scene": scene.name,
                                  "sky_masks": (scene / "sky_masks").is_dir(),
                                  "objects": (scene / "instances").is_dir() or (scene / "objects").is_dir()})
        return {"raw": raw, "processed": processed}

    def report(self) -> dict:
        nurec_ok = (self.nurec_repo / "run_inference.py").is_file() and self.nurec_python.is_file()
        return {
            "workspace": str(self.workspace),
            "nurec": {"ok": nurec_ok, "repo": self.rel(self.nurec_repo) if self.nurec_repo.exists() else None,
                      "python": self.rel(self.nurec_python) if self.nurec_python.is_file() else None,
                      "checkpoints": self.checkpoints() if nurec_ok else {}},
            "gsplat": {"ok": self.gsplat_python is not None, "cuda_home": _cuda_home()},
            "colab": {"ok": self.colab_bin is not None, "bin": self.colab_bin},
            "gpus": _gpus(),
            "datasets": self.datasets(),
            "clips": self.clips(),
            "reconstructions": self.reconstructions(),
            "evaluations": self.evaluations(),
            "cameras": KITTI_CAMERAS,
        }


# ------------------------------------------------------------------ jobs
class PipelineManager:
    """Runs conversion and evaluation jobs; one job of each kind at a time."""

    def __init__(self, workspace: Path):
        self.discovery = Discovery(workspace)
        self.workspace = self.discovery.workspace
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._closed = False

    # ---- validation helpers
    def _path(self, value, *, directory=False, suffix=None) -> Path:
        if (not isinstance(value, str) or not value or "\\" in value or ":" in value
                or Path(value).is_absolute() or ".." in Path(value).parts):
            raise ValueError("paths must be relative workspace paths using forward slashes")
        path = (self.workspace / value).resolve(strict=True)
        if not path.is_relative_to(self.workspace):
            raise ValueError("path escapes workspace")
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"incorrect path type: {value}")
        if suffix and path.suffix.lower() != suffix:
            raise ValueError(f"path must end in {suffix}")
        return path

    def _env(self, gpu, extra_path=()):
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
        env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        if gpu is not None:
            if type(gpu) is not int or gpu < 0 or gpu not in {g["index"] for g in _gpus()}:
                raise ValueError("unknown GPU index")
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if extra_path:
            env["PATH"] = os.pathsep.join([*map(str, extra_path), env.get("PATH", "")])
        return env

    def _new_output(self, kind, name):
        root = self.workspace / "outputs" / kind
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        if path.exists():
            raise ValueError(f"output already exists: outputs/{kind}/{name}")
        path.mkdir()
        return path

    # ---- job types
    def convert(self, cfg: dict) -> dict:
        allowed = {"dataset", "sequence", "start", "frames", "cameras", "lidar", "name", "fisheye_model"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown convert fields")
        if not self.discovery.nurec_python.is_file():
            raise ValueError("Instant NuRec venv not found at backends/instant-nurec/.venv (it provides nvidia-ncore)")
        data = self._path(cfg.get("dataset", ""), directory=True)
        seq = cfg.get("sequence", "")
        if not SEQUENCE.fullmatch(str(seq)):
            raise ValueError("sequence must look like 2013_05_28_drive_0003_sync")
        start, frames = cfg.get("start", 0), cfg.get("frames", 40)
        if type(start) is not int or start < 0 or type(frames) is not int or not 2 <= frames <= 2000:
            raise ValueError("start must be >= 0 and frames in 2..2000")
        cams = cfg.get("cameras") or list(KITTI_CAMERAS)
        if not isinstance(cams, list) or not cams or set(cams) - set(KITTI_CAMERAS):
            raise ValueError(f"cameras must be a subset of {list(KITTI_CAMERAS)}")
        fisheye_model = cfg.get("fisheye_model", "rectify120")
        if fisheye_model not in FISHEYE_MODELS:
            raise ValueError(f"fisheye_model must be one of {FISHEYE_MODELS}")
        lidar = cfg.get("lidar", True)
        if type(lidar) is not bool:
            raise ValueError("lidar must be boolean")
        default = f"kitti360_{seq.split('_drive_')[1][:4]}_{start:06d}_{frames}f"
        name = cfg.get("name") or default
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
            raise ValueError("name: letters, digits, _ . - only")
        out = self._new_output("ncore", name)
        py = str(self.discovery.nurec_python)
        argv = [py, "-u", str(TOOLS / "kitti360_to_ncore.py"), "--data", str(data), "--out", str(out),
                "--sequence", seq, "--start", str(start), "--frames", str(frames), "--cameras", ",".join(cams),
                "--sequence-id", name, "--fisheye-model", fisheye_model]
        if not lidar:
            argv.append("--no-lidar")
        steps = [("convert", argv)]
        if lidar:
            steps.append(("projection check", [py, "-u", str(TOOLS / "ncore_projection_check.py"),
                                               str(out / f"{name}.json"), "--out", str(out / "check"), "--frame", "0"]))
        return self._start("convert", out, steps, self._env(None),
                           {"clip": self.discovery.rel(out / f"{name}.json"), "fisheye_model": fisheye_model})

    def prepare_drivestudio(self, cfg: dict, venv: Path) -> dict:
        """KITTI-360 -> NCore (--target drivestudio) -> DriveStudio+ preprocessing -> sky masks, in the DriveStudio+
        venv (it has nvidia-ncore). Writes data/kitti360/raw_<name> and data/kitti360/processed_<name>/000."""
        allowed = {"dataset", "sequence", "start", "frames", "name", "objects"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown prepare fields")
        py = Path(venv) / "bin" / "python"
        if not py.is_file():
            raise ValueError(f"DriveStudio+ environment not found at {venv}")
        data = self._path(cfg.get("dataset", ""), directory=True)
        seq = cfg.get("sequence", "")
        if not SEQUENCE.fullmatch(str(seq)):
            raise ValueError("sequence must look like 2013_05_28_drive_0003_sync")
        start, frames, objects = cfg.get("start", 0), cfg.get("frames", 100), cfg.get("objects", True)
        if type(start) is not int or start < 0 or type(frames) is not int or not 2 <= frames <= 2000:
            raise ValueError("start must be >= 0 and frames in 2..2000")
        if type(objects) is not bool:
            raise ValueError("objects must be boolean")
        name = cfg.get("name") or f"k360_{seq.split('_drive_')[1][:4]}_{start:06d}_{frames}f"
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
            raise ValueError("name: letters, digits, _ . - only")
        root = self.workspace / "data" / "kitti360"
        raw, processed = root / f"raw_{name}", root / f"processed_{name}"
        if raw.exists() or processed.exists():
            raise ValueError(f"data/kitti360/raw_{name} or processed_{name} already exists")
        raw.mkdir(parents=True)
        convert = [str(py), "-u", str(TOOLS / "kitti360_to_ncore.py"), "--data", str(data), "--out", str(raw / name),
                   "--sequence", seq, "--start", str(start), "--frames", str(frames), "--target", "drivestudio",
                   "--cameras", "image_00,image_02,image_03"]
        keys = ["images", "lidar", "calib", "pose"] + (["objects", "dynamic_masks"] if objects else [])
        if not objects:
            convert.append("--no-objects")
        repo = self.workspace
        steps = [
            ("convert", convert),
            ("preprocess", [str(py), "-u", str(repo / "datasets" / "preprocess.py"), "--data_root", str(raw),
                            "--target_dir", str(processed), "--dataset", "physicalai", "--frame_stride", "1",
                            "--scene_ids", "0", "--workers", "1", "--process_keys", *keys]),
            ("sky masks", [str(py), "-u", str(repo / "datasets" / "tools" / "extract_masks_hf.py"),
                           "--data_root", str(processed), "--scene_ids", "0"]),
        ]
        env = self._env(None)
        env["PYTHONPATH"] = str(repo)
        return self._start("prepare", raw, steps, env, {
            "data_root": self.discovery.rel(processed), "objects": objects,
            "dataset_config": "kitti360/ncore_3cams_objects" if objects else "kitti360/ncore_3cams"})

    def preprocess(self, cfg: dict, venv: Path) -> dict:
        """DriveStudio datasets/preprocess.py (+ optional SegFormer sky masks) for one raw dataset folder."""
        allowed = {"dataset", "split", "scene_ids", "split_file", "start_idx", "num_scenes", "workers", "sky_masks"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown preprocess fields")
        key = cfg.get("dataset")
        prof = PROFILES.get(key)
        if not prof or "preprocess" not in prof:
            raise ValueError("dataset này không tiền xử lý bằng DriveStudio preprocess.py")
        preset = prof["preprocess"]
        py = Path(venv) / "bin" / "python"
        if not py.is_file():
            raise ValueError(f"DriveStudio+ environment not found at {venv}")
        raw = self.workspace / prof["raw"]
        if not raw.is_dir() or not any(raw.iterdir()):
            raise ValueError(f"{prof['raw']} chưa có dữ liệu")
        target = self.workspace / preset["target"]
        split = str(cfg.get("split") or preset.get("split") or "")
        if split and not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", split):
            raise ValueError("invalid split")
        ids = [int(x) for x in re.split(r"[ ,]+", str(cfg.get("scene_ids") or "").strip()) if x]
        if len(ids) > 200:
            raise ValueError("too many scene ids")
        split_file = str(cfg.get("split_file") or "")
        if split_file and split_file not in self.discovery.split_files():
            raise ValueError("unknown split file")
        start, num = cfg.get("start_idx", 0), cfg.get("num_scenes", 0)
        workers = cfg.get("workers", 8)
        if type(start) is not int or type(num) is not int or type(workers) is not int or start < 0 or num < 0 \
                or not 1 <= workers <= 64:
            raise ValueError("invalid start_idx / num_scenes / workers")
        if ids:
            select = ["--scene_ids", *map(str, ids)]
        elif split_file:
            select = ["--split_file", str(self.workspace / split_file)]
        elif num:
            select = ["--start_idx", str(start), "--num_scenes", str(num)]
        elif preset.get("scene_ids"):
            select = ["--scene_ids", *preset["scene_ids"].split()]
        elif preset.get("split_file"):
            select = ["--split_file", str(self.workspace / preset["split_file"])]
        else:
            select = ["--start_idx", str(preset.get("start_idx", 0)), "--num_scenes", str(preset.get("num_scenes", 1))]
        keys = ["images", "lidar", "calib", "dynamic_masks", "objects"] + ([] if key == "nuscenes" else ["pose"])
        argv = [str(py), "-u", str(self.workspace / "datasets" / "preprocess.py"), "--data_root", str(raw),
                "--target_dir", str(target), "--dataset", key, "--workers", str(workers), *select,
                "--process_keys", *keys]
        if split:
            argv += ["--split", split]
        for opt in ("interpolate_N", "start_frame_idx", "max_frame_limit"):
            if opt in preset:
                argv += [f"--{opt}", str(preset[opt])]
        steps = [("preprocess", argv)]
        if cfg.get("sky_masks", True):
            steps.append(("sky masks", [str(py), "-u", str(self.workspace / "datasets" / "tools" / "extract_masks_hf.py"),
                                        "--data_root", str(target), *select]))
        env = self._env(None)
        env["PYTHONPATH"] = str(self.workspace)
        log_dir = self._new_output("preprocess", f"{key}_{uuid.uuid4().hex[:8]}")
        return self._start("preprocess", log_dir, steps, env, {"dataset": key, "data_root": preset["target"]})

    def evaluate(self, cfg: dict) -> dict:
        allowed = {"ply", "clip", "every", "gpu"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown evaluate fields")
        ply = self._path(cfg.get("ply", ""), suffix=".ply")
        clip = self._path(cfg.get("clip", ""), suffix=".json")
        every = cfg.get("every", 4)
        if type(every) is not int or not 1 <= every <= 100:
            raise ValueError("every must be in 1..100")
        gsplat = self.discovery.gsplat_python
        if gsplat is None:
            raise ValueError("gsplat venv not found; run `fflab setup` (or set FFLAB_VENV)")
        out = self._new_output("eval", uuid.uuid4().hex[:12])
        (out / "job.json").write_text(json.dumps({"ply": self.discovery.rel(ply), "clip": self.discovery.rel(clip)}))
        cuda = _cuda_home()
        env = self._env(cfg.get("gpu"), [gsplat.parent] + ([Path(cuda) / "bin"] if cuda else []))
        if cuda:
            env.setdefault("CUDA_HOME", cuda)
        steps = [
            ("export views", [str(self.discovery.nurec_python), "-u", str(TOOLS / "ncore_eval_export.py"),
                              str(clip), "--out", str(out / "views.npz"), "--every", str(every)]),
            ("render + PSNR", [str(gsplat), "-u", str(TOOLS / "ncore_eval_render.py"), str(ply),
                               str(out / "views.npz"), "--out", str(out)]),
        ]
        return self._start("evaluate", out, steps, env, {"ply": self.discovery.rel(ply), "clip": self.discovery.rel(clip)})

    def sync_drive(self, cfg: dict) -> dict:
        allowed = {"mode", "drive_path", "sequence", "session", "dest", "dataset", "items"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown sync_drive fields")
        colab = self.discovery.colab_bin
        if not colab:
            raise ValueError("colab CLI not found; install google-colab-cli")
        mode = cfg.get("mode", "dataset")
        if mode not in ("dataset", "runs", "folder"):
            raise ValueError("mode must be dataset, runs or folder")
        items = cfg.get("items") or []
        if mode == "folder":
            prof = PROFILES.get(cfg.get("dataset") or "")
            if not prof or cfg.get("dataset") == "kitti360":
                raise ValueError("folder mode needs a dataset other than kitti360")
            if not isinstance(items, list) or len(items) > 200 or any(
                    not isinstance(i, str) or not i or i.startswith("/") or ".." in i.split("/") or "\\" in i for i in items):
                raise ValueError("invalid items")
            cfg["dest"] = prof["raw"]
        drive_path = str(cfg.get("drive_path") or "").strip()
        if not drive_path.startswith("/content/drive/") or ".." in Path(drive_path).parts:
            raise ValueError("drive_path must start with /content/drive/")
        session = str(cfg.get("session") or "fisheye-lab").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", session):
            raise ValueError("invalid session name")
        seq = str(cfg.get("sequence") or "2013_05_28_drive_0003_sync")
        if mode == "dataset" and not SEQUENCE.fullmatch(seq):
            raise ValueError("sequence must look like 2013_05_28_drive_0003_sync")
        dest_rel = str(cfg.get("dest") or ("kitti360" if mode == "dataset" else "fflab-runs")).strip().strip("/")
        if not dest_rel or chr(92) in dest_rel or ":" in dest_rel or ".." in Path(dest_rel).parts:
            raise ValueError("invalid destination folder")
        dest = (self.workspace / dest_rel).resolve()
        if not dest.is_relative_to(self.workspace) or dest == self.workspace:
            raise ValueError("destination must be a subfolder inside workspace")
        dest.mkdir(parents=True, exist_ok=True)
        py = sys.executable
        argv = [py, "-u", str(TOOLS / "colab_sync_drive.py"), "--colab", colab, "--session", session,
                "--mode", mode, "--drive-path", drive_path, "--sequence", seq, "--dest", str(dest)]
        for item in items if mode == "folder" else []:
            argv += ["--item", item]
        env = self._env(None, [Path(colab).parent])
        env.setdefault("PYTHON_KEYRING_BACKEND", "keyrings.alt.file.PlaintextKeyring")
        log_dir = self._new_output("drive_sync", uuid.uuid4().hex[:12])
        return self._start("drive_sync", log_dir, [(f"sync {mode}", argv)], env,
                           {"mode": mode, "dest": self.discovery.rel(dest), "session": session,
                            "dataset": cfg.get("dataset") or ("kitti360" if mode == "dataset" else "")})

    # ---- runner
    def _start(self, kind, out, steps, env, info):
        with self._lock:
            if self._closed:
                raise ValueError("pipeline manager is shut down")
            if any(j["kind"] == kind and j["state"] == "running" for j in self._jobs.values()):
                shutil.rmtree(out, ignore_errors=True)
                raise ValueError(f"a {kind} job is already running")
            job_id = uuid.uuid4().hex
            job = {"job_id": job_id, "kind": kind, "state": "running", "step": None, "error": None,
                   "tail": bytearray(), "process": None, "output": out, "info": info, "cancel": False}
            self._jobs[job_id] = job
            job["thread"] = threading.Thread(target=self._run, args=(job, steps, env), daemon=True)
            job["thread"].start()
            return self.status(job_id)

    def _run(self, job, steps, env):
        try:
            for label, argv in steps:
                with self._lock:
                    if job["cancel"]:
                        break
                    job["step"] = label
                    job["tail"].extend(f"\n=== {label} ===\n".encode())
                    job["process"] = subprocess.Popen(
                        argv, cwd=str(self.workspace), env=env, shell=False, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                _tail(job["process"].stdout, job["tail"])
                code = job["process"].wait()
                if code:
                    raise RuntimeError(f"{label} exited {code}")
            with self._lock:
                job["state"] = "cancelled" if job["cancel"] else "succeeded"
        except Exception as error:  # noqa: BLE001 - reported to the client
            with self._lock:
                job["state"] = "cancelled" if job["cancel"] else "failed"
                job["error"] = str(error)

    def _job(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id) or job_id not in self._jobs:
            raise ValueError("unknown pipeline job")
        return self._jobs[job_id]

    def _artifacts(self, job):
        out = job["output"]
        if not out.is_dir():
            return []
        return sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                      if p.is_file() and not p.is_symlink() and p.suffix.lower() in ARTIFACTS and p.stat().st_size)

    def status(self, job_id) -> dict:
        with self._lock:
            job = self._job(job_id)
            result = {k: job[k] for k in ("job_id", "kind", "state", "step", "error", "info")}
            result["log"] = bytes(job["tail"]).decode("utf-8", errors="replace")
            result["output"] = self.discovery.rel(job["output"])
            result["artifacts"] = self._artifacts(job)
            metrics = job["output"] / "metrics.json"
            if job["kind"] == "evaluate" and metrics.is_file():
                result["metrics"] = json.loads(metrics.read_text())
            return result

    def cancel(self, job_id) -> dict:
        with self._lock:
            job = self._job(job_id)
            job["cancel"] = True
            process = job["process"]
        if process is not None:
            _stop(process)
        return self.status(job_id)

    def artifact(self, job_id, relative) -> Path:
        with self._lock:
            job = self._job(job_id)
            if relative not in self._artifacts(job):
                raise ValueError("unknown pipeline artifact")
            path = (job["output"] / relative).resolve()
            if not path.is_relative_to(job["output"].resolve()):
                raise ValueError("artifact escapes output")
            return path

    def shutdown(self):
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
            for job in jobs:
                job["cancel"] = True
        for job in jobs:
            if job["process"] is not None:
                _stop(job["process"])
            if "thread" in job:
                job["thread"].join(timeout=5)
