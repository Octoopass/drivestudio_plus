"""Deployment settings: one place for every value that differs between installations.

Precedence (later wins): built-in DEFAULTS < JSON file < environment variables.
The JSON file is FFLAB_SETTINGS, else <workspace>/fflab.settings.json (see fflab.settings.example.json);
it only needs the keys it changes, nested objects are merged.

Nothing user- or machine-specific is hard-coded in the UI: it reads the public part through /api/config,
and paths are shown through display_path() (home -> "~", workspace -> ".").
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

DEFAULTS: dict = {
    "app": {
        "name": "Fisheye Fusion Lab Studio",
        "tagline": "Benchmark & Dựng cảnh lái xe đa phương pháp",
        "badge": "3DGS · FISHEYE · LIDAR",
        "ssh_user": "",            # shown in the "copy with scp" hint; empty = the account running the server
        "ssh_host": "",            # empty = the host name the browser used
    },
    "paths": {
        "ds_venv": "~/venvs/ds",   # DriveStudio+ training environment
        "eval_venv": "~/.fflab/venv",
    },
    "colab": {
        "data_session": "fflab-data",      # default name for a CPU session used to read Google Drive
        "gpu_session": "fflab-gpu",        # default name for a GPU training session
        "drive_cache": "fflab_cache",      # folder under MyDrive for cached CUDA wheels
        "default_gpu": "T4",
        "gpus": [                          # what `colab new --gpu` accepts, with UI labels and rough speed
            {"id": "T4", "label": "T4", "memory_gb": 16, "tier": "free", "sec_per_step": 0.45},
            {"id": "L4", "label": "L4", "memory_gb": 24, "tier": "paid", "sec_per_step": 0.22},
            {"id": "A100", "label": "A100", "memory_gb": 40, "tier": "paid", "sec_per_step": 0.12},
            {"id": "H100", "label": "H100", "memory_gb": 80, "tier": "paid", "sec_per_step": 0.08},
            {"id": "G4", "label": "G4", "memory_gb": 24, "tier": "paid", "sec_per_step": 0.3},
        ],
    },
}
ENV = {  # environment variable -> (section, key)
    "FFLAB_DS_VENV": ("paths", "ds_venv"),
    "FFLAB_VENV": ("paths", "eval_venv"),
    "FFLAB_APP_NAME": ("app", "name"),
    "FFLAB_SSH_USER": ("app", "ssh_user"),
    "FFLAB_SSH_HOST": ("app", "ssh_host"),
    "FFLAB_COLAB_GPU_SESSION": ("colab", "gpu_session"),
    "FFLAB_COLAB_DATA_SESSION": ("colab", "data_session"),
}


def _merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


class Settings:
    def __init__(self, workspace: Path, data: dict | None = None):
        self.workspace = Path(workspace).resolve()
        self.data = copy.deepcopy(DEFAULTS)
        if data is None:
            path = Path(os.environ.get("FFLAB_SETTINGS") or self.workspace / "fflab.settings.json")
            if path.is_file():
                _merge(self.data, json.loads(path.read_text(encoding="utf-8")))
            for var, (section, key) in ENV.items():
                if os.environ.get(var):
                    self.data[section][key] = os.environ[var]
        else:
            _merge(self.data, data)
        ids = [g["id"] for g in self.gpus]
        if self.data["colab"]["default_gpu"] not in ids:
            self.data["colab"]["default_gpu"] = ids[0]

    def path(self, key: str) -> Path:
        return Path(os.path.expanduser(self.data["paths"][key]))

    @property
    def gpus(self) -> list[dict]:
        return self.data["colab"]["gpus"]

    def gpu(self, gpu_id: str) -> dict | None:
        return next((g for g in self.gpus if g["id"] == gpu_id), None)

    def display_path(self, p) -> str:
        return display_path(p, self.workspace)

    def public(self) -> dict:
        """What the browser may see: no absolute paths."""
        c = self.data["colab"]
        return {"app": self.data["app"],
                "colab": {k: c[k] for k in ("data_session", "gpu_session", "default_gpu", "gpus")}}


def display_path(p, workspace: Path | None = None) -> str:
    if p is None:
        return ""
    s = os.path.abspath(os.path.expanduser(str(p)))
    if workspace is not None:
        w = str(Path(workspace))
        if s == w or s.startswith(w + os.sep):
            return "." + s[len(w):] if s != w else "."
    home = str(Path.home())
    if s == home or s.startswith(home + os.sep):
        return "~" + s[len(home):]
    return s
