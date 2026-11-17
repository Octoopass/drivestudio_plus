"""fflab entry point for the DriveStudio+ backend: reads DRIVESTUDIO_* variables and runs tools/train.py.

fflab passes every option as an environment variable (see fflab/backends/drivestudio.json), so this
script only turns them into the command line that `tools/train.py` expects and runs it from the
repository root (train.py reads `configs/datasets/...` relative to the working directory).

    DRIVESTUDIO_CONFIG=configs/streetgs_ut.yaml DRIVESTUDIO_DATASET=waymo/3cams \
    DRIVESTUDIO_OUT=./fflab-runs/r1 DRIVESTUDIO_RUN_NAME=r1 python tools/drivestudio_run.py
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _flag(env: dict, name: str, default: bool = False) -> bool:
    return env.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def build_command(env: dict, python: str = sys.executable) -> list[str]:
    """The `train.py` command line for a set of DRIVESTUDIO_* variables."""
    config = env.get("DRIVESTUDIO_CONFIG", "").strip()
    if not config:
        raise ValueError("DRIVESTUDIO_CONFIG is required (e.g. configs/streetgs_ut.yaml)")
    cmd = [
        python, "tools/train.py",
        "--config_file", config,
        "--output_root", env.get("DRIVESTUDIO_OUT", "./work_dirs") or "./work_dirs",
        "--project", env.get("DRIVESTUDIO_PROJECT", "drivestudio"),
        "--run_name", env.get("DRIVESTUDIO_RUN_NAME", "run"),
    ]
    if env.get("DRIVESTUDIO_RESUME", "").strip():
        cmd += ["--resume_from", env["DRIVESTUDIO_RESUME"].strip()]
    if _flag(env, "DRIVESTUDIO_VIEWER"):
        cmd += ["--enable_viewer", "--viewer_port", env.get("DRIVESTUDIO_VIEWER_PORT", "8080")]
    # `opts` is argparse.REMAINDER in train.py, so every dotlist override must come last.
    overrides = []
    if env.get("DRIVESTUDIO_DATASET", "").strip():
        overrides.append(f"dataset={env['DRIVESTUDIO_DATASET'].strip()}")
    if env.get("DRIVESTUDIO_SCENE_IDX", "").strip():
        overrides.append(f"data.scene_idx={env['DRIVESTUDIO_SCENE_IDX'].strip()}")
    if env.get("DRIVESTUDIO_RENDER_MODE", "").strip():
        overrides.append(f"trainer.render.render_mode={env['DRIVESTUDIO_RENDER_MODE'].strip()}")
    if env.get("DRIVESTUDIO_FASTGS", "").strip():
        overrides.append(f"trainer.fastgs.enabled={'true' if _flag(env, 'DRIVESTUDIO_FASTGS') else 'false'}")
    if env.get("DRIVESTUDIO_STEPS", "").strip():
        overrides.append(f"trainer.optim.num_iters={env['DRIVESTUDIO_STEPS'].strip()}")
    overrides += shlex.split(env.get("DRIVESTUDIO_OVERRIDES", ""))
    return cmd + overrides


def main() -> int:
    env = dict(os.environ)
    cmd = build_command(env)
    child_env = dict(env, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH", "")])))
    log_path = env.get("DRIVESTUDIO_LOG")
    print("[drivestudio] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    log = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=REPO, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        if log:
            log.write(line)
            log.flush()
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
