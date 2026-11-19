"""Pull KITTI-360 sequence files or run artifacts from Google Drive mounted on a Colab session via `colab` CLI."""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import subprocess
import zipfile
from pathlib import Path

SEQUENCE = re.compile(r"^2013_05_28_drive_\d{4}_sync$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def run_colab(colab: str, args: list[str], stdin_text: str | None = None, check: bool = True) -> str:
    env = dict(os.environ, PYTHON_KEYRING_BACKEND=os.environ.get("PYTHON_KEYRING_BACKEND", "keyrings.alt.file.PlaintextKeyring"))
    r = subprocess.run([colab, *args], input=stdin_text, text=True, capture_output=True, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise SystemExit(f"colab {' '.join(args)} failed ({r.returncode}):\n{out.strip()}")
    return out


def remote_json(colab: str, session: str, code: str) -> dict:
    out = run_colab(colab, ["exec", "-s", session], stdin_text=code)
    for line in reversed(out.splitlines()):
        if line.startswith("FFLAB_JSON:"):
            return json.loads(line[len("FFLAB_JSON:"):])
    raise SystemExit(f"Unexpected remote output:\n{out}")


def ensure_session_and_drive(colab: str, session: str, drive_path: str):
    sessions_out = run_colab(colab, ["sessions"], check=False)
    if session not in sessions_out:
        print(f"[colab] Creating CPU session '{session}'...", flush=True)
        run_colab(colab, ["new", "-s", session])
    probe = remote_json(colab, session, "import json, os\nprint('FFLAB_JSON:' + json.dumps({'mounted': os.path.isdir('/content/drive/MyDrive')}))\n")
    if not probe.get("mounted"):
        # ponytail: `colab drivemount` needs interactive browser consent once per fresh Colab VM; tell user clearly if headless fails
        raise SystemExit(
            f"Google Drive chưa mount trên session '{session}'. Chạy 1 lần trong terminal máy ảo:\n"
            f"  colab drivemount -s {session}\n"
            "rồi bấm lại Đồng bộ trên Web UI."
        )
    info = remote_json(
        colab,
        session,
        f"import json, os\np={drive_path!r}\n"
        "items = sorted(os.listdir(p)) if os.path.isdir(p) else None\n"
        "print('FFLAB_JSON:' + json.dumps({'exists': items is not None, 'items': items or []}))\n",
    )
    if not info["exists"]:
        raise SystemExit(f"Không tìm thấy thư mục trên Drive: {drive_path}")
    return info["items"]


def sync_dataset(colab: str, session: str, drive_root: str, seq: str, dest: Path):
    ensure_session_and_drive(colab, session, drive_root)
    wanted = [
        "data_poses.zip",
        "calibration.zip",
        "ImageSets-rosa-6class/kitti360_rosa_infos_train.pkl",
        "ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl",
        *(f"data_2d_raw/{seq}_image_0{i}.tar" for i in range(4)),
        f"data_3d_raw/{seq}.tar",
    ]
    check_code = (
        "import json, os\n"
        f"root={drive_root!r}; wanted={wanted!r}\n"
        "present=[rel for rel in wanted if os.path.isfile(os.path.join(root, rel))]\n"
        "has_calib_dir=os.path.isdir(os.path.join(root, 'calibration'))\n"
        "calib_files=[]\n"
        "if has_calib_dir:\n"
        "  for dp,_,fs in os.walk(os.path.join(root,'calibration')):\n"
        "    for f in fs:\n"
        "      calib_files.append(os.path.relpath(os.path.join(dp,f), root).replace('\\\\','/'))\n"
        "print('FFLAB_JSON:' + json.dumps({'present': present, 'calib_files': sorted(calib_files)[:60]}))\n"
    )
    found = remote_json(colab, session, check_code)
    files = list(found["present"]) + [f for f in found["calib_files"] if "calibration.zip" not in found["present"]]
    if not files:
        raise SystemExit(f"Không thấy file KITTI-360 cho {seq} trong {drive_root}")
    dest.mkdir(parents=True, exist_ok=True)
    for rel in files:
        target = (dest / rel).resolve()
        if not target.is_relative_to(dest.resolve()):
            raise SystemExit(f"Refusing unsafe path: {rel}")
        if target.is_file() and target.stat().st_size > 0:
            print(f"[skip] {rel} ({target.stat().st_size / 2**20:.1f} MiB)", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        remote_path = posixpath.join(drive_root, rel)
        print(f"[download] {remote_path} -> {target}", flush=True)
        run_colab(colab, ["download", "-s", session, remote_path, str(target)])
    cal_zip = dest / "calibration.zip"
    if cal_zip.is_file() and not (dest / "calibration").is_dir():
        with zipfile.ZipFile(cal_zip) as zf:
            for m in zf.namelist():
                out_p = (dest / m).resolve()
                if not out_p.is_relative_to(dest.resolve()):
                    raise SystemExit("Unsafe zip entry in calibration.zip")
            zf.extractall(dest)
        print("[extract] calibration.zip -> calibration/", flush=True)
    print(f"[done] synced {len(files)} file(s) into {dest}", flush=True)


def sync_runs(colab: str, session: str, drive_runs: str, dest: Path):
    run_names = [r for r in ensure_session_and_drive(colab, session, drive_runs) if SAFE_NAME.match(r)]
    allowed_ext = (".json", ".log", ".png", ".jpg", ".mp4", ".yaml", ".ply")
    code = (
        "import json, os\n"
        f"root={drive_runs!r}; runs={run_names!r}; exts={allowed_ext!r}\n"
        "out=[]\n"
        "for r in runs:\n"
        "  rp=os.path.join(root, r)\n"
        "  if not os.path.isdir(rp): continue\n"
        "  for dp,_,fs in os.walk(rp):\n"
        "    for f in fs:\n"
        "      if f.endswith('partial.ply') or f.endswith('.pt') or f.endswith('.pth'): continue\n"
        "      if f.lower().endswith(exts):\n"
        "        rel=os.path.relpath(os.path.join(dp,f), root).replace('\\\\','/')\n"
        "        out.append(rel)\n"
        "print('FFLAB_JSON:' + json.dumps({'files': sorted(out)[:400]}))\n"
    )
    files = remote_json(colab, session, code)["files"]
    dest.mkdir(parents=True, exist_ok=True)
    for rel in files:
        target = (dest / rel).resolve()
        if not target.is_relative_to(dest.resolve()):
            continue
        if target.is_file() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {rel}", flush=True)
        run_colab(colab, ["download", "-s", session, posixpath.join(drive_runs, rel), str(target)])
    print(f"[done] synced {len(files)} run artifact(s) into {dest}", flush=True)


MAX_FILES = 5000
LIST_FOLDER = r"""
import json, os
root = __ROOT__; items = __ITEMS__; cap = __CAP__
out = []
names = items or sorted(n for n in os.listdir(root) if not n.startswith("."))
for n in names:
    p = os.path.join(root, n)
    if os.path.isfile(p):
        out.append([n, os.path.getsize(p)])
        continue
    for dp, ds, fs in os.walk(p):
        ds[:] = [d for d in ds if not d.startswith(".")]
        for f in fs:
            if f.startswith("."):
                continue
            fp = os.path.join(dp, f)
            out.append([os.path.relpath(fp, root).replace(os.sep, "/"), os.path.getsize(fp)])
        if len(out) > cap:
            break
print("FFLAB_JSON:" + json.dumps({"files": out[:cap + 1]}))
"""


def sync_folder(colab: str, session: str, drive_root: str, items: list[str], dest: Path):
    """Copy chosen entries (files or folders) of a Drive folder into dest, keeping their structure."""
    ensure_session_and_drive(colab, session, drive_root)
    code = LIST_FOLDER.replace("__ROOT__", repr(drive_root)).replace("__ITEMS__", repr(items)).replace(
        "__CAP__", str(MAX_FILES))
    files = remote_json(colab, session, code)["files"]
    if len(files) > MAX_FILES:
        raise SystemExit(f"Thư mục có hơn {MAX_FILES} file. Hãy nén thành .zip/.tar trên Drive rồi tải file nén, "
                         "hoặc chọn từng thư mục con nhỏ hơn.")
    if not files:
        raise SystemExit(f"Không có file nào để tải trong {drive_root}")
    total = sum(size for _, size in files)
    print(f"[plan] {len(files)} file, {total / 2**30:.2f} GiB -> {dest}", flush=True)
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, (rel, size) in enumerate(files, 1):
        target = (dest / rel).resolve()
        if not target.is_relative_to(dest):
            raise SystemExit(f"Refusing unsafe path: {rel}")
        if target.is_file() and target.stat().st_size == size:
            print(f"[skip {i}/{len(files)}] {rel}", flush=True)
            done += size
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download {i}/{len(files)}] {rel} ({size / 2**20:.1f} MiB)", flush=True)
        run_colab(colab, ["download", "-s", session, posixpath.join(drive_root, rel), str(target)])
        done += size
        print(f"[progress] {done / max(total, 1) * 100:.1f}%", flush=True)
    print(f"[done] {len(files)} file vào {dest}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--colab", default="colab")
    p.add_argument("--session", default="fisheye-lab")
    p.add_argument("--mode", choices=("dataset", "runs", "folder"), default="dataset")
    p.add_argument("--item", action="append", default=[])
    p.add_argument("--drive-path", required=True)
    p.add_argument("--sequence", default="2013_05_28_drive_0003_sync")
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    if not a.drive_path.startswith("/content/drive/"):
        raise SystemExit("drive-path must start with /content/drive/")
    if a.mode == "dataset":
        if not SEQUENCE.match(a.sequence):
            raise SystemExit("invalid sequence name")
        sync_dataset(a.colab, a.session, a.drive_path.rstrip("/"), a.sequence, Path(a.dest))
    elif a.mode == "folder":
        for item in a.item:
            if item.startswith("/") or ".." in item.split("/") or not item:
                raise SystemExit(f"invalid item: {item}")
        sync_folder(a.colab, a.session, a.drive_path.rstrip("/"), a.item, Path(a.dest))
    else:
        sync_runs(a.colab, a.session, a.drive_path.rstrip("/"), Path(a.dest))


if __name__ == "__main__":
    main()
