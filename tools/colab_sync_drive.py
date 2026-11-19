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


def run_colab(colab: str, args: list[str], stdin_text: str | None = None, check: bool = True, timeout: float = 900) -> str:
    env = dict(os.environ, PYTHON_KEYRING_BACKEND=os.environ.get("PYTHON_KEYRING_BACKEND", "keyrings.alt.file.PlaintextKeyring"))
    try:
        # hard limit: `colab exec --timeout` only counts silence, and a stalled connection can hang forever
        r = subprocess.run([colab, *args], input=stdin_text, text=True, capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        if not check:
            return ""
        raise SystemExit(f"colab {' '.join(args[:3])} quá {timeout:.0f}s, đã ngắt")
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


# `colab download` goes through the runtime's Jupyter contents API behind Google's proxy. It reads a whole file
# into memory and base64-encodes it, so files of ~1 GB fail with HTTP 500, and on some runtimes the API fails for
# every file while code execution still works. So: small files try `colab download` first; big files, and every
# file once a download has failed, are streamed through `colab exec` in CHUNK-sized, MD5-checked pieces that are
# appended to <target>.part (an interrupted transfer resumes from the last complete piece).
CHUNK = 32 * 2**20
DIRECT_MAX = 200 * 2**20
_STREAM_ONLY = False        # set after the first failed `colab download` in this run


def remote_size(colab: str, session: str, path: str) -> int:
    info = remote_json(colab, session, f"import json, os\np={path!r}\n"
                       "print('FFLAB_JSON:' + json.dumps({'size': os.path.getsize(p) if os.path.isfile(p) else -1}))\n")
    if info["size"] < 0:
        raise SystemExit(f"Không thấy file trên Drive: {path}")
    return int(info["size"])


def _retry(fn, what: str, attempts: int = 3):
    import time
    for i in range(attempts):
        try:
            return fn()
        except SystemExit as exc:
            if i + 1 == attempts:
                raise
            print(f"[retry {i + 1}/{attempts - 1}] {what}: {str(exc).splitlines()[0][:160]}", flush=True)
            time.sleep(5 * (i + 1))


def _read_chunk(colab: str, session: str, remote: str, offset: int, length: int) -> bytes:
    import base64
    import hashlib
    code = (f"import base64, hashlib\nwith open({remote!r}, 'rb') as f:\n    f.seek({offset})\n    d = f.read({length})\n"
            "print('FFLAB_MD5 ' + hashlib.md5(d).hexdigest())\nprint('FFLAB_B64 ' + base64.b64encode(d).decode())\n")
    out = run_colab(colab, ["exec", "-s", session, "--timeout", "600"], stdin_text=code)
    md5 = b64 = None
    for line in out.splitlines():
        if line.startswith("FFLAB_MD5 "):
            md5 = line[10:].strip()
        elif line.startswith("FFLAB_B64 "):
            b64 = line[10:].strip()
    if md5 is None or b64 is None:
        raise SystemExit(f"Colab không trả dữ liệu: {out.strip()[-200:]}")
    data = base64.b64decode(b64)
    if len(data) != length or hashlib.md5(data).hexdigest() != md5:
        raise SystemExit(f"khối tại {offset} hỏng ({len(data)}/{length} byte)")
    return data


def _stream(colab: str, session: str, remote: str, target: Path, size: int, label: str):
    part_file = target.with_name(target.name + ".part")
    have = part_file.stat().st_size if part_file.exists() else 0
    if have % CHUNK or have > size:          # not on a chunk boundary: start over
        part_file.unlink()
        have = 0
    chunks = max(1, -(-size // CHUNK))
    print(f"[stream] {label} ({size / 2**20:.1f} MiB, {chunks} khối){' tiếp từ khối ' + str(have // CHUNK + 1) if have else ''}", flush=True)
    for i in range(have // CHUNK, chunks):
        off = i * CHUNK
        data = _retry(lambda: _read_chunk(colab, session, remote, off, min(CHUNK, size - off)), f"{label} khối {i + 1}/{chunks}")
        with part_file.open("ab") as out:
            out.write(data)
        print(f"[progress] {label} {min(size, off + CHUNK) / 2**20:.0f}/{size / 2**20:.0f} MiB", flush=True)
    if part_file.stat().st_size != size:
        raise SystemExit(f"Ghép file thiếu {label}: {part_file.stat().st_size}/{size} byte")
    part_file.replace(target)


def download_file(colab: str, session: str, remote: str, target: Path, size: int | None = None, label: str = "") -> bool:
    """Download remote -> target unless it is already complete; returns True if something was transferred."""
    global _STREAM_ONLY
    label = label or remote
    size = remote_size(colab, session, remote) if size is None else size
    if target.is_file() and target.stat().st_size == size:
        print(f"[skip] {label} ({size / 2**20:.1f} MiB)", flush=True)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()                      # incomplete leftover from an older run
    if size <= DIRECT_MAX and not _STREAM_ONLY:
        print(f"[download] {label} ({size / 2**20:.1f} MiB)", flush=True)
        try:
            _retry(lambda: run_colab(colab, ["download", "-s", session, remote, str(target)]), label, attempts=2)
            if target.is_file() and target.stat().st_size == size:
                return True
        except SystemExit as exc:
            print(f"[fallback] colab download lỗi ({str(exc).splitlines()[-1][:120]}); chuyển sang truyền qua exec", flush=True)
        _STREAM_ONLY = True
        if target.exists():
            target.unlink()
    _stream(colab, session, remote, target, size, label)
    return True


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
        download_file(colab, session, posixpath.join(drive_root, rel), target, label=rel)
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
        download_file(colab, session, posixpath.join(drive_runs, rel), target, label=rel)
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
        print(f"[file {i}/{len(files)}]", flush=True)
        download_file(colab, session, posixpath.join(drive_root, rel), target, size=size, label=rel)
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
