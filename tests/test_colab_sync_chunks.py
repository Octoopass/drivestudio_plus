"""Drive downloads in tools/colab_sync_drive.py against a fake `colab` CLI that maps /content to a temp dir:
small files via `colab download`, big ones (or all, once that fails) streamed through `colab exec` in checked chunks."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

FAKE = r'''#!__PY__
import os, shutil, subprocess, sys
ROOT = os.environ["FAKE_ROOT"]
args = sys.argv[1:]
cmd, pos, i = args[0], [], 1
while i < len(args):
    if args[i] in ("-s", "--timeout"):
        i += 2; continue
    pos.append(args[i]); i += 1
m = lambda p: ROOT + p if p.startswith("/content") else p
if cmd == "exec":
    code = sys.stdin.read()
    flaky = os.environ.get("FAKE_FLAKY")
    if flaky and flaky in code and not os.path.exists(ROOT + "/flaked"):
        open(ROOT + "/flaked", "w").close(); print("Connection was lost"); sys.exit(1)
    code = code.replace("'/content", "'" + ROOT + "/content")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)
if cmd == "download":
    src = m(pos[0])
    if os.environ.get("FAKE_DOWNLOAD_BROKEN") or os.path.getsize(src) > int(os.environ["FAKE_MAX"]):
        print("[colab] Download failed: 500 Server Error"); sys.exit(1)
    shutil.copy(src, pos[1])
'''


@pytest.fixture
def sync(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("colab_sync_drive", Path(__file__).resolve().parents[1] / "tools" / "colab_sync_drive.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "CHUNK", 1000)
    monkeypatch.setattr(mod, "DIRECT_MAX", 1500)
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = tmp_path / "colab"
    fake.write_text(FAKE.replace("__PY__", sys.executable))
    fake.chmod(0o755)
    root = tmp_path / "remote"
    (root / "content" / "drive").mkdir(parents=True)
    monkeypatch.setenv("FAKE_ROOT", str(root))
    monkeypatch.setenv("FAKE_MAX", "1500")
    monkeypatch.delenv("FAKE_DOWNLOAD_BROKEN", raising=False)
    return mod, str(fake), root


def test_big_file_streamed_with_retry_and_resume(sync, tmp_path, monkeypatch):
    mod, colab, root = sync
    data = os.urandom(3500)
    (root / "content" / "drive" / "big.tar").write_bytes(data)
    target = tmp_path / "out" / "big.tar"
    monkeypatch.setenv("FAKE_FLAKY", "f.seek(2000)")        # third chunk fails once, then succeeds
    assert mod.download_file(colab, "s", "/content/drive/big.tar", target) is True
    assert target.read_bytes() == data and not target.with_name("big.tar.part").exists()
    assert mod.download_file(colab, "s", "/content/drive/big.tar", target) is False    # complete -> skip
    target.unlink()                                          # resume from a .part holding two chunks
    target.with_name("big.tar.part").write_bytes(data[:2000])
    assert mod.download_file(colab, "s", "/content/drive/big.tar", target)
    assert target.read_bytes() == data


def test_small_file_direct_and_bad_leftover_replaced(sync, tmp_path):
    mod, colab, root = sync
    (root / "content" / "drive" / "a.zip").write_bytes(b"x" * 900)
    target = tmp_path / "a.zip"
    target.write_bytes(b"broken")                            # leftover of an interrupted run: wrong size
    assert mod.download_file(colab, "s", "/content/drive/a.zip", target)
    assert target.read_bytes() == b"x" * 900 and mod._STREAM_ONLY is False


def test_broken_download_api_falls_back_to_exec(sync, tmp_path, monkeypatch):
    mod, colab, root = sync
    monkeypatch.setenv("FAKE_DOWNLOAD_BROKEN", "1")
    (root / "content" / "drive" / "s.bin").write_bytes(b"abc" * 100)
    target = tmp_path / "s.bin"
    assert mod.download_file(colab, "s", "/content/drive/s.bin", target)
    assert target.read_bytes() == b"abc" * 100 and mod._STREAM_ONLY is True
