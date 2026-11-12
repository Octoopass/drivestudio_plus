"""Chạy một run trên Kaggle: đẩy lên, hỏi trạng thái, tải kết quả về (ARCHITECTURE mục 2.8).

Luồng:
1. ``push_run``: upload cảnh thành dataset private (một lần cho mỗi cảnh và phiên bản converter),
   sinh notebook từ ``templates/run.ipynb`` có nhúng gói job, đẩy notebook; Kaggle chạy ngay.
   Ghi ``d2s/kaggle.json`` trong thư mục run để biết notebook nào đang chạy run này.
2. ``kernel_state``: hỏi trạng thái notebook.
3. ``pull_run``: tải output, ghép ``runs/`` và ``cache/`` vào thư mục làm việc trên máy.

Phiên Kaggle bị ngắt giữa chừng thì ``pull_run`` rồi ``push_run`` lần nữa: lần đẩy sau gắn output
của notebook trước làm đầu vào, nên stage đã xong không chạy lại và train chạy tiếp từ
``checkpoint_resume.pth``.

Notebook cài code bằng ``git clone`` đúng commit của repo, nên commit phải có trên remote;
``push_run`` từ chối nếu chưa push.

File này chứa ``KaggleState``, ``push_run``, ``kernel_state``, ``pull_run``.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Optional

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine.cache import PARTIAL_SUFFIX
from drive2sim.engine.run import D2S_DIR, RUN_FILE
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle.bundle import write_bundle
from drive2sim.kaggle.client import KaggleClient, KaggleError

STATE_FILE = "kaggle.json"
TEMPLATE = paths.PACKAGE_DIR / "kaggle" / "templates" / "run.ipynb"
STAGING_DIR = ".kaggle"  # trong thư mục làm việc: thư mục tạm khi upload, tải về
DEFAULT_BRANCH = "drive2sim"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"  # machine_shape của Kaggle: GPU T4 x2


@dataclass
class KaggleState:
    """Nội dung ``d2s/kaggle.json``: run này đang ở notebook nào trên Kaggle.

    Attributes:
        username: Người dùng Kaggle.
        kernel: Notebook của lần đẩy mới nhất, ``<người dùng>/<tên>``.
        attempt: Lần đẩy thứ mấy (1, 2...); mỗi lần là một notebook riêng.
        dataset: Dataset chứa cảnh.
        commit: Commit của code mà notebook dùng.
        pushed_utc: Thời điểm đẩy.
        pulled_utc: Thời điểm tải kết quả về gần nhất.
    """

    username: str
    kernel: str
    attempt: int
    dataset: str
    commit: str
    pushed_utc: str
    pulled_utc: Optional[str] = None

    def to_dict(self) -> dict:
        """Nội dung dạng dict, để ghi JSON."""
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def load_state(run_dir: Path) -> Optional[KaggleState]:
    """Đọc ``d2s/kaggle.json``; ``None`` nếu run chưa từng đẩy lên Kaggle."""
    path = run_dir / D2S_DIR / STATE_FILE
    return KaggleState(**json.loads(path.read_text("utf-8"))) if path.is_file() else None


def _save_state(run_dir: Path, state: KaggleState) -> None:
    (run_dir / D2S_DIR / STATE_FILE).write_text(json.dumps(asdict(state), indent=2), "utf-8")


def _git(*args: str) -> str:
    """Chạy git trong repo chứa package; trả stdout, chuỗi rỗng nếu lỗi."""
    try:
        result = subprocess.run(
            ["git", "-C", str(paths.PACKAGE_DIR), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def code_source() -> tuple[str, str]:
    """Commit hiện tại và URL của remote ``origin``.

    Returns:
        ``(commit đầy đủ, url)``.

    Raises:
        KaggleError: Nếu không đọc được git, hoặc commit chưa có trên remote nào (notebook sẽ
            không clone được).
    """
    commit = _git("rev-parse", "HEAD")
    url = os.environ.get("D2S_KAGGLE_REPO") or _git("remote", "get-url", "origin")
    if not commit or not url:
        raise KaggleError("không đọc được commit hoặc remote origin của repo")
    if not _git("branch", "-r", "--contains", commit):
        raise KaggleError(f"commit {commit[:7]} chưa có trên remote; push code trước rồi chạy lại")
    return commit, url


def _ensure_scene_dataset(
    client: KaggleClient, workdir: Workdir, username: str, scene_id: str, converter_version: str
) -> str:
    """Upload cảnh thành dataset private nếu chưa có; trả định danh dataset.

    Tên dataset gồm băm của ``scene_id`` và phiên bản converter, nên ingest lại bằng converter
    khác thì upload dataset mới. Thư mục upload dựng bằng hard link khi được, để không tốn gấp
    đôi dung lượng.
    """
    slug = f"{username}/d2s-{_hash8(f'{scene_id}@{converter_version}')}"
    if client.dataset_exists(slug):
        return slug
    staging = workdir.root / STAGING_DIR / "upload" / slug.split("/")[1]
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    def link_or_copy(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(workdir.scene_dir(scene_id), staging / scene_id, copy_function=link_or_copy)
    data_license = workdir.load_scene(scene_id).manifest.source.data_license
    metadata = {
        "title": f"d2s {scene_id}"[:50],
        "id": slug,
        "description": f"Cảnh {scene_id} của drive2sim. License dữ liệu gốc: "
        f"{data_license or 'xem dataset gốc'}.",
        "licenses": [{"name": "other"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    try:
        client.create_dataset(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return slug


def build_notebook(bundle: dict, run_id: str, commit: str, repo_url: str, branch: str) -> str:
    """Điền template notebook.

    Returns:
        Nội dung file ``.ipynb``.
    """
    values = {
        "run_id": run_id,
        "commit": commit,
        "repo_url": repo_url,
        "branch": branch,
        "bundle_json": json.dumps(bundle),
    }
    notebook = json.loads(TEMPLATE.read_text("utf-8"))
    for cell in notebook["cells"]:
        cell["source"] = [Template(line).safe_substitute(values) for line in cell["source"]]
    return json.dumps(notebook, indent=1, ensure_ascii=False)


def push_run(
    workdir: Workdir,
    run_id: str,
    client: Optional[KaggleClient] = None,
    branch: str = DEFAULT_BRANCH,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> KaggleState:
    """Đẩy một run lên Kaggle và cho chạy.

    Lần đẩy đầu chạy từ đầu; các lần sau chạy tiếp từ output của notebook trước.

    Args:
        workdir: Thư mục làm việc trên máy.
        run_id: Run đã tạo (``executor: kaggle``).
        client: Kaggle; mặc định gọi lệnh ``kaggle``.
        branch: Branch để clone.
        accelerator: ``machine_shape`` của notebook, ví dụ ``NvidiaTeslaT4``, ``NvidiaL4``.

    Returns:
        Trạng thái đã ghi vào ``d2s/kaggle.json``.

    Raises:
        KaggleError: Nếu chưa cài hoặc chưa đăng nhập Kaggle, commit chưa push, run đã xong, hoặc
            notebook trước còn đang chạy.
    """
    client = client or KaggleClient()
    run_dir = workdir.find_run(run_id)
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    if record.status == "succeeded":
        raise KaggleError(f"run {run_id} đã xong; không cần đẩy lại")
    previous = load_state(run_dir)
    if previous is not None:
        status = client.kernel_status(previous.kernel)
        if status in ("queued", "running", "new"):
            raise KaggleError(f"notebook {previous.kernel} còn đang {status}; chờ xong rồi đẩy")
    username = client.username()
    commit, repo_url = code_source()
    scene = workdir.load_scene(record.scene_id)
    dataset = _ensure_scene_dataset(
        client, workdir, username, record.scene_id, scene.manifest.source.converter_version
    )
    attempt = previous.attempt + 1 if previous else 1
    kernel = f"{username}/d2s-{_hash8(run_id)}-{attempt}"
    resume = previous.kernel if previous else None
    bundle = write_bundle(run_dir, workdir, dataset, resume)

    with tempfile.TemporaryDirectory(dir=_staging(workdir)) as folder:
        folder_path = Path(folder)
        (folder_path / "run.ipynb").write_text(
            build_notebook(bundle, run_id, commit, repo_url, branch), "utf-8"
        )
        metadata = {
            "id": kernel,
            "title": kernel.split("/")[1],
            "code_file": "run.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": accelerator,
            "enable_internet": True,
            "dataset_sources": [dataset],
            "kernel_sources": [resume] if resume else [],
            "competition_sources": [],
        }
        (folder_path / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        client.push_kernel(folder_path)

    state = KaggleState(username, kernel, attempt, dataset, commit, _now())
    _save_state(run_dir, state)
    return state


def _staging(workdir: Workdir) -> Path:
    path = workdir.root / STAGING_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def kernel_state(workdir: Workdir, run_id: str, client: Optional[KaggleClient] = None) -> dict:
    """Trạng thái notebook của một run.

    Returns:
        Nội dung ``kaggle.json`` cộng ``status`` (trạng thái Kaggle).

    Raises:
        KaggleError: Nếu run chưa đẩy lên Kaggle.
    """
    client = client or KaggleClient()
    state = load_state(workdir.find_run(run_id))
    if state is None:
        raise KaggleError(f"run {run_id} chưa đẩy lên Kaggle; chạy d2s kaggle push {run_id}")
    return {**asdict(state), "status": client.kernel_status(state.kernel)}


def pull_run(workdir: Workdir, run_id: str, client: Optional[KaggleClient] = None) -> RunRecord:
    """Tải output của notebook và ghép vào thư mục làm việc.

    Thay nội dung ``d2s/`` của run bằng bản trên Kaggle (giữ ``kaggle.json``); chép thêm các kết
    quả cache mà máy chưa có.

    Returns:
        ``run.json`` sau khi ghép.

    Raises:
        KaggleError: Nếu run chưa đẩy lên Kaggle, hoặc output không có run này.
    """
    client = client or KaggleClient()
    run_dir = workdir.find_run(run_id)
    state = load_state(run_dir)
    if state is None:
        raise KaggleError(f"run {run_id} chưa đẩy lên Kaggle")
    with tempfile.TemporaryDirectory(dir=_staging(workdir)) as folder:
        output = Path(folder)
        client.kernel_output(state.kernel, output)
        relative = run_dir.relative_to(workdir.root)
        found = [p for p in output.glob(f"**/{relative.as_posix()}") if (p / D2S_DIR).is_dir()]
        if not found:
            raise KaggleError(f"output của {state.kernel} không có {relative}")
        remote_d2s = found[0] / D2S_DIR
        for item in remote_d2s.iterdir():
            if item.name == STATE_FILE:
                continue
            target = run_dir / D2S_DIR / item.name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(item), str(target))
        remote_cache = found[0].parents[2].parent / "cache"
        for entry in sorted(remote_cache.glob("*/*")):
            target = workdir.cache_dir / entry.parent.name / entry.name
            if entry.is_dir() and not entry.name.endswith(PARTIAL_SUFFIX) and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(entry), str(target))
    state.pulled_utc = _now()
    _save_state(run_dir, state)
    return files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
