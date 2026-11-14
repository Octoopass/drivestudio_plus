"""Phần dùng chung của các executor chạy run ở máy khác (Kaggle, Colab; ARCHITECTURE mục 2.8).

Hai executor gửi run đi theo cách khác nhau (notebook Kaggle, VM Colab) nhưng cùng: clone code
đúng commit từ remote git, chạy cùng gói job bằng ``d2s kaggle run``, rồi tải output về và ghép vào
thư mục làm việc. File này chứa phần giống nhau đó, để hai bên cho ra cùng kết quả:

- ``RemoteError``: lỗi chung; ``KaggleError``, ``ColabError`` đổi nó sang lỗi của mình.
- ``git``: chạy ``git`` trong repo chứa package.
- ``code_source``: commit hiện tại và URL repo máy kia sẽ clone; từ chối khi commit chưa push.
- ``merge_d2s``: ghép ``d2s/`` của run tải về vào ``d2s/`` của run trên máy.
- ``merge_output``: tìm run trong thư mục output đã tải, ghép ``d2s/`` và chép cache còn thiếu.

``drive2sim/kaggle/executor.py`` và ``drive2sim/colab/executor.py`` gọi các hàm này. Xem
CONTRACTS mục 9.2 và 9.3.
"""

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import EnvRecord, RunRecord
from drive2sim.engine.cache import PARTIAL_SUFFIX
from drive2sim.engine.machine import same_machine
from drive2sim.engine.run import CANCEL_FILE, D2S_DIR, ENV_FILE, PID_FILE, RUN_FILE
from drive2sim.engine.workdir import Workdir

STAGES_DIR = "stages"
# File trạng thái của từng executor trong d2s/ (d2s/kaggle.json, d2s/colab.json): chỉ có nghĩa trên
# máy đã đẩy run, nên bản trên máy luôn được giữ khi ghép output tải về.
EXECUTOR_STATE_FILES = ("kaggle.json", "colab.json")

Git = Callable[..., str]


class RemoteError(RuntimeError):
    """Lỗi chung khi chuẩn bị chạy run ở máy khác: không đọc được git, commit chưa push."""


def git(*args: str) -> str:
    """Chạy ``git <args>`` trong repo chứa package.

    Returns:
        stdout đã bỏ khoảng trắng hai đầu; chuỗi rỗng nếu git lỗi hoặc không có git.
    """
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


def _remote_for(url: str, run_git: Git) -> Optional[str]:
    """Tên remote của repo có URL ``url`` (ví dụ ``origin``), nếu có."""
    for line in run_git("remote", "-v").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == url:
            return parts[0]
    return None


def code_source(branch: str, repo_env: str, run_git: Git = git) -> tuple[str, str]:
    """Commit hiện tại và URL repo mà máy kia sẽ clone (biến ``repo_env`` hoặc remote ``origin``).

    Máy kia chạy ``git clone --branch <branch> <url>`` rồi ``git checkout <commit>``, nên commit
    phải có trên **đúng remote đó** và remote phải có nhánh ``branch``; có trên một remote khác
    (ví dụ ``fork``) thì máy kia không checkout được.

    Args:
        branch: Nhánh sẽ clone, ví dụ ``drive2sim``.
        repo_env: Tên biến môi trường cho phép đổi URL repo, ví dụ ``D2S_KAGGLE_REPO``.
        run_git: Hàm chạy git; test truyền hàm giả.

    Returns:
        ``(commit đầy đủ, url)``.

    Raises:
        RemoteError: Nếu không đọc được git, remote không có nhánh ``branch``, hoặc commit chưa
            push lên remote đó.
    """
    commit = run_git("rev-parse", "HEAD")
    url = os.environ.get(repo_env) or run_git("remote", "get-url", "origin")
    if not commit or not url:
        raise RemoteError("không đọc được commit hoặc remote origin của repo")
    remote = _remote_for(url, run_git)
    if remote is None:
        # URL không phải remote nào của repo này: không kiểm tra được, máy kia sẽ báo nếu thiếu.
        return commit, url
    if not run_git("rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{branch}"):
        raise RemoteError(f"remote {remote} ({url}) không có nhánh {branch}; push nhánh đó trước")
    if not run_git("branch", "-r", "--contains", commit, "--list", f"{remote}/*"):
        raise RemoteError(
            f"commit {commit[:7]} chưa có trên remote {remote} ({url}), repo mà máy chạy sẽ "
            f"clone; git push {remote} {branch} rồi chạy lại"
        )
    return commit, url


def _load_or_none(path: Path, model: type) -> Optional[object]:
    """Đọc một file JSON theo schema; thiếu file hoặc file hỏng thì trả ``None``."""
    try:
        return files.load(path, model)
    except (OSError, ValueError):
        return None


def _replace(source: Path, target: Path) -> None:
    """Chuyển ``source`` vào chỗ ``target``, xoá ``target`` cũ (file hoặc thư mục) nếu có."""
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    elif target.exists() or target.is_symlink():
        target.unlink()
    shutil.move(str(source), str(target))


def merge_d2s(local: Path, remote: Path, keep: Sequence[str] = EXECUTOR_STATE_FILES) -> list[str]:
    """Ghép ``d2s/`` tải từ máy kia vào ``d2s/`` của run trên máy.

    - Mỗi stage có trong bản tải về thay stage cùng tên trên máy (bản tải về mới hơn).
    - Stage chỉ có trên máy, đã có thư mục kết quả, được giữ lại: thư mục giữ nguyên, bản ghi của
      nó nối vào cuối ``run.json`` lấy từ bản tải về.
    - ``env.json`` lấy theo bản tải về, cộng thêm các máy chỉ có trong bản trên máy.
    - File trạng thái của executor (``keep``) trên máy giữ nguyên; ``pid``, ``cancel`` của máy kia
      bỏ đi; các file khác lấy theo bản tải về.

    Args:
        local: ``d2s/`` của run trên máy.
        remote: ``d2s/`` của run trong output đã tải (sẽ bị chuyển đi).
        keep: Tên file trong ``d2s/`` của máy này không bị ghi đè.

    Returns:
        Các ``stage_id`` chỉ có trên máy đã được giữ lại.
    """
    local_record = _load_or_none(local / RUN_FILE, RunRecord)
    remote_record = _load_or_none(remote / RUN_FILE, RunRecord)
    local_env = _load_or_none(local / ENV_FILE, EnvRecord)
    remote_ids = {s.stage_id for s in remote_record.stages} if remote_record else set()
    kept = [
        stage
        for stage in (local_record.stages if local_record else [])
        if stage.stage_id not in remote_ids and (local / STAGES_DIR / stage.stage_id).is_dir()
    ]
    for item in remote.iterdir():
        # pid, cancel: trạng thái tiến trình của máy kia (pid còn sót khi phiên bị ngắt), không có
        # nghĩa gì trên máy này; chép về thì run bị coi là đang chạy mãi.
        if item.name in (*keep, PID_FILE, CANCEL_FILE):
            continue
        target = local / item.name
        if item.name == STAGES_DIR and item.is_dir() and target.is_dir():
            for stage_dir in item.iterdir():
                _replace(stage_dir, target / stage_dir.name)
            continue
        _replace(item, target)
    if kept and remote_record is not None:
        merged = remote_record.model_copy(update={"stages": [*remote_record.stages, *kept]})
        files.dump(merged, local / RUN_FILE)
    env = _load_or_none(local / ENV_FILE, EnvRecord)
    if local_env is not None and env is not None:
        extra = [
            m for m in local_env.machines if not any(same_machine(m, s) for s in env.machines)
        ]
        if extra:
            files.dump(
                env.model_copy(update={"machines": [*env.machines, *extra]}), local / ENV_FILE
            )
    return [stage.stage_id for stage in kept]


def merge_output(workdir: Workdir, run_dir: Path, output: Path) -> Path:
    """Ghép output đã tải của một run vào thư mục làm việc trên máy.

    Output là một cây thư mục chứa ``runs/`` và ``cache/`` giống thư mục làm việc của máy kia
    (có thể nằm sâu bên dưới, ví dụ ``<output>/<notebook>/runs/...``). ``d2s/`` của run được ghép
    theo ``merge_d2s``; các kết quả cache máy chưa có được chuyển vào ``cache/``, cache đã có giữ
    nguyên, thư mục ``.partial`` (ghi dở) bỏ qua.

    Args:
        workdir: Thư mục làm việc trên máy.
        run_dir: Thư mục run trên máy.
        output: Thư mục chứa output đã tải (các mục bên trong bị chuyển đi).

    Returns:
        Thư mục run tìm thấy trong ``output``.

    Raises:
        FileNotFoundError: Nếu output không có run này.
    """
    relative = run_dir.relative_to(workdir.root)
    found = [p for p in output.glob(f"**/{relative.as_posix()}") if (p / D2S_DIR).is_dir()]
    if not found:
        raise FileNotFoundError(relative.as_posix())
    merge_d2s(run_dir / D2S_DIR, found[0] / D2S_DIR)
    remote_cache = found[0].parents[2].parent / "cache"
    for entry in sorted(remote_cache.glob("*/*")):
        target = workdir.cache_dir / entry.parent.name / entry.name
        if entry.is_dir() and not entry.name.endswith(PARTIAL_SUFFIX) and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(target))
    return found[0]
