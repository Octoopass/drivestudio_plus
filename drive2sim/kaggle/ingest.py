"""Ingest một cảnh ngay trên Kaggle, từ dữ liệu gốc đã nằm trên Kaggle (CONTRACTS mục 9.4).

Dữ liệu gốc lớn (ví dụ KITTI-360 đủ khung, khoảng 8 GB một chuỗi) không cần nằm trên máy: nó là
một dataset private hoặc output của một notebook trên Kaggle. ``d2s kaggle ingest`` đẩy một
notebook chỉ dùng CPU gắn nguồn đó, notebook chạy ``d2s scene ingest`` rồi giữ cảnh trong output
của nó. Trên máy chỉ còn các file nhỏ của cảnh (``manifest.yaml``, ``reports/``) và file
``kaggle_scene.json`` ghi notebook nào giữ cảnh; ``d2s kaggle push`` gắn output notebook đó vào
notebook train thay vì upload cảnh.

Luồng của ``ingest_on_kaggle``:
1. Tên notebook ``<người dùng>/d2s-ingest-<hash8>`` băm từ nguồn và tham số ingest, nên gọi lại
   cùng tham số thì dùng lại notebook đã chạy (chờ tiếp nếu đang chạy, tải về nếu đã xong).
2. Đẩy notebook (template ``templates/ingest.ipynb``) khi chưa có hoặc lần trước lỗi.
3. Chờ notebook xong, tải các file nhỏ của cảnh về ``data/<scene_id>/``, ghi ``kaggle_scene.json``.

File này chứa ``KaggleSceneRef``, ``IngestRequest``, ``load_scene_ref``, ``build_ingest_notebook``,
``ingest_on_kaggle``.
"""

import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Template
from typing import Optional

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.workdir import KAGGLE_SCENE_FILE, Workdir
from drive2sim.kaggle.client import KaggleClient, KaggleError
from drive2sim.kaggle.executor import code_source

TEMPLATE = paths.PACKAGE_DIR / "kaggle" / "templates" / "ingest.ipynb"
RESULT_FILE = "ingest_result.json"  # notebook ghi scene_id vào output
# File nhỏ của cảnh được tải về máy; ảnh, LiDAR, mặt nạ ở lại Kaggle.
LIGHT_FILES = r"(ingest_result\.json|manifest\.yaml|reports/[^/]+\.json)$"
POLL_S = 60
WAIT_S = 6 * 3600  # notebook ingest lâu nhất 6 giờ thì thôi chờ (chạy lại lệnh để chờ tiếp)
_RUNNING = ("queued", "running", "new", "new_script", "cancel_requested")


@dataclass
class KaggleSceneRef:
    """Nội dung ``data/<scene_id>/kaggle_scene.json``: cảnh nằm trong output của notebook nào.

    Attributes:
        username: Tài khoản Kaggle sở hữu notebook (notebook private).
        kernel: Notebook ingest ``<người dùng>/d2s-ingest-<hash8>``.
        source: Dataset hoặc notebook chứa dữ liệu gốc.
        commit: Commit của code đã ingest.
        pulled_utc: Thời điểm tải các file nhỏ về.
    """

    username: str
    kernel: str
    source: str
    commit: str
    pulled_utc: str


@dataclass(frozen=True)
class IngestRequest:
    """Nguồn và tham số của một lần ingest trên Kaggle.

    Attributes:
        source: Dataset hoặc notebook chứa dữ liệu gốc, ``<người dùng>/<tên>``.
        path: Thư mục con trong nguồn chứa dữ liệu gốc, ví dụ ``kitti360``; rỗng là gốc nguồn.
        args: Tham số truyền nguyên cho ``d2s scene ingest`` (``--sequence``, ``--scene``...).
    """

    source: str
    path: str = ""
    args: Sequence[str] = field(default_factory=tuple)

    def kernel_name(self) -> str:
        """Tên notebook (không có người dùng), băm từ nguồn và tham số."""
        text = json.dumps([self.source, self.path, list(self.args)])
        return f"d2s-ingest-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def load_scene_ref(scene_root: Path) -> Optional[KaggleSceneRef]:
    """Đọc ``kaggle_scene.json`` của cảnh; ``None`` nếu cảnh nằm trên máy."""
    path = scene_root / KAGGLE_SCENE_FILE
    if not path.is_file():
        return None
    return KaggleSceneRef(**json.loads(path.read_text("utf-8")))


def build_ingest_notebook(request: IngestRequest, commit: str, repo_url: str, branch: str) -> str:
    """Điền template notebook ingest.

    Returns:
        Nội dung file ``.ipynb``.
    """
    values = {
        "source": request.source,
        "source_name": request.source.split("/")[-1],
        "source_path": request.path,
        "ingest_args": json.dumps(list(request.args)),
        "commit": commit,
        "repo_url": repo_url,
        "branch": branch,
    }
    notebook = json.loads(TEMPLATE.read_text("utf-8"))
    for cell in notebook["cells"]:
        cell["source"] = [Template(line).safe_substitute(values) for line in cell["source"]]
    return json.dumps(notebook, indent=1, ensure_ascii=False)


def _push(client: KaggleClient, request: IngestRequest, kernel: str, branch: str) -> str:
    """Đẩy notebook ingest; trả commit của code."""
    commit, repo_url = code_source(branch)
    # Nguồn là dataset hay output của notebook: Kaggle gắn hai loại qua hai trường khác nhau.
    is_dataset = client.dataset_exists(request.source)
    with tempfile.TemporaryDirectory() as folder:
        folder_path = Path(folder)
        (folder_path / "ingest.ipynb").write_text(
            build_ingest_notebook(request, commit, repo_url, branch), "utf-8"
        )
        metadata = {
            "id": kernel,
            "title": kernel.split("/")[1],
            "code_file": "ingest.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [request.source] if is_dataset else [],
            "kernel_sources": [] if is_dataset else [request.source],
            "competition_sources": [],
        }
        (folder_path / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        client.push_kernel(folder_path)
    return commit


def _wait(
    client: KaggleClient,
    kernel: str,
    timeout_s: float,
    sleep: Callable[[float], None],
    notify: Callable[[str], None],
) -> str:
    """Chờ notebook rời trạng thái đang chạy; trả trạng thái cuối."""
    waited = 0.0
    last = ""
    while True:
        status = client.kernel_status(kernel)
        if status != last:
            notify(f"{kernel}: {status}")
            last = status
        if status not in _RUNNING:
            return status
        if waited >= timeout_s:
            raise KaggleError(
                f"notebook {kernel} vẫn {status} sau {int(timeout_s)} giây; chạy lại lệnh để chờ "
                "tiếp"
            )
        sleep(POLL_S)
        waited += POLL_S


def ingest_on_kaggle(
    workdir: Workdir,
    request: IngestRequest,
    client: Optional[KaggleClient] = None,
    branch: str = "drive2sim",
    timeout_s: float = WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
    notify: Callable[[str], None] = lambda message: None,
    now: Callable[[], str] = lambda: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
) -> Path:
    """Ingest một cảnh trên Kaggle và ghi các file nhỏ của nó vào ``data/<scene_id>/``.

    Args:
        workdir: Thư mục làm việc trên máy.
        request: Nguồn và tham số ingest.
        client: Kaggle; mặc định gọi lệnh ``kaggle``.
        branch: Branch notebook clone.
        timeout_s: Chờ notebook tối đa bao lâu.
        sleep: Hàm chờ giữa hai lần hỏi trạng thái (test thay).
        notify: Nhận từng thay đổi trạng thái để in ra.
        now: Thời điểm hiện tại dạng ISO (test thay).

    Returns:
        Thư mục cảnh trên máy (chỉ có ``manifest.yaml``, ``reports/``, ``kaggle_scene.json``).

    Raises:
        KaggleError: Nếu notebook lỗi, output không có cảnh, hoặc cảnh đã có trên máy.
    """
    client = client or KaggleClient()
    kernel = f"{client.username()}/{request.kernel_name()}"
    try:
        status = client.kernel_status(kernel)
    except KaggleError:
        status = ""  # notebook chưa có
    commit = ""
    if status in ("", "error", "cancel_acknowledged"):
        commit = _push(client, request, kernel, branch)
        notify(f"đã đẩy notebook ingest {kernel}")
    status = _wait(client, kernel, timeout_s, sleep, notify)
    if status != "complete":
        raise KaggleError(
            f"notebook {kernel} kết thúc với trạng thái {status}; xem log trên Kaggle"
        )
    with tempfile.TemporaryDirectory(dir=workdir.root) as folder:
        output = Path(folder)
        client.kernel_output(kernel, output, file_pattern=LIGHT_FILES)
        result_path = next(output.rglob(RESULT_FILE), None)
        if result_path is None:
            raise KaggleError(f"output của {kernel} không có {RESULT_FILE}: ingest chưa xong")
        result = json.loads(result_path.read_text("utf-8"))
        scene_id = result["scene_id"]
        remote_dir = next(
            (p.parent for p in output.rglob("manifest.yaml") if p.parent.name == scene_id), None
        )
        if remote_dir is None:
            raise KaggleError(f"output của {kernel} không có manifest.yaml của {scene_id}")
        files.load(remote_dir / "manifest.yaml", Manifest)  # kiểm schema trước khi ghi
        scene_dir = workdir.scene_dir(scene_id)
        if (scene_dir / "manifest.yaml").exists():
            raise KaggleError(
                f"cảnh {scene_id} đã có ở {workdir.relative(scene_dir)}; xoá bằng "
                f"d2s scene delete {scene_id} rồi chạy lại"
            )
        scene_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_dir / "manifest.yaml", scene_dir / "manifest.yaml")
        if (remote_dir / "reports").is_dir():
            shutil.copytree(remote_dir / "reports", scene_dir / "reports", dirs_exist_ok=True)
    ref = KaggleSceneRef(
        username=kernel.split("/")[0],
        kernel=kernel,
        source=request.source,
        commit=commit or result.get("commit", ""),
        pulled_utc=now(),
    )
    (scene_dir / KAGGLE_SCENE_FILE).write_text(json.dumps(asdict(ref), indent=2), "utf-8")
    return scene_dir
