"""Ingest một cảnh ngay trên Kaggle, từ dữ liệu gốc đã nằm trên Kaggle (CONTRACTS mục 9.4).

Dữ liệu gốc lớn (ví dụ KITTI-360 đủ khung, khoảng 8 GB một chuỗi) không cần nằm trên máy: nó là
một dataset private hoặc output của một notebook trên Kaggle. ``d2s kaggle ingest`` đẩy một
notebook chỉ dùng CPU gắn nguồn đó, notebook chạy ``d2s scene ingest`` rồi giữ cảnh trong output
của nó. Trên máy chỉ còn các file nhỏ của cảnh (``manifest.yaml``, ``reports/``) và file
``kaggle_scene.json`` ghi notebook nào giữ cảnh; ``d2s kaggle push`` gắn output notebook đó vào
notebook train thay vì upload cảnh.

Luồng của ``ingest_on_kaggle``:
1. Tên notebook ``<người dùng>/d2s-ingest-<nguồn>-<chuỗi>`` (``kaggle/names.py``), nên gọi lại
   cùng tham số thì dùng lại notebook đã chạy (chờ tiếp nếu đang chạy, tải về nếu đã xong).
2. Đẩy notebook (template ``templates/ingest.ipynb``) khi chưa có hoặc lần trước lỗi.
3. Chờ notebook xong, đọc các file nhỏ của cảnh từ log notebook (``read_light_files``), ghi vào
   ``data/<scene_id>/`` cùng ``kaggle_scene.json``.

File này chứa ``KaggleSceneRef``, ``IngestRequest``, ``load_scene_refs``, ``load_scene_ref``,
``build_ingest_notebook``, ``ingest_on_kaggle``.
"""

import base64
import json
import re
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
from drive2sim.kaggle import names
from drive2sim.kaggle.client import KaggleClient, KaggleError
from drive2sim.kaggle.executor import code_source

TEMPLATE = paths.PACKAGE_DIR / "kaggle" / "templates" / "ingest.ipynb"
RESULT_FILE = "ingest_result.json"  # notebook ghi scene_id vào output
# Notebook in file nhỏ của cảnh (kết quả, manifest.yaml, reports/*.json) vào log, sau chữ này,
# dạng base64 của JSON; ảnh, LiDAR, mặt nạ ở lại Kaggle.
LIGHT_MARKER = "D2S_INGEST_LIGHT"
POLL_S = 60
WAIT_S = 6 * 3600  # notebook ingest lâu nhất 6 giờ thì thôi chờ (chạy lại lệnh để chờ tiếp)
_RUNNING = ("queued", "running", "new", "new_script", "cancel_requested")


@dataclass
class KaggleSceneRef:
    """Một mục của ``data/<scene_id>/kaggle_scene.json``: notebook ingest của một tài khoản.

    Attributes:
        username: Tài khoản Kaggle sở hữu notebook (notebook private).
        kernel: Notebook ingest, ví dụ ``<người dùng>/d2s-ingest-nuscenes-mini-0061``.
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

    def option(self, flag: str) -> Optional[str]:
        """Giá trị của một tuỳ chọn trong ``args`` (ví dụ ``--sequence``), hoặc ``None``."""
        args = list(self.args)
        return args[args.index(flag) + 1] if flag in args[:-1] else None

    def kernel_name(self) -> str:
        """Tên notebook (không có người dùng), theo ``--scene`` hoặc nguồn và ``--sequence``."""
        return names.ingest_kernel(self.option("--scene"), self.option("--sequence"), self.source)

    def signature(self) -> dict:
        """Nguồn và tham số; notebook ghi lại để biết lần ingest trước có cùng yêu cầu không."""
        return {"source": self.source, "path": self.path, "args": list(self.args)}


def load_scene_refs(scene_root: Path) -> list[KaggleSceneRef]:
    """Các notebook ingest giữ cảnh, mỗi tài khoản một notebook; rỗng nếu cảnh nằm trên máy.

    Notebook private nên mỗi tài khoản Kaggle cần notebook ingest của riêng nó
    (``kaggle_scene.json`` là ``{"kernels": [...]}``, CONTRACTS mục 9.4).
    """
    path = scene_root / KAGGLE_SCENE_FILE
    if not path.is_file():
        return []
    return [KaggleSceneRef(**entry) for entry in json.loads(path.read_text("utf-8"))["kernels"]]


def load_scene_ref(scene_root: Path, username: str) -> Optional[KaggleSceneRef]:
    """Notebook ingest của ``username`` giữ cảnh; ``None`` nếu tài khoản này chưa ingest."""
    return next((ref for ref in load_scene_refs(scene_root) if ref.username == username), None)


def _same_scene(local: Manifest, remote: Manifest) -> bool:
    """Hai manifest mô tả cùng dữ liệu (bỏ qua phiên bản converter và thời điểm ingest)."""

    def comparable(manifest: Manifest) -> dict:
        data = manifest.model_dump(mode="json")
        data["source"].pop("converter_version", None)
        data.pop("created_utc", None)
        return data

    return comparable(local) == comparable(remote)


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
        "signature": json.dumps(request.signature()),
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


def read_light_files(client: KaggleClient, kernel: str) -> Optional[dict]:
    """File nhỏ của cảnh mà notebook ingest in vào log (dòng ``LIGHT_MARKER``).

    Không dùng ``kaggle kernels output``: lệnh đó liệt kê mọi file của output trước khi lọc,
    mà output của một cảnh lớn có hàng chục nghìn file, đủ để Kaggle trả 429 (đã gặp với
    KITTI-360 0003). Đọc log chỉ tốn một lệnh API.

    Returns:
        ``{"result": <ingest_result.json>, "files": {đường dẫn trong cảnh: nội dung}}``, hoặc
        ``None`` nếu log không có dòng đánh dấu (notebook theo template cũ, hoặc chưa xong).
    """
    match = re.search(LIGHT_MARKER + r" ([A-Za-z0-9+/=]+)", client.kernel_logs(kernel))
    if match is None:
        return None
    return json.loads(base64.b64decode(match.group(1)).decode("utf-8"))


def _signature(light: dict) -> dict:
    """Nguồn và tham số mà notebook ingest đã chạy."""
    return {key: light["result"].get(key) for key in ("source", "path", "args")}


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
        KaggleError: Nếu notebook lỗi, output không có cảnh, hoặc cảnh đã có trên máy mà không
            phải cảnh ingest trên Kaggle với cùng dữ liệu (chạy bằng tài khoản thứ hai thì thêm
            notebook của tài khoản đó vào ``kaggle_scene.json``).
    """
    client = client or KaggleClient()
    kernel = f"{client.username()}/{request.kernel_name()}"
    try:
        status = client.kernel_status(kernel)
    except KaggleError:
        status = ""  # notebook chưa có
    commit = ""
    light = read_light_files(client, kernel) if status == "complete" else None
    if status == "complete" and light is None:
        # Notebook theo template cũ (không in file nhỏ vào log): chạy lại, chỉ tốn CPU.
        notify(f"{kernel} chưa có kết quả trong log, chạy lại notebook")
        status = ""
    if light is not None and _signature(light) != request.signature():
        # Cùng tên nhưng lần trước ingest với nguồn hoặc tham số khác: chạy lại sẽ ghi đè output
        # mà run cũ có thể đang dùng, nên từ chối.
        raise KaggleError(
            f"notebook {kernel} đã ingest với nguồn hoặc tham số khác; đặt --scene <tên khác> để "
            "có notebook riêng"
        )
    if status in ("", "error", "cancel_acknowledged"):
        commit = _push(client, request, kernel, branch)
        notify(f"đã đẩy notebook ingest {kernel}")
        light = None
    status = _wait(client, kernel, timeout_s, sleep, notify)
    if status != "complete":
        raise KaggleError(
            f"notebook {kernel} kết thúc với trạng thái {status}; xem log trên Kaggle"
        )
    light = light or read_light_files(client, kernel)
    if light is None:
        raise KaggleError(f"log của {kernel} không có kết quả ingest ({LIGHT_MARKER})")
    result = light["result"]
    scene_id = result["scene_id"]
    if "manifest.yaml" not in light["files"]:
        raise KaggleError(f"kết quả của {kernel} không có manifest.yaml của {scene_id}")
    with tempfile.TemporaryDirectory(dir=workdir.root) as folder:
        remote_dir = Path(folder)
        for name, text in light["files"].items():
            target = (remote_dir / name).resolve()
            if remote_dir.resolve() not in target.parents:  # chỉ ghi bên trong thư mục tạm
                raise KaggleError(f"đường dẫn lạ trong kết quả của {kernel}: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, "utf-8")
        remote_manifest = files.load(remote_dir / "manifest.yaml", Manifest)  # kiểm schema
        scene_dir = workdir.scene_dir(scene_id)
        refs = load_scene_refs(scene_dir)
        if (scene_dir / "manifest.yaml").exists():
            # Cảnh đã có: chỉ nhận thêm notebook của tài khoản khác cho cùng dữ liệu.
            if not refs:
                raise KaggleError(
                    f"cảnh {scene_id} đã có trên máy ở {workdir.relative(scene_dir)}; xoá bằng "
                    f"d2s scene delete {scene_id}, hoặc đặt --scene khác"
                )
            if not _same_scene(files.load(scene_dir / "manifest.yaml", Manifest), remote_manifest):
                raise KaggleError(
                    f"cảnh {scene_id} trên máy khác cảnh trong {kernel} (camera, khung...); "
                    "đặt --scene khác"
                )
        else:
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
    refs = [other for other in refs if other.username != ref.username] + [ref]
    content = {"kernels": [asdict(entry) for entry in refs]}
    (scene_dir / KAGGLE_SCENE_FILE).write_text(json.dumps(content, indent=2), "utf-8")
    return scene_dir
