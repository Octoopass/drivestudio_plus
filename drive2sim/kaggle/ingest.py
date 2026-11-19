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
3. Chờ notebook xong, đọc các file nhỏ của cảnh từ log notebook (``read_light_files``).
4. Manifest trong đó ghi phiên bản đầu ra cũ hơn phiên bản hiện tại của converter
   (``outdated_output_version``): giữ nguyên notebook cũ (run cũ còn gắn output của nó), làm lại
   bước 2–3 với notebook ``...-c<phiên bản hiện tại>``.
5. Ghi các file nhỏ vào ``data/<scene_id>/`` cùng ``kaggle_scene.json``.

File này chứa ``KaggleSceneRef``, ``IngestRequest``, ``load_scene_refs``, ``load_scene_ref``,
``build_ingest_notebook``, ``read_light_files``, ``ingest_on_kaggle``.
"""

import json
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Template
from typing import Optional

import yaml

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import Manifest
from drive2sim.engine.workdir import KAGGLE_SCENE_FILE, Workdir
from drive2sim.ingest.manifest import outdated_output_version
from drive2sim.kaggle import names
from drive2sim.kaggle.client import KaggleClient, KaggleError
from drive2sim.kaggle.executor import code_source
from drive2sim.kaggle.scene_files import missing_view_files, read_log_lines, write_scene_files

TEMPLATE = paths.PACKAGE_DIR / "kaggle" / "templates" / "ingest.ipynb"
SCENE_FILES_TEMPLATE = paths.PACKAGE_DIR / "kaggle" / "templates" / "scene_files.ipynb"
RESULT_FILE = "ingest_result.json"  # notebook ghi scene_id vào output
# Notebook in file nhỏ của cảnh (kết quả, manifest.yaml, reports/*.json, pose, calib:
# kaggle/scene_files.py) vào log, sau chữ này, dạng base64 của JSON; ảnh, LiDAR, mặt nạ ở lại
# Kaggle.
LIGHT_MARKER = "D2S_INGEST_LIGHT"
# Notebook lấy bù file nhỏ (fetch_scene_files) in ``{"scene_id", "files"}`` sau chữ này.
SCENE_FILES_MARKER = "D2S_SCENE_FILES"
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

    def kernel_name(self, output_version: int = 1) -> str:
        """Tên notebook (không có người dùng), theo ``--scene`` hoặc nguồn và ``--sequence``.

        Args:
            output_version: Phiên bản đầu ra của converter; lớn hơn 1 thì tên có thêm
                ``-c<phiên bản>`` (``names.ingest_kernel``).
        """
        return names.ingest_kernel(
            self.option("--scene"), self.option("--sequence"), self.source, output_version
        )

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
        ``None`` nếu log không có dòng đánh dấu (notebook theo template cũ, hoặc chưa xong) hoặc
        dữ liệu trong log hỏng (khi đó đẩy lại notebook).
    """
    try:
        return read_log_lines(LIGHT_MARKER, client.kernel_logs(kernel))
    except ValueError:
        return None


def _signature(light: dict) -> dict:
    """Nguồn và tham số mà notebook ingest đã chạy."""
    return {key: light["result"].get(key) for key in ("source", "path", "args")}


def _outdated(light: dict) -> Optional[tuple[int, int]]:
    """Notebook đã ingest bằng converter cũ chưa, theo ``manifest.yaml`` trong file nhỏ của nó.

    Returns:
        ``(phiên bản đầu ra trong manifest, phiên bản hiện tại)`` nếu cũ; ``None`` nếu mới,
        converter không có phiên bản đầu ra, hoặc kết quả không có manifest (bước sau báo lỗi).
    """
    text = light["files"].get("manifest.yaml")
    if text is None:
        return None
    return outdated_output_version(Manifest.model_validate(yaml.safe_load(text)).source)


def _finish_kernel(
    client: KaggleClient,
    request: IngestRequest,
    kernel: str,
    branch: str,
    timeout_s: float,
    sleep: Callable[[float], None],
    notify: Callable[[str], None],
) -> tuple[dict, str]:
    """Dùng lại, chờ hoặc đẩy một notebook ingest tới khi nó xong, rồi đọc file nhỏ của cảnh.

    Notebook đã xong thì chỉ đọc log; đang chạy thì chờ; chưa có, lỗi, hoặc xong theo template cũ
    (log không có file nhỏ) thì đẩy.

    Returns:
        ``(file nhỏ, commit)``: file nhỏ như ``read_light_files``; commit của code nếu lần này đẩy
        notebook, chuỗi rỗng nếu dùng lại.

    Raises:
        KaggleError: Nếu notebook cùng tên đã ingest với nguồn hoặc tham số khác, notebook kết
            thúc không thành công, hoặc log không có file nhỏ.
    """
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
    return light, commit


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

    Notebook theo tên của ``request`` đã ingest bằng converter cũ (phiên bản đầu ra trong
    manifest của nó khác phiên bản hiện tại) thì không đẩy lại notebook đó, vì đẩy lại sẽ thay
    output mà run cũ đang gắn; thay vào đó dùng notebook ``...-c<phiên bản hiện tại>``
    (``names.ingest_kernel``). Gọi lại cùng lệnh thì đi đúng con đường đó và dùng lại notebook
    mới.

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
        KaggleError: Nếu notebook lỗi, output không có cảnh, notebook mới vẫn ingest bằng
            converter cũ, hoặc cảnh đã có trên máy mà không phải cảnh ingest trên Kaggle với
            cùng dữ liệu (chạy bằng tài khoản thứ hai thì thêm notebook của tài khoản đó vào
            ``kaggle_scene.json``; cảnh trên máy ingest bằng converter cũ thì báo cách thay).
    """
    client = client or KaggleClient()
    username = client.username()
    kernel = f"{username}/{request.kernel_name()}"
    light, commit = _finish_kernel(client, request, kernel, branch, timeout_s, sleep, notify)
    outdated = _outdated(light)
    if outdated is not None:
        newer = f"{username}/{request.kernel_name(outdated[1])}"
        if newer == kernel:
            raise KaggleError(
                f"notebook {kernel} ingest bằng đầu ra phiên bản {outdated[0]}, mới hơn converter "
                f"hiện tại ({outdated[1]}); cập nhật code rồi chạy lại"
            )
        notify(
            f"{kernel} ingest bằng converter cũ (đầu ra phiên bản {outdated[0]}, hiện tại "
            f"{outdated[1]}); giữ nguyên notebook đó cho run cũ, dùng notebook {newer}"
        )
        kernel = newer
        light, commit = _finish_kernel(client, request, kernel, branch, timeout_s, sleep, notify)
        outdated = _outdated(light)
        if outdated is not None:
            raise KaggleError(
                f"notebook {kernel} vẫn ingest bằng converter cũ (đầu ra phiên bản "
                f"{outdated[0]}, hiện tại {outdated[1]}); kiểm tra commit đẩy lên đã có converter "
                "mới chưa"
            )
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
            local_manifest = files.load(scene_dir / "manifest.yaml", Manifest)
            local_outdated = outdated_output_version(local_manifest.source)
            if local_outdated is not None:
                # Không tự thay cảnh trên máy: split, cache, run cũ được làm từ dữ liệu cũ.
                raise KaggleError(
                    f"cảnh {scene_id} trên máy được ingest bằng converter cũ (đầu ra phiên bản "
                    f"{local_outdated[0]}); {kernel} đã ingest lại bằng phiên bản "
                    f"{local_outdated[1]}. Xoá cảnh cũ bằng d2s scene delete {scene_id} (xoá cả "
                    "split, cache, run của nó; notebook cũ trên Kaggle giữ nguyên) rồi chạy lại "
                    "lệnh, hoặc đặt --scene khác"
                )
            if not _same_scene(local_manifest, remote_manifest):
                raise KaggleError(
                    f"cảnh {scene_id} trên máy khác cảnh trong {kernel} (camera, khung...); "
                    "đặt --scene khác"
                )
        else:
            scene_dir.mkdir(parents=True, exist_ok=True)
    # Cảnh mới: ghi mọi file nhỏ. Cảnh đã có (tài khoản thứ hai): chỉ thêm file còn thiếu (pose,
    # calib của cảnh ingest trước khi notebook in chúng), giữ nguyên manifest, split trên máy.
    write_scene_files(scene_dir, light["files"])
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


def files_kernel(ingest_kernel: str) -> str:
    """Tên notebook lấy bù file nhỏ: ``<user>/d2s-ingest-x`` thành ``<user>/d2s-files-x``.

    Ngắn hơn tên notebook ingest nên không vượt giới hạn 50 ký tự của Kaggle.
    """
    user, name = ingest_kernel.split("/", 1)
    return f"{user}/{name.replace('d2s-ingest', 'd2s-files', 1)}"


def build_scene_files_notebook(
    scene_id: str, ingest_kernel: str, commit: str, repo_url: str, branch: str
) -> str:
    """Điền template notebook lấy bù file nhỏ của cảnh.

    Returns:
        Nội dung file ``.ipynb``.
    """
    values = {
        "scene_id": scene_id,
        "ingest_kernel": ingest_kernel,
        "commit": commit,
        "repo_url": repo_url,
        "branch": branch,
    }
    notebook = json.loads(SCENE_FILES_TEMPLATE.read_text("utf-8"))
    for cell in notebook["cells"]:
        cell["source"] = [Template(line).safe_substitute(values) for line in cell["source"]]
    return json.dumps(notebook, indent=1, ensure_ascii=False)


def _read_scene_files(client: KaggleClient, kernel: str) -> Optional[dict]:
    """``{"scene_id", "files"}`` mà notebook lấy bù in vào log.

    Returns:
        Payload; ``None`` nếu log chưa có, hoặc dữ liệu hỏng (notebook ghi theo dạng một dòng cũ,
        bị stdout chen vào): khi đó đẩy lại notebook.
    """
    try:
        return read_log_lines(SCENE_FILES_MARKER, client.kernel_logs(kernel))
    except ValueError:
        return None


def fetch_scene_files(
    workdir: Workdir,
    scene_id: str,
    client: Optional[KaggleClient] = None,
    branch: str = "drive2sim",
    timeout_s: float = WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
    notify: Callable[[str], None] = lambda message: None,
) -> list[str]:
    """Tải pose, calib của một cảnh ingest trên Kaggle về máy, không ingest lại.

    Cảnh ingest trước khi notebook ingest in pose, calib vào log chỉ có ``manifest.yaml``,
    ``reports/`` trên máy, nên không xem được model bằng renderer gốc trên máy. Hàm đẩy notebook
    CPU ``<user>/d2s-files-...`` gắn output notebook ingest của tài khoản đang đăng nhập; notebook
    in file nhỏ của cảnh vào log (``kaggle/scene_files.py``), máy đọc log rồi ghi file còn thiếu.
    Dùng lại notebook đã chạy xong.

    Args:
        workdir: Thư mục làm việc trên máy.
        scene_id: Cảnh cần lấy bù.
        client: Kaggle; mặc định gọi lệnh ``kaggle``.
        branch: Branch notebook clone.
        timeout_s: Chờ notebook tối đa bao lâu.
        sleep: Hàm chờ giữa hai lần hỏi trạng thái (test thay).
        notify: Nhận từng thay đổi trạng thái để in ra.

    Returns:
        Đường dẫn các file vừa ghi trong cảnh; rỗng nếu cảnh đã đủ.

    Raises:
        KaggleError: Nếu cảnh không phải cảnh ingest trên Kaggle, tài khoản đang đăng nhập không có
            notebook ingest của cảnh, notebook lỗi, hoặc sau khi ghi cảnh vẫn thiếu pose, calib.
    """
    scene_dir = workdir.scene_dir(scene_id)
    refs = load_scene_refs(scene_dir)
    if not refs:
        raise KaggleError(
            f"cảnh {scene_id} không phải cảnh ingest trên Kaggle (không có {KAGGLE_SCENE_FILE})"
        )
    if not missing_view_files(scene_dir):
        return []
    client = client or KaggleClient()
    username = client.username()
    ref = load_scene_ref(scene_dir, username)
    if ref is None:
        owners = ", ".join(entry.username for entry in refs)
        raise KaggleError(
            f"tài khoản {username} không có notebook ingest của cảnh {scene_id}; đăng nhập một "
            f"trong các tài khoản: {owners}"
        )
    kernel = files_kernel(ref.kernel)
    try:
        status = client.kernel_status(kernel)
    except KaggleError:
        status = ""  # notebook chưa có
    light = _read_scene_files(client, kernel) if status == "complete" else None
    if light is None and status not in _RUNNING:
        commit, repo_url = code_source(branch)
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            (folder_path / "scene_files.ipynb").write_text(
                build_scene_files_notebook(scene_id, ref.kernel, commit, repo_url, branch),
                "utf-8",
            )
            metadata = {
                "id": kernel,
                "title": kernel.split("/")[1],
                "code_file": "scene_files.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "kernel_sources": [ref.kernel],
                "competition_sources": [],
            }
            (folder_path / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
            client.push_kernel(folder_path)
        notify(f"đã đẩy notebook {kernel}")
    if light is None:
        status = _wait(client, kernel, timeout_s, sleep, notify)
        if status != "complete":
            raise KaggleError(
                f"notebook {kernel} kết thúc với trạng thái {status}; xem log trên Kaggle"
            )
        light = _read_scene_files(client, kernel)
    if light is None or light.get("scene_id") != scene_id:
        raise KaggleError(f"log của {kernel} không có file nhỏ của cảnh {scene_id}")
    try:
        written = write_scene_files(scene_dir, light["files"])
    except ValueError as error:
        raise KaggleError(f"{kernel}: {error}") from None
    still = missing_view_files(scene_dir)
    if still:
        raise KaggleError(f"cảnh {scene_id} trong {kernel} vẫn thiếu: {', '.join(still)}")
    return written
