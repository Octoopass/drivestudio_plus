"""Chạy một run trên Kaggle: đẩy lên, hỏi trạng thái, tải kết quả về (ARCHITECTURE mục 2.8).

Luồng:
1. ``push_run``: upload cảnh thành dataset private (một lần cho mỗi cảnh và phiên bản converter),
   sinh notebook từ ``templates/run.ipynb`` có nhúng gói job, đẩy notebook; Kaggle chạy ngay.
   Ghi ``d2s/kaggle.json`` trong thư mục run để biết notebook nào đang chạy run này.
2. ``kernel_state``: hỏi trạng thái notebook.
3. ``pull_run``: tải output, ghép ``runs/`` và ``cache/`` vào thư mục làm việc trên máy. Bản trên
   Kaggle thắng ở mọi stage nó có; stage chỉ có trên máy (ví dụ ``export.preview`` chạy trên máy
   sau lần tải trước) được giữ lại và ghép vào ``run.json``; ``env.json`` giữ cả các máy đã chạy
   trên máy này.

Phiên Kaggle bị ngắt giữa chừng thì ``pull_run`` rồi ``push_run`` lần nữa: lần đẩy sau gắn output
của notebook trước làm đầu vào, nên stage đã xong không chạy lại và train chạy tiếp từ
``checkpoint_resume.pth``.

Notebook cài code bằng ``git clone`` đúng commit của repo, nên commit phải có trên remote;
``push_run`` từ chối nếu chưa push.

File này chứa ``KaggleState``, ``push_run``, ``kernel_state``, ``pull_run``.
"""

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Optional

from drive2sim import paths
from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine import remote
from drive2sim.engine.approaches import load_approach
from drive2sim.engine.cache import ARTIFACT_FILE
from drive2sim.engine.registry import produces_artifact
from drive2sim.engine.run import (
    D2S_DIR,
    RUN_FILE,
    run_is_active,
)
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle import names
from drive2sim.kaggle.bundle import write_bundle
from drive2sim.kaggle.client import KaggleClient, KaggleError

if TYPE_CHECKING:
    from drive2sim.kaggle.ingest import KaggleSceneRef

STATE_FILE = "kaggle.json"
TEMPLATE = paths.PACKAGE_DIR / "kaggle" / "templates" / "run.ipynb"
STAGING_DIR = ".kaggle"  # trong thư mục làm việc: thư mục tạm khi upload, tải về
DEFAULT_BRANCH = "drive2sim"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"  # machine_shape của Kaggle: GPU T4 x2
MAX_RESUME_SOURCES = 5  # số lần đẩy trước được gắn vào notebook mới
WHEEL_CANDIDATES = 5  # số notebook mới nhất thử khi tìm nguồn wheel
MAX_PARALLEL_RUNS = 2  # số run mỗi notebook: một run một GPU
PARALLEL_ACCELERATOR = "NvidiaTeslaT4"  # máy Kaggle có 2 GPU (T4 x2)
DATASET_WAIT_S = 900  # chờ Kaggle xử lý xong dataset cảnh vừa upload, tối đa 15 phút
DATASET_POLL_S = 10
SMPL_DATASET_ENV = "D2S_KAGGLE_SMPL_DATASET"  # dataset private chứa SMPL_NEUTRAL.pkl
REPO_ENV = "D2S_KAGGLE_REPO"  # đổi repo notebook clone (mặc định remote origin)


@dataclass
class KaggleState:
    """Nội dung ``d2s/kaggle.json``: run này đang ở notebook nào trên Kaggle.

    Attributes:
        username: Người dùng Kaggle.
        kernel: Notebook của lần đẩy mới nhất, ``<người dùng>/<tên>``.
        attempt: Lần đẩy thứ mấy (1, 2...); mỗi lần là một notebook riêng.
        dataset: Dataset chứa cảnh, hoặc notebook ingest giữ cảnh (cảnh ingest trên Kaggle).
        commit: Commit của code mà notebook dùng.
        pushed_utc: Thời điểm đẩy.
        pulled_utc: Thời điểm tải kết quả về gần nhất.
        kernels: Notebook của mọi lần đẩy, cũ trước mới sau (``kernel`` là phần tử cuối); để
            chạy tiếp gắn đúng output các lần trước dù quy tắc đặt tên đổi.
        remote_status: Trạng thái notebook ``kernel`` lần hỏi Kaggle gần nhất (``running``,
            ``complete``...), để Studio hiện run đang chạy hay đã xong trên Kaggle trước khi tải
            về; ``None`` nếu chưa hỏi được.
        checked_utc: Thời điểm hỏi đó.
    """

    username: str
    kernel: str
    attempt: int
    dataset: str
    commit: str
    pushed_utc: str
    pulled_utc: Optional[str] = None
    kernels: list[str] = field(default_factory=list)
    remote_status: Optional[str] = None
    checked_utc: Optional[str] = None

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


def _other_account(client: KaggleClient, state: KaggleState) -> Optional[str]:
    """Lý do khi lệnh ``kaggle`` đang đăng nhập tài khoản khác tài khoản đã đẩy run.

    ``None`` nếu cùng tài khoản hoặc không biết đang đăng nhập ai. Notebook và dataset của tool
    là private, nên chỉ tài khoản đã đẩy run mới xem, tải được.
    """
    username, _ = client.login()
    if username is None or username == state.username:
        return None
    return (
        f"run đã đẩy bằng tài khoản Kaggle {state.username} (notebook private), còn lệnh kaggle "
        f"đang đăng nhập {username}: đổi lại khoá API của {state.username} để xem và tải run này"
    )


def record_status(run_dir: Path, kernel: str, status: str) -> None:
    """Ghi trạng thái vừa hỏi được của notebook ``kernel`` vào ``kaggle.json``.

    Bỏ qua nếu run đã được đẩy lại sang notebook khác trong lúc hỏi.
    """
    state = load_state(run_dir)
    if state is None or state.kernel != kernel:
        return
    state.remote_status, state.checked_utc = status, _now()
    _save_state(run_dir, state)


def _save_state(run_dir: Path, state: KaggleState) -> None:
    (run_dir / D2S_DIR / STATE_FILE).write_text(json.dumps(asdict(state), indent=2), "utf-8")


def _git(*args: str) -> str:
    """Chạy git trong repo chứa package; trả stdout, chuỗi rỗng nếu lỗi (``remote.git``)."""
    return remote.git(*args)


def code_source(branch: str = DEFAULT_BRANCH) -> tuple[str, str]:
    """Commit hiện tại và URL repo mà notebook sẽ clone (``D2S_KAGGLE_REPO`` hoặc ``origin``).

    Kiểm tra theo ``remote.code_source``: commit phải có trên đúng remote notebook clone và remote
    phải có nhánh ``branch``.

    Args:
        branch: Nhánh notebook clone.

    Returns:
        ``(commit đầy đủ, url)``.

    Raises:
        KaggleError: Nếu không đọc được git, remote không có nhánh ``branch``, hoặc commit chưa
            push lên remote đó.
    """
    try:
        return remote.code_source(branch, REPO_ENV, _git)
    except remote.RemoteError as error:
        raise KaggleError(str(error)) from None


def _previous_kernels(previous: Optional[KaggleState], run_id: str) -> list[str]:
    """Notebook của các lần đẩy trước, cũ trước mới sau.

    ``kaggle.json`` ghi từ trước khi có trường ``kernels`` thì dựng lại tên theo quy tắc cũ
    ``d2s-<hash8 của run_id>-<lần đẩy>``.
    """
    if previous is None:
        return []
    if previous.kernels:
        return list(previous.kernels)
    return [
        f"{previous.username}/d2s-{_hash8(run_id)}-{number}"
        for number in range(1, previous.attempt + 1)
    ]


def cache_kernels(
    client: KaggleClient, workdir: Workdir, record: RunRecord, username: str
) -> list[str]:
    """Notebook của run khác đã có kết quả các stage cache (mask, SMPL) mà run này cần.

    Mỗi run Kaggle tính lại mask và SMPL từ đầu (cỡ 1,5 giờ GPU với nuScenes ``scene-0061``) dù
    output notebook của run trước cùng cảnh đã có đúng kết quả đó trong ``cache/``. Gắn output
    ấy vào notebook mới thì ``materialize`` chép cache sang và engine dùng lại.

    Chọn theo ``run.json`` trên máy (run đã tải về): run khác cùng cảnh, đẩy bằng cùng tài khoản
    (notebook private), có stage cùng ``stage_id``, cùng tham số, đã xong và có ``cache_key``. Đây
    chỉ là phép chọn sơ bộ: trên Kaggle engine vẫn so khoá cache đầy đủ (gồm phiên bản stage và
    đầu vào), khớp mới dùng lại, không khớp thì chạy như thường. Notebook không còn (đã xoá) hoặc
    đang chạy thì bỏ qua.

    Returns:
        Định danh notebook, mỗi notebook một lần, run mới nhất trước.
    """
    return _cache_sources(client, workdir, record, username)[0]


def _cache_sources(
    client: KaggleClient, workdir: Workdir, record: RunRecord, username: str
) -> tuple[list[str], set[str]]:
    """Như ``cache_kernels``, kèm các ``stage_id`` chưa notebook nào cùng tài khoản có cache."""
    wanted = {stage.stage_id: stage.params for stage in record.stages}
    missing = set(wanted)
    found: list[str] = []
    scene_runs = workdir.runs_dir / record.scene_id
    # Tên thư mục run bắt đầu bằng thời điểm tạo, nên sắp theo tên là run mới nhất trước.
    states = scene_runs.glob(f"*/*/{D2S_DIR}/{STATE_FILE}")
    for state_path in sorted(states, key=lambda path: path.parents[1].name, reverse=True):
        run_dir = state_path.parents[1]
        if not missing:
            break
        if run_dir.name == record.run_id:
            continue
        state = load_state(run_dir)
        if state is None or state.username != username or state.kernel in found:
            continue
        try:
            other = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
        except (OSError, ValueError):
            continue
        covers = {
            stage.stage_id
            for stage in other.stages
            if stage.stage_id in missing
            and stage.cache_key
            and stage.status == "succeeded"
            and stage.params == wanted[stage.stage_id]
        }
        if not covers:
            continue
        try:
            status = client.kernel_status(state.kernel)
        except KaggleError:
            continue
        if status not in ("complete", "error"):
            continue
        found.append(state.kernel)
        missing -= covers
    return found, missing


def cache_dataset(
    client: KaggleClient, workdir: Workdir, record: RunRecord, username: str, stage_ids: set[str]
) -> Optional[str]:
    """Dataset private chứa cache trên máy (mask, SMPL) cho các stage ``stage_ids`` của run.

    Output notebook chỉ tài khoản đã chạy mới đọc được, nên tài khoản mới (chưa có run nào)
    phải tính lại mask, SMPL (khoảng 1,6 giờ GPU với nuScenes ``scene-0061``) dù máy đã có sẵn.
    Ở đây upload kết quả cache trên máy (vài chục MB) thành dataset của tài khoản đang đăng
    nhập, gắn vào notebook; ``materialize`` chép ``cache/<cảnh>/<khoá>/`` sang như với output
    notebook, engine vẫn so khoá.

    Chọn khoá như ``cache_kernels``: stage của run khác cùng cảnh, cùng ``stage_id``, cùng tham
    số, đã xong, có ``cache_key`` mà máy có thư mục cache đầy đủ (``artifact.json``). Tên dataset
    gồm băm các khoá, nên dataset đã có thì dùng lại, không upload lại.

    Returns:
        Định danh dataset, hoặc ``None`` nếu máy không có cache nào dùng được.
    """
    wanted = {s.stage_id: s.params for s in record.stages if s.stage_id in stage_ids}
    keys: set[str] = set()
    for run_path in (workdir.runs_dir / record.scene_id).glob(f"*/*/{D2S_DIR}/{RUN_FILE}"):
        try:
            other = files.load(run_path, RunRecord)
        except (OSError, ValueError):
            continue
        for stage in other.stages:
            if (
                stage.stage_id in wanted
                and stage.cache_key
                and stage.status == "succeeded"
                and stage.params == wanted[stage.stage_id]
                and (workdir.cache_dir / stage.cache_key / ARTIFACT_FILE).is_file()
            ):
                keys.add(stage.cache_key)
    if not keys:
        return None
    slug = f"{username}/" + names.cache_dataset(record.scene_id, _hash8("|".join(sorted(keys))))
    if client.dataset_exists(slug):
        wait_dataset_ready(client, slug)
        return slug
    staging = workdir.root / STAGING_DIR / "upload" / slug.split("/")[1]
    shutil.rmtree(staging, ignore_errors=True)
    for key in sorted(keys):
        target = staging / "cache" / key
        shutil.copytree(workdir.cache_dir / key, target, copy_function=_link_or_copy)
    metadata = {
        "title": f"d2s cache {record.scene_id}"[:50],
        "id": slug,
        "description": "Kết quả cache (mask, SMPL) của drive2sim, dùng lại giữa các run.",
        "licenses": [{"name": "other"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    try:
        client.create_dataset(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    wait_dataset_ready(client, slug)
    return slug


def _link_or_copy(src: str, dst: str) -> None:
    """Hard link nếu được (không tốn thêm dung lượng), không thì chép."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_scene_refs(scene_root: Path) -> list["KaggleSceneRef"]:
    """``kaggle_scene.json`` của cảnh (đọc trễ để tránh vòng import với ``kaggle.ingest``)."""
    from drive2sim.kaggle.ingest import load_scene_refs

    return load_scene_refs(scene_root)


def _ensure_scene_dataset(
    client: KaggleClient, workdir: Workdir, username: str, scene_id: str, converter_version: str
) -> str:
    """Upload cảnh thành dataset private nếu chưa có; trả định danh dataset.

    Tên dataset gồm băm của ``scene_id`` và phiên bản converter, nên ingest lại bằng converter
    khác thì upload dataset mới. Thư mục upload dựng bằng hard link khi được, để không tốn gấp
    đôi dung lượng.
    """
    slug = f"{username}/{names.scene_dataset(scene_id, converter_version)}"
    # Dataset đặt tên theo quy tắc cũ (băm) vẫn dùng lại, để không upload lại cảnh đã có.
    legacy = f"{username}/d2s-{_hash8(f'{scene_id}@{converter_version}')}"
    for existing in (slug, legacy):
        if client.dataset_exists(existing):
            wait_dataset_ready(client, existing)
            return existing
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
    wait_dataset_ready(client, slug)
    return slug


def wait_dataset_ready(
    client: KaggleClient,
    slug: str,
    timeout_s: float = DATASET_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Chờ Kaggle xử lý xong dataset (giải nén, lập chỉ mục) trước khi đẩy notebook gắn nó.

    Dataset vừa upload còn ở trạng thái xử lý; notebook chạy ngay lúc đó không thấy cảnh dưới
    ``/kaggle/input`` và lỗi ở bước tìm cảnh.

    Raises:
        KaggleError: Nếu Kaggle báo dataset lỗi, hoặc quá ``timeout_s`` vẫn chưa xong.
    """
    waited = 0.0
    while True:
        try:
            status = client.dataset_status(slug)
        except KaggleError as error:
            # Ngay sau khi tạo, Kaggle có thể chưa nhận ra dataset và trả 403 cho lệnh hỏi trạng
            # thái (đã gặp với Kaggle CLI 2.2.5); hỏi lại tới khi hết thời gian chờ.
            if waited >= timeout_s:
                raise
            status = f"chưa hỏi được trạng thái: {error}"
        if "ready" in status:
            return
        if "error" in status:
            raise KaggleError(f"Kaggle báo dataset {slug} lỗi khi xử lý: {status}")
        if waited >= timeout_s:
            raise KaggleError(
                f"dataset {slug} vẫn đang xử lý sau {int(timeout_s)} giây ({status}); đẩy lại sau"
            )
        sleep(DATASET_POLL_S)
        waited += DATASET_POLL_S


def build_notebook(bundles: dict[str, dict], commit: str, repo_url: str, branch: str) -> str:
    """Điền template notebook.

    Args:
        bundles: ``{run_id: gói job}``, một hoặc hai run.
        commit: Commit notebook clone.
        repo_url: Repo để clone.
        branch: Branch để clone.

    Returns:
        Nội dung file ``.ipynb``.
    """
    values = {
        "run_ids": ", ".join(bundles),
        "commit": commit,
        "repo_url": repo_url,
        "branch": branch,
        "bundles_json": json.dumps(bundles),
    }
    notebook = json.loads(TEMPLATE.read_text("utf-8"))
    for cell in notebook["cells"]:
        cell["source"] = [Template(line).safe_substitute(values) for line in cell["source"]]
    return json.dumps(notebook, indent=1, ensure_ascii=False)


@dataclass
class _Prepared:
    """Một run đã sẵn sàng đẩy: gói job và nguồn cần gắn vào notebook."""

    run_id: str
    run_dir: Path
    record: RunRecord
    attempt: int
    history: list[str]
    dataset: str
    bundle: dict
    datasets: list[str]
    kernels: list[str]


def _prepare(client: KaggleClient, workdir: Workdir, run_id: str, username: str) -> _Prepared:
    """Kiểm tra run đẩy được, upload cảnh (và cache cho tài khoản mới) nếu cần, dựng gói job."""
    run_dir = workdir.find_run(run_id)
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    if record.status == "succeeded":
        raise KaggleError(f"run {run_id} đã xong; không cần đẩy lại")
    previous = load_state(run_dir)
    if previous is not None:
        status = client.kernel_status(previous.kernel)
        if status in ("queued", "running", "new", "new_script", "cancel_requested"):
            raise KaggleError(f"notebook {previous.kernel} còn đang {status}; chờ xong rồi đẩy")
    if previous is not None and previous.username != username:
        # Lần đẩy sau chạy tiếp từ output của các notebook trước, private của tài khoản cũ.
        raise KaggleError(
            f"run đã đẩy bằng tài khoản Kaggle {previous.username}, còn lệnh kaggle đang đăng "
            f"nhập {username}: tài khoản mới không đọc được notebook cũ để chạy tiếp. Đổi lại "
            f"khoá của {previous.username}, hoặc xoá {D2S_DIR}/{STATE_FILE} của run để chạy lại "
            f"từ đầu bằng {username}"
        )
    scene = workdir.load_scene(record.scene_id)
    scene_kernels: list[str] = []
    remote_scenes = _load_scene_refs(scene.root)
    if remote_scenes:
        # Cảnh đã ingest trên Kaggle (d2s kaggle ingest): gắn output notebook ingest của chính
        # tài khoản đang đăng nhập (notebook private), không upload cảnh.
        remote_scene = next((ref for ref in remote_scenes if ref.username == username), None)
        if remote_scene is None:
            owners = ", ".join(ref.username for ref in remote_scenes)
            raise KaggleError(
                f"cảnh {record.scene_id} mới ingest trên Kaggle bằng tài khoản {owners}, còn lệnh "
                f"kaggle đang đăng nhập {username}: notebook private nên tài khoản này không đọc "
                f"được; chạy d2s kaggle ingest bằng {username} trước (cùng tham số)"
            )
        dataset = remote_scene.kernel
        scene_kernels = [remote_scene.kernel]
        sources: list[str] = []
    else:
        dataset = _ensure_scene_dataset(
            client, workdir, username, record.scene_id, scene.manifest.source.converter_version
        )
        sources = [dataset]
    # Run có dựng người (prep.smpl) thì trainer cần file mô hình SMPL, file có bản quyền nên không
    # đi kèm repo: người dùng tự đưa nó lên một dataset private và đặt tên dataset ở biến này.
    if produces_artifact((stage.stage_id for stage in record.stages), "smpl"):
        smpl_dataset = os.environ.get(SMPL_DATASET_ENV, "").strip()
        if not smpl_dataset:
            raise KaggleError(
                "run dựng người (prep.smpl) cần file mô hình SMPL trên Kaggle: upload "
                "SMPL_NEUTRAL.pkl thành một dataset private rồi đặt biến "
                f"{SMPL_DATASET_ENV}=<người dùng>/<tên dataset> trước khi đẩy"
            )
        sources.append(smpl_dataset)
    history = _previous_kernels(previous, run_id)
    # Gắn output của tối đa MAX_RESUME_SOURCES lần đẩy trước, mới nhất trước: lần gần nhất có thể
    # lỗi trước khi ghi được gì (ví dụ lúc cài môi trường), khi đó notebook chạy tiếp từ lần cũ hơn
    # còn kết quả thay vì train lại từ đầu.
    resume = list(reversed(history))[:MAX_RESUME_SOURCES]
    bundle = write_bundle(run_dir, workdir, dataset, resume)
    # Gắn thêm output của run khác đã có mask, SMPL mà run này cần, để khỏi tính lại.
    kernels, missing = _cache_sources(client, workdir, record, username)
    # Tài khoản chưa có notebook nào có cache đó (ví dụ tài khoản mới): upload cache trên máy.
    if missing and not resume:
        cached = cache_dataset(client, workdir, record, username, missing)
        if cached and cached not in sources:
            sources.append(cached)
    return _Prepared(
        run_id=run_id,
        run_dir=run_dir,
        record=record,
        attempt=previous.attempt + 1 if previous else 1,
        history=history,
        dataset=dataset,
        bundle=bundle,
        datasets=sources,
        kernels=scene_kernels + resume + kernels,
    )


def _unique(items: Sequence[str]) -> list[str]:
    """Bỏ phần tử lặp, giữ thứ tự."""
    return list(dict.fromkeys(items))


def push_run(
    workdir: Workdir,
    run_id: str,
    client: Optional[KaggleClient] = None,
    branch: str = DEFAULT_BRANCH,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> KaggleState:
    """Đẩy một run lên Kaggle và cho chạy (``push_runs`` với một run)."""
    return push_runs(workdir, [run_id], client, branch, accelerator)[0]


def push_runs(
    workdir: Workdir,
    run_ids: Sequence[str],
    client: Optional[KaggleClient] = None,
    branch: str = DEFAULT_BRANCH,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> list[KaggleState]:
    """Đẩy một run, hoặc hai run chung một notebook, lên Kaggle và cho chạy.

    Lần đẩy đầu chạy từ đầu; các lần sau chạy tiếp từ output của notebook trước. Hai run thì
    notebook chạy song song, mỗi run một GPU (máy ``NvidiaTeslaT4`` của Kaggle có 2 GPU T4);
    quota GPU tính theo giờ phiên, nên hai run tốn giờ như một. Mỗi run giữ ``kaggle.json`` riêng
    (cùng notebook); đẩy lại thì mỗi run chạy tiếp từ output của chính nó, đẩy riêng hay đẩy
    đôi đều được.

    Args:
        workdir: Thư mục làm việc trên máy.
        run_ids: Một hoặc hai run đã tạo (``executor: kaggle``).
        client: Kaggle; mặc định gọi lệnh ``kaggle``.
        branch: Branch để clone.
        accelerator: ``machine_shape`` của notebook, ví dụ ``NvidiaTeslaT4``, ``NvidiaL4``.

    Returns:
        Trạng thái đã ghi vào ``d2s/kaggle.json`` của từng run, cùng thứ tự ``run_ids``.

    Raises:
        KaggleError: Nếu chưa cài hoặc chưa đăng nhập Kaggle, commit chưa push, run đã xong, hoặc
            notebook trước còn đang chạy, hoặc lần đẩy trước dùng tài khoản Kaggle khác; thêm khi
            số run không hợp lệ, hoặc hai run mà máy không phải ``NvidiaTeslaT4``.
    """
    run_ids = list(run_ids)
    if not 1 <= len(run_ids) <= MAX_PARALLEL_RUNS or len(set(run_ids)) != len(run_ids):
        raise KaggleError(f"đẩy được 1 tới {MAX_PARALLEL_RUNS} run khác nhau mỗi notebook")
    if len(run_ids) > 1 and accelerator != PARALLEL_ACCELERATOR:
        raise KaggleError(
            f"chạy song song cần máy {PARALLEL_ACCELERATOR} (2 GPU), không phải {accelerator}"
        )
    client = client or KaggleClient()
    username = client.username()
    commit, repo_url = code_source(branch)
    prepared = [_prepare(client, workdir, run_id, username) for run_id in run_ids]
    if len(prepared) == 1:
        one = prepared[0]
        defaults = {
            name: option.default
            for name, option in load_approach(one.record.approach.approach_id).options.items()
        }
        name = names.train_kernel(
            one.record.scene_id, one.run_id, one.attempt, one.record.approach.options, defaults
        )
    else:
        name = names.pair_kernel(
            prepared[0].record.scene_id, run_ids, [p.attempt for p in prepared]
        )
    kernel = f"{username}/{name}"
    # Output có wheel đã build (khỏi build lại môi trường khoảng 25 phút).
    wheels = wheel_kernel(client, workdir, username)
    datasets = _unique([d for p in prepared for d in p.datasets])
    kernels = _unique([k for p in prepared for k in p.kernels] + ([wheels] if wheels else []))

    with tempfile.TemporaryDirectory(dir=_staging(workdir)) as folder:
        folder_path = Path(folder)
        notebook = build_notebook({p.run_id: p.bundle for p in prepared}, commit, repo_url, branch)
        (folder_path / "run.ipynb").write_text(notebook, "utf-8")
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
            "dataset_sources": datasets,
            "kernel_sources": kernels,
            "competition_sources": [],
        }
        (folder_path / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        client.push_kernel(folder_path)

    states = []
    for p in prepared:
        state = KaggleState(
            username, kernel, p.attempt, p.dataset, commit, _now(), kernels=p.history + [kernel]
        )
        _save_state(p.run_dir, state)
        states.append(state)
    return states


def _staging(workdir: Workdir) -> Path:
    path = workdir.root / STAGING_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def kernel_state(workdir: Workdir, run_id: str, client: Optional[KaggleClient] = None) -> dict:
    """Trạng thái notebook của một run.

    Returns:
        Nội dung ``kaggle.json`` cộng ``status`` (trạng thái Kaggle, ``None`` nếu không hỏi được)
        và ``status_error`` (lý do không hỏi được, hoặc ``None``).

    Raises:
        KaggleError: Nếu run chưa đẩy lên Kaggle.
    """
    client = client or KaggleClient()
    state = load_state(workdir.find_run(run_id))
    if state is None:
        raise KaggleError(f"run {run_id} chưa đẩy lên Kaggle; chạy d2s kaggle push {run_id}")
    # Không hỏi được Kaggle (chưa cài lệnh, hết hạn khoá, mất mạng) vẫn trả thông tin đã lưu trong
    # kaggle.json, để Studio còn hiện notebook nào, đẩy lúc nào và cho tải kết quả về.
    try:
        status, error = client.kernel_status(state.kernel), None
        record_status(workdir.find_run(run_id), state.kernel, status)
    except KaggleError as failure:
        status, error = None, _other_account(client, state) or str(failure)
    return {**asdict(state), "status": status, "status_error": error}


def _merge_d2s(local: Path, remote_d2s: Path) -> list[str]:
    """Ghép ``d2s/`` tải từ Kaggle vào ``d2s/`` của run trên máy (``remote.merge_d2s``).

    Bản Kaggle thắng ở mọi stage nó có; stage chỉ có trên máy được giữ lại; ``env.json`` cộng các
    máy chỉ có trên máy; ``kaggle.json`` trên máy giữ nguyên; ``pid``, ``cancel`` không chép về.

    Returns:
        Các ``stage_id`` chỉ có trên máy đã được giữ lại.
    """
    return remote.merge_d2s(local, remote_d2s)


# Notebook train in các file kết quả nhỏ vào log sau chữ này (base64 của JSON
# {đường dẫn: nội dung}), templates/run.ipynb ô cuối.
RUN_LIGHT_MARKER = "D2S_RUN_LIGHT"


def pull_run_light(
    workdir: Workdir, run_id: str, client: Optional[KaggleClient] = None
) -> RunRecord:
    """Tải nhanh file kết quả nhỏ của run (run.json, summary, metric, thống kê) từ log notebook.

    Một lệnh API (``kaggle kernels logs``) thay vì tải cả output: output của một run có hàng nghìn
    file (cache mặt nạ), tải mất cỡ một giờ và dễ bị 429. Ghép như ``pull_run``; không đánh dấu đã
    tải về (``pulled_utc``), nên ``pull_run`` (hoặc Studio) vẫn tải đủ sau.

    Raises:
        KaggleError: Như ``pull_run``; thêm khi run đã tải đủ (ghép bản nhỏ sẽ thay các stage đầy
            đủ trên máy) hoặc log không có kết quả (notebook theo template cũ, hoặc chưa xong).
    """
    client = client or KaggleClient()
    run_dir = workdir.find_run(run_id)
    state = load_state(run_dir)
    if state is None:
        raise KaggleError(f"run {run_id} chưa đẩy lên Kaggle")
    if state.pulled_utc is not None:
        raise KaggleError(f"run {run_id} đã tải đủ kết quả về; không cần --light")
    reason = _other_account(client, state)
    if reason:
        raise KaggleError(reason)
    match = re.search(RUN_LIGHT_MARKER + r" ([A-Za-z0-9+/=]+)", client.kernel_logs(state.kernel))
    if match is None:
        raise KaggleError(
            f"log của {state.kernel} không có kết quả nhỏ (notebook chưa xong, hoặc theo template "
            f"cũ); tải đủ bằng d2s kaggle pull {run_id}"
        )
    light = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
    with tempfile.TemporaryDirectory(dir=_staging(workdir)) as folder:
        output = Path(folder)
        for name, text in light.items():
            target = (output / name).resolve()
            if output.resolve() not in target.parents:  # chỉ ghi bên trong thư mục tạm
                raise KaggleError(f"đường dẫn lạ trong kết quả của {state.kernel}: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, "utf-8")
        try:
            remote.merge_output(workdir, run_dir, output)
        except FileNotFoundError:
            raise KaggleError(f"kết quả nhỏ của {state.kernel} không có run {run_id}") from None
    return files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)


def skip_local_cache(workdir: Workdir) -> Optional[str]:
    """Biểu thức cho ``--file-pattern`` bỏ qua phần output máy không cần tải.

    Output của notebook chứa cả ``cache/<cảnh>/<stage>@<khoá>/`` (mask, SMPL: hàng nghìn file nhỏ)
    và ``wheels/`` (gói đã build để notebook sau dùng lại, vài trăm MB, máy không dùng). Lệnh
    ``kaggle`` tải từng file một, nên tải lại cache đã có tốn hàng chục phút; ``merge_output`` vốn
    cũng bỏ các mục cache đã có.
    """
    entries = sorted(p for p in workdir.cache_dir.glob("*/*") if p.is_dir())
    skipped = "|".join(
        ["wheels/"] + [re.escape(f"cache/{p.parent.name}/{p.name}/") for p in entries]
    )
    return f"^(?!.*(?:{skipped}))"


def wheel_kernel(client: KaggleClient, workdir: Workdir, username: str) -> Optional[str]:
    """Notebook mới nhất đã chạy xong của cùng tài khoản, để notebook mới lấy wheel đã build.

    Notebook nào cũng lưu wheel của gsplat-geer, nvdiffrast, pytorch3d... vào ``wheels/`` trong
    output (``envs/install_main.sh``, biến ``D2S_WHEEL_CACHE``): wheel tự build hoặc wheel lấy lại
    từ notebook trước. Gắn notebook mới nhất là đủ; nó chưa có wheel (chạy trước khi có cơ chế này)
    thì notebook mới build như cũ rồi lưu. Duyệt ``kaggle.json`` của mọi run trên máy, mới nhất
    trước; bỏ notebook đã xoá hoặc chưa xong.
    """
    states = []
    for state_path in workdir.runs_dir.glob(f"*/*/*/{D2S_DIR}/{STATE_FILE}"):
        try:
            state = load_state(state_path.parents[1])
        except (OSError, ValueError, TypeError):
            continue
        if state is not None and state.username == username:
            states.append(state)
    for state in sorted(states, key=lambda s: s.pushed_utc, reverse=True)[:WHEEL_CANDIDATES]:
        try:
            status = client.kernel_status(state.kernel)
        except KaggleError:
            continue
        if status in ("complete", "error"):
            return state.kernel
    return None


def pull_run(workdir: Workdir, run_id: str, client: Optional[KaggleClient] = None) -> RunRecord:
    """Tải output của notebook và ghép vào thư mục làm việc.

    Ghép ``d2s/`` của run theo ``_merge_d2s``: bản Kaggle thắng ở mọi stage nó có, stage chỉ có
    trên máy được giữ lại. Chép thêm các kết quả cache mà máy chưa có; cache đã có không tải lại
    (``skip_local_cache``).

    Returns:
        ``run.json`` sau khi ghép.

    Raises:
        KaggleError: Nếu run chưa đẩy lên Kaggle, đang chạy trên máy, lệnh ``kaggle`` đang đăng
            nhập tài khoản khác tài khoản đã đẩy run, hoặc output không có run này.
    """
    client = client or KaggleClient()
    run_dir = workdir.find_run(run_id)
    if run_is_active(run_dir):
        raise KaggleError(f"run {run_id} đang chạy trên máy (ví dụ đang xuất); chờ xong rồi tải")
    state = load_state(run_dir)
    if state is None:
        raise KaggleError(f"run {run_id} chưa đẩy lên Kaggle")
    reason = _other_account(client, state)
    if reason:
        raise KaggleError(reason)
    with tempfile.TemporaryDirectory(dir=_staging(workdir)) as folder:
        output = Path(folder)
        client.kernel_output(state.kernel, output, skip_local_cache(workdir))
        try:
            remote.merge_output(workdir, run_dir, output)
        except FileNotFoundError:
            relative = run_dir.relative_to(workdir.root)
            raise KaggleError(f"output của {state.kernel} không có {relative}") from None
        # Notebook chạy hai run (push_runs): ghép luôn run kia từ cùng output, khỏi tải lại.
        for other_dir, other in _runs_on_kernel(workdir, state.kernel):
            if other_dir == run_dir or run_is_active(other_dir):
                continue
            try:
                remote.merge_output(workdir, other_dir, output)
            except FileNotFoundError:
                continue
            other.pulled_utc = _now()
            _save_state(other_dir, other)
    state.pulled_utc = _now()
    _save_state(run_dir, state)
    return files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)


def _runs_on_kernel(workdir: Workdir, kernel: str) -> list[tuple[Path, KaggleState]]:
    """Các run trên máy có lần đẩy mới nhất là notebook ``kernel``."""
    found = []
    for state_path in workdir.runs_dir.glob(f"*/*/*/{D2S_DIR}/{STATE_FILE}"):
        run_dir = state_path.parents[1]
        try:
            state = load_state(run_dir)
        except (OSError, ValueError, TypeError):
            continue
        if state is not None and state.kernel == kernel:
            found.append((run_dir, state))
    return found
