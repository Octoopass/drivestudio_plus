"""Chạy một run trên VM Google Colab: đẩy lên, theo dõi log, tải kết quả về (CONTRACTS mục 9.3).

Executor này đi theo đúng khuôn của executor Kaggle (``drive2sim/kaggle/executor.py``) để cùng một
run cho ra cùng kết quả: cùng gói job (``drive2sim/kaggle/bundle.py``), cùng lệnh chạy trên máy
kia (``d2s kaggle run``), cùng cách cài môi trường (``envs/install_main.sh``), cùng cách ghép
output tải về (``drive2sim/engine/remote.py``). Khác Kaggle ở cách đưa mọi thứ lên máy chạy:

1. ``push_run``: tạo hoặc dùng lại phiên Colab có GPU (mặc định phiên ``d2s``, GPU T4); hỏi VM
   đã có gì (cảnh, run, file SMPL, job đang chạy); upload cảnh thành một file tar cắt thành từng
   phần (một lần cho mỗi cảnh và phiên bản converter trong một phiên), upload ``SMPL_NEUTRAL.pkl``
   nếu run dựng người, upload bản run và cache trên máy nếu VM chưa có run (chạy tiếp sau khi mất
   phiên); rồi chép ``templates/job.sh`` và gói job lên VM và chạy job **tách khỏi kết nối**
   (log ra file, mã thoát ghi ra file ``.exit``). Ghi ``d2s/colab.json`` trong thư mục run.
2. ``job_state``, ``follow_log``: đọc trạng thái và log của job theo từng đoạn ngắn, mỗi lần một
   lệnh ``colab exec`` vài giây, nên kết nối rớt chỉ làm mất một lần hỏi, job vẫn chạy.
3. ``pull_run``: trên VM gói thư mục run và cache của run thành tar cắt từng phần, tải về, kiểm
   tra sha256, ghép vào ``runs/`` và ``cache/`` như Kaggle. Run xong thì tắt phiên (trả VM) trừ khi
   còn run khác của phiên đó chưa tải về.

Phiên Colab mất giữa chừng (VM bị thu hồi): mọi thứ trên VM mất theo, gồm kết quả chưa tải về.
``job_state`` và ``pull_run`` báo rõ và ghi ``lost_utc``; ``push_run`` lần nữa tạo phiên mới,
upload lại cảnh và bản run đã tải về gần nhất, nên stage đã tải về không chạy lại.

File này chứa ``ColabState``, ``push_run``, ``job_state``, ``follow_log``, ``pull_run``,
``stop_session``, ``watched_runs``.
"""

import hashlib
import json
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.colab.client import GPUS, ColabClient, ColabError, ColabSessionLostError
from drive2sim.core import files
from drive2sim.core.schema import RunRecord
from drive2sim.engine import remote
from drive2sim.engine.run import D2S_DIR, RUN_FILE, run_is_active
from drive2sim.engine.workdir import Workdir
from drive2sim.kaggle.bundle import write_bundle

STATE_FILE = "colab.json"
TEMPLATE = paths.PACKAGE_DIR / "colab" / "templates" / "job.sh"
STAGING_DIR = ".colab"  # trong thư mục làm việc: thư mục tạm khi đóng gói, tải về
REMOTE_ROOT = "/content/d2s"  # trên VM: repo/, venv/, work/, input/, upload/, jobs/, out/
DEFAULT_BRANCH = "drive2sim"
DEFAULT_SESSION = "d2s"
DEFAULT_GPU = "T4"
REPO_ENV = "D2S_COLAB_REPO"  # đổi repo VM clone (mặc định remote origin)
# Upload và download đi qua API nội dung của Jupyter, cả file trong một request (base64), nên cắt
# file lớn thành phần 32 MiB. Giới hạn kích thước request của Colab chưa kiểm tra; 32 MiB là mức
# an toàn chọn trước, chưa đo.
CHUNK_BYTES = 32 * 1024 * 1024
TRANSFER_ATTEMPTS = 3
LOG_TAIL_LINES = 40


@dataclass
class ColabState:
    """Nội dung ``d2s/colab.json``: run này đang chạy ở job nào, trên phiên Colab nào.

    Attributes:
        session: Tên phiên Colab (``colab -s``).
        gpu: GPU của phiên, ví dụ ``T4``.
        attempt: Lần đẩy thứ mấy (1, 2...).
        job: Tên job trên VM, ``<run_id>-<lần đẩy>``; log ở ``/content/d2s/jobs/<job>.log``.
        commit: Commit của code mà job dùng.
        branch: Nhánh đã clone.
        pushed_utc: Thời điểm đẩy.
        pulled_utc: Thời điểm tải kết quả về gần nhất.
        lost_utc: Thời điểm phát hiện phiên đã mất (sau lần đẩy mới nhất), nếu có.
        stopped_utc: Thời điểm tool tắt phiên, nếu có.
    """

    session: str
    gpu: str
    attempt: int
    job: str
    commit: str
    branch: str
    pushed_utc: str
    pulled_utc: Optional[str] = None
    lost_utc: Optional[str] = None
    stopped_utc: Optional[str] = None

    def to_dict(self) -> dict:
        """Nội dung dạng dict, để ghi JSON."""
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def load_state(run_dir: Path) -> Optional[ColabState]:
    """Đọc ``d2s/colab.json``; ``None`` nếu run chưa từng đẩy lên Colab."""
    path = run_dir / D2S_DIR / STATE_FILE
    return ColabState(**json.loads(path.read_text("utf-8"))) if path.is_file() else None


def _save_state(run_dir: Path, state: ColabState) -> None:
    (run_dir / D2S_DIR / STATE_FILE).write_text(json.dumps(asdict(state), indent=2), "utf-8")


def code_source(branch: str = DEFAULT_BRANCH) -> tuple[str, str]:
    """Commit hiện tại và URL repo mà VM sẽ clone (``D2S_COLAB_REPO`` hoặc ``origin``).

    Kiểm tra như Kaggle (``remote.code_source``): commit phải có trên đúng remote đó.

    Raises:
        ColabError: Nếu không đọc được git hoặc commit chưa push.
    """
    try:
        return remote.code_source(branch, REPO_ENV)
    except remote.RemoteError as error:
        raise ColabError(str(error)) from None


def _staging(workdir: Workdir) -> Path:
    path = workdir.root / STAGING_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ mã Python chạy trên VM

# Kiểm tra job còn chạy: pid còn và không phải zombie. Tiến trình job là con của kernel; xong mà
# kernel chưa thu dọn thì pid vẫn còn ở trạng thái Z, os.kill(pid, 0) vẫn thành công.
_ALIVE = """
def alive(pid_file):
    try:
        pid = int(pid_file.read_text().split()[0])
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def busy_job(jobs):
    for pid_file in sorted(jobs.glob("*.pid")) if jobs.is_dir() else []:
        if not (jobs / (pid_file.stem + ".exit")).exists() and alive(pid_file):
            return pid_file.stem
    return None
"""

_PROBE = """
root = pathlib.Path(cfg["root"])
print("@@PROBE " + json.dumps({
    "busy": busy_job(root / "jobs"),
    "has_scene": (root / "input" / cfg["scene_key"]).is_dir(),
    "has_run": (root / "work" / cfg["run_rel"] / "d2s" / "run.json").is_file(),
    "has_smpl": (root / "work" / "smpl_models" / "SMPL_NEUTRAL.pkl").is_file(),
}))
"""

_PREPARE_PARTS = """
parts = pathlib.Path(cfg["parts"])
shutil.rmtree(parts, ignore_errors=True)
parts.mkdir(parents=True)
pathlib.Path(cfg["target"]).parent.mkdir(parents=True, exist_ok=True)
print("@@READY")
"""

# Ghép các phần đã upload, kiểm tra sha256, rồi (với tar) giải nén. Giải nén vào thư mục tạm rồi
# mới đổi tên, nên lần chạy lỗi giữa chừng không để lại cảnh dở mà lần sau tưởng là đủ.
_ASSEMBLE = """
def assemble():
    parts = pathlib.Path(cfg["parts"])
    target = pathlib.Path(cfg["target"])
    if not parts.is_dir():
        # Lần gọi trước đã ghép xong nhưng kết nối rớt trước khi máy nhận được kết quả.
        dest = pathlib.Path(cfg["extract_to"] or target.parent)
        names = cfg["paths"] if cfg["extract_to"] else [target.name]
        return "OK" if all((dest / name).exists() for name in names) else "MISSING"
    digest = hashlib.sha256()
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as out:
        for index in range(cfg["count"]):
            data = (parts / f"part{index:04d}").read_bytes()
            digest.update(data)
            out.write(data)
    shutil.rmtree(parts, ignore_errors=True)
    if digest.hexdigest() != cfg["sha256"]:
        tmp.unlink()
        return "BAD " + digest.hexdigest()
    if cfg["extract_to"] is None:
        os.replace(tmp, target)
        return "OK"
    dest = pathlib.Path(cfg["extract_to"])
    staging = dest.with_name(dest.name + ".extract")
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(tmp) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(staging, filter="data")
        else:
            tar.extractall(staging)
    tmp.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    # Chỉ thêm mục chưa có: cảnh đã có thì giữ, run đã có trên VM (mới hơn) thì giữ.
    for rel in cfg["paths"]:
        source, final = staging / rel, dest / rel
        if source.exists() and not final.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, final)
    shutil.rmtree(staging, ignore_errors=True)
    return "OK"


print("@@" + assemble())
"""

_LAUNCH = """
root = pathlib.Path(cfg["root"])
jobs = root / "jobs"
jobs.mkdir(parents=True, exist_ok=True)
busy = busy_job(jobs)
if busy:
    print("@@BUSY " + busy)
else:
    job = cfg["job"]
    bundle = root / "bundles" / job
    bundle.mkdir(parents=True, exist_ok=True)
    for name, content in cfg["bundle"].items():
        (bundle / name).write_text(json.dumps(content, indent=2))
    script = jobs / (job + ".sh")
    script.write_text(cfg["script"])
    log, exit_file = jobs / (job + ".log"), jobs / (job + ".exit")
    exit_file.unlink(missing_ok=True)
    args = [str(script), str(root), cfg["repo_url"], cfg["branch"], cfg["commit"], str(bundle),
            str(root / "input" / cfg["scene_key"])]
    command = (f"bash {shlex.join(args)} > {shlex.quote(str(log))} 2>&1; "
               f"echo $? > {shlex.quote(str(exit_file))}")
    # Môi trường của job: của kernel, bỏ PYTHONPATH (đường dẫn gói của Python Colab, không hợp với
    # venv Python 3.10 của job). Không dùng colab exec --env (biến giữ lại trong kernel).
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(["bash", "-c", command], cwd=str(root), env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    (jobs / (job + ".pid")).write_text(str(process.pid))
    print("@@LAUNCHED " + str(process.pid))
"""

# Đọc log từ offset (offset < 0: chỉ lấy các dòng cuối), chỉ trả dòng trọn vẹn trừ khi job đã xong
# và đã đọc hết; kèm mã thoát nếu có và job còn sống không.
_POLL = """
jobs = pathlib.Path(cfg["root"]) / "jobs"
job = cfg["job"]
log, exit_file, pid_file = (jobs / (job + suffix) for suffix in (".log", ".exit", ".pid"))
exit_code = None
if exit_file.exists():
    text = exit_file.read_text().strip()
    exit_code = int(text) if text.lstrip("-").isdigit() else 1
size = log.stat().st_size if log.exists() else 0
offset = cfg["offset"]
if offset < 0:
    offset = max(0, size - 65536)
data = b""
if log.exists():
    with open(log, "rb") as stream:
        stream.seek(offset)
        data = stream.read(cfg["max_bytes"])
more = offset + len(data) < size
if exit_code is None or more:
    data = data[: data.rfind(b"\\n") + 1]
lines = data.decode("utf-8", "replace").splitlines()
if cfg["offset"] < 0:
    lines = lines[-cfg["tail"]:]
print("@@STATE " + json.dumps({"known": pid_file.exists(), "alive": alive(pid_file),
                               "exit_code": exit_code, "offset": offset + len(data)}))
for line in lines:
    print("@@L " + line)
"""

_PACK = """
root = pathlib.Path(cfg["root"])
work, jobs, job = root / "work", root / "jobs", cfg["job"]
run = work / cfg["run_rel"]
if not (jobs / (job + ".exit")).exists() and alive(jobs / (job + ".pid")):
    print("@@BUSY " + job)
elif not (run / "d2s" / "run.json").is_file():
    print("@@NORUN")
else:
    record = json.loads((run / "d2s" / "run.json").read_text())
    members = [cfg["run_rel"]] + [
        "cache/" + stage["cache_key"] for stage in record.get("stages", [])
        if stage.get("cache_key") and (work / "cache" / stage["cache_key"]).is_dir()
    ]
    out = root / "out" / job
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    archive = out / "output.tar"
    with tarfile.open(archive, "w") as tar:
        for member in members:
            tar.add(work / member, arcname=member)
    digest, parts = hashlib.sha256(), []
    with open(archive, "rb") as stream:
        while True:
            data = stream.read(cfg["chunk"])
            if not data:
                break
            digest.update(data)
            part = out / f"part{len(parts):04d}"
            part.write_bytes(data)
            parts.append(str(part))
    archive.unlink()
    print("@@PACK " + json.dumps({"sha256": digest.hexdigest(), "parts": parts}))
"""

_CLEAN = """
shutil.rmtree(pathlib.Path(cfg["path"]), ignore_errors=True)
print("@@OK")
"""


def _snippet(body: str, config: dict) -> str:
    """Ghép một đoạn mã chạy trên VM: import, ``cfg`` (JSON nhúng sẵn), hàm chung, rồi thân."""
    header = (
        "import hashlib, json, os, pathlib, shlex, shutil, subprocess, tarfile\n"
        f"cfg = json.loads({json.dumps(config)!r})\n"
    )
    return header + _ALIVE + body


def _markers(output: str, name: str) -> list[str]:
    """Giá trị của các dòng ``@@<name> <giá trị>`` trong stdout của một đoạn mã."""
    prefix = f"@@{name}"
    return [line[len(prefix) :].strip() for line in output.splitlines() if line.startswith(prefix)]


def _run_on_vm(
    client: ColabClient,
    session: str,
    body: str,
    config: dict,
    timeout_s: float = 60,
    attempts: int = 3,
) -> str:
    """Chạy một đoạn mã trên VM và trả stdout (thử lại ``attempts`` lần khi kết nối lỗi)."""
    return client.exec_python(
        session, _snippet(body, config), timeout_s=timeout_s, attempts=attempts
    )


# ------------------------------------------------------------------ chuyển file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retry(action: Callable[[], None], attempts: int = TRANSFER_ATTEMPTS) -> None:
    """Chạy ``action``, thử lại khi lỗi Colab (trừ khi phiên đã mất)."""
    for attempt in range(attempts):
        try:
            action()
            return
        except ColabSessionLostError:
            raise
        except ColabError:
            if attempt + 1 == attempts:
                raise


def upload_file(
    client: ColabClient,
    session: str,
    local: Path,
    target: str,
    extract_to: Optional[str] = None,
    extract_paths: Iterable[str] = (),
) -> None:
    """Upload một file lên VM theo từng phần ``CHUNK_BYTES``, ghép lại và kiểm tra sha256.

    Với ``extract_to``, file là tar: sau khi ghép thì giải nén, và chỉ chuyển các mục
    ``extract_paths`` (đường dẫn tương đối trong tar) chưa có vào ``extract_to``.

    Args:
        client: Colab.
        session: Tên phiên.
        local: File trên máy.
        target: Đường dẫn file trên VM (với tar: nơi để file tạm khi ghép).
        extract_to: Thư mục giải nén trên VM, hoặc ``None`` để giữ nguyên file.
        extract_paths: Các mục cần chuyển vào ``extract_to``.

    Raises:
        ColabError: Nếu upload lỗi sau vài lần thử, hoặc sha256 trên VM không khớp.
    """
    sha = _sha256(local)
    parts = f"{REMOTE_ROOT}/upload/{sha[:16]}"
    prepared = _run_on_vm(client, session, _PREPARE_PARTS, {"parts": parts, "target": target})
    if not _markers(prepared, "READY"):
        raise ColabError(f"không tạo được thư mục upload trên VM: {prepared.strip()[-200:]}")
    count = 0
    with tempfile.TemporaryDirectory() as folder, open(local, "rb") as stream:
        part = Path(folder) / "part"
        while True:
            data = stream.read(CHUNK_BYTES)
            if not data and count > 0:
                break
            part.write_bytes(data)
            remote_part = f"{parts}/part{count:04d}"
            _retry(lambda: client.upload(session, part, remote_part))  # noqa: B023
            count += 1
            if not data:
                break
    config = {
        "parts": parts,
        "target": target,
        "count": count,
        "sha256": sha,
        "extract_to": extract_to,
        "paths": list(extract_paths),
    }
    result = _run_on_vm(client, session, _ASSEMBLE, config, timeout_s=1800)
    if not _markers(result, "OK"):
        raise ColabError(f"ghép {local.name} trên VM lỗi (sha256 khác?): {result.strip()[-200:]}")


def _make_tar(archive: Path, root: Path, members: Iterable[str]) -> None:
    """Gói các mục ``members`` (tương đối với ``root``) thành tar không nén, theo symlink."""
    with tarfile.open(archive, "w", dereference=True) as tar:
        for member in members:
            tar.add(root / member, arcname=member)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Giải nén tar tải về, chỉ nhận file và thư mục nằm bên trong ``destination``.

    Bỏ qua symlink, hard link và đường dẫn tuyệt đối hay có ``..``: output là kết quả của tool,
    không có những thứ này (job xoá symlink trước khi gói), nên gặp thì coi là rác.
    """
    with tarfile.open(archive) as tar:
        members = [
            member
            for member in tar.getmembers()
            if (member.isfile() or member.isdir())
            and not member.name.startswith("/")
            and ".." not in Path(member.name).parts
        ]
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, members=members, filter="data")
        else:
            tar.extractall(destination, members=members)  # noqa: S202 (đã lọc ở trên)


# ------------------------------------------------------------------ đẩy, theo dõi, tải về


def _ensure_session(client: ColabClient, session: str, gpu: str) -> str:
    """Dùng lại phiên ``session`` nếu đang có, không thì tạo mới với ``gpu``.

    Returns:
        GPU của phiên (phiên có sẵn có thể có GPU khác ``gpu``).

    Raises:
        ColabError: Nếu phiên có sẵn là CPU, hoặc không tạo được phiên.
    """
    for found in client.sessions():
        if found["name"] == session:
            if found["hardware"] not in GPUS:
                raise ColabError(
                    f"phiên Colab {session} đang có sẵn nhưng không có GPU "
                    f"({found['hardware'] or '?'}); tắt bằng colab stop -s {session} hoặc chọn "
                    "phiên khác"
                )
            return found["hardware"]
    client.create_session(session, gpu)
    return gpu


def _resume_members(workdir: Workdir, run_dir: Path, record: RunRecord) -> list[str]:
    """Các mục cần upload để chạy tiếp từ bản trên máy: thư mục run và cache run đã dùng.

    Chỉ khi run đã có kết quả trên máy (đã tải về từ lần trước); run mới tạo thì không cần gì,
    job dựng ``run.json`` từ gói job.
    """
    stages_dir = run_dir / D2S_DIR / remote.STAGES_DIR
    if not stages_dir.is_dir() or not any(stages_dir.iterdir()):
        return []
    members = [run_dir.relative_to(workdir.root).as_posix()]
    for stage in record.stages:
        if stage.cache_key and (workdir.cache_dir / stage.cache_key).is_dir():
            members.append(f"cache/{stage.cache_key}")
    return members


def push_run(
    workdir: Workdir,
    run_id: str,
    client: Optional[ColabClient] = None,
    branch: str = DEFAULT_BRANCH,
    gpu: str = DEFAULT_GPU,
    session: str = DEFAULT_SESSION,
) -> ColabState:
    """Đẩy một run lên VM Colab và cho chạy tách khỏi kết nối.

    Lần đẩy sau của cùng run chạy tiếp: nếu VM vẫn còn run (cùng phiên) thì chạy tiếp từ bản trên
    VM; nếu không (phiên mới) thì upload bản run đã tải về trên máy.

    Args:
        workdir: Thư mục làm việc trên máy.
        run_id: Run đã tạo (``executor: colab``).
        client: Colab; mặc định gọi lệnh ``colab``.
        branch: Nhánh để clone.
        gpu: GPU khi phải tạo phiên mới: ``T4`` (mặc định), ``L4``, ``G4``, ``A100``, ``H100``.
        session: Tên phiên Colab; các run dùng chung một phiên thì môi trường chỉ cài một lần.

    Returns:
        Trạng thái đã ghi vào ``d2s/colab.json``.

    Raises:
        ColabError: Nếu chưa cài hoặc chưa đăng nhập Colab, GPU lạ, commit chưa push, run đã xong,
            thiếu file SMPL, VM đang chạy job khác, hoặc upload lỗi.
    """
    client = client or ColabClient()
    if gpu not in GPUS:
        raise ColabError(f"GPU {gpu} không có trên Colab; chọn một trong {', '.join(GPUS)}")
    run_dir = workdir.find_run(run_id)
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    if record.status == "succeeded":
        raise ColabError(f"run {run_id} đã xong; không cần đẩy lại")
    if run_is_active(run_dir):
        raise ColabError(f"run {run_id} đang chạy trên máy; chờ xong rồi đẩy")
    commit, repo_url = code_source(branch)
    # Run dựng người (prep.smpl) cần file mô hình SMPL, có bản quyền nên không đi kèm repo: lấy
    # file trên máy (như run chạy trên máy) rồi upload riêng lên VM, không qua git.
    smpl_model = None
    if any(stage.stage_id == "prep.smpl" for stage in record.stages):
        smpl_model = paths.find_smpl_model(workdir.root)
        if smpl_model is None:
            raise ColabError(
                f"run dựng người (prep.smpl) cần file {paths.SMPL_MODEL_NAME} trên máy để upload "
                f"lên Colab: đặt ở <thư mục làm việc>/smpl_models/ hoặc trỏ biến "
                f"{paths.SMPL_MODEL_ENV} tới nó (tải ở https://smpl.is.tue.mpg.de)"
            )
    scene = workdir.load_scene(record.scene_id)
    scene_key = _hash8(f"{record.scene_id}@{scene.manifest.source.converter_version}")
    run_rel = run_dir.relative_to(workdir.root).as_posix()
    session_gpu = _ensure_session(client, session, gpu)

    probe_out = _run_on_vm(
        client, session, _PROBE, {"root": REMOTE_ROOT, "scene_key": scene_key, "run_rel": run_rel}
    )
    probes = _markers(probe_out, "PROBE")
    if not probes:
        raise ColabError(f"không đọc được trạng thái VM: {probe_out.strip()[-200:]}")
    probe = json.loads(probes[-1])
    if probe["busy"]:
        raise ColabError(
            f"phiên Colab {session} đang chạy job {probe['busy']}; chờ xong rồi đẩy, hoặc chọn "
            "phiên khác (--session)"
        )
    with tempfile.TemporaryDirectory(dir=_staging(workdir)) as folder:
        staging = Path(folder)
        if not probe["has_scene"]:
            archive = staging / "scene.tar"
            _make_tar(archive, workdir.scene_dir(record.scene_id).parent, [record.scene_id])
            upload_file(
                client,
                session,
                archive,
                f"{REMOTE_ROOT}/upload/scene-{scene_key}.tar",
                extract_to=f"{REMOTE_ROOT}/input/{scene_key}",
                extract_paths=[record.scene_id],
            )
            archive.unlink()
        resume = [] if probe["has_run"] else _resume_members(workdir, run_dir, record)
        if resume:
            archive = staging / "resume.tar"
            _make_tar(archive, workdir.root, resume)
            upload_file(
                client,
                session,
                archive,
                f"{REMOTE_ROOT}/upload/resume-{run_id}.tar",
                extract_to=f"{REMOTE_ROOT}/work",
                extract_paths=resume,
            )
            archive.unlink()
    if smpl_model is not None and not probe["has_smpl"]:
        upload_file(
            client, session, smpl_model, f"{REMOTE_ROOT}/work/smpl_models/{paths.SMPL_MODEL_NAME}"
        )

    previous = load_state(run_dir)
    attempt = previous.attempt + 1 if previous else 1
    job = f"{run_id}-{attempt}"
    config = {
        "root": REMOTE_ROOT,
        "job": job,
        "bundle": write_bundle(run_dir, workdir, None),
        "script": TEMPLATE.read_text("utf-8"),
        "repo_url": repo_url,
        "branch": branch,
        "commit": commit,
        "scene_key": scene_key,
    }
    launched = _run_on_vm(client, session, _LAUNCH, config)
    busy = _markers(launched, "BUSY")
    # BUSY với chính job này: lần gọi trước đã chạy job nhưng kết nối rớt trước khi trả lời.
    if busy and busy[0] != job:
        raise ColabError(f"phiên Colab {session} đang chạy job {busy[0]}; chờ xong rồi đẩy")
    if not busy and not _markers(launched, "LAUNCHED"):
        raise ColabError(f"không chạy được job trên VM: {launched.strip()[-200:]}")
    state = ColabState(session, session_gpu, attempt, job, commit, branch, _now())
    _save_state(run_dir, state)
    return state


def _require_state(workdir: Workdir, run_id: str) -> tuple[Path, ColabState]:
    run_dir = workdir.find_run(run_id)
    state = load_state(run_dir)
    if state is None:
        raise ColabError(f"run {run_id} chưa đẩy lên Colab; chạy d2s colab push {run_id}")
    return run_dir, state


def _lost_message(state: ColabState, run_id: str) -> str:
    return (
        f"phiên Colab {state.session} đã mất (VM bị thu hồi, hết giờ hoặc đã tắt); kết quả chưa "
        f"tải về của lần đẩy {state.attempt} mất theo VM. Chạy tiếp từ bản đã tải về trên máy: "
        f"d2s colab push {run_id}"
    )


def _mark_lost(run_dir: Path, state: ColabState) -> None:
    state.lost_utc = _now()
    _save_state(run_dir, state)


def _poll(
    client: ColabClient, state: ColabState, offset: int, max_bytes: int = 65536
) -> tuple[dict, list[str]]:
    """Một lần đọc trạng thái và log của job (``_POLL``)."""
    config = {
        "root": REMOTE_ROOT,
        "job": state.job,
        "offset": offset,
        "max_bytes": max_bytes,
        "tail": LOG_TAIL_LINES,
    }
    # Hỏi trạng thái chỉ thử một lần: nơi gọi (follow_log, Studio, vòng theo dõi) tự hỏi lại sau.
    output = _run_on_vm(client, state.session, _POLL, config, attempts=1)
    states = _markers(output, "STATE")
    if not states:
        raise ColabError(f"không đọc được trạng thái job {state.job}: {output.strip()[-200:]}")
    lines = [raw[4:] for raw in output.splitlines() if raw == "@@L" or raw.startswith("@@L ")]
    return json.loads(states[-1]), lines


def _job_status(info: dict) -> str:
    """Trạng thái job từ kết quả ``_POLL``.

    ``running`` (còn chạy), ``succeeded`` hoặc ``failed`` (mã thoát 0 hoặc khác 0), ``died``
    (tiến trình mất mà không ghi mã thoát), ``missing`` (VM không biết job này: phiên cùng tên
    nhưng là VM mới).
    """
    if not info["known"]:
        return "missing"
    if info["exit_code"] is not None:
        return "succeeded" if info["exit_code"] == 0 else "failed"
    return "running" if info["alive"] else "died"


def job_state(workdir: Workdir, run_id: str, client: Optional[ColabClient] = None) -> dict:
    """Trạng thái job Colab của một run, kèm các dòng log cuối.

    Returns:
        Nội dung ``colab.json`` cộng ``status`` (``running``, ``succeeded``, ``failed``, ``died``,
        ``lost``, ``stopped`` khi phiên đã tắt sau khi tải về, hoặc ``None`` khi không hỏi
        được), ``exit_code``, ``log_tail`` (tối đa
        ``LOG_TAIL_LINES`` dòng) và ``status_error`` (lý do không hỏi được, hoặc ``None``).

    Raises:
        ColabError: Nếu run chưa đẩy lên Colab.
    """
    client = client or ColabClient()
    run_dir, state = _require_state(workdir, run_id)

    def answer(status: Optional[str], error: Optional[str] = None, **extra: object) -> dict:
        return {**asdict(state), "status": status, "exit_code": None, "log_tail": [],
                "status_error": error, **extra}  # fmt: skip

    if state.lost_utc and state.lost_utc >= state.pushed_utc:
        return answer("lost", _lost_message(state, run_id))
    if state.stopped_utc and state.stopped_utc >= state.pushed_utc:
        # Tool đã tắt phiên sau khi tải kết quả về: không còn gì để hỏi.
        return answer("stopped")
    # Không hỏi được Colab (mất mạng, hết hạn đăng nhập) vẫn trả thông tin đã lưu, như Kaggle.
    try:
        info, lines = _poll(client, state, -1)
    except ColabSessionLostError:
        _mark_lost(run_dir, state)
        return answer("lost", _lost_message(state, run_id))
    except ColabError as error:
        return answer(None, str(error))
    status = _job_status(info)
    if status == "missing":
        _mark_lost(run_dir, state)
        return answer("lost", _lost_message(state, run_id))
    return answer(status, exit_code=info["exit_code"], log_tail=lines)


def follow_log(
    workdir: Workdir,
    run_id: str,
    on_line: Callable[[str], None],
    client: Optional[ColabClient] = None,
    interval_s: float = 15,
    max_failures: int = 20,
    sleep: Optional[Callable[[float], None]] = None,
) -> str:
    """Đọc log của job từ đầu và in từng dòng tới khi job kết thúc.

    Mỗi lần đọc là một lệnh ``colab exec`` ngắn; lần đọc lỗi (rớt kết nối) thì nghỉ rồi thử lại,
    quá ``max_failures`` lần liên tiếp mới bỏ, job vẫn chạy trên VM. Ctrl-C chỉ dừng việc theo dõi.

    Args:
        workdir: Thư mục làm việc.
        run_id: Run đã đẩy.
        on_line: Hàm nhận từng dòng log.
        client: Colab.
        interval_s: Nghỉ giữa hai lần đọc khi chưa có dòng mới, giây.
        max_failures: Số lần đọc lỗi liên tiếp tối đa.
        sleep: Hàm nghỉ; test truyền hàm không nghỉ.

    Returns:
        Trạng thái cuối của job (như ``job_state``).

    Raises:
        ColabError: Nếu run chưa đẩy, phiên đã mất, hoặc đọc lỗi quá ``max_failures`` lần.
    """
    client = client or ColabClient()
    pause = sleep or time.sleep
    run_dir, state = _require_state(workdir, run_id)
    offset, failures = 0, 0
    while True:
        try:
            info, lines = _poll(client, state, offset)
            failures = 0
        except ColabSessionLostError:
            _mark_lost(run_dir, state)
            raise ColabError(_lost_message(state, run_id)) from None
        except ColabError as error:
            failures += 1
            if failures >= max_failures:
                raise ColabError(
                    f"không đọc được log sau {failures} lần ({error}); job vẫn chạy trên VM: "
                    f"d2s colab logs {run_id} --follow để theo dõi lại"
                ) from None
            pause(interval_s)
            continue
        for line in lines:
            on_line(line)
        offset = info["offset"]
        status = _job_status(info)
        if status == "missing":
            _mark_lost(run_dir, state)
            raise ColabError(_lost_message(state, run_id))
        if status != "running" and not lines:
            return status
        if not lines:
            pause(interval_s)


def watched_runs(workdir: Workdir) -> list[tuple[Path, ColabState]]:
    """Các run đã đẩy lên Colab mà chưa tải kết quả của lần đẩy mới nhất và phiên chưa mất.

    Returns:
        ``(thư mục run, colab.json)`` theo thứ tự đường dẫn.
    """
    found = []
    for state_file in sorted(workdir.runs_dir.glob(f"*/*/*/{D2S_DIR}/{STATE_FILE}")):
        run_dir = state_file.parents[1]
        try:
            state = load_state(run_dir)
        except (OSError, ValueError, TypeError):
            continue
        if state is None:
            continue
        # Thời điểm ghi theo ISO 8601 UTC cùng một định dạng, nên so chuỗi là so thời gian.
        pulled = state.pulled_utc and state.pulled_utc >= state.pushed_utc
        lost = state.lost_utc and state.lost_utc >= state.pushed_utc
        if not pulled and not lost:
            found.append((run_dir, state))
    return found


def pull_run(
    workdir: Workdir,
    run_id: str,
    client: Optional[ColabClient] = None,
    keep_session: bool = False,
) -> RunRecord:
    """Tải kết quả của job về và ghép vào thư mục làm việc, như ``d2s kaggle pull``.

    Ghép theo ``remote.merge_output``: bản trên VM thắng ở mọi stage nó có, stage chỉ có trên máy
    được giữ lại, cache máy chưa có được chép thêm. Run đã xong (``succeeded``) thì tắt phiên để
    không tốn compute unit, trừ khi ``keep_session`` hoặc còn run khác của cùng phiên chưa tải về.

    Args:
        workdir: Thư mục làm việc trên máy.
        run_id: Run đã đẩy.
        client: Colab.
        keep_session: Không tắt phiên kể cả khi run đã xong.

    Returns:
        ``run.json`` sau khi ghép.

    Raises:
        ColabError: Nếu run chưa đẩy, đang chạy trên máy, job còn chạy trên VM, VM không có run,
            phiên đã mất, hoặc tải về lỗi.
    """
    client = client or ColabClient()
    run_dir, state = _require_state(workdir, run_id)
    if run_is_active(run_dir):
        raise ColabError(f"run {run_id} đang chạy trên máy (ví dụ đang xuất); chờ xong rồi tải")
    run_rel = run_dir.relative_to(workdir.root).as_posix()
    config = {"root": REMOTE_ROOT, "job": state.job, "run_rel": run_rel, "chunk": CHUNK_BYTES}
    try:
        packed = _run_on_vm(client, state.session, _PACK, config, timeout_s=1800)
        if _markers(packed, "BUSY"):
            raise ColabError(f"job {state.job} còn đang chạy trên Colab; chờ xong rồi tải")
        if _markers(packed, "NORUN"):
            raise ColabError(f"VM của phiên {state.session} không có {run_rel}")
        packs = _markers(packed, "PACK")
        if not packs:
            raise ColabError(f"không gói được output trên VM: {packed.strip()[-200:]}")
        pack = json.loads(packs[-1])
        with tempfile.TemporaryDirectory(dir=_staging(workdir)) as folder:
            staging = Path(folder)
            archive = staging / "output.tar"
            with open(archive, "wb") as output:
                for remote_part in pack["parts"]:
                    local_part = staging / "part"
                    _retry(lambda: client.download(state.session, remote_part, local_part))  # noqa: B023
                    output.write(local_part.read_bytes())
                    local_part.unlink()
            if _sha256(archive) != pack["sha256"]:
                raise ColabError("output tải từ Colab bị hỏng (sha256 khác); tải lại")
            extracted = staging / "output"
            _safe_extract(archive, extracted)
            archive.unlink()
            remote.merge_output(workdir, run_dir, extracted)
        try:
            _run_on_vm(client, state.session, _CLEAN, {"path": f"{REMOTE_ROOT}/out/{state.job}"})
        except ColabError:
            pass  # chỉ là dọn file tạm trên VM
    except ColabSessionLostError:
        _mark_lost(run_dir, state)
        raise ColabError(_lost_message(state, run_id)) from None
    state.pulled_utc = _now()
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    others = [s for d, s in watched_runs(workdir) if d != run_dir and s.session == state.session]
    if record.status == "succeeded" and not keep_session and not others:
        try:
            client.stop(state.session)
            state.stopped_utc = _now()
        except ColabError:
            pass  # tắt phiên lỗi không làm hỏng kết quả đã tải; người dùng tự tắt sau
    _save_state(run_dir, state)
    return record


def stop_session(workdir: Workdir, run_id: str, client: Optional[ColabClient] = None) -> dict:
    """Tắt phiên Colab của một run (trả VM); mọi thứ trên VM chưa tải về sẽ mất.

    Returns:
        Nội dung ``colab.json`` sau khi ghi ``stopped_utc``.

    Raises:
        ColabError: Nếu run chưa đẩy lên Colab hoặc lệnh lỗi.
    """
    client = client or ColabClient()
    run_dir, state = _require_state(workdir, run_id)
    client.stop(state.session)
    state.stopped_utc = _now()
    if not (state.pulled_utc and state.pulled_utc >= state.pushed_utc):
        state.lost_utc = state.stopped_utc
    _save_state(run_dir, state)
    return asdict(state)
