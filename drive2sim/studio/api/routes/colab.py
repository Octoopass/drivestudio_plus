"""Route Colab: ``/api/colab/...`` (như ``d2s colab``; CONTRACTS mục 9.3 và 11.1).

Đẩy và tải có thể mất nhiều phút (tạo phiên, upload cảnh, tải output), nên chạy như việc nền
(``/api/jobs``); hỏi trạng thái và tắt phiên thì trả lời ngay (một lệnh ``colab`` vài giây).

File này chứa:
- ``GET /api/colab/setup/status``: kiểm tra lệnh ``colab``, đăng nhập, compute unit, file SMPL và
  code đã push chưa (``drive2sim/colab/setup.py``).
- ``GET /api/colab/{run_id}``: ``colab.json`` của run cộng trạng thái job và log cuối.
- ``POST /api/colab/{run_id}/push``: đẩy (lần sau chạy tiếp); thân tuỳ chọn ``gpu``, ``session``;
  trả ``202`` kèm việc.
- ``POST /api/colab/{run_id}/pull``: tải kết quả về; trả ``202`` kèm việc.
- ``POST /api/colab/{run_id}/stop``: tắt phiên Colab của run.
- ``GET /api/colab/accounts``, ``POST /api/colab/accounts/{email}/use``,
  ``DELETE /api/colab/accounts/{email}``: tài khoản Google của lệnh ``colab``
  (``drive2sim/colab/accounts.py``).
- ``GET /api/colab/login``, ``POST /api/colab/login``, ``POST /api/colab/login/code``,
  ``POST /api/colab/login/cancel``: thêm tài khoản ngay trên Studio (trang đăng nhập Google rồi dán
  mã xác thực).
- ``GET /api/colab/cli``, ``POST /api/colab/cli/update``: phiên bản google-colab-cli và cập nhật
  (việc nền ``colab_update``).
- ``/api/colab/drive/...``: lấy dữ liệu gốc từ Google Drive qua một phiên Colab CPU (phiên, mount
  có bước đồng ý của Google, duyệt thư mục, tải về ``uploads/<tên>/``; ``colab/drive.py``).
"""

from collections.abc import Callable
from typing import Literal, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from drive2sim.colab import accounts, drive, executor, setup
from drive2sim.colab.client import ColabClient, ColabError
from drive2sim.engine.workdir import Workdir
from drive2sim.studio.api import uploads
from drive2sim.studio.api.jobs import RunQueue
from drive2sim.studio.api.state import StudioState, state_of

router = APIRouter(prefix="/api/colab", tags=["colab"])


class PushBody(BaseModel):
    """Lựa chọn khi đẩy run lên Colab; bỏ trống thì dùng mặc định của ``d2s colab push``.

    Attributes:
        gpu: GPU khi phải tạo phiên mới.
        session: Tên phiên Colab.
    """

    model_config = ConfigDict(extra="forbid")

    gpu: Literal["T4", "L4", "G4", "A100", "H100"] = executor.DEFAULT_GPU
    session: str = Field(default=executor.DEFAULT_SESSION, pattern=r"^[A-Za-z0-9_-]{1,40}$")


@router.get("/setup/status")
def setup_status(request: Request) -> dict:
    """Kiểm tra Colab đã sẵn sàng chạy run chưa; gọi ``colab sessions`` và ``colab usage``.

    Returns:
        Kết quả ``setup.check`` (``ready``, ``cli``, ``account``, ``gpu``, ``smpl``, ``code``).
    """
    return setup.check(workdir_root=state_of(request).workdir.root)


def submit_push(state: StudioState, run_id: str, body: Optional[PushBody] = None) -> dict:
    """Xếp việc đẩy một run lên Colab; ``POST /api/runs`` với ``executor: colab`` cũng gọi.

    Returns:
        Việc vừa xếp hàng.
    """
    options = body or PushBody()

    def work() -> str:
        executor.push_run(state.workdir, run_id, gpu=options.gpu, session=options.session)
        return run_id

    return state.jobs.submit("colab_push", f"đẩy {run_id} lên Colab", work)


class CodeBody(BaseModel):
    """Mã xác thực Google người dùng dán vào sau khi đăng nhập.

    Attributes:
        code: Mã Google hiển thị (dạng ``4/0A...``).
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=accounts.CODE_PATTERN)


def _login(request: Request) -> accounts.LoginFlow:
    """Lần đăng nhập của Studio: một bản cho cả app, tạo khi cần."""
    app_state = request.app.state
    if not hasattr(app_state, "colab_login"):
        app_state.colab_login = accounts.LoginFlow()
    return app_state.colab_login


@router.get("/accounts")
def list_accounts() -> dict:
    """Tài khoản Google đã đăng nhập vào lệnh ``colab`` và tài khoản đang dùng.

    Returns:
        ``{"accounts": [email], "active": email hoặc null}``.
    """
    return accounts.list_accounts()


@router.post("/accounts/{email}/use")
def use_account(email: str) -> dict:
    """Chuyển lệnh ``colab`` (phiên, Drive) sang một tài khoản đã đăng nhập.

    Returns:
        Danh sách tài khoản sau khi chuyển.
    """
    return accounts.use_account(email)


@router.delete("/accounts/{email}")
def forget_account(email: str) -> dict:
    """Bỏ một tài khoản khỏi danh sách; token đang dùng giữ nguyên.

    Returns:
        Danh sách tài khoản sau khi bỏ.
    """
    return accounts.forget_account(email)


@router.get("/login")
def login_status(request: Request) -> dict:
    """Trạng thái lần thêm tài khoản.

    Returns:
        ``{"state", "url", "email", "error"}``; ``state`` là ``idle``, ``waiting_code``,
        ``checking``, ``succeeded`` hoặc ``failed``.
    """
    return _login(request).view()


@router.post("/login")
def login_start(request: Request) -> dict:
    """Bắt đầu thêm tài khoản; trả ``url`` trang đăng nhập Google để mở trong trình duyệt."""
    return _login(request).start()


@router.post("/login/code")
def login_code(body: CodeBody, request: Request) -> dict:
    """Gửi mã xác thực đã dán; trả trạng thái sau khi lệnh ``colab`` lưu token."""
    return _login(request).submit_code(body.code)


@router.post("/login/cancel")
def login_cancel(request: Request) -> dict:
    """Huỷ lần thêm tài khoản đang chờ mã."""
    return _login(request).cancel()


@router.get("/cli")
def cli_versions() -> dict:
    """Phiên bản google-colab-cli đang cài và bản mới nhất.

    Returns:
        ``{"version", "latest", "up_to_date"}``.
    """
    return accounts.cli_versions()


@router.post("/cli/update", status_code=202)
def cli_update(request: Request) -> dict:
    """Cập nhật google-colab-cli ở nền (``uv tool upgrade``); trả việc vừa xếp hàng."""

    def work() -> None:
        accounts.update_cli()
        return None

    return state_of(request).jobs.submit("colab_update", "cập nhật google-colab-cli", work)


class DriveSessionBody(BaseModel):
    """Phiên Colab dùng để đọc Drive.

    Attributes:
        session: Tên phiên; chưa có thì ``POST /api/colab/drive/session`` tạo phiên CPU.
    """

    model_config = ConfigDict(extra="forbid")

    session: str = Field(default="d2s-drive", pattern=drive.SESSION_PATTERN)


class DrivePullBody(DriveSessionBody):
    """Thân của ``POST /api/colab/drive/pull``.

    Attributes:
        path: Thư mục hoặc file trong ``MyDrive``.
        name: Thư mục đích ``uploads/<tên>/`` trong thư mục làm việc.
        include: Chỉ tải file có đường dẫn bắt đầu bằng một trong các tiền tố này (một chuỗi
            KITTI-360: ``pull_sets`` của ``/drive/inspect``; mục đã chọn: ``"<thư mục>/"``,
            ``"<file>"``); rỗng là tải hết.
    """

    path: str = Field(min_length=1, max_length=1024)
    name: str = Field(pattern=uploads.NAME_PATTERN)
    include: list[str] = Field(default_factory=list, max_length=200)


def _mount(request: Request) -> drive.MountFlow:
    """Lần mount Drive của Studio: một bản cho cả app, tạo khi cần."""
    app_state = request.app.state
    if not hasattr(app_state, "colab_mount"):
        app_state.colab_mount = drive.MountFlow()
    return app_state.colab_mount


@router.get("/drive/sessions")
def drive_sessions() -> dict:
    """Các phiên Colab đang có (``colab sessions``).

    Returns:
        ``{"sessions": [{"name", "hardware"}]}``.
    """
    return {"sessions": ColabClient().sessions()}


@router.post("/drive/session", status_code=202)
def drive_session(body: DriveSessionBody, request: Request) -> dict:
    """Tạo phiên CPU để đọc Drive ở nền (việc ``colab_session``); phiên đã có thì không tạo lại."""

    def work() -> None:
        client = ColabClient()
        if not any(item["name"] == body.session for item in client.sessions()):
            client.create_session(body.session, None)
        return None

    label = f"tạo phiên Colab {body.session} (CPU)"
    return state_of(request).jobs.submit("colab_session", label, work)


@router.post("/drive/session/stop")
def drive_session_stop(body: DriveSessionBody) -> dict:
    """Tắt phiên đọc Drive (trả VM cho Colab)."""
    ColabClient().stop(body.session)
    return {"session": body.session, "stopped": True}


@router.get("/drive/mount")
def drive_mount_status(request: Request) -> dict:
    """Trạng thái lần mount Drive: ``{"state", "session", "url", "log"}``."""
    return _mount(request).view()


@router.post("/drive/mount")
def drive_mount(body: DriveSessionBody, request: Request) -> dict:
    """Mount Drive vào phiên; ``needs_consent`` kèm ``url`` trang cấp quyền của Google."""
    return _mount(request).start(body.session)


@router.post("/drive/mount/confirm")
def drive_mount_confirm(request: Request) -> dict:
    """Người dùng đã đồng ý trên trang Google: cho ``colab drivemount`` đi tiếp."""
    return _mount(request).confirm()


@router.post("/drive/mount/cancel")
def drive_mount_cancel(request: Request) -> dict:
    """Dừng lần mount đang chạy."""
    return _mount(request).cancel()


@router.get("/drive/browse")
def drive_browse(session: str = "d2s-drive", path: str = "") -> dict:
    """Liệt kê một thư mục trong ``MyDrive``.

    Returns:
        ``{"path", "mounted", "exists", "dirs", "files": [[tên, byte]], "truncated"}``.
    """
    body = DriveSessionBody(session=session)
    return drive.browse(ColabClient(), body.session, path)


@router.get("/drive/inspect")
def drive_inspect(session: str = "d2s-drive", path: str = "") -> dict:
    """Nhận dạng dataset ở một thư mục trong ``MyDrive`` (``drive.inspect``).

    Returns:
        ``{"path", "mounted", "exists", "truncated", "report", "sequence_bytes", "pull_sets"}``.
    """
    body = DriveSessionBody(session=session)
    return drive.inspect(ColabClient(), body.session, path)


@router.post("/drive/pull", status_code=202)
def drive_pull(body: DrivePullBody, request: Request) -> dict:
    """Tải một thư mục trên Drive về ``uploads/<tên>/`` ở nền (việc ``drive_pull``, có tiến độ).

    Kết quả của việc là đường dẫn tuyệt đối của thư mục đích, để ô nạp cảnh xem thư mục đó.
    """
    state = state_of(request)
    drive.safe_drive_path(body.path)
    for prefix in body.include:
        drive.safe_drive_path(prefix)
    target = state.workdir.uploads_dir / body.name

    def work() -> str:
        result = drive.pull_folder(
            ColabClient(), body.session, body.path, target, progress=state.jobs.report,
            include=body.include,
        )  # fmt: skip
        return result["path_abs"]

    what = f"{body.path} ({len(body.include)} mục chọn)" if body.include else body.path
    return state.jobs.submit("drive_pull", f"Drive: {what} → uploads/{body.name}", work)


@router.get("/{run_id}")
def colab_status(run_id: str, request: Request) -> dict:
    """Trạng thái job Colab của run.

    Returns:
        Nội dung ``colab.json`` (``session``, ``gpu``, ``attempt``, ``job``, ``commit``,
        ``branch``, ``pushed_utc``, ``pulled_utc``, ``lost_utc``, ``stopped_utc``) cộng
        ``status``, ``exit_code``, ``log_tail``, ``status_error``.
    """
    return executor.job_state(state_of(request).workdir, run_id)


@router.post("/{run_id}/push", status_code=202)
def push(run_id: str, request: Request, body: Optional[PushBody] = None) -> dict:
    """Đẩy run lên Colab ở nền."""
    state = state_of(request)
    state.workdir.find_run(run_id)
    return submit_push(state, run_id, body)


@router.post("/{run_id}/pull", status_code=202)
def pull(run_id: str, request: Request) -> dict:
    """Tải kết quả của run trên Colab về ở nền."""
    state = state_of(request)
    state.workdir.find_run(run_id)
    return state.jobs.submit(
        "colab_pull", f"tải {run_id} từ Colab", pull_work(state.workdir, state.queue, run_id)
    )


@router.post("/{run_id}/stop")
def stop(run_id: str, request: Request) -> dict:
    """Tắt phiên Colab của run; mọi thứ trên VM chưa tải về sẽ mất.

    Returns:
        ``colab.json`` sau khi ghi ``stopped_utc``.
    """
    return executor.stop_session(state_of(request).workdir, run_id)


def pull_work(workdir: Workdir, queue: RunQueue, run_id: str) -> Callable[[], str]:
    """Việc tải kết quả Colab về, có giữ chỗ run trong lúc ghép (như Kaggle).

    Không giữ chỗ thì tải về chạy cùng lúc với xuất lại hay chạy tiếp trên cùng thư mục run, hai
    bên xoá và ghi đè run.json, thư mục stage của nhau.
    """

    def work() -> str:
        run_dir = workdir.find_run(run_id)
        if not queue.claim(run_dir):
            raise ColabError(f"run {run_id} đang chạy hoặc xếp hàng trên máy; chờ xong rồi tải")
        try:
            executor.pull_run(workdir, run_id)
        finally:
            queue.release(run_dir)
        return run_id

    return work
