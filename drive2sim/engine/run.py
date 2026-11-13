"""Tạo một run và chạy lần lượt các stage của nó.

Một *run* là một lần chạy một approach trên một cảnh, với một split và một bộ tham số (SPEC
mục 3). File này làm bước 3–5 của engine (ARCHITECTURE mục 4.5): tạo thư mục run, ghi
``d2s/run.json``, rồi chạy từng stage và cập nhật trạng thái.

File này chứa:
- ``create_run``: nhận ``Plan`` từ ``resolve.py``, tạo thư mục
  ``runs/<scene_id>/<approach_id>/<run_id>/`` và ghi ``d2s/run.json`` (mọi stage ở trạng thái
  ``pending``) cùng ``d2s/env.json``.
- ``execute_run``: chạy hoặc chạy tiếp một run đã tạo. Stage có cache thì tìm trong ``cache/``
  trước; stage không cache thì ghi vào ``d2s/stages/<stage_id>/`` và đánh dấu xong bằng file
  ``done``. Mỗi khi một stage đổi trạng thái, ``run.json`` được ghi lại và một sự kiện được phát
  ra cho CLI hoặc Studio.
- ``list_runs``, ``load_summary``, ``summary_path``: đọc các run đã có và kết quả đánh giá.
- ``request_cancel``, ``run_is_active``: huỷ một run, biết run còn chạy không.
- ``reset_stage``: chạy lại hoặc thêm một stage (cho ``d2s export``).
- ``EventSink``: kiểu của hàm nhận sự kiện tiến độ (cùng dạng với sự kiện SSE, CONTRACTS mục
  11.4).

Nhờ cache và file ``done``, chạy lại một run dở (``d2s run resume``, hoặc phiên Kaggle bị ngắt)
chỉ làm tiếp phần còn thiếu. Stage lỗi làm run dừng ở trạng thái ``failed``, lỗi ghi vào
``run.json`` và traceback ghi vào ``d2s/log.txt``. Định dạng các file ở CONTRACTS mục 5 và 6.
"""

import functools
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from drive2sim import paths
from drive2sim.core import files, identifiers
from drive2sim.core.schema import (
    SCHEMA_VERSIONS,
    ArtifactRecord,
    EnvRecord,
    ErrorInfo,
    Producer,
    RunApproach,
    RunRecord,
    Split,
    StageRecord,
    Summary,
)
from drive2sim.engine.cache import (
    ARTIFACT_FILE,
    CacheStore,
    input_key_for_run_stage,
    input_key_for_scene,
    input_key_for_split,
)
from drive2sim.engine.machine import current_machine, package_versions, same_machine
from drive2sim.engine.registry import REGISTRY, StageRegistry
from drive2sim.engine.resolve import Plan, validate_params, wire_inputs
from drive2sim.engine.stage import SCENE_INPUT, Scene, StageCancelledError
from drive2sim.engine.workdir import Workdir

D2S_DIR = "d2s"
RUN_FILE = "run.json"
ENV_FILE = "env.json"
DONE_FILE = "done"
LOG_FILE = "log.txt"
CANCEL_FILE = "cancel"  # có file này trong d2s/ nghĩa là người dùng yêu cầu huỷ run
# "<pid> <tên máy>" của tiến trình đang chạy run; có file chưa chắc run còn chạy (run_is_active)
PID_FILE = "pid"


class RunActiveError(RuntimeError):
    """Run đang được một tiến trình khác chạy; không chạy thêm lần nữa cùng lúc."""


# Nhận một sự kiện dạng {"event": ..., "run_id": ..., ...} như SSE (CONTRACTS mục 11.4).
EventSink = Callable[[dict[str, object]], None]


def _ignore_event(event: dict[str, object]) -> None:
    """Nhận một sự kiện và bỏ qua nó; là giá trị mặc định khi không ai theo dõi run.

    Args:
        event: Sự kiện tiến độ.
    """


def _now_utc() -> datetime:
    """Trả về thời điểm hiện tại theo giờ UTC, dùng cho các trường ``*_utc`` (CONTRACTS 10.3)."""
    return datetime.now(timezone.utc)


@functools.lru_cache(maxsize=1)
def code_commit() -> str:
    """Đọc commit hiện tại của repo chứa package, để ghi vào ``env.json`` và ``artifact.json``.

    Ghi commit giúp biết kết quả được tạo bằng phiên bản code nào, để chạy lại được. Kết quả được
    nhớ sau lần gọi đầu, vì commit không đổi trong lúc chạy.

    Returns:
        Mã commit ngắn, ví dụ ``"35bc590"``; ``"unknown"`` nếu không có git hoặc không phải repo.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(paths.PACKAGE_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _peak_vram_mb() -> int:
    """Đọc lượng VRAM cao nhất đã dùng từ lần gọi trước, rồi đặt lại bộ đếm.

    Engine gọi hàm này trước mỗi stage (để đặt lại) và sau mỗi stage (để đọc), nên giá trị đọc
    được là VRAM cao nhất của riêng stage đó. Hàm chỉ dùng PyTorch nếu stage đã import nó, để
    engine không phải nạp PyTorch khi chạy các stage không cần GPU.

    Returns:
        VRAM cao nhất tính bằng megabyte; ``0`` nếu không có PyTorch hoặc không có GPU.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return 0
    peak = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
    torch.cuda.reset_peak_memory_stats()
    return peak


def _artifact_duration(output_dir: Path) -> Optional[float]:
    """Đọc thời gian lần chạy gốc trong ``artifact.json`` của một kết quả trong cache.

    Returns:
        Số giây; ``None`` nếu file không ghi (cache tạo trước ``d2s-artifact/0.2``) hoặc lỗi.
    """
    try:
        return files.load(output_dir / ARTIFACT_FILE, ArtifactRecord).duration_s
    except (OSError, ValueError):
        return None


def _env_record(executor: str) -> EnvRecord:
    """Thu thập thông tin môi trường đang chạy run, để ghi vào ``d2s/env.json``.

    Thông tin này giúp chạy lại run trên máy khác (SPEC mục 11). Trường nào không đọc được, ví
    dụ phiên bản PyTorch khi PyTorch chưa được import, thì để trống (CONTRACTS mục 6.4).

    Args:
        executor: Nơi chạy, ``local`` hoặc ``kaggle``.

    Returns:
        Bản ghi ``env.json``.
    """
    return EnvRecord(
        schema_version=SCHEMA_VERSIONS["env"],
        code_commit=code_commit(),
        python=platform.python_version(),
        executor=executor,
        **package_versions(),
    )


def record_machine(run_dir: Path) -> None:
    """Cập nhật ``d2s/env.json`` theo máy đang bắt đầu chạy (hoặc chạy tiếp) run.

    Phiên bản code, Python, gói và tên GPU lấy theo máy này; máy chưa có trong ``machines`` thì
    được thêm vào cuối. Gọi ở đầu ``execute_run``, nên với run Kaggle đây là máy của notebook.

    Args:
        run_dir: Thư mục run.
    """
    path = run_dir / D2S_DIR / ENV_FILE
    try:
        previous = files.load(path, EnvRecord)
    except (OSError, ValueError):
        previous = None
    machine = current_machine()
    env = _env_record(previous.executor if previous else "local")
    env.seed = previous.seed if previous else None
    env.machines = list(previous.machines) if previous else []
    if not any(same_machine(machine, seen) for seen in env.machines):
        env.machines.append(machine.model_copy(update={"first_seen_utc": _now_utc()}))
    env.gpu = machine.gpu
    files.dump(env, path)


def create_run(
    plan: Plan,
    workdir: Workdir,
    scene: Scene,
    split: Split,
    executor: str = "local",
    run_id: Optional[str] = None,
) -> Path:
    """Tạo thư mục cho một run mới và ghi ``d2s/run.json``, ``d2s/env.json``.

    ``run.json`` lưu đầy đủ lựa chọn của các mục cấu hình và tham số đã gộp của từng stage, để
    ``execute_run`` chạy hoặc chạy tiếp mà không cần đọc lại file approach (file approach có
    thể đã bị sửa sau đó). Hàm này chưa chạy stage nào; gọi ``execute_run`` sau đó.

    Args:
        plan: Kết quả của ``resolve``.
        workdir: Thư mục làm việc.
        scene: Cảnh của run.
        split: Split của run.
        executor: Nơi chạy, ``local`` hoặc ``kaggle``.
        run_id: Định danh run; mặc định sinh mới dạng
            ``YYYYMMDD-HHMMSS-<approach_id>-<4 hex>`` (CONTRACTS mục 10.1).

    Returns:
        Thư mục run vừa tạo, ví dụ ``runs/nuscenes_scene-0061/omnire/20261112-081000-omnire-a1b2``.

    Raises:
        FileExistsError: Nếu thư mục run đã tồn tại.
    """
    run_id = run_id or identifiers.make_run_id(plan.approach_id)
    run_dir = workdir.run_dir(scene.scene_id, plan.approach_id, run_id)
    (run_dir / D2S_DIR).mkdir(parents=True, exist_ok=False)
    record = RunRecord(
        schema_version=SCHEMA_VERSIONS["run"],
        run_id=run_id,
        scene_id=scene.scene_id,
        split_id=split.split_id,
        approach=RunApproach(
            approach_id=plan.approach_id,
            version=plan.approach_version,
            preset=plan.preset,
            options=plan.options,
            quick=plan.quick,
        ),
        level=plan.level,
        executor=executor,
        status="pending",
        created_utc=_now_utc(),
        lidar_usage=plan.lidar_usage,
        stages=[
            StageRecord(stage_id=stage.stage_id, status="pending", params=stage.params)
            for stage in plan.stages
        ],
    )
    files.dump(record, run_dir / D2S_DIR / RUN_FILE)
    files.dump(_env_record(executor), run_dir / D2S_DIR / ENV_FILE)
    return run_dir


def preview_reuse(
    plan: Plan, scene: Scene, split: Split, workdir: Workdir, registry: StageRegistry = REGISTRY
) -> list[bool]:
    """Đoán trước stage nào của một run mới sẽ dùng lại kết quả có sẵn trong cache.

    Dùng cho ``d2s approach check`` và màn hình chọn approach của Studio, để người dùng biết
    trước phần nào chạy lại, phần nào không. Khoá cache được tính giống hệt lúc chạy thật
    (``identifiers.make_cache_key`` trên tham số và khoá đầu vào). Stage không cache luôn chạy
    trong run mới; stage có cache nhận đầu vào từ một stage không cache thì khoá phụ thuộc
    ``run_id`` chưa có, nên cũng tính là sẽ chạy.

    Args:
        plan: Kết quả của ``resolve``.
        scene: Cảnh của run.
        split: Split của run.
        workdir: Thư mục làm việc, để tìm trong ``cache/``.
        registry: Bảng stage.

    Returns:
        Mỗi stage một giá trị, cùng thứ tự với ``plan.stages``: ``True`` nếu sẽ dùng lại cache.
    """
    cache = CacheStore(workdir.cache_dir)
    keys: list[Optional[str]] = []
    reused: list[bool] = []
    for stage in plan.stages:
        input_keys: list[Optional[str]] = []
        for source in stage.input_sources.values():
            if source == SCENE_INPUT:
                input_keys.append(
                    input_key_for_scene(scene.scene_id, scene.manifest.source.converter_version)
                )
            elif isinstance(source, int):
                input_keys.append(keys[source])
            else:
                input_keys.append(input_key_for_split(split.split_id))
        key = None
        if stage.cacheable and None not in input_keys:
            key = identifiers.make_cache_key(
                scene.scene_id,
                stage.stage_id,
                registry.get(stage.stage_id).version,
                stage.params,
                [k for k in input_keys if k is not None],
            )
        keys.append(key)
        reused.append(key is not None and cache.lookup(key) is not None)
    return reused


@dataclass
class _StageContext:
    """Ngữ cảnh engine đưa cho một stage đang chạy (cài interface ``StageContext``).

    Khi stage gọi ``progress`` hoặc ``log_metric``, lớp này gắn thêm ``run_id`` và ``stage_id``
    rồi chuyển thành sự kiện cho ``sink`` (CONTRACTS mục 11.4).

    Attributes:
        output_dir: Thư mục stage ghi kết quả vào.
        scene: Cảnh của run.
        split: Split của run.
        run_id: Run đang chạy.
        stage_id: Stage đang chạy.
        sink: Hàm nhận sự kiện.
        cancel_file: File ``d2s/cancel`` của run; có file này là người dùng đã yêu cầu huỷ.
    """

    output_dir: Path
    scene: Scene
    split: Split
    run_id: str
    stage_id: str
    sink: EventSink
    cancel_file: Path

    def should_stop(self) -> bool:
        """Cho biết người dùng đã yêu cầu huỷ run chưa (có file ``d2s/cancel``)."""
        return self.cancel_file.exists()

    def progress(self, fraction: float, message: str = "") -> None:
        """Phát sự kiện ``progress`` của stage.

        Args:
            fraction: Phần đã xong, từ 0 đến 1; giá trị ngoài khoảng bị kẹp lại.
            message: Mô tả ngắn việc đang làm.
        """
        self.sink(
            {
                "event": "progress",
                "run_id": self.run_id,
                "stage_id": self.stage_id,
                "fraction": min(max(float(fraction), 0.0), 1.0),
                "message": message,
            }
        )

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Phát sự kiện ``metric`` của stage.

        Args:
            name: Tên metric, ví dụ ``"loss"``.
            value: Giá trị.
            step: Bước (iteration) ứng với giá trị.
        """
        self.sink(
            {
                "event": "metric",
                "run_id": self.run_id,
                "stage_id": self.stage_id,
                "name": name,
                "value": float(value),
                "step": int(step),
            }
        )


@dataclass
class _Runner:
    """Giữ trạng thái và làm các việc cụ thể trong lúc chạy một run.

    Chỉ dùng bên trong file này; bên ngoài gọi ``execute_run``. Khi tạo, lớp đọc manifest và
    split của run từ thư mục làm việc.

    Attributes:
        run_dir: Thư mục run.
        record: Nội dung ``run.json``, được sửa dần trong lúc chạy và ghi lại sau mỗi thay đổi.
        workdir: Thư mục làm việc.
        registry: Bảng stage.
        sink: Hàm nhận sự kiện.
        scene: Cảnh của run.
        split: Split của run.
    """

    run_dir: Path
    record: RunRecord
    workdir: Workdir
    registry: StageRegistry
    sink: EventSink
    scene: Scene = field(init=False)
    split: Split = field(init=False)

    def __post_init__(self) -> None:
        """Đọc cảnh và split của run sau khi dataclass gán xong các trường."""
        self.scene = self.workdir.load_scene(self.record.scene_id)
        self.split = self.workdir.load_split(self.record.scene_id, self.record.split_id)

    def save(self) -> None:
        """Ghi ``record`` hiện tại ra ``d2s/run.json`` để Studio và CLI đọc được trạng thái mới."""
        files.dump(self.record, self.run_dir / D2S_DIR / RUN_FILE)

    def emit_status(self, stage_id: Optional[str], status: str) -> None:
        """Phát sự kiện ``status`` khi run hoặc một stage đổi trạng thái.

        Args:
            stage_id: Stage đổi trạng thái, hoặc ``None`` khi là trạng thái của cả run.
            status: Trạng thái mới, ví dụ ``"running"``, ``"succeeded"``.
        """
        self.sink(
            {
                "event": "status",
                "run_id": self.record.run_id,
                "stage_id": stage_id,
                "status": status,
            }
        )

    def input_key(self, kind: str) -> str:
        """Tạo khoá cache cho đầu vào ``"scene"`` hoặc ``"split"`` của run này.

        Args:
            kind: ``"scene"`` hoặc ``"split"``.

        Returns:
            Khoá do ``input_key_for_scene`` hoặc ``input_key_for_split`` tạo.
        """
        if kind == SCENE_INPUT:
            return input_key_for_scene(
                self.scene.scene_id, self.scene.manifest.source.converter_version
            )
        return input_key_for_split(self.split.split_id)

    def write_artifact(
        self,
        output_dir: Path,
        stage_id: str,
        stage_record: StageRecord,
        input_keys: list[str],
        output_paths: dict[str, Path],
        duration_s: Optional[float] = None,
    ) -> None:
        """Kiểm tra đầu ra stage trả về, rồi ghi ``artifact.json`` mô tả kết quả.

        Đầu ra phải đúng tên đã khai báo, phải tồn tại và phải nằm trong ``output_dir``; sai một
        điều thì stage bị coi là lỗi. ``artifact.json`` ghi loại artifact, khoá cache, stage và
        phiên bản đã tạo ra nó, tham số, khoá các đầu vào và danh sách file (CONTRACTS mục 5.3).

        Args:
            output_dir: Thư mục kết quả của stage.
            stage_id: Stage vừa chạy.
            stage_record: Bản ghi của stage trong ``run.json``.
            input_keys: Khoá của các đầu vào.
            output_paths: Kết quả ``Stage.run`` trả về: tên đầu ra → đường dẫn.
            duration_s: Thời gian ``Stage.run`` đã chạy, giây.

        Raises:
            RuntimeError: Nếu đầu ra sai tên, không tồn tại, hoặc nằm ngoài ``output_dir``.
        """
        cls = self.registry.get(stage_id)
        if set(output_paths) != set(cls.outputs):
            raise RuntimeError(
                f"stage {stage_id} trả về đầu ra {sorted(output_paths)}, "
                f"nhưng khai báo {sorted(cls.outputs)}"
            )
        relative_files = []
        for name, path in output_paths.items():
            path = Path(path).resolve()
            if not path.exists():
                raise RuntimeError(f"stage {stage_id}: đầu ra {name!r} không tồn tại: {path}")
            try:
                relative_files.append(path.relative_to(output_dir.resolve()).as_posix())
            except ValueError:
                raise RuntimeError(
                    f"stage {stage_id}: đầu ra {name!r} nằm ngoài thư mục của stage: {path}"
                ) from None
        record = ArtifactRecord(
            schema_version=SCHEMA_VERSIONS["artifact"],
            artifact_type=next(iter(cls.outputs.values())),
            cache_key=stage_record.cache_key,
            producer=Producer(stage_id=stage_id, version=cls.version, code_commit=code_commit()),
            params=stage_record.params,
            inputs=sorted(input_keys),
            created_utc=_now_utc(),
            files=relative_files,
            duration_s=duration_s,
        )
        files.dump(record, output_dir / ARTIFACT_FILE)

    def run_stage(
        self,
        stage_record: StageRecord,
        output_dir: Path,
        inputs: dict[str, Path],
        input_keys: list[str],
    ) -> None:
        """Gọi ``run`` của stage để ghi kết quả vào ``output_dir``, rồi ghi ``artifact.json``.

        Args:
            stage_record: Bản ghi của stage trong ``run.json``, chứa tham số đã gộp.
            output_dir: Thư mục stage ghi kết quả vào (``.partial`` của cache, hoặc thư mục
                stage trong run).
            inputs: Tên đầu vào → thư mục artifact của stage nguồn.
            input_keys: Khoá của các đầu vào, ghi vào ``artifact.json``.
        """
        cls = self.registry.get(stage_record.stage_id)
        params = cls.Params.model_validate(stage_record.params)
        ctx = _StageContext(
            output_dir,
            self.scene,
            self.split,
            self.record.run_id,
            stage_record.stage_id,
            self.sink,
            self.run_dir / D2S_DIR / CANCEL_FILE,
        )
        started = time.monotonic()
        output_paths = cls().run(ctx, inputs, params)
        self.write_artifact(
            output_dir,
            stage_record.stage_id,
            stage_record,
            input_keys,
            output_paths,
            round(time.monotonic() - started, 3),
        )

    def execute(self) -> RunRecord:
        """Chạy lần lượt mọi stage của run, bỏ qua stage đã có kết quả.

        Với mỗi stage: lấy thư mục và khoá của các đầu vào từ stage nguồn; nếu stage có cache thì
        tính khoá cache và dùng lại kết quả nếu đã có, chưa có thì chạy vào thư mục ``.partial``
        rồi ``commit``; nếu stage không cache thì bỏ qua khi đã có file ``done``, chưa có thì
        chạy rồi ghi ``done``. Stage đã ``succeeded`` từ lần chạy trước giữ nguyên thời gian và
        VRAM đã đo.

        Returns:
            ``run.json`` sau khi chạy: ``succeeded`` nếu mọi stage xong, ``failed`` nếu một stage
            lỗi (các stage sau không chạy).

        Raises:
            KeyboardInterrupt: Khi người dùng ngắt; run và stage đang chạy được ghi
                ``cancelled`` trước khi ném lại lỗi.
        """
        stage_ids = [stage.stage_id for stage in self.record.stages]
        wiring = wire_inputs(stage_ids, self.registry)
        cache = CacheStore(self.workdir.cache_dir)
        artifact_dirs: list[Optional[Path]] = [None] * len(stage_ids)
        artifact_keys: list[Optional[str]] = [None] * len(stage_ids)

        pid_file = self.run_dir / D2S_DIR / PID_FILE
        # Hai tiến trình cùng chạy một run (Studio và lệnh d2s, hay hai lệnh) sẽ cùng ghi
        # run.json và thư mục stage; tiến trình xong trước còn xoá file pid của tiến trình kia.
        if run_is_active(self.run_dir):
            raise RunActiveError(f"run {self.record.run_id} đang chạy ở tiến trình khác")
        pid_file.write_text(f"{os.getpid()} {platform.node()}\n")
        try:
            return self._execute_stages(wiring, cache, artifact_dirs, artifact_keys)
        finally:
            pid_file.unlink(missing_ok=True)

    def _execute_stages(
        self,
        wiring: list[dict[str, object]],
        cache: CacheStore,
        artifact_dirs: list[Optional[Path]],
        artifact_keys: list[Optional[str]],
    ) -> RunRecord:
        """Phần thân của ``execute``: chạy từng stage; tách ra để ``execute`` lo file ``pid``."""
        self.record.status = "running"
        self.save()
        self.emit_status(None, "running")
        for index, stage_record in enumerate(self.record.stages):
            stage_id = stage_record.stage_id
            if (self.run_dir / D2S_DIR / CANCEL_FILE).exists():
                return self.cancel(None)
            cls = self.registry.get(stage_id)
            # Kiểm tra lại tham số: code của stage có thể đã đổi từ lúc tạo run.
            stage_record.params = validate_params(stage_id, stage_record.params, self.registry)
            inputs: dict[str, Path] = {}
            input_keys: list[str] = []
            for input_name, source in wiring[index].items():
                if isinstance(source, int):
                    source_dir, source_key = artifact_dirs[source], artifact_keys[source]
                    assert source_dir is not None and source_key is not None
                    inputs[input_name] = source_dir
                    input_keys.append(source_key)
                else:
                    input_keys.append(self.input_key(str(source)))

            started = time.monotonic()
            _peak_vram_mb()  # đặt lại bộ đếm VRAM trước stage
            ran = False
            try:
                if cls.cacheable:
                    key = identifiers.make_cache_key(
                        self.scene.scene_id, stage_id, cls.version, stage_record.params, input_keys
                    )
                    output_dir = cache.lookup(key)
                    if output_dir is None:
                        stage_record.cache_key, stage_record.cache_hit = key, False
                        self.start_stage(stage_record)
                        partial = cache.begin(key)
                        try:
                            self.run_stage(stage_record, partial, inputs, input_keys)
                        except BaseException:
                            cache.abort(partial)
                            raise
                        output_dir = cache.commit(partial)
                        ran = True
                    elif stage_record.cache_key != key:
                        stage_record.cache_key, stage_record.cache_hit = key, True
                        stage_record.reused_duration_s = _artifact_duration(output_dir)
                    artifact_keys[index] = key
                else:
                    output_dir = self.run_dir / D2S_DIR / "stages" / stage_id
                    done = output_dir / DONE_FILE
                    if not done.is_file():
                        self.start_stage(stage_record)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        self.run_stage(stage_record, output_dir, inputs, input_keys)
                        done.write_text(_now_utc().isoformat() + "\n")
                        ran = True
                    artifact_keys[index] = input_key_for_run_stage(self.record.run_id, stage_id)
            except KeyboardInterrupt:
                self.finish_stage(stage_record, "cancelled", started, measured=True)
                self.record.status = "cancelled"
                self.save()
                self.emit_status(None, "cancelled")
                raise
            except StageCancelledError:
                self.finish_stage(stage_record, "cancelled", started, measured=True)
                return self.cancel(stage_record)
            except Exception as error:
                self.fail_stage(stage_record, error, started)
                return self.record
            artifact_dirs[index] = output_dir
            if ran or stage_record.status != "succeeded":
                # Stage đã xong ở lần chạy trước giữ nguyên thời gian và VRAM đã đo.
                self.finish_stage(stage_record, "succeeded", started, measured=ran)

        self.record.status = "succeeded"
        self.save()
        self.emit_status(None, "succeeded")
        self.sink({"event": "done", "run_id": self.record.run_id, "status": "succeeded"})
        return self.record

    def cancel(self, stage_record: Optional[StageRecord]) -> RunRecord:
        """Kết thúc run ở trạng thái ``cancelled`` theo yêu cầu huỷ của người dùng.

        Xoá file ``d2s/cancel`` để lần chạy tiếp sau không bị huỷ ngay, ghi ``run.json`` và phát
        sự kiện ``status``, ``done``.

        Args:
            stage_record: Stage đang chạy khi bị huỷ (đã được ghi ``cancelled``), hoặc ``None``
                nếu huỷ trước khi bắt đầu stage tiếp theo.

        Returns:
            ``run.json`` sau khi huỷ.
        """
        (self.run_dir / D2S_DIR / CANCEL_FILE).unlink(missing_ok=True)
        self.record.status = "cancelled"
        self.save()
        self.emit_status(None, "cancelled")
        self.sink({"event": "done", "run_id": self.record.run_id, "status": "cancelled"})
        return self.record

    def start_stage(self, stage_record: StageRecord) -> None:
        """Đánh dấu một stage bắt đầu chạy thật, tức là không dùng lại kết quả cũ.

        Xoá lỗi của lần chạy trước (nếu có), ghi ``run.json`` và phát sự kiện ``running``.

        Args:
            stage_record: Bản ghi của stage trong ``run.json``.
        """
        stage_record.status = "running"
        stage_record.error = None
        self.save()
        self.emit_status(stage_record.stage_id, "running")

    def finish_stage(
        self, stage_record: StageRecord, status: str, started: float, measured: bool
    ) -> None:
        """Ghi trạng thái cuối của một stage, kèm thời gian chạy và VRAM, rồi phát sự kiện.

        Args:
            stage_record: Stage trong ``run.json``.
            status: Trạng thái cuối.
            started: Thời điểm bắt đầu theo ``time.monotonic``.
            measured: ``True`` khi stage chạy thật lần này: cộng thời gian lần này vào
                ``duration_s`` và giữ VRAM lớn nhất; ``False`` khi dùng lại kết quả cũ: giữ nguyên.
        """
        stage_record.status = status  # type: ignore[assignment]
        if measured:
            # Cộng dồn: một stage bị ngắt rồi chạy tiếp có thời gian của mọi lần chạy.
            elapsed = time.monotonic() - started
            stage_record.duration_s = round(stage_record.duration_s + elapsed, 3)
            stage_record.peak_vram_mb = max(stage_record.peak_vram_mb, _peak_vram_mb())
        self.save()
        self.emit_status(stage_record.stage_id, status)

    def fail_stage(self, stage_record: StageRecord, error: Exception, started: float) -> None:
        """Ghi lại một stage bị lỗi và chuyển cả run sang ``failed``.

        Lỗi ghi vào ``run.json`` với mã ``stage_failed`` và thông báo tiếng Việt cho người dùng;
        traceback đầy đủ ghi nối vào ``d2s/log.txt`` cho người sửa lỗi. Sau đó phát sự kiện
        ``log``, ``status`` và ``done``.

        Args:
            stage_record: Bản ghi của stage trong ``run.json``.
            error: Lỗi stage ném ra.
            started: Thời điểm stage bắt đầu, theo ``time.monotonic``.
        """
        stage_id = stage_record.stage_id
        stage_record.error = ErrorInfo(
            code="stage_failed",
            message=f"stage {stage_id} lỗi: {type(error).__name__}: {error}",
            details={"log": f"{D2S_DIR}/{LOG_FILE}"},
        )
        log_path = self.run_dir / D2S_DIR / LOG_FILE
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{_now_utc().isoformat()}] stage {stage_id} failed\n")
            log_file.write(
                "".join(traceback.format_exception(type(error), error, error.__traceback__))
            )
        self.sink(
            {
                "event": "log",
                "run_id": self.record.run_id,
                "stage_id": stage_id,
                "level": "error",
                "message": stage_record.error.message,
            }
        )
        self.finish_stage(stage_record, "failed", started, measured=True)
        self.record.status = "failed"
        self.save()
        self.emit_status(None, "failed")
        self.sink({"event": "done", "run_id": self.record.run_id, "status": "failed"})


def execute_run(
    run_dir: Path,
    workdir: Workdir,
    registry: StageRegistry = REGISTRY,
    sink: EventSink = _ignore_event,
) -> RunRecord:
    """Chạy, hoặc chạy tiếp, một run đã tạo bằng ``create_run``.

    Đây là hàm CLI (``d2s run start``, ``d2s run resume``) và Studio gọi để chạy run. Stage đã
    xong được bỏ qua: stage có cache nhận lại thư mục cache, stage khác nhận lại thư mục có file
    ``done``. Stage lỗi làm run dừng ở ``failed``; gọi lại hàm này sau khi sửa lỗi thì run chạy
    tiếp từ stage lỗi.

    Args:
        run_dir: Thư mục run.
        workdir: Thư mục làm việc.
        registry: Bảng stage; mặc định là bảng chung.
        sink: Hàm nhận sự kiện tiến độ dạng ``{"event": ..., "run_id": ..., ...}`` (CONTRACTS
            mục 11.4); mặc định bỏ qua sự kiện.

    Returns:
        Nội dung ``run.json`` sau khi chạy, với ``status`` là ``succeeded`` hoặc ``failed``.

    Raises:
        KeyboardInterrupt: Khi người dùng ngắt; run được ghi ``cancelled`` trước.
        ResolveError: Nếu tham số đã lưu không còn hợp lệ với code hiện tại của stage.
    """
    record = files.load(Path(run_dir) / D2S_DIR / RUN_FILE, RunRecord)
    record_machine(Path(run_dir))
    return _Runner(Path(run_dir), record, workdir, registry, sink).execute()


def list_runs(workdir: Workdir, scene_id: Optional[str] = None) -> list[RunRecord]:
    """Đọc ``run.json`` của mọi run trong thư mục làm việc, mới nhất trước.

    Args:
        workdir: Thư mục làm việc.
        scene_id: Chỉ lấy run của cảnh này; mặc định lấy mọi cảnh.

    Returns:
        Các run, sắp theo ``run_id`` giảm dần (``run_id`` bắt đầu bằng thời điểm tạo).
    """
    pattern = f"{scene_id or '*'}/*/*/{D2S_DIR}/{RUN_FILE}"
    records = [files.load(path, RunRecord) for path in workdir.runs_dir.glob(pattern)]
    return sorted(records, key=lambda record: record.run_id, reverse=True)


def summary_path(run_dir: Path) -> Path:
    """Trả về đường dẫn ``summary.json`` của một run (CONTRACTS mục 6.3).

    ``eval.standard`` ghi file này vào thư mục của nó: ``d2s/stages/eval.standard/summary.json``.
    """
    return Path(run_dir) / D2S_DIR / "stages" / "eval.standard" / "summary.json"


def load_summary(run_dir: Path) -> Optional[Summary]:
    """Đọc ``summary.json`` của một run nếu đã có.

    Args:
        run_dir: Thư mục run.

    Returns:
        Kết quả đánh giá, hoặc ``None`` nếu run chưa tới bước đánh giá.
    """
    path = summary_path(run_dir)
    return files.load(path, Summary) if path.is_file() else None


def load_env(run_dir: Path) -> Optional[EnvRecord]:
    """Đọc ``d2s/env.json`` của một run: phiên bản gói và các máy đã chạy run.

    Returns:
        Bản ghi, hoặc ``None`` nếu không có hoặc đọc lỗi.
    """
    try:
        return files.load(Path(run_dir) / D2S_DIR / ENV_FILE, EnvRecord)
    except (OSError, ValueError):
        return None


def run_is_active(run_dir: Path) -> bool:
    """Cho biết run có đang được một tiến trình còn sống chạy không.

    Engine ghi pid vào ``d2s/pid`` khi bắt đầu chạy và xoá khi xong. File còn nhưng tiến trình đã
    chết (máy tắt, tiến trình bị kill) thì run không còn chạy, dù ``run.json`` có thể vẫn ghi
    ``running``.

    Args:
        run_dir: Thư mục run.

    Returns:
        ``True`` nếu có tiến trình còn sống đang chạy run.
    """
    pid_file = Path(run_dir) / D2S_DIR / PID_FILE
    try:
        fields = pid_file.read_text().split()
        pid = int(fields[0])
    except (OSError, ValueError, IndexError):
        return False
    # File pid ghi trên máy khác (ví dụ notebook Kaggle bị ngắt rồi output được tải về): pid đó
    # không nói gì về máy này; trước đây nó trùng một tiến trình bất kỳ ở đây và run bị coi là
    # đang chạy mãi.
    if len(fields) > 1 and fields[1] != platform.node():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def request_cancel(run_dir: Path) -> RunRecord:
    """Yêu cầu huỷ một run.

    Nếu run đang chạy, ghi file ``d2s/cancel``: engine dừng trước stage tiếp theo, và stage đang
    chạy lệnh ngoài dừng lệnh đó trong khoảng một giây. Nếu run không chạy (chờ trong hàng đợi
    sẽ bị huỷ khi tới lượt, còn run ghi ``running`` nhưng tiến trình đã chết) thì ghi luôn
    ``cancelled`` vào ``run.json``.

    Args:
        run_dir: Thư mục run.

    Returns:
        ``run.json`` sau khi ghi yêu cầu.
    """
    run_dir = Path(run_dir)
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    if record.status in ("succeeded", "failed", "cancelled"):
        return record
    (run_dir / D2S_DIR / CANCEL_FILE).write_text(_now_utc().isoformat() + "\n")
    if not run_is_active(run_dir) and record.status == "running":
        for stage in record.stages:
            if stage.status == "running":
                stage.status = "cancelled"
        record.status = "cancelled"
        (run_dir / D2S_DIR / CANCEL_FILE).unlink(missing_ok=True)
        files.dump(record, run_dir / D2S_DIR / RUN_FILE)
    return record


def reset_stage(run_dir: Path, stage_id: str, registry: StageRegistry = REGISTRY) -> RunRecord:
    """Chuẩn bị chạy lại một stage của run, hoặc thêm stage đó vào cuối run nếu chưa có.

    Dùng cho ``d2s export ply|preview <run_id>``: stage export được chạy lại mà không chạy lại
    train. Stage không cache bị xoá thư mục kết quả (kể cả file ``done``); stage có cache chỉ được
    đặt lại trạng thái, vì kết quả cache dùng chung. Sau đó gọi ``execute_run`` để chạy: các stage
    khác đã xong được bỏ qua.

    Args:
        run_dir: Thư mục run.
        stage_id: Stage cần chạy lại hoặc thêm, ví dụ ``export.ply_world``.
        registry: Bảng stage.

    Returns:
        ``run.json`` sau khi sửa.

    Raises:
        KeyError: Nếu ``stage_id`` chưa đăng ký.
        ResolveError: Nếu stage cần một đầu vào mà không stage nào của run tạo ra.
    """
    run_dir = Path(run_dir)
    cls = registry.get(stage_id)
    record = files.load(run_dir / D2S_DIR / RUN_FILE, RunRecord)
    existing = [stage for stage in record.stages if stage.stage_id == stage_id]
    if existing:
        stage = existing[0]
        stage.status = "pending"
        stage.error = None
        if not cls.cacheable:
            shutil.rmtree(run_dir / D2S_DIR / "stages" / stage_id, ignore_errors=True)
    else:
        record.stages.append(
            StageRecord(
                stage_id=stage_id,
                status="pending",
                params=validate_params(stage_id, {}, registry),
            )
        )
        wire_inputs([stage.stage_id for stage in record.stages], registry)
    record.status = "pending"
    files.dump(record, run_dir / D2S_DIR / RUN_FILE)
    return record
