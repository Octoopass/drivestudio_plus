"""Kiểm tra môi trường Python của máy này có đúng môi trường mà approach cần không.

Approach dòng DriveStudio chạy trong môi trường chính, mô tả ở ``envs/main.yml`` và cài bằng
``envs/install_main.sh`` (cùng một script cho máy cá nhân và notebook Kaggle). Lệch phiên bản
PyTorch hay gsplat có thể vẫn chạy được nhưng cho kết quả khác đi, nên bảng kiểm tra tách ba mức:
``ok``, ``warn`` (có nhưng khác phiên bản: vẫn cho chạy, kết quả có thể lệch so với run khác) và
``missing`` (thiếu: run sẽ lỗi).

Chỉ kiểm cho run chạy trên máy này; run trên Kaggle cài môi trường trong notebook.

File này chứa:
- ``EXPECTED``, ``environment_check``: bảng kiểm tra cho ``approach check`` và Studio; kết quả có
  ``message`` là câu báo lỗi chung khi mức ``missing``.
- ``EnvironmentMissingError``, ``require_local_environment``: ``d2s run start --executor local``,
  ``d2s run resume`` và ``POST /api/runs`` gọi trước khi tạo hoặc chạy run trên máy, để dừng sớm
  bằng một câu tiếng Việt (mã lỗi ``env_missing``) thay vì để stage đầu tiên lỗi giữa chừng.
- ``MissingModuleError``, ``missing_module_message``: lệnh con của stage, ingest không nạp được
  một gói Python (``ModuleNotFoundError``, ``ImportError``); ``run_logged``
  (``drive2sim/stages/commands.py``) đổi lỗi đó thành câu cùng hướng dẫn, không in traceback.

Lý do kiểm sớm: stage chạy script bằng chính Python đang chạy ``d2s`` (``sys.executable``). Chạy
``d2s`` từ ``.venv`` nhẹ (chỉ để test, lint) thì script lỗi ngay ở ``import numpy``; người dùng mới
chỉ thấy traceback mà không biết phải cài môi trường nào hay có thể chạy trên Kaggle, Colab.
"""

import time
from pathlib import Path
from typing import Any, Optional

from drive2sim.engine.machine import _gpus, package_versions
from drive2sim.engine.render_check import _torch_cuda
from drive2sim.paths import SMPL_MODEL_NAME, find_smpl_model

# Phiên bản của môi trường chính; phải khớp envs/main.yml và envs/install_main.sh (có test kiểm).
EXPECTED = {"main": {"pytorch": "2.4.1", "cuda": "12.1", "gsplat": "1.5.3"}}
# Codebase của approach → môi trường cần; codebase không có ở đây (ví dụ approach mẫu) thì không
# cần môi trường GPU nào.
CODEBASE_ENV = {"drivestudio": "main", "external:nurec": "nurec", "external:xsim": "xsim"}
# Lệnh dựng môi trường chính và cách chạy từ xa, dùng trong câu báo thiếu môi trường.
INSTALL_STEPS = {
    "main": "conda env create -f envs/main.yml && conda activate drive2sim && "
    "bash envs/install_main.sh",
    "nurec": "bash envs/install_nurec.sh",
    "xsim": "bash envs/install_xsim.sh",
}
REMOTE_HINT = "hoặc chạy trên GPU của Kaggle, Colab: --executor kaggle hoặc --executor colab"
_CHECK_LABELS = {
    "gpu": "GPU NVIDIA",
    "pytorch": "PyTorch",
    "cuda": "CUDA của PyTorch",
    "nurec_source": "mã nguồn Instant NuRec",
    "nurec_env": "môi trường Python của Instant NuRec",
    "nurec_checkpoint": "checkpoint Instant NuRec",
    "xsim_source": "mã nguồn XSIM",
    "xsim_env": "môi trường Python của XSIM",
}
_MAIN_PACKAGES = ("pytorch", "cuda", "gsplat")
# Hỏi PyTorch tốn vài giây (tiến trình con), nên giữ kết quả một lúc.
_CUDA_CACHE_S = 120
_cuda_cache: dict[str, Any] = {}


def _cached_torch_cuda() -> tuple[bool, str]:
    now = time.monotonic()
    if _cuda_cache and now - _cuda_cache["at"] < _CUDA_CACHE_S:
        return _cuda_cache["value"]
    value = _torch_cuda()
    _cuda_cache.update(at=now, value=value)
    return value


def _level(found: Optional[str], expected: str) -> str:
    if not found:
        return "missing"
    return "ok" if found.split("+")[0] == expected else "warn"


def environment_check(
    codebase: str,
    needs_smpl: bool = False,
    workdir_root: Optional[Path] = None,
    smpl_optional: bool = False,
) -> dict[str, Any]:
    """Bảng kiểm tra môi trường cho một codebase.

    Args:
        codebase: Trường ``codebase`` của file approach, ví dụ ``drivestudio``.
        needs_smpl: Approach có dựng người bằng SMPL không (có stage ``prep.smpl``); khi đó cần
            thêm file mô hình SMPL (``drive2sim.paths.find_smpl_model``).
        workdir_root: Thư mục làm việc, một trong các chỗ tìm file SMPL.
        smpl_optional: SMPL chỉ có thể cần (tuỳ chọn ``auto`` chưa biết split có người không):
            thiếu file thì ``warn`` thay vì ``missing``.

    Returns:
        ``{"env": tên hoặc None, "file", "install", "level": "ok" | "warn" | "missing",
        "checks": [{"id", "level", "found", "expected"}], "message"}``. ``env`` là ``None`` khi
        codebase không cần môi trường GPU (``checks`` rỗng, ``level`` ``ok``). ``message`` là câu
        báo lỗi tiếng Việt khi ``level`` là ``missing`` (``missing_environment_message``), không
        thì ``None``.
    """
    env = CODEBASE_ENV.get(codebase)
    if env is None:
        return {
            "env": None,
            "file": None,
            "install": None,
            "level": "ok",
            "checks": [],
            "message": None,
        }
    if env == "nurec":
        return _nurec_check()
    if env == "xsim":
        # Chuẩn bị dữ liệu và chấm điểm của run XSIM chạy ở môi trường chính: kiểm cả hai.
        return _xsim_check(
            environment_check("drivestudio", needs_smpl, workdir_root, smpl_optional)
        )
    expected = EXPECTED[env]
    versions = package_versions()
    gpus = _gpus()
    checks = [
        {
            "id": "gpu",
            "level": "ok" if gpus else "missing",
            "found": gpus[0][0] if gpus else None,
            "expected": None,
        },
        {
            "id": "pytorch",
            "level": _level(versions["pytorch"], expected["pytorch"]),
            "found": versions["pytorch"],
            "expected": expected["pytorch"],
        },
    ]
    if versions["pytorch"]:
        ok, info = _cached_torch_cuda()
        cuda_level = _level(versions["cuda"], expected["cuda"]) if ok else "missing"
        checks.append(
            {
                "id": "cuda",
                "level": cuda_level,
                "found": versions["cuda"] if ok else (info or None),
                "expected": expected["cuda"],
            }
        )
    else:
        checks.append(
            {"id": "cuda", "level": "missing", "found": None, "expected": expected["cuda"]}
        )
    checks.append(
        {
            "id": "gsplat",
            "level": _level(versions["gsplat"], expected["gsplat"]),
            "found": versions["gsplat"],
            "expected": expected["gsplat"],
        }
    )
    if needs_smpl:
        model = find_smpl_model(workdir_root)
        checks.append(
            {
                "id": "smpl_model",
                "level": "ok" if model else "warn" if smpl_optional else "missing",
                "found": str(model) if model else None,
                "expected": SMPL_MODEL_NAME,
            }
        )
    levels = {check["level"] for check in checks}
    level = "missing" if "missing" in levels else "warn" if "warn" in levels else "ok"
    result: dict[str, Any] = {
        "env": env,
        "file": f"envs/{env}.yml",
        "install": f"bash envs/install_{env}.sh",
        "level": level,
        "checks": checks,
    }
    result["message"] = missing_environment_message(result)
    return result


def _nurec_check() -> dict[str, Any]:
    """Bảng kiểm tra môi trường của Instant NuRec (``drive2sim/external/nurec``)."""
    from drive2sim.external import nurec

    gpus = _gpus()
    checks = [
        {
            "id": "gpu",
            "level": "ok" if gpus else "missing",
            "found": gpus[0][0] if gpus else None,
            "expected": None,
        },
        *nurec.environment_checks(),
    ]
    levels = {check["level"] for check in checks}
    level = "missing" if "missing" in levels else "warn" if "warn" in levels else "ok"
    result: dict[str, Any] = {
        "env": "nurec",
        "file": str(nurec.nurec_dir()),
        "install": INSTALL_STEPS["nurec"],
        "level": level,
        "checks": checks,
    }
    result["message"] = missing_environment_message(result)
    return result


def _xsim_check(main: dict[str, Any]) -> dict[str, Any]:
    """Bảng kiểm tra môi trường XSIM (``drive2sim/external/xsim``) cùng môi trường chính.

    Gồm các mục của môi trường chính (``main``, gồm cả file SMPL khi cần) và mã nguồn, venv của
    XSIM. GPU dưới mức README của XSIM khuyến nghị (24 GB) chỉ cảnh báo: chưa đo mức tối thiểu.
    """
    from drive2sim.external import xsim

    gpus = _gpus()
    memory = max((gpu[1] or 0 for gpu in gpus), default=0)
    checks = [
        {
            "id": "gpu",
            "level": (
                "missing" if not gpus else "ok" if memory >= xsim.RECOMMENDED_VRAM_MB else "warn"
            ),
            "found": f"{gpus[0][0]} ({memory} MB)" if gpus else None,
            "expected": f">= {xsim.RECOMMENDED_VRAM_MB // 1024} GB",
        },
        *(check for check in main["checks"] if check["id"] != "gpu"),
        *xsim.environment_checks(),
    ]
    levels = {check["level"] for check in checks}
    level = "missing" if "missing" in levels else "warn" if "warn" in levels else "ok"
    result: dict[str, Any] = {
        "env": "xsim",
        "file": str(xsim.xsim_dir()),
        "install": INSTALL_STEPS["xsim"],
        "level": level,
        "checks": checks,
    }
    result["message"] = missing_environment_message(result)
    return result


def missing_environment_message(env: dict[str, Any]) -> Optional[str]:
    """Viết câu báo thiếu môi trường từ kết quả ``environment_check``.

    Câu đầu nói thiếu gì; sau đó là cách sửa: lệnh dựng môi trường chính khi thiếu gói, chỗ đặt
    file SMPL khi thiếu file đó, và luôn kèm cách chạy trên Kaggle, Colab (không cần GPU trên máy).
    Thiếu GPU thì không gợi ý cài gói, vì cài xong vẫn không chạy được.

    Args:
        env: Kết quả của ``environment_check``.

    Returns:
        Câu tiếng Việt, ví dụ ``"chưa chạy được trên máy này: môi trường main thiếu PyTorch
        2.4.1, gsplat 1.5.3. Cài: conda env create ... ; hoặc chạy trên GPU của Kaggle, Colab:
        ..."``; ``None`` nếu ``level`` không phải ``missing``.
    """
    if env.get("level") != "missing":
        return None
    missing = [check for check in env["checks"] if check["level"] == "missing"]
    packages = [c for c in missing if c["id"] not in ("gpu", "smpl_model")]
    items = [
        " ".join(
            part for part in (_CHECK_LABELS.get(c["id"], c["id"]), c["expected"] or "") if part
        )
        for c in missing
        if c["id"] != "smpl_model"
    ]
    fixes = []
    # Run XSIM cần cả môi trường chính: gợi ý đúng script cho phần đang thiếu.
    main_missing = env["env"] == "xsim" and any(c["id"] in _MAIN_PACKAGES for c in packages)
    if main_missing:
        fixes.append(f"cài môi trường chính: {INSTALL_STEPS['main']}")
    if any(not (main_missing and c["id"] in _MAIN_PACKAGES) for c in packages):
        fixes.append(
            f"cài: {INSTALL_STEPS.get(env['env'], env['install'])}, rồi chạy lại lệnh trong "
            "môi trường đó"
        )
    if any(c["id"] == "smpl_model" for c in missing):
        items.append(SMPL_MODEL_NAME)
        fixes.append(
            f"đặt {SMPL_MODEL_NAME} ở <thư mục làm việc>/smpl_models/ hoặc trỏ biến "
            "D2S_SMPL_MODEL tới nó (tải ở https://smpl.is.tue.mpg.de)"
        )
    fixes.append(REMOTE_HINT)
    return (
        f"chưa chạy được trên máy này: môi trường {env['env']} thiếu {', '.join(items)}. "
        + "; ".join(fixes)
    )


class EnvironmentMissingError(Exception):
    """Máy này thiếu môi trường để chạy run trên máy (mã lỗi ``env_missing``).

    ``d2s run start --executor local``, ``d2s run resume`` và ``POST /api/runs`` ném lỗi này
    trước khi tạo hoặc chạy run; CLI trả mã thoát ``3``, API trả ``422`` (CONTRACTS mục 11.3).

    Attributes:
        environment: Kết quả ``environment_check`` gây ra lỗi.
    """

    def __init__(self, message: str, environment: dict[str, Any]) -> None:
        super().__init__(message)
        self.environment = environment


def require_local_environment(
    codebase: str, needs_smpl: bool = False, workdir_root: Optional[Path] = None
) -> dict[str, Any]:
    """Dừng sớm nếu máy này thiếu môi trường để chạy approach trên máy.

    Chỉ chặn mức ``missing`` (run chắc chắn lỗi); mức ``warn`` (khác phiên bản) vẫn cho chạy như
    ``approach check`` đã báo.

    Args:
        codebase: Trường ``codebase`` của file approach.
        needs_smpl: Run có stage tạo artifact ``smpl`` (cần file mô hình SMPL).
        workdir_root: Thư mục làm việc, một chỗ tìm file SMPL.

    Returns:
        Kết quả ``environment_check`` khi không thiếu gì.

    Raises:
        EnvironmentMissingError: Nếu mức là ``missing``; thông báo theo
            ``missing_environment_message``.
    """
    env = environment_check(codebase, needs_smpl=needs_smpl, workdir_root=workdir_root)
    if env["message"]:
        raise EnvironmentMissingError(env["message"], env)
    return env


class MissingModuleError(RuntimeError):
    """Lệnh con không nạp được một gói Python: môi trường đang chạy ``d2s`` thiếu gói.

    ``run_logged`` ném lỗi này thay cho ``RuntimeError`` kèm traceback khi đầu ra của lệnh kết thúc
    bằng ``ModuleNotFoundError`` hoặc ``ImportError``. Là lớp con của ``RuntimeError`` nên nơi đã
    bắt ``RuntimeError`` (ingest) vẫn bắt được; engine ghi mã ``env_missing`` cho stage.

    Attributes:
        detail: Dòng lỗi gốc, ví dụ ``"ModuleNotFoundError: No module named 'numpy'"``.
    """

    def __init__(self, message: str, detail: str) -> None:
        super().__init__(message)
        self.detail = detail


def missing_module_message(detail: str, python: str, log_name: str) -> str:
    """Viết câu báo lỗi khi lệnh con thiếu gói Python.

    Args:
        detail: Dòng lỗi gốc của Python, ví dụ ``"ModuleNotFoundError: No module named 'numpy'"``.
        python: Python đã chạy lệnh (``sys.executable``), để người dùng thấy đang ở môi trường nào.
        log_name: Tên file log có traceback đầy đủ.

    Returns:
        Câu tiếng Việt: thiếu gì, cách dựng môi trường chính, cách chạy trên Kaggle, Colab.
    """
    return (
        f"Python đang chạy d2s ({python}) thiếu gói mà bước này cần ({detail}); đây không phải "
        f"môi trường drive2sim. Cài: {INSTALL_STEPS['main']}, rồi chạy lại lệnh trong môi trường "
        "đó; hoặc để Kaggle, Colab chạy bước này (run: --executor kaggle hoặc --executor "
        f"colab; ingest: d2s kaggle ingest). Traceback đầy đủ ở {log_name}"
    )
