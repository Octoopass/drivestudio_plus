"""Kiểm tra định dạng và tạo các định danh của tool (``scene_id``, ``stage_id``, ``run_id``...).

Mọi định danh theo một quy tắc chung ở CONTRACTS mục 10.1, để tên thư mục, tên file và khoá
trong JSON đọc là hiểu và không trùng nhau.

File này chứa:
- Hằng ``DATASETS``, ``STAGE_GROUPS`` và các biểu thức chính quy của từng loại định danh.
- Các hàm ``check_*`` (``check_scene_id``, ``check_stage_id``, ``check_run_id``,
  ``check_cache_key``, ``check_version``, ``check_snake_case``): kiểm tra một giá trị đúng định
  dạng. Mỗi hàm trả lại chính giá trị nếu đúng và ném ``ValueError`` nếu sai, để dùng trực tiếp
  trong validator của Pydantic (``schema.py``).
- ``make_cache_key``, ``make_run_id``: tạo khoá cache và ``run_id`` mới.
- ``canonical_json``, ``short_hash``: chuẩn hoá dữ liệu thành JSON ổn định rồi băm, dùng cho
  khoá cache.
- ``parse_schema_version``: tách ``d2s-<loại>/<major>.<minor>``.

Thông báo lỗi viết tiếng Anh vì là thông báo kỹ thuật; CLI và API bọc lại thành tiếng Việt
(CONTRACTS mục 10.6).
"""

import hashlib
import json
import re
import secrets
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Optional

# Nguồn dữ liệu được hỗ trợ; cũng là tiền tố của scene_id.
DATASETS: tuple[str, ...] = (
    "waymo",
    "nuscenes",
    "nuplan",
    "argoverse2",
    "pandaset",
    "kitti",
    "kitti360",
    "physicalai",
    "carla",
    "colmap",
    "custom",
)
# Các nhóm stage; là phần đầu của stage_id.
STAGE_GROUPS: tuple[str, ...] = ("prep", "init", "train", "eval", "render", "export")

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SCENE_ID_RE = re.compile(r"^(" + "|".join(DATASETS) + r")_[a-z0-9][a-z0-9_-]*$")
STAGE_ID_RE = re.compile(r"^(" + "|".join(STAGE_GROUPS) + r")\.[a-z][a-z0-9_]*$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-z][a-z0-9_]*-[0-9a-f]{4}$")
CACHE_KEY_RE = re.compile(
    r"^(?P<scene_id>[a-z0-9][a-z0-9_-]*)/(?P<stage_id>[a-z]+\.[a-z][a-z0-9_]*)"
    r"@(?P<version>\d+\.\d+\.\d+)-(?P<hash8>[0-9a-f]{8})$"
)
SCHEMA_VERSION_RE = re.compile(r"^d2s-(?P<kind>[a-z]+)/(?P<major>\d+)\.(?P<minor>\d+)$")


def _check(pattern: "re.Pattern[str]", value: str, what: str) -> str:
    """Kiểm tra ``value`` khớp ``pattern``.

    Args:
        pattern: Biểu thức chính quy cần khớp từ đầu đến cuối chuỗi.
        value: Giá trị cần kiểm tra.
        what: Tên định danh, dùng trong thông báo lỗi.

    Returns:
        Chính ``value`` khi hợp lệ.

    Raises:
        ValueError: Nếu ``value`` không phải chuỗi hoặc không khớp.
    """
    if not isinstance(value, str) or not pattern.match(value):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def check_snake_case(value: str, what: str = "name") -> str:
    """Kiểm tra tên dạng ``snake_case`` (approach_id, split_id, tên mục cấu hình, tên preset).

    Args:
        value: Tên cần kiểm tra.
        what: Loại tên, dùng trong thông báo lỗi.

    Returns:
        Chính ``value`` khi hợp lệ.

    Raises:
        ValueError: Nếu sai định dạng.
    """
    return _check(SNAKE_CASE_RE, value, what)


def check_scene_id(value: str) -> str:
    """Kiểm tra ``scene_id`` dạng ``<dataset>_<sequence_id>``, ví dụ ``nuscenes_scene-0061``.

    Args:
        value: Định danh cảnh.

    Returns:
        Chính ``value`` khi hợp lệ.

    Raises:
        ValueError: Nếu tiền tố không phải nguồn dữ liệu được hỗ trợ hoặc có ký tự không hợp lệ.
    """
    return _check(SCENE_ID_RE, value, "scene_id (expected <dataset>_<sequence_id>)")


def check_stage_id(value: str) -> str:
    """Kiểm tra ``stage_id`` dạng ``<nhóm>.<tên>``, ví dụ ``prep.masks_segformer``.

    Args:
        value: Định danh stage.

    Returns:
        Chính ``value`` khi hợp lệ.

    Raises:
        ValueError: Nếu nhóm không thuộc ``STAGE_GROUPS`` hoặc tên không phải ``snake_case``.
    """
    return _check(STAGE_ID_RE, value, "stage_id (expected <group>.<name>)")


def check_version(value: str) -> str:
    """Kiểm tra phiên bản dạng ``MAJOR.MINOR.PATCH``.

    Args:
        value: Chuỗi phiên bản.

    Returns:
        Chính ``value`` khi hợp lệ.

    Raises:
        ValueError: Nếu sai định dạng.
    """
    return _check(VERSION_RE, value, "version (expected MAJOR.MINOR.PATCH)")


def check_run_id(value: str) -> str:
    """Kiểm tra ``run_id`` dạng ``YYYYMMDD-HHMMSS-<approach_id>-<4 ký tự hex>``.

    Args:
        value: Định danh run.

    Returns:
        Chính ``value`` khi hợp lệ.

    Raises:
        ValueError: Nếu sai định dạng.
    """
    return _check(RUN_ID_RE, value, "run_id (expected YYYYMMDD-HHMMSS-<approach_id>-<4 hex>)")


def check_cache_key(value: str) -> str:
    """Kiểm tra khoá cache dạng ``<scene_id>/<stage_id>@<version>-<hash8>``.

    Args:
        value: Khoá cache.

    Returns:
        Chính ``value`` khi hợp lệ.

    Raises:
        ValueError: Nếu sai định dạng chung, hoặc phần ``scene_id``, ``stage_id`` bên trong sai.
    """
    match = CACHE_KEY_RE.match(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(
            f"invalid cache_key (expected <scene_id>/<stage_id>@<version>-<hash8>): {value!r}"
        )
    check_scene_id(match.group("scene_id"))
    check_stage_id(match.group("stage_id"))
    return value


def canonical_json(obj: object) -> str:
    """Chuyển ``obj`` sang JSON dạng chuẩn để băm.

    Khoá được sắp xếp và không có khoảng trắng thừa, nên hai dict có cùng nội dung nhưng khác
    thứ tự khoá cho cùng một chuỗi.

    Args:
        obj: Dữ liệu chuyển được sang JSON.

    Returns:
        Chuỗi JSON.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def short_hash(obj: object) -> str:
    """Lấy 8 ký tự hex đầu của sha256 trên ``canonical_json(obj)``.

    Args:
        obj: Dữ liệu chuyển được sang JSON.

    Returns:
        Chuỗi 8 ký tự hex.
    """
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:8]


def make_cache_key(
    scene_id: str, stage_id: str, version: str, params: dict, input_keys: Iterable[str]
) -> str:
    """Tạo khoá cache cho kết quả của một stage có cache (CONTRACTS mục 5.2).

    Khoá chỉ băm tham số và khoá của các artifact đầu vào, không băm nội dung dữ liệu, vì dữ
    liệu chuẩn không đổi sau khi ingest.

    Args:
        scene_id: Định danh cảnh, dạng ``<dataset>_<sequence_id>``.
        stage_id: Định danh stage, dạng ``<nhóm>.<tên>``.
        version: Phiên bản của stage, dạng ``MAJOR.MINOR.PATCH``.
        params: Tham số đã gộp của stage.
        input_keys: Khoá cache của các artifact đầu vào; thứ tự không quan trọng.

    Returns:
        Khoá dạng ``<scene_id>/<stage_id>@<version>-<hash8>``.

    Raises:
        ValueError: Nếu một định danh sai định dạng.
    """
    check_scene_id(scene_id)
    check_stage_id(stage_id)
    check_version(version)
    # Sắp xếp input_keys để khoá không phụ thuộc thứ tự engine liệt kê đầu vào.
    digest = short_hash({"params": params, "inputs": sorted(input_keys)})
    return f"{scene_id}/{stage_id}@{version}-{digest}"


def make_run_id(
    approach_id: str, now: Optional[datetime] = None, suffix: Optional[str] = None
) -> str:
    """Tạo ``run_id`` dạng ``YYYYMMDD-HHMMSS-<approach_id>-<4 ký tự hex>``.

    Args:
        approach_id: Định danh approach, dạng ``snake_case``.
        now: Thời điểm tạo run; mặc định là bây giờ theo giờ UTC.
        suffix: 4 ký tự hex ở cuối; mặc định sinh ngẫu nhiên để hai run cùng giây không trùng.

    Returns:
        Định danh run.

    Raises:
        ValueError: Nếu ``approach_id`` hoặc ``suffix`` sai định dạng.
    """
    check_snake_case(approach_id, "approach_id")
    now = now or datetime.now(timezone.utc)
    suffix = suffix if suffix is not None else secrets.token_hex(2)
    return check_run_id(f"{now.strftime('%Y%m%d-%H%M%S')}-{approach_id}-{suffix}")


def parse_schema_version(value: str) -> tuple[str, int, int]:
    """Tách ``schema_version`` như ``d2s-run/0.1`` thành loại file và phiên bản.

    Args:
        value: Giá trị của trường ``schema_version``.

    Returns:
        Bộ ``(loại, major, minor)``, ví dụ ``("run", 0, 1)``.

    Raises:
        ValueError: Nếu sai định dạng ``d2s-<loại>/<major>.<minor>``.
    """
    match = SCHEMA_VERSION_RE.match(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"invalid schema_version: {value!r}")
    return match.group("kind"), int(match.group("major")), int(match.group("minor"))
