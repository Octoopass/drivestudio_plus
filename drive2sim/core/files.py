"""Đọc và ghi các file YAML/JSON của tool qua schema.

File người viết tay dùng YAML, file tool ghi ra dùng JSON (CONTRACTS mục 10.2). Mọi lần ghi đi
qua một file tạm rồi đổi tên, nên tiến trình bị ngắt giữa chừng không để lại file hỏng.
"""

import json
import os
from pathlib import Path
from typing import Any, TypeVar, Union

import yaml
from pydantic import BaseModel

from drive2sim.core import identifiers
from drive2sim.core.schema import MODELS_BY_KIND

M = TypeVar("M", bound=BaseModel)
PathLike = Union[str, Path]


def _read_raw(path: Path) -> dict[str, Any]:
    """Đọc file YAML hoặc JSON thành dict, chưa kiểm tra schema.

    Args:
        path: Đường dẫn file; đuôi ``.yaml``, ``.yml`` hoặc ``.json``.

    Returns:
        Nội dung file.

    Raises:
        ValueError: Nếu đuôi file không được hỗ trợ hoặc cấp ngoài cùng không phải mapping.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported file type: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load(path: PathLike, model: type[M]) -> M:
    """Đọc một file và kiểm tra theo schema ``model``.

    Args:
        path: Đường dẫn file YAML hoặc JSON.
        model: Lớp schema trong ``drive2sim.core.schema``, ví dụ ``Approach``.

    Returns:
        Đối tượng schema đã kiểm tra.

    Raises:
        ValueError: Nếu đọc file lỗi hoặc nội dung sai schema; thông báo có kèm tên file.
    """
    path = Path(path)
    try:
        return model.model_validate(_read_raw(path))
    except Exception as error:
        # Gắn tên file vào lỗi để người dùng biết file nào sai.
        raise ValueError(f"{path}: {error}") from error


def load_any(path: PathLike) -> BaseModel:
    """Đọc một file, tự chọn schema theo trường ``schema_version``.

    Args:
        path: Đường dẫn file YAML hoặc JSON.

    Returns:
        Đối tượng schema tương ứng với loại file.

    Raises:
        ValueError: Nếu thiếu ``schema_version``, loại file không biết, hoặc nội dung sai schema.
    """
    path = Path(path)
    data = _read_raw(path)
    kind, _, _ = identifiers.parse_schema_version(data.get("schema_version", ""))
    if kind not in MODELS_BY_KIND:
        raise ValueError(f"{path}: unknown schema kind {kind!r}")
    return load(path, MODELS_BY_KIND[kind])


def dump(obj: BaseModel, path: PathLike) -> Path:
    """Ghi một đối tượng schema ra file JSON hoặc YAML (theo đuôi file).

    Trường có giá trị ``None`` được bỏ đi. Ghi vào ``<tên>.tmp`` rồi đổi tên, nên file đích luôn
    là bản cũ đầy đủ hoặc bản mới đầy đủ.

    Args:
        obj: Đối tượng schema cần ghi.
        path: Đường dẫn file đích; thư mục cha được tạo nếu chưa có.

    Returns:
        Đường dẫn file đã ghi.

    Raises:
        ValueError: Nếu đuôi file không được hỗ trợ.
    """
    path = Path(path)
    data = obj.model_dump(mode="json", exclude_none=True)
    if path.suffix in (".yaml", ".yml"):
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    elif path.suffix == ".json":
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    else:
        raise ValueError(f"unsupported file type: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # đổi tên là thao tác nguyên tử trên cùng một ổ đĩa
    return path
