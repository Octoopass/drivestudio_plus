"""Read and write the tool's YAML/JSON files through their schemas.

YAML for hand-edited files, JSON for files the tool writes (CONTRACTS section 10.2). Writes go to a
temporary file that is then renamed, so an interrupted write never leaves a truncated file.
"""
import json
import os
from pathlib import Path
from typing import Type, TypeVar, Union

import yaml
from pydantic import BaseModel

from drive2sim.core import ids
from drive2sim.core.schema import MODELS_BY_KIND

M = TypeVar("M", bound=BaseModel)
PathLike = Union[str, Path]


def _read_raw(path: Path) -> dict:
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


def load(path: PathLike, model: Type[M]) -> M:
    """Load a file and validate it against `model`."""
    path = Path(path)
    try:
        return model.model_validate(_read_raw(path))
    except Exception as e:  # add the file name to schema errors
        raise ValueError(f"{path}: {e}") from e


def load_any(path: PathLike) -> BaseModel:
    """Load a file and pick the model from its schema_version."""
    path = Path(path)
    data = _read_raw(path)
    kind, _, _ = ids.parse_schema_version(data.get("schema_version", ""))
    if kind not in MODELS_BY_KIND:
        raise ValueError(f"{path}: unknown schema kind {kind!r}")
    return load(path, MODELS_BY_KIND[kind])


def dump(obj: BaseModel, path: PathLike) -> Path:
    """Write a model as JSON or YAML (by file extension), atomically."""
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
    os.replace(tmp, path)
    return path
