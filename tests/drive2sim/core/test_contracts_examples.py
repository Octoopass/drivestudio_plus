"""Mọi ví dụ JSON/YAML có ``schema_version`` trong CONTRACTS.md phải hợp lệ theo schema.

Test này giữ cho tài liệu và code không lệch nhau: sửa định dạng file thì phải sửa cả hai.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from drive2sim.core import identifiers
from drive2sim.core.schema import MODELS_BY_KIND

CONTRACTS = Path(__file__).resolve().parents[3] / "docs" / "drive2sim" / "CONTRACTS.md"
BLOCK_RE = re.compile(r"```(json|yaml)\n(.*?)```", re.S)


def examples() -> list:
    """Lấy mọi khối JSON/YAML có ``schema_version`` trong CONTRACTS.md, kèm loại file."""
    found = []
    for lang, body in BLOCK_RE.findall(CONTRACTS.read_text(encoding="utf-8")):
        if "schema_version" not in body:
            continue
        data = json.loads(body) if lang == "json" else yaml.safe_load(body)
        kind, _, _ = identifiers.parse_schema_version(data["schema_version"])
        found.append(pytest.param(kind, data, id=kind))
    return found


def test_found_all_kinds() -> None:
    kinds = {p.values[0] for p in examples()}
    assert kinds == set(MODELS_BY_KIND), f"examples missing for {set(MODELS_BY_KIND) - kinds}"


@pytest.mark.parametrize("kind,data", examples())
def test_example_validates(kind: str, data: dict) -> None:
    MODELS_BY_KIND[kind].model_validate(data)
