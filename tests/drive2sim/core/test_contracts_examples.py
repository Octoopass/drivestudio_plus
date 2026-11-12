"""Every JSON/YAML example with a schema_version in CONTRACTS.md must validate against its schema."""
import json
import re
from pathlib import Path

import pytest
import yaml

from drive2sim.core import ids
from drive2sim.core.schema import MODELS_BY_KIND

CONTRACTS = Path(__file__).resolve().parents[3] / "docs" / "drive2sim" / "CONTRACTS.md"
BLOCK_RE = re.compile(r"```(json|yaml)\n(.*?)```", re.S)


def examples():
    out = []
    for lang, body in BLOCK_RE.findall(CONTRACTS.read_text(encoding="utf-8")):
        if "schema_version" not in body:
            continue
        data = json.loads(body) if lang == "json" else yaml.safe_load(body)
        kind, _, _ = ids.parse_schema_version(data["schema_version"])
        out.append(pytest.param(kind, data, id=kind))
    return out


def test_found_all_kinds():
    kinds = {p.values[0] for p in examples()}
    assert kinds == set(MODELS_BY_KIND), f"examples missing for {set(MODELS_BY_KIND) - kinds}"


@pytest.mark.parametrize("kind,data", examples())
def test_example_validates(kind, data):
    MODELS_BY_KIND[kind].model_validate(data)
