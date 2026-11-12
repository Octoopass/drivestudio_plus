"""Identifier formats (docs/drive2sim/CONTRACTS.md section 10.1) and helpers that build them."""
import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

DATASETS = (
    "waymo", "nuscenes", "nuplan", "argoverse2", "pandaset",
    "kitti", "physicalai", "carla", "colmap", "custom",
)
STAGE_GROUPS = ("prep", "init", "train", "eval", "render", "export")

SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SCENE_ID_RE = re.compile(r"^(" + "|".join(DATASETS) + r")_[a-z0-9][a-z0-9_-]*$")
STAGE_ID_RE = re.compile(r"^(" + "|".join(STAGE_GROUPS) + r")\.[a-z][a-z0-9_]*$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-z][a-z0-9_]*-[0-9a-f]{4}$")
HASH8_RE = re.compile(r"^[0-9a-f]{8}$")
CACHE_KEY_RE = re.compile(
    r"^(?P<scene_id>[a-z0-9][a-z0-9_-]*)/(?P<stage_id>[a-z]+\.[a-z][a-z0-9_]*)"
    r"@(?P<version>\d+\.\d+\.\d+)-(?P<hash8>[0-9a-f]{8})$"
)
SCHEMA_VERSION_RE = re.compile(r"^d2s-(?P<kind>[a-z]+)/(?P<major>\d+)\.(?P<minor>\d+)$")


def _check(pattern: "re.Pattern[str]", value: str, what: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def check_snake(value: str, what: str = "name") -> str:
    return _check(SNAKE_RE, value, what)


def check_scene_id(value: str) -> str:
    return _check(SCENE_ID_RE, value, "scene_id (expected <dataset>_<sequence_id>)")


def check_stage_id(value: str) -> str:
    return _check(STAGE_ID_RE, value, "stage_id (expected <group>.<name>)")


def check_version(value: str) -> str:
    return _check(VERSION_RE, value, "version (expected MAJOR.MINOR.PATCH)")


def check_run_id(value: str) -> str:
    return _check(RUN_ID_RE, value, "run_id (expected YYYYMMDD-HHMMSS-<approach_id>-<4 hex>)")


def check_cache_key(value: str) -> str:
    m = CACHE_KEY_RE.match(value) if isinstance(value, str) else None
    if m is None:
        raise ValueError(f"invalid cache_key (expected <scene_id>/<stage_id>@<version>-<hash8>): {value!r}")
    check_scene_id(m.group("scene_id"))
    check_stage_id(m.group("stage_id"))
    return value


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no extra spaces, used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash8(obj: Any) -> str:
    """First 8 hex characters of sha256 over canonical_json(obj)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:8]


def make_cache_key(scene_id: str, stage_id: str, version: str, params: dict, input_keys: Iterable[str]) -> str:
    """Cache key of a cacheable stage result (CONTRACTS section 5.2).

    The hash covers the merged parameters and the keys of the input artifacts, not the data itself.
    """
    check_scene_id(scene_id)
    check_stage_id(stage_id)
    check_version(version)
    digest = hash8({"params": params, "inputs": sorted(input_keys)})
    return f"{scene_id}/{stage_id}@{version}-{digest}"


def make_run_id(approach_id: str, now: Optional[datetime] = None, suffix: Optional[str] = None) -> str:
    """Run id in the form YYYYMMDD-HHMMSS-<approach_id>-<4 hex>, time in UTC."""
    check_snake(approach_id, "approach_id")
    now = now or datetime.now(timezone.utc)
    suffix = suffix if suffix is not None else secrets.token_hex(2)
    return check_run_id(f"{now.strftime('%Y%m%d-%H%M%S')}-{approach_id}-{suffix}")


def parse_schema_version(value: str):
    """Return (kind, major, minor) of a schema_version string such as 'd2s-run/0.1'."""
    m = SCHEMA_VERSION_RE.match(value) if isinstance(value, str) else None
    if m is None:
        raise ValueError(f"invalid schema_version: {value!r}")
    return m.group("kind"), int(m.group("major")), int(m.group("minor"))
