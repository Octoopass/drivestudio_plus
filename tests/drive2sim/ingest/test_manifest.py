"""Kiểm tra phiên bản đầu ra của converter và cách nhận ra cảnh cũ (``ingest/manifest.py``).

File này kiểm tra:
- ``current_output_version`` tra đúng phiên bản theo tên converter (KITTI-360, mọi converter
  DriveStudio dùng chung một số, converter lạ không có);
- ``outdated_output_version`` coi manifest không ghi phiên bản là phiên bản 1, trừ commit đã có
  đầu ra mới mà chưa ghi trường; cảnh mới và converter lạ không bị coi là cũ;
- manifest ``d2s-manifest/1.0`` không có ``output_version`` vẫn đọc được;
- ``outdated_message`` nêu cả hai phiên bản.
"""

import copy
from typing import Optional

import pytest

from drive2sim.core.schema import Manifest
from drive2sim.ingest import kitti360
from drive2sim.ingest.manifest import (
    DRIVESTUDIO_OUTPUT_VERSION,
    KITTI360_OUTPUT_VERSION,
    current_output_version,
    outdated_message,
    outdated_output_version,
)
from tests.drive2sim.samples import MANIFEST


def manifest_with(
    converter: str, output_version: Optional[int], converter_version: str = "0.1.0+test"
) -> Manifest:
    data = copy.deepcopy(MANIFEST)
    data["source"]["converter"] = converter
    data["source"]["converter_version"] = converter_version
    if output_version is not None:
        data["source"]["output_version"] = output_version
    return Manifest.model_validate(data)


def test_current_output_version_by_converter() -> None:
    assert KITTI360_OUTPUT_VERSION == 2  # bản khớp Mei mới
    assert current_output_version(kitti360.CONVERTER) == KITTI360_OUTPUT_VERSION
    assert current_output_version("drivestudio.nuscenes") == DRIVESTUDIO_OUTPUT_VERSION
    assert current_output_version("drivestudio.kitti") == DRIVESTUDIO_OUTPUT_VERSION
    assert current_output_version("custom") is None


@pytest.mark.parametrize(
    ("converter", "output_version", "converter_version", "expected"),
    [
        ("kitti360", None, "0.1.0+e567e29", (1, 2)),  # manifest cũ, KITTI-360 đã đổi đầu ra
        ("kitti360", 1, "0.1.0+e567e29", (1, 2)),
        ("kitti360", 2, "0.1.0+abc1234", None),
        ("kitti360", None, "0.1.0+188fb9b", None),  # commit đã có đầu ra 2, chưa ghi trường
        ("drivestudio.nuscenes", None, "0.1.0+e567e29", None),  # DriveStudio chưa đổi đầu ra
        ("drivestudio.nuscenes", 1, "0.1.0+e567e29", None),
        ("drivestudio.nuscenes", None, "0.1.0+188fb9b", None),
        ("custom", None, "0.1.0+e567e29", None),  # converter lạ: không xét
    ],
)
def test_outdated_output_version(
    converter: str,
    output_version: Optional[int],
    converter_version: str,
    expected: Optional[tuple[int, int]],
) -> None:
    manifest = manifest_with(converter, output_version, converter_version)
    assert outdated_output_version(manifest.source) == expected
    assert (outdated_message(manifest) is None) == (expected is None)


def test_old_manifest_without_output_version_still_validates() -> None:
    data = copy.deepcopy(MANIFEST)
    data["schema_version"] = "d2s-manifest/1.0"
    assert "output_version" not in data["source"]
    assert Manifest.model_validate(data).source.output_version is None
    data["source"]["output_version"] = 0
    with pytest.raises(ValueError):
        Manifest.model_validate(data)


def test_outdated_message_names_both_versions() -> None:
    message = outdated_message(manifest_with("kitti360", None))
    assert message is not None
    assert "converter kitti360 cũ" in message and "phiên bản 1, hiện tại 2" in message
