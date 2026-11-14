"""Đọc config method, config dataset có khoá ``extends`` (CONTRACTS mục 10.5).

Một config có thể ghi ``extends: <tên file>`` ở cấp ngoài cùng để kế thừa một file khác cùng thư
mục và chỉ ghi phần khác. Quy tắc gộp:
- mapping gộp sâu theo khoá;
- số, chuỗi, boolean và danh sách ở file con thay hẳn giá trị ở file cha (danh sách không gộp, vì
  thứ tự và số phần tử của nó thường có nghĩa, ví dụ các mốc lịch learning rate);
- khoá có giá trị ``null`` ở file con là xoá khoá đó;
- kế thừa được nhiều tầng; vòng lặp là lỗi; khoá ``extends`` bị bỏ khỏi bản đã gộp.

YAML được đọc giống hệt OmegaConf 2.3 (``omegaconf/_utils.py``, ``get_yaml_loader``): nhận
``1e-15`` là số thực (``yaml.safe_load`` coi là chuỗi), bỏ kiểu timestamp, báo lỗi khi trùng khoá.
Nhờ vậy config gộp ở đây cho cùng giá trị như khi DriveStudio tự đọc bằng ``OmegaConf.load``, mà
không cần cài OmegaConf.

File này chứa ``load_config``, ``merge_config``, ``ConfigError``.
"""

import copy
import re
from pathlib import Path
from typing import Any, Optional, Union

import yaml

EXTENDS_KEY = "extends"


class ConfigError(ValueError):
    """Config không đọc được: YAML sai, file cha không có, ``extends`` sai, vòng lặp."""


class _OmegaConfLoader(yaml.SafeLoader):
    """SafeLoader theo đúng cách OmegaConf 2.3 đọc YAML."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        keys = set()
        for key_node, _ in node.value:
            if key_node.tag != yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG:
                continue
            if key_node.value in keys:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key_node.value}",
                    key_node.start_mark,
                )
            keys.add(key_node.value)
        return super().construct_mapping(node, deep=deep)


_OmegaConfLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?[0-9]+(?:_[0-9]+)*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9]+(?:_[0-9]+)*(?:[eE][-+]?[0-9]+)
    |\\.[0-9]+(?:_[0-9]+)*(?:[eE][-+][0-9]+)?
    |[-+]?[0-9]+(?:_[0-9]+)*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)
_OmegaConfLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for key, resolvers in _OmegaConfLoader.yaml_implicit_resolvers.items()
}


def merge_config(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Gộp ``child`` lên ``parent`` theo quy tắc của ``extends``; không sửa hai đối số.

    Args:
        parent: Config của file cha.
        child: Config của file con (đã bỏ khoá ``extends``).

    Returns:
        Config đã gộp.
    """
    merged = copy.deepcopy(parent)
    for key, value in child.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read(path: Path, child: Optional[Path]) -> dict[str, Any]:
    where = f" (được {child.name} kế thừa)" if child else ""
    if not path.is_file():
        raise ConfigError(f"không có file config {path}{where}")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_OmegaConfLoader)  # noqa: S506
    except yaml.YAMLError as error:
        raise ConfigError(f"YAML sai trong {path}{where}: {error}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}{where}: cấp ngoài cùng phải là mapping")
    return data


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Đọc một file config và gộp mọi tầng ``extends`` của nó.

    Args:
        path: Đường dẫn file config.

    Returns:
        Config đã gộp, không còn khoá ``extends``.

    Raises:
        ConfigError: Nếu file không có hoặc YAML sai, ``extends`` không phải tên một file cùng
            thư mục, file cha không có, hoặc kế thừa thành vòng; thông báo nêu tên cả file con và
            file cha.
    """
    chain: list[Path] = []
    current, child = Path(path).resolve(), None
    layers = []
    while True:
        if current in chain:
            names = " → ".join(p.name for p in [*chain, current])
            raise ConfigError(f"extends thành vòng: {names}")
        chain.append(current)
        data = _read(current, child)
        parent_name = data.pop(EXTENDS_KEY, None)
        layers.append(data)
        if parent_name is None:
            break
        if not isinstance(parent_name, str) or Path(parent_name).name != parent_name:
            raise ConfigError(
                f"{current.name}: extends phải là tên một file cùng thư mục, nhận {parent_name!r}"
            )
        child, current = current, current.parent / parent_name
    # File gốc giữ nguyên, kể cả giá trị null; chỉ ở file con null mới nghĩa là xoá khoá.
    merged = layers[-1]
    for layer in reversed(layers[:-1]):
        merged = merge_config(merged, layer)
    return merged
