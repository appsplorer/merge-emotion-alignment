from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping: %s" % path)
    return data


def compose_config(path: Path, project_root: Optional[Path] = None) -> Dict[str, Any]:
    path = path.resolve()
    project_root = (project_root or path.parents[2]).resolve()
    current = load_yaml(path)
    merged: Dict[str, Any] = {}
    for include in current.pop("includes", []):
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = project_root / include_path
        merged = deep_merge(merged, load_yaml(include_path))
    merged = deep_merge(merged, current)
    return merged


def set_by_dotted_key(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_override_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError("Override must use key=value syntax: %s" % item)
        key, value = item.split("=", 1)
        set_by_dotted_key(result, key.strip(), parse_override_value(value.strip()))
    return result


def save_yaml(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
