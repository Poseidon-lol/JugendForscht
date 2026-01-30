"""
Configuration utilities for the OSC discovery stack.
====================================================

The project relies on light-weight YAML configuration files that describe
experiment hyper-parameters (e.g. surrogate training schedule, generator
architecture, acquisition policies).  This module provides helpers to

* load YAML/JSON dictionaries into dot-accessible ``ConfigNode`` objects
* merge overrides coming from CLI arguments or notebooks
* validate/instantiate optional ``dataclass`` schemas
* persist the resolved configuration back to disk for experiment tracking

Example
-------
>>> from pathlib import Path
>>> from src.utils.config import load_config, save_config
>>> cfg = load_config(Path('configs/train_conf.yaml'))
>>> cfg.training.batch_size
32
>>> cfg.training.lr = 5e-4
>>> save_config(cfg, Path('experiments/run_001/train_used.yaml'))

The helper intentionally avoids heavy dependencies (OmegaConf / hydra)
to keep portability high for notebook-driven workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar, Union

import json
import yaml

__all__ = [
    "ConfigNode",
    "load_config",
    "merge_overrides",
    "save_config",
    "dataclass_from_dict",
]

T = TypeVar("T")


class ConfigNode(dict):
    """Dictionary with attribute-style access and recursive conversion.

    The object behaves like a standard ``dict`` but also allows
    ``node.attr`` access.  Nested dictionaries are lazily wrapped in
    ``ConfigNode`` to keep memory usage modest.
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        data = data or {}
        self.update(data, **kwargs)

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, Mapping) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            super().__setitem__(item, value)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def update(self, *args: Mapping[str, Any], **kwargs: Any) -> None:  # type: ignore[override]
        for mapping in args:
            for k, v in mapping.items():
                super().__setitem__(k, self._maybe_wrap(v))
        for k, v in kwargs.items():
            super().__setitem__(k, self._maybe_wrap(v))

    @staticmethod
    def _maybe_wrap(value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, ConfigNode):
            return ConfigNode(value)
        if isinstance(value, list):
            return [ConfigNode._maybe_wrap(v) for v in value]
        return value


def _deep_merge(base: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge incoming mapping into base."""

    for key, value in incoming.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            _deep_merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value  # type: ignore[index]
    return base


def merge_overrides(cfg: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge overrides into cfg and return a fresh dictionary."""
    data = json.loads(json.dumps(cfg))  # deep copy via json ensures primitives only
    if overrides:
        _deep_merge(data, overrides)
    return data


def _read_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format '{suffix}' for path {path}")


def load_config(
    path: Union[str, Path],
    *,
    schema: Optional[Type[T]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Union[ConfigNode, T]:
    """Load configuration file with optional overrides and schema casting.

    Parameters
    ----------
    path:
        Location of the YAML/JSON configuration file.
    schema:
        Optional dataclass type; if provided the dictionary will be cast
        into an instance of this dataclass.
    overrides:
        Mapping of override values. Keys follow dotted notation, e.g.
        ``{"training.lr": 1e-3}``. Nested dictionaries are also accepted.
    """

    path = Path(path)
    cfg_dict = _read_file(path)
    if overrides:
        # dotted notation -> nested dict
        processed = {}
        for key, value in overrides.items():
            if "." in key:
                current = processed
                parts = key.split(".")
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                processed[key] = value
        cfg_dict = merge_overrides(cfg_dict, processed)

    if schema is None:
        return ConfigNode(cfg_dict)
    return dataclass_from_dict(schema, cfg_dict)


def dataclass_from_dict(schema: Type[T], data: Mapping[str, Any]) -> T:
    """Instantiate dataclass ``schema`` from dictionary with validation."""

    if not is_dataclass(schema):
        raise TypeError(f"Schema {schema} must be a dataclass type.")

    from dataclasses import MISSING  # local import to avoid polluting __all__

    kwargs: Dict[str, Any] = {}
    allowed = {f.name for f in fields(schema)}
    for key in data.keys():
        if key not in allowed:
            raise KeyError(f"Unexpected key '{key}' for schema {schema.__name__}")

    for f in fields(schema):
        raw_value = data.get(f.name, MISSING)
        if raw_value is MISSING:
            if f.default is not MISSING or f.default_factory is not MISSING:  # type: ignore[attr-defined]
                continue
            raise KeyError(f"Missing required key '{f.name}' for schema {schema.__name__}")

        if is_dataclass(f.type) and isinstance(raw_value, Mapping):
            kwargs[f.name] = dataclass_from_dict(f.type, raw_value)
        else:
            kwargs[f.name] = raw_value

    return schema(**kwargs)  # type: ignore[arg-type]


def save_config(cfg: Union[ConfigNode, Mapping[str, Any], Any], path: Union[str, Path]) -> None:
    """Persist configuration object (ConfigNode or dataclass) to YAML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif isinstance(cfg, ConfigNode):
        payload = json.loads(json.dumps(cfg))
    elif isinstance(cfg, Mapping):
        payload = json.loads(json.dumps(cfg))
    else:
        raise TypeError(f"Unsupported config type {type(cfg)}")
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
