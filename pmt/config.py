"""Immutable configurations, YAML overlays and the paper's compact run registry."""

from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Config(Mapping):
    """Expose a recursively immutable mapping through keys and attributes."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        def freeze(value: Any) -> Any:
            if isinstance(value, Mapping):
                return Config(value)
            return tuple(freeze(v) for v in value) if isinstance(value, (list, tuple)) else value

        object.__setattr__(self, '_values', MappingProxyType({k: freeze(v) for k, v in values.items()}))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError('Config is read-only')

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, serializable mutable copy."""

        def thaw(value: Any) -> Any:
            if isinstance(value, Config):
                return value.to_dict()
            return [thaw(v) for v in value] if isinstance(value, tuple) else value

        return {k: thaw(v) for k, v in self.items()}


def _merge(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        base[key] = _merge(base.get(key, {}), value) if isinstance(value, dict) else value
    return base


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f'Cyclic config inheritance: {path}')
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f'Config must be a mapping: {path}')
    parents = values.pop('extends', [])
    base: dict = {}
    for parent in [parents] if isinstance(parents, str) else parents:
        _merge(base, _read(path.parent / parent, (*seen, path)))
    return _merge(base, values)


def _override(values: dict, key: str, value: Any, *, strict: bool) -> None:
    node = values
    parts = key.split('.')
    for part in parts[:-1]:
        if strict and part not in node:
            raise KeyError(key)
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise KeyError(key)
    if strict and parts[-1] not in node:
        raise KeyError(key)
    old = node.get(parts[-1])
    if strict and old is not None and value is not None:
        if not (type(value) is type(old) or (type(old) is float and type(value) is int)):
            raise TypeError(f'{key}: expected {type(old).__name__}, received {type(value).__name__}')
    node[parts[-1]] = value


def _finish(values: dict, overrides: Sequence[str]) -> Config:
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'Expected key=value, received {item!r}')
        _override(values, key, yaml.safe_load(value), strict=True)
    return Config(values)


def load(path: str, overrides: Sequence[str] = ()) -> Config:
    """Load a YAML file with relative inheritance and typed dotted overrides."""
    return _finish(_read(Path(path)), overrides)


@lru_cache(maxsize=1)
def _registry() -> dict:
    return _read(ROOT / 'configs/runs.yaml')


@lru_cache(maxsize=1)
def runs() -> dict[str, dict]:
    """Return the registered runs with presets expanded into arm overrides."""
    registry = _registry()
    entries = {}
    for arm, row in registry['arms'].items():
        metadata = dict(row)
        overrides = {}
        for preset in metadata.pop('presets', ()):
            overrides.update(registry['presets'][preset])
        overrides.update(metadata.pop('overrides', {}))
        for seed in row['seeds']:
            entries[f'{arm}_s{seed}'] = dict(metadata, arm=arm, seed=seed, overrides=overrides)
    return entries


def load_run(run_id: str, overrides: Sequence[str] = ()) -> Config:
    """Resolve an exact reference run ID through dataset defaults and arm overlays."""
    row = runs()[run_id]
    values = _read(ROOT / 'configs' / f'{row["dataset"]}.yaml')
    values['data'].update(_registry()['data_shapes'][row['dataset']])
    for key, value in row.get('overrides', {}).items():
        _override(values, key, value, strict=False)
    values['train'].update(seed=row['seed'], run_id=run_id, precision=row['precision'])
    return _finish(values, overrides)
