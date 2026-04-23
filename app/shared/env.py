from __future__ import annotations

import os
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dotenv_path() -> Path:
    return repo_root() / ".env"


def dotenv_values() -> dict[str, str]:
    path = dotenv_path()
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]

        values[key] = value

    return values


def env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    return dotenv_values().get(name, default)


def env_int(name: str, default: int) -> int:
    raw_value = env_text(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw_value = env_text(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw_value = env_text(name, "true" if default else "false").strip().lower()
    if raw_value in _TRUE_VALUES:
        return True
    if raw_value in _FALSE_VALUES:
        return False
    return default
