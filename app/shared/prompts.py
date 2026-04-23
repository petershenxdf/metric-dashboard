from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prompt_path(*parts: str) -> Path:
    path = repo_root() / "prompts"
    for part in parts:
        path = path / part
    return path


@lru_cache(maxsize=None)
def load_prompt_text(*parts: str) -> str:
    return prompt_path(*parts).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_prompt_template(*parts: str) -> Template:
    return Template(load_prompt_text(*parts))
