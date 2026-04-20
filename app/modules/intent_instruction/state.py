from __future__ import annotations

from .fixtures import default_feature_names
from .service import IntentInstructionProvider

_debug_provider: IntentInstructionProvider | None = None


def get_debug_provider() -> IntentInstructionProvider:
    global _debug_provider
    if _debug_provider is None:
        _debug_provider = IntentInstructionProvider(
            feature_name_lookup=default_feature_names,
        )
    return _debug_provider


def reset_debug_provider() -> None:
    """Drop the cached provider so the next caller gets a fresh one."""

    global _debug_provider
    _debug_provider = None
