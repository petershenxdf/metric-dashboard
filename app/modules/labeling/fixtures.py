from __future__ import annotations

from app.modules.selection.service import get_selection_context
from app.modules.selection.state import get_debug_store


def current_selection_context():
    return get_selection_context(get_debug_store())
