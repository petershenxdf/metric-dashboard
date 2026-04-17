from __future__ import annotations

from app.modules.selection.fixtures import initial_selected_point_ids, selection_fixture_dataset
from app.modules.selection.service import get_selection_context
from app.modules.selection.state import get_debug_store_for_dataset


def current_selection_context():
    dataset = selection_fixture_dataset()
    store = get_debug_store_for_dataset(dataset, initial_selected_point_ids())
    return get_selection_context(store)
