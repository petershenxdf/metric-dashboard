from __future__ import annotations

from app.modules.selection.schemas import SelectionContext

from .service import create_labeling_store

_debug_stores_by_dataset = {}


def get_debug_store_for_context(selection_context: SelectionContext):
    if not isinstance(selection_context, SelectionContext):
        raise ValueError("selection_context must be a SelectionContext")

    if selection_context.dataset_id not in _debug_stores_by_dataset:
        _debug_stores_by_dataset[selection_context.dataset_id] = create_labeling_store(selection_context.dataset_id)

    return _debug_stores_by_dataset[selection_context.dataset_id]


def reset_debug_store_for_context(selection_context: SelectionContext):
    if not isinstance(selection_context, SelectionContext):
        raise ValueError("selection_context must be a SelectionContext")

    _debug_stores_by_dataset[selection_context.dataset_id] = create_labeling_store(selection_context.dataset_id)
    return _debug_stores_by_dataset[selection_context.dataset_id]
