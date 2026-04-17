from __future__ import annotations

from app.shared.schemas import Dataset

from .fixtures import initial_selected_point_ids, selection_fixture_dataset
from .service import create_selection_store

_debug_stores_by_dataset = {}


def get_debug_store():
    """Convenience wrapper: returns the per-dataset store for the selection
    module's own fixture dataset.  Internally delegates to
    ``get_debug_store_for_dataset`` so there is only one storage dict."""
    return get_debug_store_for_dataset(
        selection_fixture_dataset(),
        initial_selected_point_ids(),
    )


def reset_debug_store():
    return reset_debug_store_for_dataset(
        selection_fixture_dataset(),
        initial_selected_point_ids(),
    )


def get_debug_store_for_dataset(
    dataset: Dataset,
    initial_selected_point_ids=None,
):
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be a Dataset")

    point_ids = _dataset_point_ids(dataset)
    existing_store = _debug_stores_by_dataset.get(dataset.dataset_id)
    if existing_store is None or existing_store.known_point_ids != point_ids:
        _debug_stores_by_dataset[dataset.dataset_id] = create_selection_store(
            dataset,
            initial_selected_point_ids,
        )

    return _debug_stores_by_dataset[dataset.dataset_id]


def reset_debug_store_for_dataset(
    dataset: Dataset,
    initial_selected_point_ids=None,
):
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be a Dataset")

    _debug_stores_by_dataset[dataset.dataset_id] = create_selection_store(
        dataset,
        initial_selected_point_ids,
    )
    return _debug_stores_by_dataset[dataset.dataset_id]


def _dataset_point_ids(dataset: Dataset):
    return tuple(point.point_id for point in dataset.points)
