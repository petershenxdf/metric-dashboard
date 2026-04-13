from __future__ import annotations

from typing import Iterable

from app.shared.schemas import Dataset

from .schemas import SelectionAction, SelectionActionResult, SelectionContext, SelectionGroup, SelectionState
from .store import SelectionStore


def create_selection_store(
    dataset: Dataset,
    initial_selected_point_ids: Iterable[str] | None = None,
) -> SelectionStore:
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be a Dataset")

    store = SelectionStore(
        dataset_id=dataset.dataset_id,
        known_point_ids=tuple(point.point_id for point in dataset.points),
    )

    if initial_selected_point_ids is not None:
        replace_selection(store, initial_selected_point_ids)

    return store


def get_selection_state(store: SelectionStore) -> SelectionState:
    _validate_store(store)
    return SelectionState(
        dataset_id=store.dataset_id,
        known_point_ids=store.known_point_ids,
        selected_point_ids=store.ordered_selected_point_ids(),
    )


def get_selection_context(store: SelectionStore) -> SelectionContext:
    return get_selection_state(store).to_context()


def select_points(store: SelectionStore, point_ids: Iterable[str]) -> SelectionState:
    point_ids = _known_point_ids(store, point_ids)
    store.selected_point_ids.update(point_ids)
    return get_selection_state(store)


def deselect_points(store: SelectionStore, point_ids: Iterable[str]) -> SelectionState:
    point_ids = _known_point_ids(store, point_ids)
    store.selected_point_ids.difference_update(point_ids)
    return get_selection_state(store)


def replace_selection(store: SelectionStore, point_ids: Iterable[str]) -> SelectionState:
    point_ids = _known_point_ids(store, point_ids)
    store.selected_point_ids = set(point_ids)
    return get_selection_state(store)


def toggle_points(store: SelectionStore, point_ids: Iterable[str]) -> SelectionState:
    point_ids = _known_point_ids(store, point_ids)
    for point_id in point_ids:
        if point_id in store.selected_point_ids:
            store.selected_point_ids.remove(point_id)
        else:
            store.selected_point_ids.add(point_id)
    return get_selection_state(store)


def clear_selection(store: SelectionStore) -> SelectionState:
    _validate_store(store)
    store.selected_point_ids.clear()
    return get_selection_state(store)


def list_selection_groups(store: SelectionStore) -> tuple[SelectionGroup, ...]:
    _validate_store(store)
    return tuple(store.selection_groups.values())


def save_selection_group(
    store: SelectionStore,
    group_name: str,
    point_ids: Iterable[str] | None = None,
    metadata: dict | None = None,
) -> SelectionGroup:
    _validate_store(store)
    if point_ids is None:
        point_ids = store.ordered_selected_point_ids()

    point_ids = _known_point_ids(store, point_ids)
    if not point_ids:
        raise ValueError("selection group must include at least one point")

    group_name_key = _group_name_key(group_name)
    if any(_group_name_key(group.group_name) == group_name_key for group in store.selection_groups.values()):
        raise ValueError(f"selection group name already exists: {group_name.strip()}")

    group = SelectionGroup(
        group_id=_next_group_id(store),
        group_name=group_name,
        dataset_id=store.dataset_id,
        point_ids=point_ids,
        metadata=metadata or {},
    )
    store.selection_groups[group.group_id] = group
    return group


def select_selection_group(store: SelectionStore, group_id: str) -> SelectionActionResult:
    group = get_selection_group(store, group_id)
    action = SelectionAction(
        action="replace",
        point_ids=group.point_ids,
        source="selection_group",
        mode="replace",
        metadata={"group_id": group.group_id, "group_name": group.group_name},
    )
    state = replace_selection(store, group.point_ids)
    return SelectionActionResult(
        action=action,
        state=state,
        context=state.to_context(),
    )


def delete_selection_group(store: SelectionStore, group_id: str) -> SelectionGroup:
    group = get_selection_group(store, group_id)
    del store.selection_groups[group.group_id]
    return group


def get_selection_group(store: SelectionStore, group_id: str) -> SelectionGroup:
    _validate_store(store)
    group_id = _clean_group_id(group_id)
    try:
        return store.selection_groups[group_id]
    except KeyError as exc:
        raise ValueError(f"selection group does not exist: {group_id}") from exc


def apply_selection_action(store: SelectionStore, action: SelectionAction) -> SelectionActionResult:
    _validate_store(store)
    if not isinstance(action, SelectionAction):
        raise ValueError("action must be a SelectionAction")

    handlers = {
        "select": select_points,
        "deselect": deselect_points,
        "replace": replace_selection,
        "toggle": toggle_points,
    }

    if action.action == "clear":
        state = clear_selection(store)
    else:
        state = handlers[action.action](store, action.point_ids)

    return SelectionActionResult(
        action=action,
        state=state,
        context=state.to_context(),
    )


def _known_point_ids(store: SelectionStore, point_ids: Iterable[str]):
    _validate_store(store)
    return store.validate_known_points(point_ids)


def _validate_store(store: SelectionStore) -> None:
    if not isinstance(store, SelectionStore):
        raise ValueError("store must be a SelectionStore")


def _next_group_id(store: SelectionStore) -> str:
    index = len(store.selection_groups) + 1
    while True:
        group_id = f"group_{index:03d}"
        if group_id not in store.selection_groups:
            return group_id
        index += 1


def _group_name_key(group_name: str) -> str:
    return SelectionGroup(
        group_id="name_check",
        group_name=group_name,
        dataset_id="name_check_dataset",
        point_ids=("name_check_point",),
    ).group_name.casefold()


def _clean_group_id(group_id: str) -> str:
    return SelectionGroup(
        group_id=group_id,
        group_name="group_id_check",
        dataset_id="group_id_check_dataset",
        point_ids=("group_id_check_point",),
    ).group_id
