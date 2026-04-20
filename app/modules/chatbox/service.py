from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from app.modules.selection.schemas import SelectionContext, SelectionGroup
from app.shared.schemas import clean_text

from .providers.base import IntentProvider
from .schemas import (
    ChatboxState,
    ChatMessagePayload,
    ChatResponse,
    ChatTurn,
    DEFAULT_HISTORY_WINDOW,
    SuggestionChip,
)
from .store import ChatboxStore


def create_chatbox_store(dataset_id: str) -> ChatboxStore:
    return ChatboxStore(dataset_id=clean_text(dataset_id, "dataset_id"))


def get_chatbox_state(store: ChatboxStore, provider: IntentProvider) -> ChatboxState:
    _validate_store(store)
    snapshot = provider.current_snapshot(store.dataset_id)
    return ChatboxState(
        dataset_id=store.dataset_id,
        turns=tuple(store.turns),
        instruction_snapshot=snapshot,
    )


def clear_history(store: ChatboxStore) -> ChatboxStore:
    _validate_store(store)
    store.turns.clear()
    return store


def build_payload(
    store: ChatboxStore,
    message: str,
    selection_context: SelectionContext,
    selection_groups: Sequence[SelectionGroup],
    label_context: Mapping,
    history_window_size: int = DEFAULT_HISTORY_WINDOW,
) -> ChatMessagePayload:
    _validate_store(store)
    _validate_selection_context(store, selection_context)
    message = clean_text(message, "message")
    if history_window_size < 0:
        raise ValueError("history_window_size must be non-negative")

    window = tuple(turn.to_dict() for turn in store.turns[-history_window_size:]) if history_window_size else ()

    return ChatMessagePayload(
        message=message,
        dataset_id=store.dataset_id,
        selected_point_ids=tuple(selection_context.selected_point_ids),
        unselected_point_ids=tuple(selection_context.unselected_point_ids),
        selection_groups=tuple(group.to_dict() for group in selection_groups),
        label_context=dict(label_context or {}),
        history_window=window,
    )


def submit_message(
    store: ChatboxStore,
    provider: IntentProvider,
    message: str,
    selection_context: SelectionContext,
    selection_groups: Sequence[SelectionGroup] = (),
    label_context: Mapping | None = None,
    history_window_size: int = DEFAULT_HISTORY_WINDOW,
) -> tuple[ChatMessagePayload, ChatResponse]:
    payload = build_payload(
        store,
        message,
        selection_context,
        selection_groups,
        label_context or {},
        history_window_size,
    )

    response = provider.respond(payload)

    user_turn = ChatTurn(turn_id=store.next_turn_id(), role="user", text=payload.message)
    assistant_turn = ChatTurn(turn_id=store.next_turn_id(), role="assistant", text=response.reply)
    store.turns.append(user_turn)
    store.turns.append(assistant_turn)

    return payload, response


def suggestion_chips() -> tuple[SuggestionChip, ...]:
    base = (
        SuggestionChip("Make feature petal_length more important", "feature_weight"),
        SuggestionChip("Treat these points as similar", "group_similar"),
        SuggestionChip("Push cluster 1 and cluster 3 apart", "group_dissimilar"),
        SuggestionChip("Merge clusters 1 and 2", "merge_clusters"),
        SuggestionChip("Treat p42 as a typical example for cluster 2", "anchor_point"),
        SuggestionChip("Ignore cluster 5 for now", "ignore_cluster"),
    )
    path_b_only = (
        SuggestionChip("Split cluster 2 into two", "split_cluster", requires_path_b=True),
        SuggestionChip("Reclassify p7 as not an outlier", "reclassify_outlier", requires_path_b=True),
    )
    return base + path_b_only


def _validate_store(store: ChatboxStore) -> None:
    if not isinstance(store, ChatboxStore):
        raise ValueError("store must be a ChatboxStore")


def _validate_selection_context(store: ChatboxStore, selection_context: SelectionContext) -> None:
    if not isinstance(selection_context, SelectionContext):
        raise ValueError("selection_context must be a SelectionContext")
    if selection_context.dataset_id != store.dataset_id:
        raise ValueError("selection_context dataset_id must match store dataset_id")
