"""Shared request helpers for workflow and module Flask routes.

Kept under `app/shared/` so that scatterplot (a module) and the Step 1-6
workflow pages can all share the same small parsing helpers without creating
a module -> workflows layering dependency.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from flask import request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS
from app.modules.selection.http_helpers import (
    request_payload,
    selection_action_from_payload,
)
from app.modules.selection.service import (
    apply_selection_action,
    list_selection_groups,
)


def n_clusters_from_request(default: int = DEFAULT_N_CLUSTERS) -> int:
    """Read `n_clusters` from JSON body, query args, or form data (in that order).

    Returns `default` when the value is missing or unparseable. Always returns
    a positive integer (floors at 1).
    """
    raw_value = _coerce_to_raw("n_clusters")
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default

    return max(value, 1)


def dataset_id_from_request(
    default_dataset_id: str,
    is_valid: Callable[[Any], bool],
) -> str:
    """Read `dataset_id` from JSON body, query args, or form data.

    Returns `default_dataset_id` if the value is missing or fails `is_valid`.
    """
    raw_value = _coerce_to_raw("dataset_id")
    if is_valid(raw_value):
        return raw_value
    return default_dataset_id


def selection_groups_payload(store) -> list:
    """Serialize selection groups to JSON-safe dicts."""
    return [group.to_dict() for group in list_selection_groups(store)]


def apply_selection_action_or_error(
    store,
    action_name: str,
    payload: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
):
    """Build a selection action from the payload and apply it against `store`.

    Returns `(result, None)` on success or `(None, error_message)` on ValueError.
    """
    try:
        action = selection_action_from_payload(
            action_name,
            payload,
            metadata=dict(metadata) if metadata else {},
        )
        return apply_selection_action(store, action), None
    except ValueError as exc:
        return None, str(exc)


def _coerce_to_raw(key: str):
    payload: Dict[str, Any] = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}

    return (
        payload.get(key)
        or request.args.get(key)
        or request.form.get(key)
    )


__all__ = [
    "apply_selection_action_or_error",
    "dataset_id_from_request",
    "n_clusters_from_request",
    "request_payload",
    "selection_groups_payload",
]
