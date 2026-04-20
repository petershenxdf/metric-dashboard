from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.shared.flask_helpers import api_error, api_success

from .fixtures import (
    current_label_context,
    current_selection_context,
    current_selection_groups,
    example_messages,
)
from .schemas import DEFAULT_HISTORY_WINDOW, ROUTER_CATEGORIES
from .service import clear_history, get_chatbox_state, submit_message, suggestion_chips
from .state import get_debug_provider, get_debug_store_for_context, reset_debug_store_for_context


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "chatbox",
        __name__,
        template_folder="templates",
        url_prefix="/modules/chatbox",
    )

    @blueprint.get("/")
    def index():
        context = current_selection_context()
        store = get_debug_store_for_context(context)
        provider = get_debug_provider()
        state = get_chatbox_state(store, provider)
        groups = current_selection_groups()
        label_context = current_label_context()
        return render_template(
            "chatbox/index.html",
            state=state,
            selection_context=context,
            selection_groups_payload=[group.to_dict() for group in groups],
            label_context=label_context,
            chips=suggestion_chips(),
            router_categories=ROUTER_CATEGORIES,
            examples=example_messages(),
            history_window=DEFAULT_HISTORY_WINDOW,
            provider_label=provider.label,
            dependency_mode="real selection + labeling debug state, mocked intent provider",
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "chatbox", "status": "working"},
                diagnostics={
                    "dependency_mode": "real selection + labeling debug state, mocked intent provider"
                },
            )
        )

    @blueprint.get("/api/context")
    def context_api():
        context = current_selection_context()
        groups = current_selection_groups()
        label_context = current_label_context()
        store = get_debug_store_for_context(context)
        provider = get_debug_provider()
        snapshot = provider.current_snapshot(store.dataset_id)
        return jsonify(
            api_success(
                {
                    "dataset_id": context.dataset_id,
                    "selection_context": context.to_dict(),
                    "selection_groups": [group.to_dict() for group in groups],
                    "label_context": label_context,
                    "instruction_snapshot": snapshot.to_dict(),
                    "suggestion_chips": [chip.to_dict() for chip in suggestion_chips()],
                },
                diagnostics={"provider": provider.label},
            )
        )

    @blueprint.get("/api/history")
    def history_api():
        context = current_selection_context()
        store = get_debug_store_for_context(context)
        provider = get_debug_provider()
        state = get_chatbox_state(store, provider)
        return jsonify(
            api_success(
                {
                    "dataset_id": state.dataset_id,
                    "turns": [turn.to_dict() for turn in state.turns],
                    "turn_count": len(state.turns),
                },
                diagnostics={"provider": provider.label},
            )
        )

    @blueprint.post("/api/messages")
    def messages_api():
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "")
        context = current_selection_context()
        groups = current_selection_groups()
        label_context = current_label_context()
        store = get_debug_store_for_context(context)
        provider = get_debug_provider()
        history_window = _history_window_from_payload(payload)

        try:
            forwarded, response = submit_message(
                store,
                provider,
                message=message,
                selection_context=context,
                selection_groups=groups,
                label_context=label_context,
                history_window_size=history_window,
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_chat_message", str(exc))), 400

        state = get_chatbox_state(store, provider)
        return jsonify(
            api_success(
                {
                    "forwarded_payload": forwarded.to_dict(),
                    "response": response.to_dict(),
                    "state": state.to_dict(),
                },
                diagnostics={"provider": provider.label, "history_window": history_window},
            )
        )

    @blueprint.post("/api/reset")
    def reset_api():
        context = current_selection_context()
        store = reset_debug_store_for_context(context)
        provider = get_debug_provider()
        state = get_chatbox_state(store, provider)
        return jsonify(api_success(state.to_dict(), diagnostics={"provider": provider.label}))

    @blueprint.post("/api/clear")
    def clear_api():
        context = current_selection_context()
        store = get_debug_store_for_context(context)
        clear_history(store)
        provider = get_debug_provider()
        state = get_chatbox_state(store, provider)
        return jsonify(api_success(state.to_dict(), diagnostics={"provider": provider.label}))

    return blueprint


def _history_window_from_payload(payload) -> int:
    if not isinstance(payload, dict):
        return DEFAULT_HISTORY_WINDOW
    raw = payload.get("history_window")
    if raw is None:
        return DEFAULT_HISTORY_WINDOW
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("history_window must be an integer") from exc
    if value < 0:
        raise ValueError("history_window must be non-negative")
    return value
