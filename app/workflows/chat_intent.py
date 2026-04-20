from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

from flask import Blueprint, jsonify, render_template, request

from app.modules.chatbox.schemas import DEFAULT_HISTORY_WINDOW
from app.modules.chatbox.service import (
    clear_history,
    create_chatbox_store,
    get_chatbox_state,
    submit_message,
    suggestion_chips,
)
from app.modules.intent_instruction.service import IntentInstructionProvider
from app.shared.flask_helpers import api_error, api_success
from app.shared.chat_grounding import build_grounded_chat_context
from app.shared.request_helpers import n_clusters_from_request


@dataclass
class _WorkflowRuntime:
    """Per-dataset chatbox store + shared real intent provider for the
    chat-intent workflow. Separate from the Step 7 debug runtime so
    isolation tests of the chatbox module page stay untouched."""

    provider: IntentInstructionProvider
    stores: Dict[str, object] = field(default_factory=dict)

    def store_for(self, dataset_id: str):
        if dataset_id not in self.stores:
            self.stores[dataset_id] = create_chatbox_store(dataset_id)
        return self.stores[dataset_id]

    def reset(self, dataset_id: str) -> None:
        self.stores[dataset_id] = create_chatbox_store(dataset_id)
        self.provider.reset(dataset_id)


_runtime = _WorkflowRuntime(
    provider=IntentInstructionProvider(),
)


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "chat_intent_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/chat-intent",
    )

    @blueprint.get("/")
    def index():
        grounded = _grounded_state()
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        provider = _runtime.provider
        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(grounded.selection_context.dataset_id)
        return render_template(
            "workflows/chat_intent.html",
            selection_context=grounded.selection_context,
            selection_groups_payload=[group.to_dict() for group in grounded.selection_groups],
            label_context=grounded.label_context_payload(),
            analysis_context=grounded.analysis_context_payload(),
            state=state,
            chips_payload=[chip.to_dict() for chip in suggestion_chips()],
            provider_label=provider.label,
            llm_label=provider.llm.label,
            instruction=instruction,
            history_window=DEFAULT_HISTORY_WINDOW,
            n_clusters=grounded.n_clusters,
        )

    @blueprint.post("/api/messages")
    def messages_api():
        body: Mapping = request.get_json(silent=True) or {}
        message = body.get("message", "")
        grounded = _grounded_state()
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        provider = _runtime.provider

        try:
            forwarded, response = submit_message(
                store,
                provider,
                message=message,
                selection_context=grounded.selection_context,
                selection_groups=grounded.selection_groups,
                label_context=grounded.label_context_payload(),
                history_window_size=DEFAULT_HISTORY_WINDOW,
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_chat_message", str(exc))), 400

        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(grounded.selection_context.dataset_id)
        return jsonify(
            api_success(
                {
                    "forwarded_payload": forwarded.to_dict(),
                    "response": response.to_dict(),
                    "state": state.to_dict(),
                    "current_instruction": instruction.to_dict(),
                    "analysis_context": grounded.analysis_context_payload(),
                },
                diagnostics={"provider": provider.label, "llm": provider.llm.label},
            )
        )

    @blueprint.post("/api/reset")
    def reset_api():
        grounded = _grounded_state()
        _runtime.reset(grounded.selection_context.dataset_id)
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        provider = _runtime.provider
        state = get_chatbox_state(store, provider)
        instruction = provider.store.get(grounded.selection_context.dataset_id)
        return jsonify(
            api_success(
                {"state": state.to_dict(), "current_instruction": instruction.to_dict()}
            )
        )

    @blueprint.post("/api/clear")
    def clear_api():
        grounded = _grounded_state()
        store = _runtime.store_for(grounded.selection_context.dataset_id)
        clear_history(store)
        provider = _runtime.provider
        state = get_chatbox_state(store, provider)
        return jsonify(api_success(state.to_dict()))

    return blueprint


def _grounded_state():
    return build_grounded_chat_context(n_clusters=n_clusters_from_request())
