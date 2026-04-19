from __future__ import annotations

from flask import Blueprint, render_template

from app.modules.chatbox.fixtures import (
    current_label_context,
    current_selection_context,
    current_selection_groups,
)
from app.modules.chatbox.service import get_chatbox_state, suggestion_chips_for_strategy
from app.modules.chatbox.state import get_debug_provider, get_debug_store_for_context


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "chat_selection_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/chat-selection",
    )

    @blueprint.get("/")
    def index():
        selection_context = current_selection_context()
        groups = current_selection_groups()
        label_context = current_label_context()
        store = get_debug_store_for_context(selection_context)
        provider = get_debug_provider()
        state = get_chatbox_state(store, provider)
        chips = suggestion_chips_for_strategy(state.strategy)
        return render_template(
            "workflows/chat_selection.html",
            selection_context=selection_context,
            selection_groups_payload=[group.to_dict() for group in groups],
            label_context=label_context,
            state=state,
            chips_payload=[chip.to_dict() for chip in chips],
            provider_label=provider.label,
        )

    return blueprint
