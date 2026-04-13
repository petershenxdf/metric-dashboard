from __future__ import annotations

from flask import Blueprint, render_template

from app.modules.selection.fixtures import selection_fixture_dataset
from app.modules.selection.service import get_selection_context, get_selection_state
from app.modules.selection.state import get_debug_store


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "selection_context_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/selection-context",
    )

    @blueprint.get("/")
    def index():
        dataset = selection_fixture_dataset()
        store = get_debug_store()
        state = get_selection_state(store)
        context = get_selection_context(store)

        return render_template(
            "workflows/selection_context.html",
            dataset=dataset,
            state=state,
            context=context,
        )

    return blueprint
