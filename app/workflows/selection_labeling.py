from __future__ import annotations

from flask import Blueprint, render_template

from app.modules.labeling.service import get_labeling_state
from app.modules.labeling.state import get_debug_store_for_context
from app.modules.selection.fixtures import selection_fixture_dataset
from app.modules.selection.service import get_selection_context, get_selection_state
from app.modules.selection.state import get_debug_store


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "selection_labeling_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/selection-labeling",
    )

    @blueprint.get("/")
    def index():
        dataset = selection_fixture_dataset()
        selection_store = get_debug_store()
        selection_state = get_selection_state(selection_store)
        context = get_selection_context(selection_store)
        labeling_state = get_labeling_state(get_debug_store_for_context(context))

        return render_template(
            "workflows/selection_labeling.html",
            dataset=dataset,
            selection_state=selection_state,
            context=context,
            labeling_state=labeling_state,
        )

    return blueprint
