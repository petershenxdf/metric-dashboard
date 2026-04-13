from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.selection.http_helpers import optional_point_ids_from_payload, request_payload
from app.shared.flask_helpers import api_error, api_success

from .fixtures import current_selection_context
from .schemas import LABELING_ACTION_TYPES
from .service import apply_labeling_action, clear_annotations, get_labeling_state
from .state import get_debug_store_for_context, reset_debug_store_for_context


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "labeling",
        __name__,
        template_folder="templates",
        url_prefix="/modules/labeling",
    )

    @blueprint.get("/")
    def index():
        context = current_selection_context()
        store = get_debug_store_for_context(context)
        state = get_labeling_state(store)
        return render_template(
            "labeling/index.html",
            context=context,
            state=state,
            action_types=LABELING_ACTION_TYPES,
            dependency_mode="real selection debug state",
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "labeling", "status": "working"},
                diagnostics={"dependency_mode": "real selection debug state"},
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        context = current_selection_context()
        state = get_labeling_state(get_debug_store_for_context(context))
        payload = state.to_dict()
        payload.update({"module": "labeling", "status": "working"})
        return jsonify(
            api_success(
                payload,
                diagnostics={"dependency_mode": "real selection debug state"},
            )
        )

    @blueprint.get("/api/annotations")
    def annotations_api():
        context = current_selection_context()
        state = get_labeling_state(get_debug_store_for_context(context))
        return jsonify(
            api_success(
                {"annotations": [annotation.to_dict() for annotation in state.annotations]},
                diagnostics={"dependency_mode": "real selection debug state"},
            )
        )

    @blueprint.post("/api/apply")
    def apply_api():
        context = current_selection_context()
        store = get_debug_store_for_context(context)
        payload = request_payload(request)
        try:
            annotation = apply_labeling_action(
                store,
                context,
                action=str(payload.get("action", "")),
                label_value=payload.get("label_value"),
                point_ids=optional_point_ids_from_payload(payload),
            )
            state = get_labeling_state(store)
        except ValueError as exc:
            return jsonify(api_error("invalid_labeling_action", str(exc))), 400

        return jsonify(
            api_success(
                {
                    "annotation": annotation.to_dict(),
                    "state": state.to_dict(),
                },
                diagnostics={"dependency_mode": "real selection debug state"},
            )
        )

    @blueprint.post("/api/reset")
    def reset_api():
        context = current_selection_context()
        store = reset_debug_store_for_context(context)
        return jsonify(
            api_success(
                get_labeling_state(store).to_dict(),
                diagnostics={"dependency_mode": "real selection debug state"},
            )
        )

    @blueprint.post("/api/clear")
    def clear_api():
        context = current_selection_context()
        store = get_debug_store_for_context(context)
        return jsonify(
            api_success(
                clear_annotations(store).to_dict(),
                diagnostics={"dependency_mode": "real selection debug state"},
            )
        )

    return blueprint
