from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.shared.flask_helpers import api_error, api_success

from .fixtures import fixture_group_point_ids, selection_fixture_dataset
from .schemas import SELECTION_ACTION_TYPES, SELECTION_MODES, SELECTION_SOURCES, SelectionAction
from .service import (
    apply_selection_action,
    delete_selection_group,
    get_selection_context,
    get_selection_state,
    list_selection_groups,
    save_selection_group,
    select_selection_group,
)
from .state import get_debug_store, reset_debug_store


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "selection",
        __name__,
        template_folder="templates",
        url_prefix="/modules/selection",
    )

    @blueprint.get("/")
    def index():
        store = get_debug_store()
        dataset = selection_fixture_dataset()
        state = get_selection_state(store)
        selection_groups = list_selection_groups(store)
        return render_template(
            "selection/index.html",
            dataset=dataset,
            state=state,
            context=state.to_context(),
            selection_groups=selection_groups,
            dependency_mode="real data-workspace fixture",
            action_types=SELECTION_ACTION_TYPES,
            sources=SELECTION_SOURCES,
            modes=SELECTION_MODES,
            fixture_groups={
                "setosa": fixture_group_point_ids("setosa"),
                "versicolor": fixture_group_point_ids("versicolor"),
                "virginica": fixture_group_point_ids("virginica"),
            },
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "selection", "status": "working"},
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        state = get_selection_state(get_debug_store())
        return jsonify(
            api_success(
                state.to_dict(),
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.get("/api/context")
    def context_api():
        context = get_selection_context(get_debug_store())
        return jsonify(
            api_success(
                context.to_dict(),
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.post("/api/select")
    def select_api():
        return _action_response("select")

    @blueprint.post("/api/deselect")
    def deselect_api():
        return _action_response("deselect")

    @blueprint.post("/api/replace")
    def replace_api():
        return _action_response("replace")

    @blueprint.post("/api/toggle")
    def toggle_api():
        return _action_response("toggle")

    @blueprint.post("/api/clear")
    def clear_api():
        return _action_response("clear")

    @blueprint.post("/api/reset")
    def reset_api():
        store = reset_debug_store()
        state = get_selection_state(store)
        return jsonify(
            api_success(
                state.to_dict(),
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.get("/api/groups")
    def groups_api():
        groups = [group.to_dict() for group in list_selection_groups(get_debug_store())]
        return jsonify(
            api_success(
                {"groups": groups, "group_count": len(groups)},
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.post("/api/groups")
    def save_group_api():
        payload = _request_payload()
        try:
            group = save_selection_group(
                get_debug_store(),
                group_name=payload.get("group_name", ""),
                point_ids=_optional_point_ids_from_payload(payload),
                metadata=payload.get("metadata", {}),
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": _selection_groups_payload()},
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.post("/api/groups/<group_id>/select")
    def select_group_api(group_id: str):
        try:
            result = select_selection_group(get_debug_store(), group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"selection": result.to_dict(), "groups": _selection_groups_payload()},
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.delete("/api/groups/<group_id>")
    def delete_group_api(group_id: str):
        try:
            group = delete_selection_group(get_debug_store(), group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"deleted_group": group.to_dict(), "groups": _selection_groups_payload()},
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    return blueprint


def _action_response(action_name: str):
    payload = _request_payload()
    try:
        action = SelectionAction(
            action=action_name,
            point_ids=_point_ids_from_payload(payload),
            source=str(payload.get("source", "api")),
            mode=payload.get("mode"),
            metadata=payload.get("metadata", {}),
        )
        result = apply_selection_action(get_debug_store(), action)
    except ValueError as exc:
        return jsonify(api_error("invalid_selection_action", str(exc))), 400

    return jsonify(
        api_success(
            result.to_dict(),
            diagnostics={"dependency_mode": "real data-workspace fixture"},
        )
    )


def _request_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}

    payload = dict(request.form)
    payload.update(request.args)
    return payload


def _point_ids_from_payload(payload):
    point_ids = payload.get("point_ids", [])
    if isinstance(point_ids, str):
        return [point_id.strip() for point_id in point_ids.split(",") if point_id.strip()]
    return point_ids


def _optional_point_ids_from_payload(payload):
    if "point_ids" not in payload or payload.get("point_ids") in (None, ""):
        return None
    return _point_ids_from_payload(payload)


def _selection_groups_payload():
    return [group.to_dict() for group in list_selection_groups(get_debug_store())]
