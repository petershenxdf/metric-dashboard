from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.selection.http_helpers import optional_point_ids_from_payload
from app.modules.selection.service import (
    delete_selection_group,
    get_selection_context,
    save_selection_group,
    select_selection_group,
)
from app.shared.flask_helpers import api_error, api_success
from app.shared.request_helpers import (
    apply_selection_action_or_error,
    n_clusters_from_request,
    request_payload,
    selection_groups_payload,
)

from .fixtures import scatterplot_fixture_state


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "scatterplot",
        __name__,
        template_folder="templates",
        url_prefix="/modules/scatterplot",
    )

    @blueprint.get("/")
    def index():
        return render_template(
            "scatterplot/index.html",
            dependency_mode="real Step 1-5 fixture state",
            **scatterplot_fixture_state(_n_clusters_from_request()),
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "scatterplot", "status": "working"},
                diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
            )
        )

    @blueprint.get("/api/render-payload")
    def render_payload_api():
        state = scatterplot_fixture_state(_n_clusters_from_request())
        return jsonify(
            api_success(
                state["render_payload"].to_dict(),
                diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        state = scatterplot_fixture_state(_n_clusters_from_request())
        render_payload = state["render_payload"]
        return jsonify(
            api_success(
                {
                    "module": "scatterplot",
                    "status": "working",
                    "dataset_id": state["dataset"].dataset_id,
                    "point_count": len(render_payload.points),
                    "selected_count": len(state["selection_state"].selected_point_ids),
                    "annotation_count": len(state["labeling_state"].annotations),
                    "render_id": render_payload.render_id,
                },
                diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
            )
        )

    @blueprint.post("/api/toggle")
    def toggle_api():
        return _selection_action_response("toggle")

    @blueprint.post("/api/select")
    def select_api():
        return _selection_action_response("select")

    @blueprint.post("/api/clear")
    def clear_api():
        return _selection_action_response("clear")

    @blueprint.get("/api/groups")
    def groups_api():
        groups = [group.to_dict() for group in scatterplot_fixture_state(_n_clusters_from_request())["selection_groups"]]
        return jsonify(
            api_success(
                {"groups": groups, "group_count": len(groups)},
                diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
            )
        )

    @blueprint.post("/api/groups")
    def save_group_api():
        state = scatterplot_fixture_state(_n_clusters_from_request())
        payload = request_payload(request)
        try:
            group = save_selection_group(
                state["selection_store"],
                group_name=payload.get("group_name", ""),
                point_ids=optional_point_ids_from_payload(payload),
                metadata={"module": "scatterplot"},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": selection_groups_payload(scatterplot_fixture_state(_n_clusters_from_request())["selection_store"])},
                diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
            )
        )

    @blueprint.post("/api/groups/<group_id>/select")
    def select_group_api(group_id: str):
        try:
            result = select_selection_group(scatterplot_fixture_state(_n_clusters_from_request())["selection_store"], group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"selection": result.to_dict(), "groups": selection_groups_payload(scatterplot_fixture_state(_n_clusters_from_request())["selection_store"])},
                diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
            )
        )

    @blueprint.delete("/api/groups/<group_id>")
    def delete_group_api(group_id: str):
        try:
            group = delete_selection_group(scatterplot_fixture_state(_n_clusters_from_request())["selection_store"], group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"deleted_group": group.to_dict(), "groups": selection_groups_payload(scatterplot_fixture_state(_n_clusters_from_request())["selection_store"])},
                diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
            )
        )

    return blueprint


def _selection_action_response(action_name: str):
    state = scatterplot_fixture_state(_n_clusters_from_request())
    payload = request_payload(request)
    result, error = apply_selection_action_or_error(
        state["selection_store"],
        action_name,
        payload,
        metadata={"module": "scatterplot"},
    )
    if error is not None:
        return jsonify(api_error("invalid_scatterplot_selection", error)), 400

    refreshed = scatterplot_fixture_state(_n_clusters_from_request())
    return jsonify(
        api_success(
            {
                "selection": result.to_dict(),
                "selection_context": get_selection_context(refreshed["selection_store"]).to_dict(),
                "render_payload": refreshed["render_payload"].to_dict(),
            },
            diagnostics={"dependency_mode": "real Step 1-5 fixture state"},
        )
    )


def _n_clusters_from_request() -> int:
    return n_clusters_from_request()
