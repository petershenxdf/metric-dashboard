from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.scatterplot.fixtures import scatterplot_fixture_state
from app.modules.selection.http_helpers import optional_point_ids_from_payload
from app.modules.selection.service import (
    delete_selection_group,
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


DEPENDENCY_MODE = "real scatterplot plus selection workflow fixture"


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "scatter_selection_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/scatter-selection",
    )

    @blueprint.get("/")
    def index():
        return render_template(
            "workflows/scatter_selection.html",
            dependency_mode=DEPENDENCY_MODE,
            **scatterplot_fixture_state(_n_clusters_from_request()),
        )

    @blueprint.get("/api/state")
    def state_api():
        state = scatterplot_fixture_state(_n_clusters_from_request())
        return jsonify(
            api_success(
                {
                    "dataset": state["dataset"].to_dict(),
                    "projection": state["projection"].to_dict(),
                    "clusters": state["analysis"].cluster_result.to_dict(),
                    "outliers": state["analysis"].outlier_result.to_dict(),
                    "selection": state["selection_state"].to_dict(),
                    "selection_context": state["context"].to_dict(),
                    "selection_groups": [group.to_dict() for group in state["selection_groups"]],
                    "render_payload": state["render_payload"].to_dict(),
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
                metadata={"workflow": "scatter-selection"},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": selection_groups_payload(scatterplot_fixture_state(_n_clusters_from_request())["selection_store"])},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
        metadata={"workflow": "scatter-selection"},
    )
    if error is not None:
        return jsonify(api_error("invalid_selection_action", error)), 400

    refreshed = scatterplot_fixture_state(_n_clusters_from_request())
    return jsonify(
        api_success(
            {
                "selection": result.to_dict(),
                "render_payload": refreshed["render_payload"].to_dict(),
            },
            diagnostics={"dependency_mode": DEPENDENCY_MODE},
        )
    )


def _n_clusters_from_request() -> int:
    return n_clusters_from_request()
