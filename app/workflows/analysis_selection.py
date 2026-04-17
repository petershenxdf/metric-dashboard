from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, run_default_analysis
from app.modules.data_workspace.service import create_feature_matrix
from app.modules.projection.service import project_feature_matrix, scaled_projection_points
from app.modules.selection.http_helpers import (
    optional_point_ids_from_payload,
)
from app.modules.selection.service import (
    delete_selection_group,
    get_selection_context,
    get_selection_state,
    list_selection_groups,
    save_selection_group,
    select_selection_group,
)
from app.modules.selection.state import get_debug_store_for_dataset, reset_debug_store_for_dataset
from app.shared.flask_helpers import api_error, api_success
from app.shared.request_helpers import (
    apply_selection_action_or_error,
    dataset_id_from_request,
    n_clusters_from_request,
    request_payload,
    selection_groups_payload,
)
from .fixtures import (
    ANALYSIS_SELECTION_DATASET_OPTIONS,
    DEFAULT_WORKFLOW_DATASET_ID,
    analysis_selection_dataset,
    analysis_selection_initial_selected_point_ids,
    is_analysis_selection_dataset_id,
)


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "analysis_selection_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/analysis-selection",
    )

    @blueprint.get("/")
    def index():
        view_model = _build_view_model(_n_clusters_from_request(), _dataset_id_from_request())
        return render_template("workflows/analysis_selection.html", **view_model)

    @blueprint.get("/api/state")
    def state_api():
        view_model = _build_view_model(_n_clusters_from_request(), _dataset_id_from_request())
        return jsonify(
            api_success(
                _state_payload(view_model),
                diagnostics={"dependency_mode": "real Step 1-4 workflow fixture"},
            )
        )

    @blueprint.post("/api/select")
    def select_api():
        return _selection_action_response("select")

    @blueprint.post("/api/deselect")
    def deselect_api():
        return _selection_action_response("deselect")

    @blueprint.post("/api/replace")
    def replace_api():
        return _selection_action_response("replace")

    @blueprint.post("/api/toggle")
    def toggle_api():
        return _selection_action_response("toggle")

    @blueprint.post("/api/clear")
    def clear_api():
        return _selection_action_response("clear")

    @blueprint.post("/api/reset")
    def reset_api():
        dataset = _workflow_dataset(_dataset_id_from_request())
        store = reset_debug_store_for_dataset(
            dataset,
            analysis_selection_initial_selected_point_ids(dataset.dataset_id),
        )
        state = get_selection_state(store)
        return jsonify(
            api_success(
                {"state": state.to_dict(), "groups": []},
                diagnostics={"dependency_mode": "real Step 1-4 workflow fixture"},
            )
        )

    @blueprint.get("/api/groups")
    def groups_api():
        groups = [group.to_dict() for group in list_selection_groups(_workflow_store())]
        return jsonify(
            api_success(
                {"groups": groups, "group_count": len(groups)},
                diagnostics={"dependency_mode": "real Step 1-4 workflow fixture"},
            )
        )

    @blueprint.post("/api/groups")
    def save_group_api():
        payload = request_payload(request)
        try:
            group = save_selection_group(
                _workflow_store(),
                group_name=payload.get("group_name", ""),
                point_ids=optional_point_ids_from_payload(payload),
                metadata={"workflow": "analysis-selection"},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": selection_groups_payload(_workflow_store())},
                diagnostics={"dependency_mode": "real Step 1-4 workflow fixture"},
            )
        )

    @blueprint.post("/api/groups/<group_id>/select")
    def select_group_api(group_id: str):
        try:
            result = select_selection_group(_workflow_store(), group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"selection": result.to_dict(), "groups": selection_groups_payload(_workflow_store())},
                diagnostics={"dependency_mode": "real Step 1-4 workflow fixture"},
            )
        )

    @blueprint.delete("/api/groups/<group_id>")
    def delete_group_api(group_id: str):
        try:
            group = delete_selection_group(_workflow_store(), group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"deleted_group": group.to_dict(), "groups": selection_groups_payload(_workflow_store())},
                diagnostics={"dependency_mode": "real Step 1-4 workflow fixture"},
            )
        )

    return blueprint


def _build_view_model(n_clusters: int, dataset_id: str):
    dataset = _workflow_dataset(dataset_id)
    matrix = create_feature_matrix(dataset)
    projection = project_feature_matrix(matrix)
    error = None

    try:
        analysis = run_default_analysis(matrix, n_clusters=n_clusters)
    except ValueError as exc:
        error = str(exc)
        n_clusters = DEFAULT_N_CLUSTERS
        analysis = run_default_analysis(matrix, n_clusters=n_clusters)

    cluster_labels = {
        assignment.point_id: assignment.cluster_id
        for assignment in analysis.cluster_result.assignments
    }
    outlier_ids = set(analysis.outlier_result.outlier_point_ids)
    store = _workflow_store_for_dataset(dataset)
    state = get_selection_state(store)
    selected_ids = set(state.selected_point_ids)
    point_by_id = {point.point_id: point for point in dataset.points}

    plot_points = []
    for point in scaled_projection_points(projection, cluster_labels):
        source_point = point_by_id[point["point_id"]]
        plot_points.append(
            {
                **point,
                "features": source_point.features,
                "source_label": source_point.metadata.get("label", ""),
                "cluster_id": cluster_labels.get(point["point_id"], ""),
                "is_outlier": point["point_id"] in outlier_ids,
                "is_selected": point["point_id"] in selected_ids,
            }
        )

    return {
        "dataset": dataset,
        "matrix": matrix,
        "projection": projection,
        "analysis": analysis,
        "plot_points": plot_points,
        "state": state,
        "context": get_selection_context(store),
        "selection_groups": list_selection_groups(store),
        "n_clusters": n_clusters,
        "dataset_options": ANALYSIS_SELECTION_DATASET_OPTIONS,
        "selected_dataset_id": dataset.dataset_id,
        "error": error,
    }


def _workflow_dataset(dataset_id: str | None = None):
    return analysis_selection_dataset(dataset_id)


def _workflow_store():
    dataset = _workflow_dataset(_dataset_id_from_request())
    return _workflow_store_for_dataset(dataset)


def _workflow_store_for_dataset(dataset):
    return get_debug_store_for_dataset(
        dataset,
        analysis_selection_initial_selected_point_ids(dataset.dataset_id),
    )


def _selection_action_response(action_name: str):
    payload = request_payload(request)
    result, error = apply_selection_action_or_error(
        _workflow_store(),
        action_name,
        payload,
        metadata={"workflow": "analysis-selection"},
    )
    if error is not None:
        return jsonify(api_error("invalid_selection_action", error)), 400

    return jsonify(
        api_success(
            result.to_dict(),
            diagnostics={"dependency_mode": "real Step 1-4 workflow fixture"},
        )
    )


def _state_payload(view_model):
    return {
        "dataset": view_model["dataset"].to_dict(),
        "feature_matrix": view_model["matrix"].to_dict(),
        "projection": view_model["projection"].to_dict(),
        "outliers": view_model["analysis"].outlier_result.to_dict(),
        "clusters": view_model["analysis"].cluster_result.to_dict(),
        "selection": view_model["state"].to_dict(),
        "selection_context": view_model["context"].to_dict(),
        "selection_groups": [group.to_dict() for group in view_model["selection_groups"]],
    }


def _n_clusters_from_request() -> int:
    return n_clusters_from_request()


def _dataset_id_from_request() -> str:
    return dataset_id_from_request(
        DEFAULT_WORKFLOW_DATASET_ID,
        is_analysis_selection_dataset_id,
    )
