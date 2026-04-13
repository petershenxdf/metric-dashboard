from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.fixtures import (
    default_analysis_feature_names,
    default_analysis_raw_points,
)
from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, run_default_analysis
from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.projection.service import project_feature_matrix, scaled_projection_points
from app.modules.selection.schemas import SelectionAction
from app.modules.selection.service import (
    apply_selection_action,
    delete_selection_group,
    get_selection_context,
    get_selection_state,
    list_selection_groups,
    save_selection_group,
    select_selection_group,
)
from app.modules.selection.state import get_debug_store_for_dataset, reset_debug_store_for_dataset
from app.shared.flask_helpers import api_error, api_success

WIDE_GAP_DATASET_ID = "wide_gap_analysis_debug"
DEFAULT_ANALYSIS_DATASET_ID = "default_analysis_outlier_debug"
DEFAULT_WORKFLOW_DATASET_ID = WIDE_GAP_DATASET_ID
DATASET_OPTIONS = (
    {
        "dataset_id": WIDE_GAP_DATASET_ID,
        "title": "Wide Gap Debug",
        "description": "fewer points with larger visual gaps for selection testing",
    },
    {
        "dataset_id": DEFAULT_ANALYSIS_DATASET_ID,
        "title": "Default Analysis Outlier Debug",
        "description": "three compact clusters plus three distant outlier candidates",
    },
)
INITIAL_SELECTED_BY_DATASET = {
    WIDE_GAP_DATASET_ID: ("alpha_01", "outlier_north"),
    DEFAULT_ANALYSIS_DATASET_ID: ("cluster_a_001", "outlier_far_001"),
}


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
        store = reset_debug_store_for_dataset(dataset, _initial_selected_point_ids(dataset.dataset_id))
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
        payload = _request_payload()
        try:
            group = save_selection_group(
                _workflow_store(),
                group_name=payload.get("group_name", ""),
                point_ids=_optional_point_ids_from_payload(payload),
                metadata={"workflow": "analysis-selection"},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": _selection_groups_payload()},
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
                {"selection": result.to_dict(), "groups": _selection_groups_payload()},
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
                {"deleted_group": group.to_dict(), "groups": _selection_groups_payload()},
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
        "dataset_options": DATASET_OPTIONS,
        "selected_dataset_id": dataset.dataset_id,
        "error": error,
    }


def _workflow_dataset(dataset_id: str | None = None):
    dataset_id = dataset_id or DEFAULT_WORKFLOW_DATASET_ID
    if dataset_id == WIDE_GAP_DATASET_ID:
        return create_dataset(
            _wide_gap_raw_points(),
            dataset_id=WIDE_GAP_DATASET_ID,
            feature_names=default_analysis_feature_names(),
        )

    if dataset_id == DEFAULT_ANALYSIS_DATASET_ID:
        return create_dataset(
            default_analysis_raw_points(),
            dataset_id=DEFAULT_ANALYSIS_DATASET_ID,
            feature_names=default_analysis_feature_names(),
        )

    return _workflow_dataset(DEFAULT_WORKFLOW_DATASET_ID)


def _workflow_store():
    dataset = _workflow_dataset(_dataset_id_from_request())
    return _workflow_store_for_dataset(dataset)


def _workflow_store_for_dataset(dataset):
    return get_debug_store_for_dataset(dataset, _initial_selected_point_ids(dataset.dataset_id))


def _selection_action_response(action_name: str):
    payload = _request_payload()
    try:
        action = SelectionAction(
            action=action_name,
            point_ids=_point_ids_from_payload(payload),
            source=str(payload.get("source", "api")),
            mode=payload.get("mode"),
            metadata={"workflow": "analysis-selection"},
        )
        result = apply_selection_action(_workflow_store(), action)
    except ValueError as exc:
        return jsonify(api_error("invalid_selection_action", str(exc))), 400

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
    raw_value = request.args.get("n_clusters")
    if raw_value is None:
        return DEFAULT_N_CLUSTERS

    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_N_CLUSTERS

    return max(value, 1)


def _dataset_id_from_request() -> str:
    payload = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}

    raw_value = payload.get("dataset_id") or request.args.get("dataset_id") or request.form.get("dataset_id")
    if raw_value in {option["dataset_id"] for option in DATASET_OPTIONS}:
        return raw_value
    return DEFAULT_WORKFLOW_DATASET_ID


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
    return [group.to_dict() for group in list_selection_groups(_workflow_store())]


def _initial_selected_point_ids(dataset_id: str):
    return INITIAL_SELECTED_BY_DATASET.get(dataset_id, INITIAL_SELECTED_BY_DATASET[DEFAULT_WORKFLOW_DATASET_ID])


def _wide_gap_raw_points():
    return [
        {"point_id": "alpha_01", "features": [-8.2, -5.8], "metadata": {"label": "alpha"}},
        {"point_id": "alpha_02", "features": [-7.5, -6.7], "metadata": {"label": "alpha"}},
        {"point_id": "alpha_03", "features": [-6.9, -5.2], "metadata": {"label": "alpha"}},
        {"point_id": "beta_01", "features": [6.5, -6.0], "metadata": {"label": "beta"}},
        {"point_id": "beta_02", "features": [7.4, -5.1], "metadata": {"label": "beta"}},
        {"point_id": "beta_03", "features": [8.1, -6.8], "metadata": {"label": "beta"}},
        {"point_id": "gamma_01", "features": [-2.2, 7.4], "metadata": {"label": "gamma"}},
        {"point_id": "gamma_02", "features": [-1.2, 8.3], "metadata": {"label": "gamma"}},
        {"point_id": "gamma_03", "features": [-3.2, 8.6], "metadata": {"label": "gamma"}},
        {"point_id": "outlier_north", "features": [2.0, 19.0], "metadata": {"label": "outlier_fixture"}},
        {"point_id": "outlier_west", "features": [-18.0, 2.0], "metadata": {"label": "outlier_fixture"}},
        {"point_id": "outlier_east", "features": [18.0, 6.0], "metadata": {"label": "outlier_fixture"}},
    ]
