from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, run_default_analysis
from app.modules.data_workspace.service import create_feature_matrix
from app.modules.labeling.service import (
    apply_labeling_action,
    clear_annotations,
    get_labeling_state,
)
from app.modules.labeling.state import (
    get_debug_store_for_context as get_labeling_store_for_context,
    reset_debug_store_for_context as reset_labeling_store_for_context,
)
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
from app.shared.effective_analysis import apply_manual_labels_to_analysis
from .fixtures import (
    ANALYSIS_SELECTION_DATASET_OPTIONS,
    DEFAULT_WORKFLOW_DATASET_ID,
    analysis_selection_dataset,
    analysis_selection_initial_selected_point_ids,
    is_analysis_selection_dataset_id,
)

WORKFLOW_NAME = "analysis-labeling"
DEPENDENCY_MODE = "real Step 1-5 workflow fixture"


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "analysis_labeling_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/analysis-labeling",
    )

    @blueprint.get("/")
    def index():
        view_model = _build_view_model(_n_clusters_from_request(), _dataset_id_from_request())
        return render_template("workflows/analysis_labeling.html", **view_model)

    @blueprint.get("/api/state")
    def state_api():
        view_model = _build_view_model(_n_clusters_from_request(), _dataset_id_from_request())
        return jsonify(
            api_success(
                _state_payload(view_model),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/select")
    def select_api():
        return _selection_action_response("select")

    @blueprint.post("/api/clear")
    def clear_api():
        return _selection_action_response("clear")

    @blueprint.post("/api/reset-selection")
    def reset_selection_api():
        dataset = _workflow_dataset(_dataset_id_from_request())
        store = reset_debug_store_for_dataset(
            dataset,
            analysis_selection_initial_selected_point_ids(dataset.dataset_id),
        )
        state = get_selection_state(store)
        return jsonify(
            api_success(
                {"state": state.to_dict(), "groups": []},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.get("/api/groups")
    def groups_api():
        groups = [group.to_dict() for group in list_selection_groups(_workflow_store())]
        return jsonify(
            api_success(
                {"groups": groups, "group_count": len(groups)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
                metadata={"workflow": WORKFLOW_NAME},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": selection_groups_payload(_workflow_store())},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
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
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/label")
    def label_api():
        payload = request_payload(request)
        context = get_selection_context(_workflow_store())
        store = get_labeling_store_for_context(context)
        n_clusters = _n_clusters_from_request()
        try:
            action = str(payload.get("action", ""))
            label_value = payload.get("label_value")
            _validate_workflow_label(action, label_value, n_clusters)
            annotation = apply_labeling_action(
                store,
                context,
                action=action,
                label_value=label_value,
                point_ids=optional_point_ids_from_payload(payload),
            )
            view_model = _build_view_model(n_clusters, _dataset_id_from_request())
        except ValueError as exc:
            return jsonify(api_error("invalid_labeling_action", str(exc))), 400

        return jsonify(
            api_success(
                {
                    "annotation": annotation.to_dict(),
                    "state": _state_payload(view_model),
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/clear-labels")
    def clear_labels_api():
        context = get_selection_context(_workflow_store())
        store = get_labeling_store_for_context(context)
        clear_annotations(store)
        view_model = _build_view_model(_n_clusters_from_request(), _dataset_id_from_request())
        return jsonify(
            api_success(
                {"state": _state_payload(view_model)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/reset-labels")
    def reset_labels_api():
        context = get_selection_context(_workflow_store())
        reset_labeling_store_for_context(context)
        view_model = _build_view_model(_n_clusters_from_request(), _dataset_id_from_request())
        return jsonify(
            api_success(
                {"state": _state_payload(view_model)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    return blueprint


def _build_view_model(n_clusters: int, dataset_id: str):
    dataset = _workflow_dataset(dataset_id)
    matrix = create_feature_matrix(dataset)
    projection = project_feature_matrix(matrix)
    error = None

    try:
        raw_analysis = run_default_analysis(matrix, n_clusters=n_clusters)
    except ValueError as exc:
        error = str(exc)
        n_clusters = DEFAULT_N_CLUSTERS
        raw_analysis = run_default_analysis(matrix, n_clusters=n_clusters)

    selection_store = _workflow_store_for_dataset(dataset)
    selection_state = get_selection_state(selection_store)
    context = get_selection_context(selection_store)
    labeling_state = get_labeling_state(get_labeling_store_for_context(context))
    analysis = apply_manual_labels_to_analysis(dataset, raw_analysis, labeling_state)
    cluster_labels = {
        assignment.point_id: assignment.cluster_id
        for assignment in analysis.cluster_result.assignments
    }
    outlier_ids = set(analysis.outlier_result.outlier_point_ids)
    selected_ids = set(selection_state.selected_point_ids)
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
                "manual_labels": _manual_labels_for_point(labeling_state, point["point_id"]),
            }
        )

    return {
        "dataset": dataset,
        "matrix": matrix,
        "projection": projection,
        "analysis": analysis,
        "raw_analysis": raw_analysis,
        "plot_points": plot_points,
        "selection_state": selection_state,
        "context": context,
        "selection_groups": list_selection_groups(selection_store),
        "labeling_state": labeling_state,
        "allowed_labels": _allowed_labels(n_clusters),
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
        metadata={"workflow": WORKFLOW_NAME},
    )
    if error is not None:
        return jsonify(api_error("invalid_selection_action", error)), 400

    return jsonify(
        api_success(
            result.to_dict(),
            diagnostics={"dependency_mode": DEPENDENCY_MODE},
        )
    )


def _state_payload(view_model):
    return {
        "dataset": view_model["dataset"].to_dict(),
        "feature_matrix": view_model["matrix"].to_dict(),
        "projection": view_model["projection"].to_dict(),
        "outliers": view_model["analysis"].outlier_result.to_dict(),
        "clusters": view_model["analysis"].cluster_result.to_dict(),
        "raw_outliers": view_model["raw_analysis"].outlier_result.to_dict(),
        "raw_clusters": view_model["raw_analysis"].cluster_result.to_dict(),
        "selection": view_model["selection_state"].to_dict(),
        "selection_context": view_model["context"].to_dict(),
        "selection_groups": [group.to_dict() for group in view_model["selection_groups"]],
        "labeling": view_model["labeling_state"].to_dict(),
    }


def _manual_labels_for_point(labeling_state, point_id: str):
    labels = []
    for annotation in labeling_state.annotations:
        if point_id in annotation.point_ids:
            labels.append(
                {
                    "annotation_id": annotation.annotation_id,
                    "label_type": annotation.label_type,
                    "label_value": annotation.label_value,
                    "display_label": _annotation_display_label(annotation),
                }
            )
    return labels


def _annotation_display_label(annotation):
    if annotation.label_type == "outlier" and annotation.label_value is True:
        return "outlier"
    return annotation.label_value


def _validate_workflow_label(action: str, label_value, n_clusters: int) -> None:
    if action == "assign_cluster":
        allowed_clusters = set(_allowed_cluster_labels(n_clusters))
        if label_value not in allowed_clusters:
            allowed = ", ".join([*sorted(allowed_clusters), "outlier"])
            raise ValueError(f"label_value must be one of: {allowed}")
        return

    if action == "mark_outlier":
        return

    raise ValueError("analysis-labeling only supports cluster_N labels and outlier")


def _allowed_cluster_labels(n_clusters: int):
    return [f"cluster_{index}" for index in range(1, n_clusters + 1)]


def _allowed_labels(n_clusters: int):
    return [*_allowed_cluster_labels(n_clusters), "outlier"]


def _n_clusters_from_request() -> int:
    return n_clusters_from_request()


def _dataset_id_from_request() -> str:
    return dataset_id_from_request(
        DEFAULT_WORKFLOW_DATASET_ID,
        is_analysis_selection_dataset_id,
    )
