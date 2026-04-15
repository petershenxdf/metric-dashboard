from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, run_default_analysis
from app.modules.data_workspace.service import create_feature_matrix
from app.modules.labeling.service import apply_labeling_action, clear_annotations, get_labeling_state
from app.modules.labeling.state import (
    get_debug_store_for_context as get_labeling_store_for_context,
    reset_debug_store_for_context as reset_labeling_store_for_context,
)
from app.modules.projection.service import project_feature_matrix
from app.modules.scatterplot.service import build_render_payload
from app.modules.selection.http_helpers import optional_point_ids_from_payload, request_payload, selection_action_from_payload
from app.modules.selection.service import (
    apply_selection_action,
    delete_selection_group,
    get_selection_context,
    get_selection_state,
    list_selection_groups,
    save_selection_group,
    select_selection_group,
)
from app.modules.selection.state import get_debug_store_for_dataset
from app.shared.flask_helpers import api_error, api_success
from app.workflows.effective_analysis import apply_manual_labels_to_analysis
from app.workflows.fixtures import (
    DEFAULT_WORKFLOW_DATASET_ID,
    analysis_selection_dataset,
    analysis_selection_initial_selected_point_ids,
)


DEPENDENCY_MODE = "real scatterplot plus selection and labeling workflow fixture"


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "scatter_labeling_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/scatter-labeling",
    )

    @blueprint.get("/")
    def index():
        return render_template(
            "workflows/scatter_labeling.html",
            dependency_mode=DEPENDENCY_MODE,
            **_workflow_state(_n_clusters_from_request()),
        )

    @blueprint.get("/api/state")
    def state_api():
        return jsonify(
            api_success(
                _state_payload(_workflow_state(_n_clusters_from_request())),
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
        state = _workflow_state(_n_clusters_from_request())
        groups = [group.to_dict() for group in state["selection_groups"]]
        return jsonify(
            api_success(
                {"groups": groups, "group_count": len(groups)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/groups")
    def save_group_api():
        state = _workflow_state(_n_clusters_from_request())
        payload = request_payload(request)
        try:
            group = save_selection_group(
                state["selection_store"],
                group_name=payload.get("group_name", ""),
                point_ids=optional_point_ids_from_payload(payload),
                metadata={"workflow": "scatter-labeling"},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": _selection_groups_payload()},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/groups/<group_id>/select")
    def select_group_api(group_id: str):
        state = _workflow_state(_n_clusters_from_request())
        try:
            result = select_selection_group(state["selection_store"], group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"selection": result.to_dict(), "groups": _selection_groups_payload()},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.delete("/api/groups/<group_id>")
    def delete_group_api(group_id: str):
        state = _workflow_state(_n_clusters_from_request())
        try:
            group = delete_selection_group(state["selection_store"], group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"deleted_group": group.to_dict(), "groups": _selection_groups_payload()},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/label")
    def label_api():
        payload = request_payload(request)
        state = _workflow_state(_n_clusters_from_request())
        store = get_labeling_store_for_context(state["context"])
        try:
            action = str(payload.get("action", ""))
            label_value = payload.get("label_value")
            _validate_label(action, label_value, state["n_clusters"])
            annotation = apply_labeling_action(
                store,
                state["context"],
                action=action,
                label_value=label_value,
            )
            refreshed = _workflow_state(state["n_clusters"])
        except ValueError as exc:
            return jsonify(api_error("invalid_labeling_action", str(exc))), 400

        return jsonify(
            api_success(
                {
                    "annotation": annotation.to_dict(),
                    "state": _state_payload(refreshed),
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/clear-labels")
    def clear_labels_api():
        state = _workflow_state(_n_clusters_from_request())
        clear_annotations(get_labeling_store_for_context(state["context"]))
        refreshed = _workflow_state(state["n_clusters"])
        return jsonify(
            api_success(
                {"state": _state_payload(refreshed)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/reset-labels")
    def reset_labels_api():
        state = _workflow_state(_n_clusters_from_request())
        reset_labeling_store_for_context(state["context"])
        refreshed = _workflow_state(state["n_clusters"])
        return jsonify(
            api_success(
                {"state": _state_payload(refreshed)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    return blueprint


def _workflow_state(n_clusters: int):
    dataset = analysis_selection_dataset(DEFAULT_WORKFLOW_DATASET_ID)
    matrix = create_feature_matrix(dataset)
    projection = project_feature_matrix(matrix)
    raw_analysis = run_default_analysis(matrix, n_clusters=n_clusters)
    selection_store = get_debug_store_for_dataset(
        dataset,
        analysis_selection_initial_selected_point_ids(dataset.dataset_id),
    )
    selection_state = get_selection_state(selection_store)
    context = get_selection_context(selection_store)
    labeling_state = get_labeling_state(get_labeling_store_for_context(context))
    analysis = apply_manual_labels_to_analysis(dataset, raw_analysis, labeling_state)
    render_payload = build_render_payload(
        dataset=dataset,
        projection=projection,
        clusters=analysis.cluster_result,
        outliers=analysis.outlier_result,
        selection_context=context,
        labeling_state=labeling_state,
    )

    return {
        "dataset": dataset,
        "matrix": matrix,
        "projection": projection,
        "raw_analysis": raw_analysis,
        "analysis": analysis,
        "selection_store": selection_store,
        "selection_state": selection_state,
        "selection_groups": list_selection_groups(selection_store),
        "context": context,
        "labeling_state": labeling_state,
        "render_payload": render_payload,
        "allowed_labels": _allowed_labels(n_clusters),
        "n_clusters": n_clusters,
    }


def _selection_action_response(action_name: str):
    state = _workflow_state(_n_clusters_from_request())
    payload = request_payload(request)
    try:
        action = selection_action_from_payload(
            action_name,
            payload,
            metadata={"workflow": "scatter-labeling"},
        )
        result = apply_selection_action(state["selection_store"], action)
        refreshed = _workflow_state(state["n_clusters"])
    except ValueError as exc:
        return jsonify(api_error("invalid_selection_action", str(exc))), 400

    return jsonify(
        api_success(
            {
                "selection": result.to_dict(),
                "state": _state_payload(refreshed),
            },
            diagnostics={"dependency_mode": DEPENDENCY_MODE},
        )
    )


def _state_payload(state):
    return {
        "dataset": state["dataset"].to_dict(),
        "projection": state["projection"].to_dict(),
        "raw_clusters": state["raw_analysis"].cluster_result.to_dict(),
        "raw_outliers": state["raw_analysis"].outlier_result.to_dict(),
        "clusters": state["analysis"].cluster_result.to_dict(),
        "outliers": state["analysis"].outlier_result.to_dict(),
        "selection": state["selection_state"].to_dict(),
        "selection_context": state["context"].to_dict(),
        "selection_groups": [group.to_dict() for group in state["selection_groups"]],
        "labeling": state["labeling_state"].to_dict(),
        "render_payload": state["render_payload"].to_dict(),
    }


def _validate_label(action: str, label_value, n_clusters: int) -> None:
    if action == "assign_cluster" and label_value in set(_allowed_cluster_labels(n_clusters)):
        return

    if action == "mark_outlier":
        return

    allowed = ", ".join([*_allowed_cluster_labels(n_clusters), "outlier"])
    raise ValueError(f"label must be one of: {allowed}")


def _allowed_cluster_labels(n_clusters: int):
    return [f"cluster_{index}" for index in range(1, n_clusters + 1)]


def _allowed_labels(n_clusters: int):
    return [*_allowed_cluster_labels(n_clusters), "outlier"]


def _n_clusters_from_request() -> int:
    payload = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}

    raw_value = payload.get("n_clusters") or request.args.get("n_clusters") or request.form.get("n_clusters")
    if raw_value is None:
        return DEFAULT_N_CLUSTERS

    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_N_CLUSTERS

    return max(value, 1)


def _selection_groups_payload():
    return [
        group.to_dict()
        for group in list_selection_groups(_workflow_state(_n_clusters_from_request())["selection_store"])
    ]
