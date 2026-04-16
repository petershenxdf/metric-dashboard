from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request

from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.labeling.service import (
    apply_labeling_action,
    clear_annotations,
    get_labeling_state,
)
from app.modules.labeling.state import (
    get_debug_store_for_context as get_labeling_store_for_context,
    reset_debug_store_for_context as reset_labeling_store_for_context,
)
from app.modules.selection.http_helpers import (
    optional_point_ids_from_payload,
    request_payload,
    selection_action_from_payload,
)
from app.modules.selection.service import (
    apply_selection_action,
    delete_selection_group,
    get_selection_context,
    get_selection_state,
    list_selection_groups,
    save_selection_group,
    select_selection_group,
)
from app.modules.selection.state import (
    get_debug_store_for_dataset as get_selection_store_for_dataset,
    reset_debug_store_for_dataset as reset_selection_store_for_dataset,
)
from app.shared.flask_helpers import api_error, api_success
from app.shared.schemas import Dataset, FeatureMatrix

from .fixtures import (
    normalize_ssdbcodi_dataset_id,
    ssdbcodi_dataset_id,
    ssdbcodi_dataset_options,
    ssdbcodi_feature_names,
    ssdbcodi_raw_points,
)
from .service import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_BOOTSTRAP_K,
    DEFAULT_CONTAMINATION,
    DEFAULT_MIN_PTS,
    cluster_counts,
    run_ssdbcodi,
)
from .store import get_debug_store, reset_debug_store

DEPENDENCY_MODE = "real data-workspace, selection, and labeling stores"
WORKFLOW_NAME = "ssdbcodi"
PALETTE = ("#3565a8", "#d4533a", "#46a35a", "#a063b5", "#c39a3b", "#1f8fa3", "#b03570")


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "ssdbcodi",
        __name__,
        template_folder="templates",
        url_prefix="/modules/ssdbcodi",
    )

    @blueprint.get("/")
    def index():
        dataset_id, dataset_error = _dataset_id_from_request()
        params, param_error = _params_from_request(dataset_id)
        view_model = _build_view_model(dataset_id, params, dataset_error or param_error)
        return render_template("ssdbcodi/index.html", **view_model)

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "ssdbcodi", "status": "working"},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        dataset_id, error = _dataset_id_from_request()
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400

        store = get_debug_store(dataset_id)
        selection_store = _selection_store(dataset_id)
        selection_state = get_selection_state(selection_store)
        context = selection_state.to_context()
        labeling_state = get_labeling_state(get_labeling_store_for_context(context))
        latest = store.latest_result
        payload = {
            "module": "ssdbcodi",
            "status": "working",
            "dataset_id": dataset_id,
            "history": store.history_summary(),
            "latest_run_id": latest.run_id if latest is not None else None,
            "selection": selection_state.to_dict(),
            "selection_context": context.to_dict(),
            "selection_groups": [group.to_dict() for group in list_selection_groups(selection_store)],
            "labeling": labeling_state.to_dict(),
        }
        if latest is not None:
            payload["cluster_counts"] = cluster_counts(latest)
            payload["outlier_count"] = len(latest.outlier_result.outlier_point_ids)
            payload["bootstrap_used"] = bool(latest.parameters.get("bootstrap_used"))
        return jsonify(
            api_success(
                payload,
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.get("/api/scores")
    def scores_api():
        dataset_id, error = _dataset_id_from_request()
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        store = get_debug_store(dataset_id)
        if store.latest_result is None:
            return jsonify(
                api_error(
                    "no_result",
                    "no SSDBCODI result has been computed yet; POST to /api/run first",
                )
            ), 400
        return jsonify(
            api_success(
                {
                    "run_id": store.latest_result.run_id,
                    "point_scores": [
                        score.to_dict() for score in store.latest_result.point_scores
                    ],
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.get("/api/result")
    def result_api():
        dataset_id, error = _dataset_id_from_request()
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        store = get_debug_store(dataset_id)
        if store.latest_result is None:
            return jsonify(
                api_error(
                    "no_result",
                    "no SSDBCODI result has been computed yet; POST to /api/run first",
                )
            ), 400
        return jsonify(
            api_success(
                store.latest_result.to_dict(),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/run")
    def run_api():
        payload = request_payload(request)
        dataset_id, dataset_error = _dataset_id_from_payload(payload)
        if dataset_error is not None:
            return jsonify(api_error("invalid_dataset", dataset_error)), 400
        params, error = _params_from_payload(payload, dataset_id)
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400

        matrix = _fixture_matrix(dataset_id)
        labeling_state = _labeling_state_for_current_selection(dataset_id)
        try:
            result = run_ssdbcodi(matrix, labeling_state=labeling_state, **params)
        except ValueError as exc:
            return jsonify(api_error("invalid_parameters", str(exc))), 400

        _store_result(result, dataset_id)
        return jsonify(
            api_success(
                _result_payload(result),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/select")
    def select_api():
        return _selection_action_response("select")

    @blueprint.post("/api/clear-selection")
    def clear_selection_api():
        return _selection_action_response("clear")

    @blueprint.post("/api/reset-selection")
    def reset_selection_api():
        payload = request_payload(request)
        dataset_id, error = _dataset_id_from_payload(payload)
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        store = reset_selection_store_for_dataset(_fixture_dataset(dataset_id), initial_selected_point_ids=())
        state = get_selection_state(store)
        return jsonify(
            api_success(
                {"state": state.to_dict(), "groups": []},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.get("/api/groups")
    def groups_api():
        dataset_id, error = _dataset_id_from_request()
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        groups = [group.to_dict() for group in list_selection_groups(_selection_store(dataset_id))]
        return jsonify(
            api_success(
                {"groups": groups, "group_count": len(groups)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/groups")
    def save_group_api():
        payload = request_payload(request)
        dataset_id, error = _dataset_id_from_payload(payload)
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        try:
            group = save_selection_group(
                _selection_store(dataset_id),
                group_name=payload.get("group_name", ""),
                point_ids=optional_point_ids_from_payload(payload),
                metadata={"workflow": WORKFLOW_NAME},
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"group": group.to_dict(), "groups": _selection_groups_payload(dataset_id)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/groups/<group_id>/select")
    def select_group_api(group_id: str):
        payload = request_payload(request)
        dataset_id, error = _dataset_id_from_payload(payload)
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        try:
            result = select_selection_group(_selection_store(dataset_id), group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"selection": result.to_dict(), "groups": _selection_groups_payload(dataset_id)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.delete("/api/groups/<group_id>")
    def delete_group_api(group_id: str):
        payload = request_payload(request)
        dataset_id, error = _dataset_id_from_payload(payload)
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        try:
            group = delete_selection_group(_selection_store(dataset_id), group_id)
        except ValueError as exc:
            return jsonify(api_error("invalid_selection_group", str(exc))), 400

        return jsonify(
            api_success(
                {"deleted_group": group.to_dict(), "groups": _selection_groups_payload(dataset_id)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/label")
    def label_api():
        payload = request_payload(request)
        dataset_id, dataset_error = _dataset_id_from_payload(payload)
        if dataset_error is not None:
            return jsonify(api_error("invalid_dataset", dataset_error)), 400
        params, error = _params_from_payload(payload, dataset_id)
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400

        context = get_selection_context(_selection_store(dataset_id))
        store = get_labeling_store_for_context(context)
        try:
            action = str(payload.get("action", ""))
            label_value = payload.get("label_value")
            _validate_label_action(action, label_value, params["n_clusters"])
            annotation = apply_labeling_action(
                store,
                context,
                action=action,
                label_value=label_value,
                point_ids=optional_point_ids_from_payload(payload),
            )
        except ValueError as exc:
            return jsonify(api_error("invalid_label", str(exc))), 400

        return jsonify(
            api_success(
                {
                    "annotation": annotation.to_dict(),
                    "labeling_state": get_labeling_state(store).to_dict(),
                    "pending_run": True,
                },
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/clear-labels")
    def clear_labels_api():
        payload = request_payload(request)
        dataset_id, error = _dataset_id_from_payload(payload)
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        context = get_selection_context(_selection_store(dataset_id))
        store = get_labeling_store_for_context(context)
        clear_annotations(store)
        return jsonify(
            api_success(
                {"labeling_state": get_labeling_state(store).to_dict()},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/reset-labels")
    def reset_labels_api():
        payload = request_payload(request)
        dataset_id, error = _dataset_id_from_payload(payload)
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        context = get_selection_context(_selection_store(dataset_id))
        store = reset_labeling_store_for_context(context)
        return jsonify(
            api_success(
                {"labeling_state": get_labeling_state(store).to_dict()},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    @blueprint.post("/api/reset")
    def reset_api():
        payload = request_payload(request)
        dataset_id, error = _dataset_id_from_payload(payload)
        if error is not None:
            return jsonify(api_error("invalid_dataset", error)), 400
        reset_debug_store(dataset_id)
        selection_store = reset_selection_store_for_dataset(_fixture_dataset(dataset_id), initial_selected_point_ids=())
        context = get_selection_context(selection_store)
        reset_labeling_store_for_context(context)
        return jsonify(
            api_success(
                {"reset": True, "dataset_id": dataset_id},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    return blueprint


def _build_view_model(dataset_id: str, params: Mapping[str, Any], error: Optional[str]):
    matrix = _fixture_matrix(dataset_id)
    result = None
    if error is None:
        try:
            result = run_ssdbcodi(
                matrix,
                labeling_state=_labeling_state_for_current_selection(dataset_id),
                **params,
            )
        except ValueError as exc:
            error = str(exc)

    selection_store = _selection_store(dataset_id)
    selection_state = get_selection_state(selection_store)
    context = get_selection_context(selection_store)
    labeling_state = get_labeling_state(get_labeling_store_for_context(context))

    return {
        "dataset_id": dataset_id,
        "dataset_options": ssdbcodi_dataset_options(),
        "feature_matrix": matrix,
        "result": result,
        "params": dict(params),
        "error": error,
        "plot_points": _plot_points(matrix, result, selection_state, labeling_state),
        "selection_state": selection_state,
        "selection_groups": list_selection_groups(selection_store),
        "context": context,
        "labeling_state": labeling_state,
        "allowed_labels": _allowed_labels(int(params["n_clusters"])),
        "label_color_lookup": _label_color_lookup(int(params["n_clusters"])),
        "cluster_counts": cluster_counts(result) if result is not None else {},
        "dependency_mode": DEPENDENCY_MODE,
    }


def _fixture_dataset(dataset_id: str | None = None) -> Dataset:
    dataset_id = ssdbcodi_dataset_id(dataset_id)
    return create_dataset(
        ssdbcodi_raw_points(dataset_id),
        dataset_id=dataset_id,
        feature_names=ssdbcodi_feature_names(),
    )


def _fixture_matrix(dataset_id: str | None = None) -> FeatureMatrix:
    return create_feature_matrix(_fixture_dataset(dataset_id))


def _selection_store(dataset_id: str | None = None):
    return get_selection_store_for_dataset(_fixture_dataset(dataset_id), initial_selected_point_ids=())


def _labeling_state_for_current_selection(dataset_id: str | None = None):
    context = get_selection_context(_selection_store(dataset_id))
    state = get_labeling_state(get_labeling_store_for_context(context))
    return state if state.annotations else None


def _store_result(result, dataset_id: str | None = None):
    store = get_debug_store(ssdbcodi_dataset_id(dataset_id))
    store.record_result(result)


def _selection_action_response(action_name: str):
    payload = request_payload(request)
    dataset_id, error = _dataset_id_from_payload(payload)
    if error is not None:
        return jsonify(api_error("invalid_dataset", error)), 400
    try:
        action = selection_action_from_payload(
            action_name,
            payload,
            metadata={"workflow": WORKFLOW_NAME},
        )
        result = apply_selection_action(_selection_store(dataset_id), action)
    except ValueError as exc:
        return jsonify(api_error("invalid_selection_action", str(exc))), 400

    return jsonify(
        api_success(
            result.to_dict(),
            diagnostics={"dependency_mode": DEPENDENCY_MODE},
        )
    )


def _selection_groups_payload(dataset_id: str | None = None):
    return [group.to_dict() for group in list_selection_groups(_selection_store(dataset_id))]


def _result_payload(result):
    return {
        "run_id": result.run_id,
        "cluster_counts": cluster_counts(result),
        "outlier_count": len(result.outlier_result.outlier_point_ids),
        "bootstrap_used": bool(result.parameters.get("bootstrap_used")),
        "point_scores": [score.to_dict() for score in result.point_scores],
        "seeds": [seed.to_dict() for seed in result.seeds],
        "parameters": dict(result.parameters),
        "cluster_result": result.cluster_result.to_dict(),
        "outlier_result": result.outlier_result.to_dict(),
    }


def _plot_points(
    matrix: FeatureMatrix,
    result,
    selection_state,
    labeling_state,
):
    if result is None:
        return []

    feature_lookup = {
        point_id: list(matrix.values[index])
        for index, point_id in enumerate(matrix.point_ids)
    }
    score_lookup = {score.point_id: score for score in result.point_scores}
    selected_ids = set(selection_state.selected_point_ids)
    cluster_ids = sorted({score.cluster_id for score in result.point_scores})
    color_lookup = {
        cluster_id: PALETTE[index % len(PALETTE)]
        for index, cluster_id in enumerate(cluster_ids)
    }
    screen_lookup = _screen_coordinates(feature_lookup)

    return [
        {
            "point_id": point_id,
            "features": feature_lookup[point_id],
            "screen_x": screen_lookup[point_id][0],
            "screen_y": screen_lookup[point_id][1],
            "cluster_id": score_lookup[point_id].cluster_id,
            "is_outlier": score_lookup[point_id].is_outlier,
            "is_selected": point_id in selected_ids,
            "color": color_lookup.get(score_lookup[point_id].cluster_id, "#888"),
            "scores": score_lookup[point_id].to_dict(),
            "manual_labels": _manual_labels_for_point(labeling_state, point_id),
        }
        for point_id in matrix.point_ids
    ]


def _screen_coordinates(feature_lookup):
    xs = [features[0] for features in feature_lookup.values()]
    ys = [features[1] for features in feature_lookup.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = (x_max - x_min) * 0.1 or 1.0
    pad_y = (y_max - y_min) * 0.1 or 1.0

    def scale_x(value):
        return 56 + ((value - x_min + pad_x) / (x_max - x_min + 2 * pad_x)) * 748

    def scale_y(value):
        return 464 - ((value - y_min + pad_y) / (y_max - y_min + 2 * pad_y)) * 408

    return {
        point_id: (scale_x(features[0]), scale_y(features[1]))
        for point_id, features in feature_lookup.items()
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
    if annotation.label_type == "outlier" and annotation.label_value is False:
        return "not_outlier"
    return annotation.label_value


def _validate_label_action(action: str, label_value, n_clusters: int) -> None:
    if action == "assign_cluster":
        allowed_clusters = set(_allowed_cluster_labels(n_clusters))
        if label_value not in allowed_clusters:
            allowed = ", ".join([*sorted(allowed_clusters), "outlier"])
            raise ValueError(f"label_value must be one of: {allowed}")
        return

    if action in {"mark_outlier", "mark_not_outlier"}:
        return

    raise ValueError("ssdbcodi labeling only supports cluster_N labels and outlier")


def _allowed_cluster_labels(n_clusters: int):
    return [f"cluster_{index}" for index in range(1, n_clusters + 1)]


def _allowed_labels(n_clusters: int):
    return [*_allowed_cluster_labels(n_clusters), "outlier"]


def _label_color_lookup(n_clusters: int):
    colors = {
        label: PALETTE[index % len(PALETTE)]
        for index, label in enumerate(_allowed_cluster_labels(n_clusters))
    }
    colors["outlier"] = "#d6336c"
    return colors


def _dataset_id_from_request() -> Tuple[str, Optional[str]]:
    try:
        return normalize_ssdbcodi_dataset_id(request.args.get("dataset_id")), None
    except ValueError as exc:
        return ssdbcodi_dataset_id(), str(exc)


def _dataset_id_from_payload(payload: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    try:
        return normalize_ssdbcodi_dataset_id(payload.get("dataset_id")), None
    except ValueError as exc:
        return ssdbcodi_dataset_id(), str(exc)


def _params_from_request(dataset_id: str) -> Tuple[Dict[str, Any], Optional[str]]:
    params: Dict[str, Any] = _default_params()
    for name in ("n_clusters", "min_pts"):
        raw = request.args.get(name)
        if raw is None:
            continue
        try:
            params[name] = int(raw)
        except ValueError:
            return params, f"{name} must be an integer"
    for name in ("alpha", "beta", "contamination"):
        raw = request.args.get(name)
        if raw is None:
            continue
        try:
            params[name] = float(raw)
        except ValueError:
            return params, f"{name} must be a number"
    return params, _validate_param_ranges(params, dataset_id)


def _params_from_payload(payload: Mapping[str, Any], dataset_id: str) -> Tuple[Dict[str, Any], Optional[str]]:
    params: Dict[str, Any] = _default_params()
    for name in ("n_clusters", "min_pts"):
        if name in payload and payload.get(name) is not None and payload.get(name) != "":
            try:
                params[name] = int(payload[name])
            except (TypeError, ValueError):
                return params, f"{name} must be an integer"
    for name in ("alpha", "beta", "contamination"):
        if name in payload and payload.get(name) is not None and payload.get(name) != "":
            try:
                params[name] = float(payload[name])
            except (TypeError, ValueError):
                return params, f"{name} must be a number"
    return params, _validate_param_ranges(params, dataset_id)


def _default_params() -> Dict[str, Any]:
    return {
        "n_clusters": DEFAULT_BOOTSTRAP_K,
        "min_pts": DEFAULT_MIN_PTS,
        "alpha": DEFAULT_ALPHA,
        "beta": DEFAULT_BETA,
        "contamination": DEFAULT_CONTAMINATION,
    }


def _validate_param_ranges(params: Mapping[str, Any], dataset_id: str) -> Optional[str]:
    point_count = len(ssdbcodi_raw_points(dataset_id))
    if params["n_clusters"] < 1:
        return "n_clusters must be at least 1"
    if params["n_clusters"] > point_count:
        return "n_clusters must not exceed the number of points"
    if params["min_pts"] < 1:
        return "min_pts must be at least 1"
    if params["min_pts"] >= point_count:
        return "min_pts must be less than the number of points"
    if params["alpha"] < 0 or params["beta"] < 0:
        return "alpha and beta must be non-negative"
    if params["alpha"] + params["beta"] > 1.0:
        return "alpha + beta must not exceed 1"
    if params["contamination"] <= 0 or params["contamination"] >= 0.5:
        return "contamination must be greater than 0 and less than 0.5"
    return None
