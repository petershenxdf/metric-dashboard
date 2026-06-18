from __future__ import annotations

from functools import lru_cache

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, cluster_counts, run_default_analysis
from app.modules.labeling.service import apply_labeling_action, clear_annotations, get_labeling_state
from app.modules.labeling.state import (
    get_debug_store_for_context as get_labeling_store_for_context,
    reset_debug_store_for_context as reset_labeling_store_for_context,
)
from app.modules.projection.service import project_feature_matrix
from app.modules.rule_panel.schemas import TreeConfig
from app.modules.rule_panel.service import generate_rule_set
from app.modules.rule_panel.interpretation import (
    INTERPRETER_KIND_OPTIONS,
    RULE_INTERPRETATION_CATEGORIES,
    RuleInterpretationRun,
    create_rule_interpreter,
    rule_interpretation_category_status,
)
from app.modules.scatterplot.service import build_render_payload
from app.modules.selection.http_helpers import optional_point_ids_from_payload
from app.modules.selection.service import (
    delete_selection_group,
    get_selection_context,
    get_selection_state,
    list_selection_groups,
    save_selection_group,
    select_selection_group,
)
from app.modules.selection.state import get_debug_store_for_dataset, reset_debug_store_for_dataset
from app.shared.effective_analysis import apply_manual_labels_to_analysis
from app.shared.env import env_text
from app.shared.flask_helpers import api_error, api_success
from app.shared.request_helpers import (
    apply_selection_action_or_error,
    n_clusters_from_request,
    request_payload,
    selection_groups_payload,
)
from app.shared.wine_dataset import WINE_DATASET_ID, load_wine_dataset, load_wine_feature_matrix

WORKFLOW_NAME = "wine-dashboard"
DEPENDENCY_MODE = "wine.mat integrated Step 8.8 dashboard without chatbox"
DEFAULT_TREE_DEPTH = 3
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_FOCUS_CATEGORY = "label_priority"
INITIAL_SELECTED_POINT_IDS = ("wine_001", "wine_002", "wine_003")
PLOT_WIDTH = 860
PLOT_HEIGHT = 520
DATA_PREVIEW_LIMIT = 24
DEFAULT_RULE_INTERPRETER_KIND = "deepseek"
_INTERPRETATION_RUN_CACHE_MAX = 16
_interpretation_run_cache = {}


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "wine_dashboard_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/wine-dashboard",
    )

    @blueprint.get("/")
    def index():
        view_model, error = _view_model_from_request()
        return render_template(
            "workflows/wine_dashboard.html",
            dependency_mode=DEPENDENCY_MODE,
            interpreter_options=INTERPRETER_KIND_OPTIONS,
            category_options=RULE_INTERPRETATION_CATEGORIES,
            error=error,
            **view_model,
        )

    @blueprint.get("/api/state")
    def state_api():
        view_model, error = _view_model_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400
        return jsonify(
            api_success(
                _state_payload(view_model),
                diagnostics={
                    "dependency_mode": DEPENDENCY_MODE,
                    "dataset_id": WINE_DATASET_ID,
                    "chatbox": "excluded",
                    "source_of_truth": "ssdbcodi",
                    "decision_tree_role": "explanation_only",
                },
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
        dataset = load_wine_dataset()
        store = reset_debug_store_for_dataset(dataset, INITIAL_SELECTED_POINT_IDS)
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
            view_model = _build_view_model(
                n_clusters,
                _tree_config_from_request(),
                _provider_kind_from_request(),
                _focus_category_from_request(),
            )
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
        view_model = _build_view_model(
            _n_clusters_from_request(),
            _tree_config_from_request(),
            _provider_kind_from_request(),
            _focus_category_from_request(),
        )
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
        view_model = _build_view_model(
            _n_clusters_from_request(),
            _tree_config_from_request(),
            _provider_kind_from_request(),
            _focus_category_from_request(),
        )
        return jsonify(
            api_success(
                {"state": _state_payload(view_model)},
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    return blueprint


def _view_model_from_request():
    try:
        return _build_view_model(
            _n_clusters_from_request(),
            _tree_config_from_request(),
            _provider_kind_from_request(),
            _focus_category_from_request(),
        ), None
    except ValueError as exc:
        fallback = _build_view_model(
            DEFAULT_N_CLUSTERS,
            TreeConfig(),
            "mock",
            None,
        )
        return fallback, str(exc)


def _build_view_model(
    n_clusters: int,
    tree_config: TreeConfig,
    provider_kind: str,
    focus_category: str | None,
):
    dataset, matrix, projection, raw_analysis = _base_wine_state(n_clusters)
    selection_store = _workflow_store_for_dataset(dataset)
    selection_state = get_selection_state(selection_store)
    context = get_selection_context(selection_store)
    labeling_state = get_labeling_state(get_labeling_store_for_context(context))
    provider_labeling_state = labeling_state if labeling_state.annotations else None

    if provider_labeling_state is None:
        provider_analysis = raw_analysis
        analysis = raw_analysis
        rule_set = _base_rule_set(
            n_clusters,
            tree_config.max_depth,
            tree_config.min_samples_leaf,
        )
    else:
        provider_analysis = run_default_analysis(
            matrix,
            n_clusters=n_clusters,
            labeling_state=provider_labeling_state,
        )
        analysis = apply_manual_labels_to_analysis(dataset, provider_analysis, labeling_state)
        rule_set = generate_rule_set(
            matrix,
            analysis,
            dataset_id=dataset.dataset_id,
            config=tree_config,
        )

    category_status = rule_interpretation_category_status(
        rule_set,
        analysis_result=analysis,
        feature_matrix=matrix,
    )
    interpretation_run = _interpretation_run(
        rule_set,
        analysis,
        matrix,
        provider_kind=provider_kind,
        focus_category=focus_category,
        category_status=category_status,
    )
    interpretation = interpretation_run.interpretation
    guidance_point_ids = _guidance_point_ids(interpretation)

    scatter_payload = build_render_payload(
        dataset,
        projection,
        analysis.cluster_result,
        analysis.outlier_result,
        selection_context=context,
        labeling_state=labeling_state,
    )
    plot_points = _plot_points(dataset, scatter_payload)

    return {
        "dataset": dataset,
        "matrix": matrix,
        "projection": projection,
        "raw_analysis": raw_analysis,
        "provider_analysis": provider_analysis,
        "analysis": analysis,
        "cluster_counts": cluster_counts(analysis.cluster_result),
        "scatter_payload": scatter_payload,
        "plot_points": plot_points,
        "data_preview_points": plot_points[:DATA_PREVIEW_LIMIT],
        "hidden_preview_count": max(0, len(plot_points) - DATA_PREVIEW_LIMIT),
        "selection_state": selection_state,
        "context": context,
        "selection_groups": list_selection_groups(selection_store),
        "labeling_state": labeling_state,
        "allowed_labels": _allowed_labels(n_clusters),
        "rule_set": rule_set,
        "interpretation": interpretation,
        "interpretation_run": interpretation_run,
        "guidance_point_ids": guidance_point_ids,
        "category_status": category_status,
        "provider_kind": provider_kind,
        "focus_category": focus_category,
        "n_clusters": n_clusters,
        "tree_config": tree_config,
        "plot_width": PLOT_WIDTH,
        "plot_height": PLOT_HEIGHT,
        "show_debug": _debug_enabled_from_request(),
    }


@lru_cache(maxsize=8)
def _base_wine_state(n_clusters: int):
    dataset = load_wine_dataset()
    matrix = load_wine_feature_matrix()
    projection = project_feature_matrix(matrix)
    raw_analysis = run_default_analysis(matrix, n_clusters=n_clusters)
    return dataset, matrix, projection, raw_analysis


@lru_cache(maxsize=32)
def _base_rule_set(n_clusters: int, max_depth: int, min_samples_leaf: int):
    dataset, matrix, _, raw_analysis = _base_wine_state(n_clusters)
    config = TreeConfig(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    rule_set = generate_rule_set(
        matrix,
        raw_analysis,
        dataset_id=dataset.dataset_id,
        config=config,
    )
    return rule_set


def _interpretation_run(
    rule_set,
    analysis,
    matrix,
    *,
    provider_kind: str,
    focus_category: str | None,
    category_status=(),
):
    if (
        provider_kind == "deepseek"
        and focus_category is not None
        and _category_is_missing_typical_case(category_status, focus_category)
    ):
        deterministic_run = create_rule_interpreter("mock").interpret(
            rule_set,
            analysis_result=analysis,
            feature_matrix=matrix,
            focus_category=focus_category,
        )
        diagnostics = {
            **dict(deterministic_run.diagnostics),
            "provider_kind": "deepseek",
            "provider_label": "deepseek_skipped_no_typical_case->mock_rule_interpreter",
            "requested_provider_kind": "deepseek",
            "used_fallback": False,
            "deepseek_skipped": True,
            "skip_reason": "no_typical_case_for_category",
            "validation": "local_no_case_guidance",
            "focus_category": focus_category,
        }
        interpretation = deterministic_run.interpretation
        interpretation = type(interpretation)(
            interpretation_id=interpretation.interpretation_id,
            rule_set_id=interpretation.rule_set_id,
            categories=interpretation.categories,
            target_rule_ids=interpretation.target_rule_ids,
            summary=interpretation.summary,
            evidence=interpretation.evidence,
            recommendation=interpretation.recommendation,
            category_explanation=interpretation.category_explanation,
            decision_rationale=interpretation.decision_rationale,
            label_targets=interpretation.label_targets,
            suspicion_reasons=interpretation.suspicion_reasons,
            point_label_guidance=interpretation.point_label_guidance,
            label_outcomes=interpretation.label_outcomes,
            quantitative_findings=interpretation.quantitative_findings,
            suggested_label_actions=interpretation.suggested_label_actions,
            confidence=interpretation.confidence,
            warnings=interpretation.warnings,
            provider_label=diagnostics["provider_label"],
        )
        return RuleInterpretationRun(
            interpretation=interpretation,
            request_payload=deterministic_run.request_payload,
            diagnostics=diagnostics,
        )

    cache_key = _interpretation_cache_key(rule_set, analysis, provider_kind, focus_category)
    cached_run = _cached_interpretation_run(cache_key)
    if cached_run is not None:
        return cached_run

    try:
        run = create_rule_interpreter(provider_kind).interpret(
            rule_set,
            analysis_result=analysis,
            feature_matrix=matrix,
            focus_category=focus_category,
        )
        _remember_interpretation_run(cache_key, run)
        return run
    except Exception as exc:
        fallback_run = create_rule_interpreter("mock").interpret(
            rule_set,
            analysis_result=analysis,
            feature_matrix=matrix,
            focus_category=focus_category,
        )
        diagnostics = {
            **dict(fallback_run.diagnostics),
            "provider_kind": provider_kind,
            "provider_label": f"{provider_kind}->mock_rule_interpreter",
            "requested_provider_kind": provider_kind,
            "used_fallback": True,
            "error": str(exc),
            "validation": "fallback_preserved_dashboard",
            "focus_category": focus_category,
        }
        interpretation = fallback_run.interpretation
        return RuleInterpretationRun(
            interpretation=interpretation,
            request_payload=fallback_run.request_payload,
            diagnostics=diagnostics,
        )


def _interpretation_cache_key(rule_set, analysis, provider_kind: str, focus_category: str | None):
    if provider_kind != "deepseek":
        return None
    return (
        provider_kind,
        focus_category or "",
        rule_set.rule_set_id,
        analysis.analysis_run_id,
    )


def _cached_interpretation_run(cache_key):
    if cache_key is None or cache_key not in _interpretation_run_cache:
        return None
    cached = _interpretation_run_cache[cache_key]
    diagnostics = {
        **dict(cached.diagnostics),
        "cache_hit": True,
    }
    return RuleInterpretationRun(
        interpretation=cached.interpretation,
        request_payload=cached.request_payload,
        diagnostics=diagnostics,
    )


def _remember_interpretation_run(cache_key, run: RuleInterpretationRun) -> None:
    if cache_key is None:
        return
    diagnostics = dict(run.diagnostics)
    if diagnostics.get("used_fallback") or diagnostics.get("deepseek_skipped"):
        return
    _interpretation_run_cache[cache_key] = RuleInterpretationRun(
        interpretation=run.interpretation,
        request_payload=run.request_payload,
        diagnostics={**diagnostics, "cache_hit": False},
    )
    while len(_interpretation_run_cache) > _INTERPRETATION_RUN_CACHE_MAX:
        oldest_key = next(iter(_interpretation_run_cache))
        del _interpretation_run_cache[oldest_key]


def _category_is_missing_typical_case(category_status, focus_category: str) -> bool:
    for item in category_status:
        if item.get("category") == focus_category:
            return not bool(item.get("has_typical_case"))
    return False


def _workflow_store():
    return _workflow_store_for_dataset(load_wine_dataset())


def _workflow_store_for_dataset(dataset):
    return get_debug_store_for_dataset(dataset, INITIAL_SELECTED_POINT_IDS)


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
        "workflow": WORKFLOW_NAME,
        "dataset": view_model["dataset"].to_dict(),
        "feature_matrix": view_model["matrix"].to_dict(),
        "projection": view_model["projection"].to_dict(),
        "scatterplot": view_model["scatter_payload"].to_dict(),
        "outliers": view_model["analysis"].outlier_result.to_dict(),
        "clusters": view_model["analysis"].cluster_result.to_dict(),
        "cluster_counts": view_model["cluster_counts"],
        "raw_outliers": view_model["raw_analysis"].outlier_result.to_dict(),
        "raw_clusters": view_model["raw_analysis"].cluster_result.to_dict(),
        "provider_outliers": view_model["provider_analysis"].outlier_result.to_dict(),
        "provider_clusters": view_model["provider_analysis"].cluster_result.to_dict(),
        "selection": view_model["selection_state"].to_dict(),
        "selection_context": view_model["context"].to_dict(),
        "selection_groups": [group.to_dict() for group in view_model["selection_groups"]],
        "labeling": view_model["labeling_state"].to_dict(),
        "rule_set": view_model["rule_set"].to_dict(),
        "interpretation_preview": view_model["interpretation"].to_dict(),
        "interpretation_request": view_model["interpretation_run"].request_payload,
        "interpretation_diagnostics": view_model["interpretation_run"].diagnostics,
        "guidance_point_ids": list(view_model["guidance_point_ids"]),
        "category_status": [dict(item) for item in view_model["category_status"]],
        "provider_kind": view_model["provider_kind"],
        "focus_category": view_model["focus_category"],
        "chatbox": {"included": False},
    }


def _plot_points(dataset, scatter_payload):
    source_points = {point.point_id: point for point in dataset.points}
    points = []
    for point in scatter_payload.points:
        source = source_points[point.point_id]
        points.append(
            {
                **point.to_dict(),
                "features": source.features,
                "class_label": source.metadata.get("class_label", ""),
            }
        )
    return points


def _guidance_point_ids(interpretation):
    point_ids = []
    for collection_name in ("label_targets", "suspicion_reasons", "point_label_guidance"):
        for item in getattr(interpretation, collection_name, ()):
            for point_id in item.get("point_ids", ()):
                if point_id not in point_ids:
                    point_ids.append(point_id)
    return tuple(point_ids)


def _validate_workflow_label(action: str, label_value, n_clusters: int) -> None:
    if action == "assign_cluster":
        allowed_clusters = set(_allowed_cluster_labels(n_clusters))
        if label_value not in allowed_clusters:
            allowed = ", ".join([*sorted(allowed_clusters), "outlier"])
            raise ValueError(f"label_value must be one of: {allowed}")
        return

    if action == "mark_outlier":
        return

    raise ValueError("wine-dashboard only supports cluster_N labels and outlier")


def _allowed_cluster_labels(n_clusters: int):
    return [f"cluster_{index}" for index in range(1, n_clusters + 1)]


def _allowed_labels(n_clusters: int):
    return [*_allowed_cluster_labels(n_clusters), "outlier"]


def _n_clusters_from_request() -> int:
    return n_clusters_from_request()


def _tree_config_from_request() -> TreeConfig:
    return TreeConfig(
        max_depth=_positive_int_from_request("max_depth", DEFAULT_TREE_DEPTH),
        min_samples_leaf=_positive_int_from_request("min_samples_leaf", DEFAULT_MIN_SAMPLES_LEAF),
    )


def _provider_kind_from_request() -> str:
    payload = request.get_json(silent=True) if request.is_json else {}
    raw_value = (
        (payload or {}).get("provider_kind")
        or request.args.get("provider_kind")
        or request.form.get("provider_kind")
        or _default_provider_kind()
    )
    provider_kind = str(raw_value).strip().lower()
    if provider_kind not in INTERPRETER_KIND_OPTIONS:
        raise ValueError(f"provider_kind must be one of: {', '.join(INTERPRETER_KIND_OPTIONS)}")
    return provider_kind


def _focus_category_from_request() -> str | None:
    payload = request.get_json(silent=True) if request.is_json else {}
    raw_value = (
        (payload or {}).get("focus_category")
        or request.args.get("focus_category")
        or request.form.get("focus_category")
        or DEFAULT_FOCUS_CATEGORY
    )
    focus_category = str(raw_value).strip()
    if not focus_category:
        return DEFAULT_FOCUS_CATEGORY
    if focus_category not in RULE_INTERPRETATION_CATEGORIES:
        raise ValueError(f"focus_category must be one of: {', '.join(RULE_INTERPRETATION_CATEGORIES)}")
    return focus_category


def _debug_enabled_from_request() -> bool:
    payload = request.get_json(silent=True) if request.is_json else {}
    raw_value = (
        (payload or {}).get("debug")
        or request.args.get("debug")
        or request.form.get("debug")
        or ""
    )
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _default_provider_kind() -> str:
    provider_kind = env_text("METRIC_DASHBOARD_LLM_PROVIDER", DEFAULT_RULE_INTERPRETER_KIND).strip().lower()
    if provider_kind in INTERPRETER_KIND_OPTIONS:
        return provider_kind
    return DEFAULT_RULE_INTERPRETER_KIND


def _positive_int_from_request(name: str, default: int) -> int:
    payload = request.get_json(silent=True) if request.is_json else {}
    raw_value = (
        (payload or {}).get(name)
        or request.args.get(name)
        or request.form.get(name)
        or default
    )
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value
