from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS
from app.modules.rule_panel.fixtures import rule_panel_fixture_state
from app.shared.flask_helpers import api_error, api_success

DEPENDENCY_MODE = "SSDBCODI analysis fixture plus rule_panel decision-tree surrogate"


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "rule_panel_validation_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/rule-panel-validation",
    )

    @blueprint.get("/")
    def index():
        state, error = _state_from_request()
        return render_template(
            "workflows/rule_panel_validation.html",
            dependency_mode=DEPENDENCY_MODE,
            error=error,
            **state,
        )

    @blueprint.get("/api/state")
    def state_api():
        state, error = _state_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400
        return jsonify(
            api_success(
                {
                    "workflow": "rule-panel-validation",
                    "feature_matrix": state["feature_matrix"].to_dict(),
                    "analysis": state["analysis"].to_dict(),
                    "cluster_counts": state["cluster_counts"],
                    "rule_set": state["rule_set"].to_dict(),
                    "interpretation_preview": state["interpretation"].to_dict(),
                },
                diagnostics={
                    "dependency_mode": DEPENDENCY_MODE,
                    "source_of_truth": "ssdbcodi",
                    "decision_tree_role": "explanation_only",
                },
            )
        )

    return blueprint


def _state_from_request():
    params, error = _params_from_request()
    if error is not None:
        return _fallback_state(), error
    try:
        return rule_panel_fixture_state(**params), None
    except ValueError as exc:
        return _fallback_state(), str(exc)


def _fallback_state():
    return rule_panel_fixture_state(
        n_clusters=DEFAULT_N_CLUSTERS,
        max_depth=3,
        min_samples_leaf=1,
    )


def _params_from_request():
    n_clusters, error = _int_query("n_clusters", DEFAULT_N_CLUSTERS, minimum=1)
    if error is not None:
        return _default_params(), error

    max_depth, error = _int_query("max_depth", 3, minimum=1)
    if error is not None:
        return _default_params(), error

    min_samples_leaf, error = _int_query("min_samples_leaf", 1, minimum=1)
    if error is not None:
        return _default_params(), error

    return {
        "n_clusters": n_clusters,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }, None


def _default_params():
    return {
        "n_clusters": DEFAULT_N_CLUSTERS,
        "max_depth": 3,
        "min_samples_leaf": 1,
    }


def _int_query(name: str, default: int, minimum: int):
    raw_value = request.args.get(name)
    if raw_value is None:
        return default, None
    try:
        value = int(raw_value)
    except ValueError:
        return default, f"{name} must be an integer"
    if value < minimum:
        return default, f"{name} must be at least {minimum}"
    return value, None
