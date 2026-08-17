from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS
from app.shared.flask_helpers import api_error, api_success

from .fixtures import RULE_PANEL_DATASET_ID, rule_panel_fixture_state
def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "rule_panel",
        __name__,
        template_folder="templates",
        url_prefix="/modules/rule-panel",
    )

    @blueprint.get("/")
    def index():
        state, error = _state_from_request()
        return render_template(
            "rule_panel/index.html",
            dependency_mode="real SSDBCODI analysis fixture",
            error=error,
            **state,
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "rule-panel", "status": "working"},
                diagnostics={
                    "dependency_mode": "real SSDBCODI analysis fixture",
                    "decision_tree_role": "explanation_only",
                    "source_of_truth": "ssdbcodi",
                },
            )
        )

    @blueprint.get("/api/config")
    def config_api():
        state, error = _state_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400
        return jsonify(
            api_success(
                {
                    "module": "rule-panel",
                    "dataset_id": RULE_PANEL_DATASET_ID,
                    "tree_config": state["tree_config"].to_dict(),
                    "n_clusters": state["analysis"].cluster_result.n_clusters,
                },
                diagnostics={"decision_tree_role": "explanation_only"},
            )
        )

    @blueprint.get("/api/rules")
    def rules_api():
        state, error = _state_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400
        return jsonify(
            api_success(
                state["rule_set"].to_dict(),
                diagnostics={
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
