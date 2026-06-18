from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS
from app.modules.rule_panel.fixtures import rule_panel_fixture_state
from app.modules.rule_panel.interpretation import (
    DEFAULT_INTERPRETER_KIND,
    INTERPRETER_KIND_OPTIONS,
    RULE_INTERPRETATION_CATEGORIES,
    create_rule_interpreter,
    rule_interpretation_category_descriptions,
)
from app.shared.flask_helpers import api_error, api_success

DEPENDENCY_MODE = "wine.mat RuleSet plus Step 8.7 rule interpreter"
DEFAULT_TREE_DEPTH = 3
DEFAULT_MIN_SAMPLES_LEAF = 1


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "rule_interpretation_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/rule-interpretation",
    )

    @blueprint.get("/")
    def index():
        view_model, error = _view_model_from_request()
        return render_template(
            "workflows/rule_interpretation.html",
            dependency_mode=DEPENDENCY_MODE,
            interpreter_options=INTERPRETER_KIND_OPTIONS,
            category_options=RULE_INTERPRETATION_CATEGORIES,
            category_descriptions=rule_interpretation_category_descriptions(),
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
                    **dict(view_model["interpretation_run"].diagnostics),
                },
            )
        )

    return blueprint


def _view_model_from_request():
    params, error = _params_from_request()
    if error is not None:
        return _build_view_model(_default_params()), error
    try:
        return _build_view_model(params), None
    except ValueError as exc:
        return _build_view_model(_default_params()), str(exc)


def _build_view_model(params):
    state = rule_panel_fixture_state(
        n_clusters=params["n_clusters"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
    )
    interpreter = create_rule_interpreter(params["provider_kind"])
    interpretation_run = interpreter.interpret(
        state["rule_set"],
        analysis_result=state["analysis"],
        feature_matrix=state["feature_matrix"],
        focus_category=params["focus_category"],
    )
    return {
        **state,
        "interpretation": interpretation_run.interpretation,
        "interpretation_run": interpretation_run,
        "provider_kind": params["provider_kind"],
        "focus_category": params["focus_category"],
        "n_clusters": params["n_clusters"],
    }


def _state_payload(view_model):
    return {
        "workflow": "rule-interpretation",
        "dataset": view_model["dataset"].to_dict(),
        "feature_matrix": view_model["feature_matrix"].to_dict(),
        "analysis": view_model["analysis"].to_dict(),
        "cluster_counts": view_model["cluster_counts"],
        "tree_config": view_model["tree_config"].to_dict(),
        "rule_set": view_model["rule_set"].to_dict(),
        "interpretation": view_model["interpretation"].to_dict(),
        "interpretation_request": view_model["interpretation_run"].request_payload,
        "interpretation_diagnostics": view_model["interpretation_run"].diagnostics,
        "focus_category": view_model["focus_category"],
    }


def _params_from_request():
    n_clusters, error = _int_query("n_clusters", DEFAULT_N_CLUSTERS, minimum=1)
    if error is not None:
        return _default_params(), error

    max_depth, error = _int_query("max_depth", DEFAULT_TREE_DEPTH, minimum=1)
    if error is not None:
        return _default_params(), error

    min_samples_leaf, error = _int_query("min_samples_leaf", DEFAULT_MIN_SAMPLES_LEAF, minimum=1)
    if error is not None:
        return _default_params(), error

    provider_kind = request.args.get("provider_kind", DEFAULT_INTERPRETER_KIND).strip().lower()
    if provider_kind not in INTERPRETER_KIND_OPTIONS:
        return _default_params(), f"provider_kind must be one of: {', '.join(INTERPRETER_KIND_OPTIONS)}"

    focus_category = request.args.get("focus_category", "").strip()
    if focus_category and focus_category not in RULE_INTERPRETATION_CATEGORIES:
        return _default_params(), f"focus_category must be one of: {', '.join(RULE_INTERPRETATION_CATEGORIES)}"

    return {
        "n_clusters": n_clusters,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "provider_kind": provider_kind,
        "focus_category": focus_category or None,
    }, None


def _default_params():
    return {
        "n_clusters": DEFAULT_N_CLUSTERS,
        "max_depth": DEFAULT_TREE_DEPTH,
        "min_samples_leaf": DEFAULT_MIN_SAMPLES_LEAF,
        "provider_kind": DEFAULT_INTERPRETER_KIND,
        "focus_category": None,
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
