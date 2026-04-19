from __future__ import annotations

from flask import Blueprint, jsonify, render_template

from app.modules.algorithm_adapters.fixtures import (
    default_analysis_feature_names,
    default_analysis_raw_points,
)
from app.modules.algorithm_adapters.service import (
    DEFAULT_N_CLUSTERS,
    cluster_counts as analysis_cluster_counts,
    run_default_analysis,
)
from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.ssdbcodi.fixtures import (
    ssdbcodi_dataset_id,
    ssdbcodi_feature_names,
    ssdbcodi_raw_points,
)
from app.modules.ssdbcodi.service import (
    cluster_counts as ssdbcodi_cluster_counts,
    run_ssdbcodi,
)
from app.shared.flask_helpers import api_success
from app.shared.request_helpers import n_clusters_from_request

DEPENDENCY_MODE = "algorithm_adapters default provider plus standalone ssdbcodi preview"


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "provider_feedback_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/provider-feedback",
    )

    @blueprint.get("/")
    def index():
        state = _workflow_state(_n_clusters_from_request())
        return render_template(
            "workflows/provider_feedback.html",
            dependency_mode=DEPENDENCY_MODE,
            **state,
        )

    @blueprint.get("/api/state")
    def state_api():
        state = _workflow_state(_n_clusters_from_request())
        return jsonify(
            api_success(
                _state_payload(state),
                diagnostics={"dependency_mode": DEPENDENCY_MODE},
            )
        )

    return blueprint


def _workflow_state(n_clusters: int):
    adapter_matrix = _adapter_fixture_matrix()
    ssdbcodi_matrix = _ssdbcodi_fixture_matrix()
    error = None

    try:
        adapter_analysis = run_default_analysis(adapter_matrix, n_clusters=n_clusters)
        ssdbcodi_result = run_ssdbcodi(ssdbcodi_matrix, n_clusters=n_clusters)
    except ValueError as exc:
        error = str(exc)
        n_clusters = DEFAULT_N_CLUSTERS
        adapter_analysis = run_default_analysis(adapter_matrix, n_clusters=n_clusters)
        ssdbcodi_result = run_ssdbcodi(ssdbcodi_matrix, n_clusters=n_clusters)

    return {
        "n_clusters": n_clusters,
        "error": error,
        "adapter_matrix": adapter_matrix,
        "adapter_analysis": adapter_analysis,
        "adapter_cluster_counts": analysis_cluster_counts(adapter_analysis.cluster_result),
        "ssdbcodi_matrix": ssdbcodi_matrix,
        "ssdbcodi_result": ssdbcodi_result,
        "ssdbcodi_cluster_counts": ssdbcodi_cluster_counts(ssdbcodi_result),
    }


def _state_payload(state):
    return {
        "module_boundary": "algorithm_adapters",
        "active_provider": state["adapter_analysis"].diagnostics["provider"],
        "adapter_feature_matrix": state["adapter_matrix"].to_dict(),
        "adapter_analysis": state["adapter_analysis"].to_dict(),
        "adapter_cluster_counts": state["adapter_cluster_counts"],
        "standalone_provider": "ssdbcodi",
        "ssdbcodi_feature_matrix": state["ssdbcodi_matrix"].to_dict(),
        "ssdbcodi_result": state["ssdbcodi_result"].to_dict(),
        "ssdbcodi_cluster_counts": state["ssdbcodi_cluster_counts"],
    }


def _adapter_fixture_matrix():
    dataset = create_dataset(
        default_analysis_raw_points(),
        dataset_id="default_analysis_outlier_debug",
        feature_names=default_analysis_feature_names(),
    )
    return create_feature_matrix(dataset)


def _ssdbcodi_fixture_matrix():
    dataset_id = ssdbcodi_dataset_id()
    dataset = create_dataset(
        ssdbcodi_raw_points(dataset_id),
        dataset_id=dataset_id,
        feature_names=ssdbcodi_feature_names(),
    )
    return create_feature_matrix(dataset)


def _n_clusters_from_request() -> int:
    return n_clusters_from_request()
