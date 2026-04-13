from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.shared.flask_helpers import api_error, api_success

from .fixtures import default_analysis_feature_names, default_analysis_raw_points
from .service import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_OUTLIER_CONTAMINATION,
    DEFAULT_OUTLIER_N_NEIGHBORS,
    cluster_counts,
    run_default_analysis,
)


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "algorithm_adapters",
        __name__,
        template_folder="templates",
        url_prefix="/modules/algorithm-adapters",
    )

    @blueprint.get("/")
    def index():
        matrix = _fixture_matrix()
        params, error = _analysis_params_from_request()
        analysis = None
        counts = {}

        if error is None:
            try:
                analysis = run_default_analysis(matrix, **params)
                counts = cluster_counts(analysis.cluster_result)
            except ValueError as exc:
                error = str(exc)

        return render_template(
            "algorithm_adapters/index.html",
            feature_matrix=matrix,
            analysis=analysis,
            cluster_counts=counts,
            params=params,
            error=error,
            dependency_mode="real data-workspace fixture",
        )

    @blueprint.get("/health")
    def health():
        return jsonify(
            api_success(
                {"module": "algorithm-adapters", "status": "working"},
                diagnostics={
                    "dependency_mode": "real data-workspace fixture",
                    "provider": "sequential_lof_then_kmeans",
                },
            )
        )

    @blueprint.get("/api/outliers")
    def outliers_api():
        matrix = _fixture_matrix()
        params, error = _analysis_params_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400

        try:
            analysis = run_default_analysis(matrix, **params)
        except ValueError as exc:
            return jsonify(api_error("invalid_parameters", str(exc))), 400

        return jsonify(
            api_success(
                analysis.outlier_result.to_dict(),
                diagnostics={
                    "dependency_mode": "real data-workspace fixture",
                    "execution_order": "outlier_detection_first",
                },
            )
        )

    @blueprint.get("/api/clusters")
    def clusters_api():
        matrix = _fixture_matrix()
        params, error = _analysis_params_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400

        try:
            analysis = run_default_analysis(matrix, **params)
        except ValueError as exc:
            return jsonify(api_error("invalid_parameters", str(exc))), 400

        return jsonify(
            api_success(
                analysis.cluster_result.to_dict(),
                diagnostics={
                    "dependency_mode": "real data-workspace fixture",
                    "execution_order": "after_outlier_detection",
                },
            )
        )

    @blueprint.get("/api/analysis")
    def analysis_api():
        matrix = _fixture_matrix()
        params, error = _analysis_params_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400

        try:
            analysis = run_default_analysis(matrix, **params)
        except ValueError as exc:
            return jsonify(api_error("invalid_parameters", str(exc))), 400

        return jsonify(
            api_success(
                analysis.to_dict(),
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    @blueprint.get("/api/state")
    def state_api():
        matrix = _fixture_matrix()
        params, error = _analysis_params_from_request()
        if error is not None:
            return jsonify(api_error("invalid_parameters", error)), 400

        try:
            analysis = run_default_analysis(matrix, **params)
        except ValueError as exc:
            return jsonify(api_error("invalid_parameters", str(exc))), 400

        return jsonify(
            api_success(
                {
                    "module": "algorithm-adapters",
                    "status": "working",
                    "point_count": len(matrix.point_ids),
                    "outlier_count": len(analysis.outlier_result.outlier_point_ids),
                    "clustered_point_count": len(analysis.cluster_result.assignments),
                    "n_clusters": analysis.cluster_result.n_clusters,
                    "cluster_counts": cluster_counts(analysis.cluster_result),
                    "provider": analysis.diagnostics["provider"],
                },
                diagnostics={"dependency_mode": "real data-workspace fixture"},
            )
        )

    return blueprint


def _fixture_matrix():
    dataset = create_dataset(
        default_analysis_raw_points(),
        dataset_id="default_analysis_outlier_debug",
        feature_names=default_analysis_feature_names(),
    )
    return create_feature_matrix(dataset)


def _analysis_params_from_request():
    n_clusters, error = _int_query("n_clusters", DEFAULT_N_CLUSTERS, minimum=1)
    if error is not None:
        return _default_params(), error

    outlier_n_neighbors, error = _int_query(
        "outlier_n_neighbors",
        DEFAULT_OUTLIER_N_NEIGHBORS,
        minimum=1,
    )
    if error is not None:
        return _default_params(), error

    outlier_contamination, error = _float_query(
        "outlier_contamination",
        DEFAULT_OUTLIER_CONTAMINATION,
        minimum=0.01,
        maximum=0.49,
    )
    if error is not None:
        return _default_params(), error

    return {
        "n_clusters": n_clusters,
        "outlier_n_neighbors": outlier_n_neighbors,
        "outlier_contamination": outlier_contamination,
    }, None


def _default_params():
    return {
        "n_clusters": DEFAULT_N_CLUSTERS,
        "outlier_n_neighbors": DEFAULT_OUTLIER_N_NEIGHBORS,
        "outlier_contamination": DEFAULT_OUTLIER_CONTAMINATION,
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


def _float_query(name: str, default: float, minimum: float, maximum: float):
    raw_value = request.args.get(name)
    if raw_value is None:
        return default, None

    try:
        value = float(raw_value)
    except ValueError:
        return default, f"{name} must be a number"

    if value < minimum or value > maximum:
        return default, f"{name} must be between {minimum} and {maximum}"

    return value, None
