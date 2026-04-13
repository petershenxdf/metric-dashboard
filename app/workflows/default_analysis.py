from __future__ import annotations

from flask import Blueprint, render_template, request

from app.modules.algorithm_adapters.fixtures import (
    default_analysis_feature_names,
    default_analysis_raw_points,
)
from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, run_default_analysis
from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.projection.service import project_feature_matrix, scaled_projection_points


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "default_analysis_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/default-analysis",
    )

    @blueprint.get("/")
    def index():
        n_clusters = _n_clusters_from_request()
        dataset = create_dataset(
            default_analysis_raw_points(),
            dataset_id="default_analysis_outlier_debug",
            feature_names=default_analysis_feature_names(),
        )
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
        scaled_points = scaled_projection_points(projection, cluster_labels)
        outlier_ids = set(analysis.outlier_result.outlier_point_ids)

        return render_template(
            "workflows/default_analysis.html",
            dataset=dataset,
            projection=projection,
            analysis=analysis,
            scaled_points=scaled_points,
            outlier_ids=outlier_ids,
            n_clusters=n_clusters,
            error=error,
        )

    return blueprint


def _n_clusters_from_request() -> int:
    raw_value = request.args.get("n_clusters")
    if raw_value is None:
        return DEFAULT_N_CLUSTERS

    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_N_CLUSTERS

    return max(value, 1)
