from __future__ import annotations

from flask import Blueprint, render_template

from app.modules.data_workspace.fixtures import iris_feature_names, iris_raw_points
from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.modules.projection.service import project_feature_matrix, scaled_projection_points


def create_blueprint() -> Blueprint:
    blueprint = Blueprint(
        "data_projection_workflow",
        __name__,
        template_folder="templates",
        url_prefix="/workflows/data-projection",
    )

    @blueprint.get("/")
    def index():
        dataset = create_dataset(
            iris_raw_points(),
            dataset_id="iris_debug_sample",
            feature_names=iris_feature_names(),
        )
        matrix = create_feature_matrix(dataset)
        projection = project_feature_matrix(matrix)
        labels = {
            point.point_id: str(point.metadata.get("label", ""))
            for point in dataset.points
        }

        return render_template(
            "workflows/data_projection.html",
            dataset=dataset,
            feature_matrix=matrix,
            projection=projection,
            scaled_points=scaled_projection_points(projection, labels),
        )

    return blueprint
