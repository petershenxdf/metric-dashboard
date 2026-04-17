from __future__ import annotations

from app.modules.data_workspace.fixtures import iris_feature_names, iris_raw_points
from app.modules.data_workspace.service import create_dataset, create_feature_matrix

from .service import project_feature_matrix


def fixture_projection():
    dataset = create_dataset(
        iris_raw_points(),
        dataset_id="iris_debug_sample",
        feature_names=iris_feature_names(),
    )
    matrix = create_feature_matrix(dataset)
    projection = project_feature_matrix(matrix)
    return dataset, projection


def labels_by_point_id(dataset):
    return {
        point.point_id: str(point.metadata.get("label", ""))
        for point in dataset.points
    }
