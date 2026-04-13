from __future__ import annotations

from typing import List

from app.modules.data_workspace.fixtures import iris_feature_names, iris_raw_points
from app.modules.data_workspace.service import create_dataset


def selection_fixture_dataset():
    return create_dataset(
        iris_raw_points(),
        dataset_id="selection_iris_debug",
        feature_names=iris_feature_names(),
    )


def initial_selected_point_ids() -> List[str]:
    return ["setosa_001", "versicolor_001"]


def fixture_group_point_ids(group_name: str) -> List[str]:
    groups = {
        "setosa": ["setosa_001", "setosa_002", "setosa_003"],
        "versicolor": ["versicolor_001", "versicolor_002", "versicolor_003"],
        "virginica": ["virginica_001", "virginica_002", "virginica_003"],
    }
    return groups.get(group_name, [])
