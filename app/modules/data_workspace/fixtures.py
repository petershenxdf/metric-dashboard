from __future__ import annotations

from typing import Any, Dict, List


def iris_feature_names() -> List[str]:
    return ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def iris_raw_points() -> List[Dict[str, Any]]:
    return [
        {
            "point_id": "setosa_001",
            "features": [5.1, 3.5, 1.4, 0.2],
            "metadata": {"label": "setosa"},
        },
        {
            "point_id": "setosa_002",
            "features": [4.9, 3.0, 1.4, 0.2],
            "metadata": {"label": "setosa"},
        },
        {
            "point_id": "setosa_003",
            "features": [4.7, 3.2, 1.3, 0.2],
            "metadata": {"label": "setosa"},
        },
        {
            "point_id": "setosa_004",
            "features": [5.0, 3.6, 1.4, 0.2],
            "metadata": {"label": "setosa"},
        },
        {
            "point_id": "setosa_005",
            "features": [5.4, 3.9, 1.7, 0.4],
            "metadata": {"label": "setosa"},
        },
        {
            "point_id": "versicolor_001",
            "features": [7.0, 3.2, 4.7, 1.4],
            "metadata": {"label": "versicolor"},
        },
        {
            "point_id": "versicolor_002",
            "features": [6.4, 3.2, 4.5, 1.5],
            "metadata": {"label": "versicolor"},
        },
        {
            "point_id": "versicolor_003",
            "features": [6.9, 3.1, 4.9, 1.5],
            "metadata": {"label": "versicolor"},
        },
        {
            "point_id": "versicolor_004",
            "features": [5.5, 2.3, 4.0, 1.3],
            "metadata": {"label": "versicolor"},
        },
        {
            "point_id": "versicolor_005",
            "features": [6.5, 2.8, 4.6, 1.5],
            "metadata": {"label": "versicolor"},
        },
        {
            "point_id": "virginica_001",
            "features": [6.3, 3.3, 6.0, 2.5],
            "metadata": {"label": "virginica"},
        },
        {
            "point_id": "virginica_002",
            "features": [5.8, 2.7, 5.1, 1.9],
            "metadata": {"label": "virginica"},
        },
        {
            "point_id": "virginica_003",
            "features": [7.1, 3.0, 5.9, 2.1],
            "metadata": {"label": "virginica"},
        },
        {
            "point_id": "virginica_004",
            "features": [6.3, 2.9, 5.6, 1.8],
            "metadata": {"label": "virginica"},
        },
        {
            "point_id": "virginica_005",
            "features": [6.5, 3.0, 5.8, 2.2],
            "metadata": {"label": "virginica"},
        },
    ]


def tiny_raw_points() -> List[Dict[str, Any]]:
    return [
        {"point_id": "p1", "features": [0, 0], "metadata": {"label": "a"}},
        {"point_id": "p2", "features": [1, 0], "metadata": {"label": "a"}},
        {"point_id": "p3", "features": [4, 4], "metadata": {"label": "b"}},
    ]


def tiny_feature_names() -> List[str]:
    return ["x", "y"]
