from __future__ import annotations

from typing import Any, Dict, List


def default_analysis_feature_names() -> List[str]:
    return ["x", "y"]


def default_analysis_raw_points() -> List[Dict[str, Any]]:
    return [
        {"point_id": "cluster_a_001", "features": [0.0, 0.1], "metadata": {"label": "cluster_a"}},
        {"point_id": "cluster_a_002", "features": [0.2, -0.1], "metadata": {"label": "cluster_a"}},
        {"point_id": "cluster_a_003", "features": [-0.2, 0.0], "metadata": {"label": "cluster_a"}},
        {"point_id": "cluster_a_004", "features": [0.1, 0.3], "metadata": {"label": "cluster_a"}},
        {"point_id": "cluster_a_005", "features": [-0.3, -0.2], "metadata": {"label": "cluster_a"}},
        {"point_id": "cluster_b_001", "features": [5.0, 5.1], "metadata": {"label": "cluster_b"}},
        {"point_id": "cluster_b_002", "features": [5.2, 4.9], "metadata": {"label": "cluster_b"}},
        {"point_id": "cluster_b_003", "features": [4.8, 5.0], "metadata": {"label": "cluster_b"}},
        {"point_id": "cluster_b_004", "features": [5.1, 5.3], "metadata": {"label": "cluster_b"}},
        {"point_id": "cluster_b_005", "features": [4.7, 4.8], "metadata": {"label": "cluster_b"}},
        {"point_id": "cluster_c_001", "features": [10.0, 0.0], "metadata": {"label": "cluster_c"}},
        {"point_id": "cluster_c_002", "features": [10.2, 0.2], "metadata": {"label": "cluster_c"}},
        {"point_id": "cluster_c_003", "features": [9.8, -0.1], "metadata": {"label": "cluster_c"}},
        {"point_id": "cluster_c_004", "features": [10.1, -0.3], "metadata": {"label": "cluster_c"}},
        {"point_id": "cluster_c_005", "features": [9.7, 0.1], "metadata": {"label": "cluster_c"}},
        {"point_id": "outlier_far_001", "features": [20.0, 20.0], "metadata": {"label": "outlier_fixture"}},
        {"point_id": "outlier_far_002", "features": [-12.0, 15.0], "metadata": {"label": "outlier_fixture"}},
        {"point_id": "outlier_far_003", "features": [18.0, -14.0], "metadata": {"label": "outlier_fixture"}},
    ]
