"""Cross-module fixture datasets.

These are deterministic debug fixtures used by multiple Flask pages (both
module debug pages and workflow pages). Keeping them here prevents the
scatterplot module from having to reach into `app.workflows` for sample data.
"""

from __future__ import annotations

from typing import Dict, Tuple

from app.modules.algorithm_adapters.fixtures import (
    default_analysis_feature_names,
    default_analysis_raw_points,
)
from app.modules.data_workspace.service import create_dataset

WIDE_GAP_DATASET_ID = "wide_gap_analysis_debug"
DEFAULT_ANALYSIS_DATASET_ID = "default_analysis_outlier_debug"
DEFAULT_WORKFLOW_DATASET_ID = WIDE_GAP_DATASET_ID

ANALYSIS_SELECTION_DATASET_OPTIONS: Tuple[Dict[str, str], ...] = (
    {
        "dataset_id": WIDE_GAP_DATASET_ID,
        "title": "Wide Gap Debug",
        "description": "fewer points with larger visual gaps for selection testing",
    },
    {
        "dataset_id": DEFAULT_ANALYSIS_DATASET_ID,
        "title": "Default Analysis Outlier Debug",
        "description": "three compact clusters plus three distant outlier candidates",
    },
)

ANALYSIS_SELECTION_INITIAL_SELECTED_BY_DATASET: Dict[str, Tuple[str, ...]] = {
    WIDE_GAP_DATASET_ID: ("alpha_01", "outlier_north"),
    DEFAULT_ANALYSIS_DATASET_ID: ("cluster_a_001", "outlier_far_001"),
}


def analysis_selection_dataset(dataset_id: str | None = None):
    dataset_id = dataset_id or DEFAULT_WORKFLOW_DATASET_ID
    if dataset_id == WIDE_GAP_DATASET_ID:
        return create_dataset(
            _wide_gap_raw_points(),
            dataset_id=WIDE_GAP_DATASET_ID,
            feature_names=default_analysis_feature_names(),
        )

    if dataset_id == DEFAULT_ANALYSIS_DATASET_ID:
        return create_dataset(
            default_analysis_raw_points(),
            dataset_id=DEFAULT_ANALYSIS_DATASET_ID,
            feature_names=default_analysis_feature_names(),
        )

    return analysis_selection_dataset(DEFAULT_WORKFLOW_DATASET_ID)


def analysis_selection_initial_selected_point_ids(dataset_id: str) -> Tuple[str, ...]:
    return ANALYSIS_SELECTION_INITIAL_SELECTED_BY_DATASET.get(
        dataset_id,
        ANALYSIS_SELECTION_INITIAL_SELECTED_BY_DATASET[DEFAULT_WORKFLOW_DATASET_ID],
    )


def is_analysis_selection_dataset_id(dataset_id) -> bool:
    return dataset_id in {
        option["dataset_id"]
        for option in ANALYSIS_SELECTION_DATASET_OPTIONS
    }


def _wide_gap_raw_points():
    return [
        {"point_id": "alpha_01", "features": [-8.2, -5.8], "metadata": {"label": "alpha"}},
        {"point_id": "alpha_02", "features": [-7.5, -6.7], "metadata": {"label": "alpha"}},
        {"point_id": "alpha_03", "features": [-6.9, -5.2], "metadata": {"label": "alpha"}},
        {"point_id": "beta_01", "features": [6.5, -6.0], "metadata": {"label": "beta"}},
        {"point_id": "beta_02", "features": [7.4, -5.1], "metadata": {"label": "beta"}},
        {"point_id": "beta_03", "features": [8.1, -6.8], "metadata": {"label": "beta"}},
        {"point_id": "gamma_01", "features": [-2.2, 7.4], "metadata": {"label": "gamma"}},
        {"point_id": "gamma_02", "features": [-1.2, 8.3], "metadata": {"label": "gamma"}},
        {"point_id": "gamma_03", "features": [-3.2, 8.6], "metadata": {"label": "gamma"}},
        {"point_id": "outlier_north", "features": [2.0, 19.0], "metadata": {"label": "outlier_fixture"}},
        {"point_id": "outlier_west", "features": [-18.0, 2.0], "metadata": {"label": "outlier_fixture"}},
        {"point_id": "outlier_east", "features": [18.0, 6.0], "metadata": {"label": "outlier_fixture"}},
    ]


__all__ = [
    "ANALYSIS_SELECTION_DATASET_OPTIONS",
    "ANALYSIS_SELECTION_INITIAL_SELECTED_BY_DATASET",
    "DEFAULT_ANALYSIS_DATASET_ID",
    "DEFAULT_WORKFLOW_DATASET_ID",
    "WIDE_GAP_DATASET_ID",
    "analysis_selection_dataset",
    "analysis_selection_initial_selected_point_ids",
    "is_analysis_selection_dataset_id",
]
