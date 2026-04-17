"""Backwards-compatible re-exports.

The canonical home for these debug fixtures is `app.shared.fixtures` so that
module-level packages (e.g. scatterplot) can reach them without importing from
`app.workflows` (which would be a reverse layering dependency).
"""

from __future__ import annotations

from app.shared.fixtures import (
    ANALYSIS_SELECTION_DATASET_OPTIONS,
    ANALYSIS_SELECTION_INITIAL_SELECTED_BY_DATASET,
    DEFAULT_ANALYSIS_DATASET_ID,
    DEFAULT_WORKFLOW_DATASET_ID,
    WIDE_GAP_DATASET_ID,
    analysis_selection_dataset,
    analysis_selection_initial_selected_point_ids,
    is_analysis_selection_dataset_id,
)

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
