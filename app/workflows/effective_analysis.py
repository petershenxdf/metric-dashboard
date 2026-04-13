from __future__ import annotations

from app.modules.algorithm_adapters.schemas import (
    AnalysisResult,
    ClusterAssignment,
    ClusterResult,
    OutlierResult,
    OutlierScore,
)
from app.modules.labeling.schemas import LabelingState
from app.shared.schemas import Dataset


def apply_manual_labels_to_analysis(
    dataset: Dataset,
    raw_analysis: AnalysisResult,
    labeling_state: LabelingState,
) -> AnalysisResult:
    """Overlay manual labels on raw algorithm output for workflow visualization."""
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be a Dataset")
    if not isinstance(raw_analysis, AnalysisResult):
        raise ValueError("raw_analysis must be an AnalysisResult")
    if not isinstance(labeling_state, LabelingState):
        raise ValueError("labeling_state must be a LabelingState")

    cluster_by_point_id = {
        assignment.point_id: assignment.cluster_id
        for assignment in raw_analysis.cluster_result.assignments
    }
    outlier_by_point_id = {
        score.point_id: score.is_outlier
        for score in raw_analysis.outlier_result.scores
    }

    override_count = 0
    for annotation in labeling_state.annotations:
        if annotation.label_type == "cluster":
            override_count += len(annotation.point_ids)
            for point_id in annotation.point_ids:
                cluster_by_point_id[point_id] = annotation.label_value
                outlier_by_point_id[point_id] = False
        elif annotation.label_type == "outlier" and annotation.label_value is True:
            override_count += len(annotation.point_ids)
            for point_id in annotation.point_ids:
                outlier_by_point_id[point_id] = True
                cluster_by_point_id.pop(point_id, None)

    effective_scores = tuple(
        OutlierScore(
            point_id=score.point_id,
            score=score.score,
            is_outlier=outlier_by_point_id.get(score.point_id, score.is_outlier),
        )
        for score in raw_analysis.outlier_result.scores
    )
    effective_outlier_ids = {
        score.point_id
        for score in effective_scores
        if score.is_outlier
    }
    known_point_ids = tuple(point.point_id for point in dataset.points)
    effective_assignments = tuple(
        ClusterAssignment(point_id=point_id, cluster_id=cluster_by_point_id[point_id])
        for point_id in known_point_ids
        if point_id in cluster_by_point_id and point_id not in effective_outlier_ids
    )

    return AnalysisResult(
        analysis_run_id=f"{raw_analysis.analysis_run_id}_manual",
        outlier_result=OutlierResult(
            outlier_run_id=f"{raw_analysis.outlier_result.outlier_run_id}_manual",
            algorithm=raw_analysis.outlier_result.algorithm,
            scores=effective_scores,
            diagnostics={
                **raw_analysis.outlier_result.diagnostics,
                "manual_label_override_count": override_count,
                "state_mode": "effective_with_manual_labels",
            },
        ),
        cluster_result=ClusterResult(
            cluster_run_id=f"{raw_analysis.cluster_result.cluster_run_id}_manual",
            algorithm=raw_analysis.cluster_result.algorithm,
            n_clusters=raw_analysis.cluster_result.n_clusters,
            assignments=effective_assignments,
            excluded_outlier_point_ids=tuple(sorted(effective_outlier_ids)),
            diagnostics={
                **raw_analysis.cluster_result.diagnostics,
                "manual_label_override_count": override_count,
                "state_mode": "effective_with_manual_labels",
            },
        ),
        diagnostics={
            **raw_analysis.diagnostics,
            "state_mode": "effective_with_manual_labels",
        },
    )
