from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Mapping, Protocol

from app.shared.schemas import FeatureMatrix

from .clustering import kmeans
from .outliers import local_outlier_factor
from .schemas import AnalysisResult, ClusterAssignment, ClusterResult, OutlierResult, OutlierScore

DEFAULT_N_CLUSTERS = 3
DEFAULT_OUTLIER_N_NEIGHBORS = 5
DEFAULT_OUTLIER_CONTAMINATION = 0.13
KMEANS_ALGORITHM_NAME = "kmeans_numpy_deterministic"
LOF_ALGORITHM_NAME = "local_outlier_factor_numpy"
ANALYSIS_PROVIDER = "sequential_lof_then_kmeans"
FUTURE_PROVIDER_SLOT = "ssdbcodi"


class AnalysisProvider(Protocol):
    name: str

    def run(
        self,
        feature_matrix: FeatureMatrix,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        outlier_n_neighbors: int = DEFAULT_OUTLIER_N_NEIGHBORS,
        outlier_contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    ) -> AnalysisResult:
        ...


class SequentialLofThenKMeansProvider:
    name = ANALYSIS_PROVIDER

    def run(
        self,
        feature_matrix: FeatureMatrix,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        outlier_n_neighbors: int = DEFAULT_OUTLIER_N_NEIGHBORS,
        outlier_contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    ) -> AnalysisResult:
        outlier_result = detect_outliers(
            feature_matrix,
            n_neighbors=outlier_n_neighbors,
            contamination=outlier_contamination,
        )
        cluster_result = cluster_non_outliers(
            feature_matrix,
            outlier_result.outlier_point_ids,
            n_clusters=n_clusters,
        )

        return AnalysisResult(
            analysis_run_id=_stable_run_id(
                "analysis",
                {
                    "outlier_run_id": outlier_result.outlier_run_id,
                    "cluster_run_id": cluster_result.cluster_run_id,
                    "provider": self.name,
                },
            ),
            outlier_result=outlier_result,
            cluster_result=cluster_result,
            diagnostics={
                "provider": self.name,
                "execution_order": ["local_outlier_factor", "kmeans_on_non_outliers"],
                "future_provider_slot": FUTURE_PROVIDER_SLOT,
                "future_algorithm_note": (
                    "SSDBCODI can replace this sequential provider with an integrated "
                    "semi-supervised density-based clustering and outlier-detection provider."
                ),
            },
        )


def detect_outliers(
    feature_matrix: FeatureMatrix,
    n_neighbors: int = DEFAULT_OUTLIER_N_NEIGHBORS,
    contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
) -> OutlierResult:
    if not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")

    scores, flags = local_outlier_factor(
        feature_matrix.values,
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    outlier_scores = tuple(
        OutlierScore(point_id=point_id, score=scores[index], is_outlier=flags[index])
        for index, point_id in enumerate(feature_matrix.point_ids)
    )

    return OutlierResult(
        outlier_run_id=_stable_run_id(
            "outlier",
            {
                "point_ids": feature_matrix.point_ids,
                "values": feature_matrix.values,
                "algorithm": LOF_ALGORITHM_NAME,
                "n_neighbors": n_neighbors,
                "contamination": contamination,
            },
        ),
        algorithm=LOF_ALGORITHM_NAME,
        scores=outlier_scores,
        diagnostics={
            "n_neighbors": n_neighbors,
            "contamination": contamination,
            "execution_order": "outlier_detection_first",
        },
    )


def cluster_non_outliers(
    feature_matrix: FeatureMatrix,
    outlier_point_ids: Iterable[str],
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> ClusterResult:
    if not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")

    excluded_outliers = tuple(outlier_point_ids)
    excluded = set(excluded_outliers)
    unknown = excluded - set(feature_matrix.point_ids)
    if unknown:
        raise ValueError(f"unknown outlier point id(s): {', '.join(sorted(unknown))}")

    non_outlier_indices = tuple(
        index
        for index, point_id in enumerate(feature_matrix.point_ids)
        if point_id not in excluded
    )
    if not non_outlier_indices:
        raise ValueError("at least one non-outlier point is required for clustering")

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise ValueError("n_clusters must be an integer")

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")

    if n_clusters > len(non_outlier_indices):
        raise ValueError("n_clusters must not exceed the number of non-outlier points")

    clustered_values = tuple(feature_matrix.values[index] for index in non_outlier_indices)
    labels = kmeans(clustered_values, n_clusters=n_clusters)
    assignments = tuple(
        ClusterAssignment(
            point_id=feature_matrix.point_ids[non_outlier_indices[index]],
            cluster_id=f"cluster_{label + 1}",
        )
        for index, label in enumerate(labels)
    )

    return ClusterResult(
        cluster_run_id=_stable_run_id(
            "cluster",
            {
                "point_ids": feature_matrix.point_ids,
                "values": feature_matrix.values,
                "algorithm": KMEANS_ALGORITHM_NAME,
                "n_clusters": n_clusters,
                "excluded_outlier_point_ids": sorted(excluded),
            },
        ),
        algorithm=KMEANS_ALGORITHM_NAME,
        n_clusters=n_clusters,
        assignments=assignments,
        excluded_outlier_point_ids=tuple(sorted(excluded)),
        diagnostics={
            "clustered_point_count": len(assignments),
            "excluded_outlier_count": len(excluded),
            "execution_order": "after_outlier_detection",
        },
    )


def run_default_analysis(
    feature_matrix: FeatureMatrix,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    outlier_n_neighbors: int = DEFAULT_OUTLIER_N_NEIGHBORS,
    outlier_contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    provider: AnalysisProvider | None = None,
) -> AnalysisResult:
    selected_provider = provider or SequentialLofThenKMeansProvider()
    return selected_provider.run(
        feature_matrix,
        n_clusters=n_clusters,
        outlier_n_neighbors=outlier_n_neighbors,
        outlier_contamination=outlier_contamination,
    )


def cluster_counts(cluster_result: ClusterResult) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for assignment in cluster_result.assignments:
        counts[assignment.cluster_id] = counts.get(assignment.cluster_id, 0) + 1
    return counts


def assignments_by_point_id(cluster_result: ClusterResult) -> Mapping[str, str]:
    return {assignment.point_id: assignment.cluster_id for assignment in cluster_result.assignments}


def scores_by_point_id(outlier_result: OutlierResult) -> Mapping[str, OutlierScore]:
    return {score.point_id: score for score in outlier_result.scores}


def _stable_run_id(prefix: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]

    if isinstance(value, list):
        return [_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}

    return value
