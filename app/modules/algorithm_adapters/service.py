from __future__ import annotations

from typing import Dict, Mapping, Protocol, TYPE_CHECKING

from app.shared.schemas import FeatureMatrix

from .schemas import AnalysisResult, ClusterResult, OutlierResult, OutlierScore

DEFAULT_N_CLUSTERS = 3
DEFAULT_OUTLIER_N_NEIGHBORS = 5
DEFAULT_OUTLIER_CONTAMINATION = 0.13
if TYPE_CHECKING:
    from app.modules.labeling.schemas import LabelingState


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


def run_default_analysis(
    feature_matrix: FeatureMatrix,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    outlier_n_neighbors: int = DEFAULT_OUTLIER_N_NEIGHBORS,
    outlier_contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    provider: AnalysisProvider | None = None,
    labeling_state: "LabelingState | None" = None,
) -> AnalysisResult:
    selected_provider = provider or _default_provider(labeling_state)
    return selected_provider.run(
        feature_matrix,
        n_clusters=n_clusters,
        outlier_n_neighbors=outlier_n_neighbors,
        outlier_contamination=outlier_contamination,
    )


def _default_provider(labeling_state: "LabelingState | None" = None) -> AnalysisProvider:
    from app.modules.ssdbcodi.service import SsdbcodiProvider

    return SsdbcodiProvider(labeling_state=labeling_state)


def cluster_counts(cluster_result: ClusterResult) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for assignment in cluster_result.assignments:
        counts[assignment.cluster_id] = counts.get(assignment.cluster_id, 0) + 1
    return counts


def assignments_by_point_id(cluster_result: ClusterResult) -> Mapping[str, str]:
    return {assignment.point_id: assignment.cluster_id for assignment in cluster_result.assignments}


def scores_by_point_id(outlier_result: OutlierResult) -> Mapping[str, OutlierScore]:
    return {score.point_id: score for score in outlier_result.scores}
