from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.modules.algorithm_adapters.clustering import kmeans
from app.modules.algorithm_adapters.schemas import (
    AnalysisResult,
    ClusterAssignment,
    ClusterResult,
    OutlierResult,
    OutlierScore,
)
from app.modules.labeling.schemas import LabelingState
from app.shared.schemas import FeatureMatrix

from .algorithm import (
    SSDBCODI_ALGORITHM_NAME,
    core_distances,
    pairwise_euclidean,
    run_ssdbcodi_core,
)
from .schemas import PointScores, SeedRecord, SsdbcodiResult

DEFAULT_BOOTSTRAP_K = 3
DEFAULT_MIN_PTS = 3
DEFAULT_ALPHA = 0.4
DEFAULT_BETA = 0.3
DEFAULT_CONTAMINATION = 0.13
PROVIDER_NAME = "ssdbcodi"
SEED_SOURCE_BOOTSTRAP = "kmeans_bootstrap"
SEED_SOURCE_LABEL = "manual_label"


def bootstrap_seeds_from_kmeans(
    feature_matrix: FeatureMatrix,
    n_clusters: int = DEFAULT_BOOTSTRAP_K,
    min_pts: int = DEFAULT_MIN_PTS,
) -> Dict[int, str]:
    """Run kmeans on dense candidates, then return centroid-nearest seed points."""
    if not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise ValueError("n_clusters must be an integer")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if n_clusters > len(feature_matrix.point_ids):
        raise ValueError("n_clusters must not exceed the number of points")
    if isinstance(min_pts, bool) or not isinstance(min_pts, int) or min_pts < 1:
        raise ValueError("min_pts must be a positive integer")

    matrix = np.asarray(feature_matrix.values, dtype=float)
    candidate_indices = _dense_candidate_indices(matrix, n_clusters, min_pts)
    candidate_matrix = matrix[candidate_indices]
    labels = kmeans(candidate_matrix, n_clusters=n_clusters)
    labels_array = np.asarray(labels, dtype=int)

    seed_candidates: List[Tuple[Tuple[float, ...], int]] = []
    for cluster_index in range(n_clusters):
        mask = labels_array == cluster_index
        if not bool(mask.any()):
            continue
        cluster_points = candidate_matrix[mask]
        centroid = cluster_points.mean(axis=0)
        distances = np.sqrt(np.sum((candidate_matrix - centroid) ** 2, axis=1))
        masked_distances = np.where(mask, distances, np.inf)
        seed_index = int(candidate_indices[int(np.argmin(masked_distances))])
        seed_candidates.append((tuple(float(value) for value in centroid), seed_index))

    seeds: Dict[int, str] = {}
    for label_index, (_, seed_index) in enumerate(sorted(seed_candidates), start=1):
        seeds[seed_index] = f"cluster_{label_index}"
    return seeds


def collect_seeds_from_labeling(
    feature_matrix: FeatureMatrix,
    labeling_state: Optional[LabelingState],
) -> Tuple[Dict[int, str], Dict[int, bool]]:
    """Convert labeling annotations into seed-index/label dict and outlier overrides."""
    seeds: Dict[int, str] = {}
    outlier_overrides: Dict[int, bool] = {}
    if labeling_state is None:
        return seeds, outlier_overrides

    if not isinstance(labeling_state, LabelingState):
        raise ValueError("labeling_state must be a LabelingState")

    point_id_to_index = {
        point_id: index for index, point_id in enumerate(feature_matrix.point_ids)
    }
    for annotation in labeling_state.annotations:
        for point_id in annotation.point_ids:
            index = point_id_to_index.get(point_id)
            if index is None:
                continue
            if annotation.label_type == "cluster":
                seeds[index] = str(annotation.label_value)
            elif annotation.label_type == "class":
                seeds[index] = f"class:{annotation.label_value}"
            elif annotation.label_type == "outlier":
                outlier_overrides[index] = bool(annotation.label_value)
    return seeds, outlier_overrides


def merge_seeds(
    bootstrap_seeds: Mapping[int, str],
    manual_seeds: Mapping[int, str],
) -> Dict[int, str]:
    """Manual labels take precedence over bootstrap labels."""
    merged: Dict[int, str] = dict(bootstrap_seeds)
    merged.update(manual_seeds)
    return merged


def run_ssdbcodi(
    feature_matrix: FeatureMatrix,
    labeling_state: Optional[LabelingState] = None,
    n_clusters: int = DEFAULT_BOOTSTRAP_K,
    min_pts: int = DEFAULT_MIN_PTS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    contamination: float = DEFAULT_CONTAMINATION,
    bootstrap: bool = True,
) -> SsdbcodiResult:
    if not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")

    manual_seeds, outlier_overrides = collect_seeds_from_labeling(
        feature_matrix, labeling_state
    )
    labeled_outlier_indices = {
        index for index, is_outlier in outlier_overrides.items() if is_outlier
    }
    manual_seeds = {
        index: label
        for index, label in manual_seeds.items()
        if index not in labeled_outlier_indices
    }

    bootstrap_seeds: Dict[int, str] = {}
    used_bootstrap = False
    if bootstrap:
        bootstrap_seeds = bootstrap_seeds_from_kmeans(feature_matrix, n_clusters, min_pts)
        bootstrap_seeds = {
            index: label
            for index, label in bootstrap_seeds.items()
            if index not in labeled_outlier_indices
        }
        used_bootstrap = True

    combined_seeds = merge_seeds(bootstrap_seeds, manual_seeds)
    if not combined_seeds:
        raise ValueError(
            "no seeds available: provide manual labels or enable bootstrap"
        )

    core = run_ssdbcodi_core(
        values=feature_matrix.values,
        seeds=combined_seeds,
        labeled_outlier_indices=labeled_outlier_indices,
        min_pts=min_pts,
        alpha=alpha,
        beta=beta,
        contamination=contamination,
    )

    point_ids = feature_matrix.point_ids
    assigned_label: Sequence[Optional[str]] = core["assigned_label"]
    outlier_indices = set(core["outlier_indices"])
    for index, is_outlier in outlier_overrides.items():
        if is_outlier:
            outlier_indices.add(index)
        else:
            outlier_indices.discard(index)

    cluster_id_lookup = {
        index: assigned_label[index] or "cluster_unassigned"
        for index in range(len(point_ids))
    }

    cluster_assignments = tuple(
        ClusterAssignment(point_id=point_ids[index], cluster_id=cluster_id_lookup[index])
        for index in range(len(point_ids))
        if index not in outlier_indices
    )

    cluster_id_set = {assignment.cluster_id for assignment in cluster_assignments}
    n_unique_clusters = len(cluster_id_set) if cluster_id_set else 1

    parameters = {
        "n_clusters_bootstrap": n_clusters,
        "min_pts": min_pts,
        "alpha": alpha,
        "beta": beta,
        "contamination": contamination,
        "bootstrap_used": used_bootstrap,
        "labeled_outlier_count": len(labeled_outlier_indices),
    }

    cluster_run_id = _stable_run_id(
        "ssdbcodi_cluster",
        {
            "point_ids": list(point_ids),
            "values": [list(row) for row in feature_matrix.values],
            "seeds": _sorted_seed_payload(combined_seeds, point_ids),
            "outlier_overrides": _sorted_overrides(outlier_overrides, point_ids),
            "labeled_outliers": sorted(point_ids[index] for index in labeled_outlier_indices),
            "parameters": parameters,
        },
    )
    outlier_run_id = _stable_run_id(
        "ssdbcodi_outlier",
        {
            "point_ids": list(point_ids),
            "values": [list(row) for row in feature_matrix.values],
            "seeds": _sorted_seed_payload(combined_seeds, point_ids),
            "outlier_overrides": _sorted_overrides(outlier_overrides, point_ids),
            "labeled_outliers": sorted(point_ids[index] for index in labeled_outlier_indices),
            "parameters": parameters,
        },
    )
    run_id = _stable_run_id(
        "ssdbcodi_run",
        {
            "cluster_run_id": cluster_run_id,
            "outlier_run_id": outlier_run_id,
            "provider": PROVIDER_NAME,
        },
    )

    cluster_result = ClusterResult(
        cluster_run_id=cluster_run_id,
        algorithm=SSDBCODI_ALGORITHM_NAME,
        n_clusters=max(1, n_unique_clusters),
        assignments=cluster_assignments,
        excluded_outlier_point_ids=tuple(
            sorted(point_ids[index] for index in outlier_indices)
        ),
        diagnostics={
            "execution_order": "ssdbscan_expansion_then_outlier_score",
            "seed_count": len(combined_seeds),
            "bootstrap_used": used_bootstrap,
        },
    )

    outlier_scores = tuple(
        OutlierScore(
            point_id=point_ids[index],
            score=float(core["t_score"][index]),
            is_outlier=index in outlier_indices,
        )
        for index in range(len(point_ids))
    )
    outlier_result = OutlierResult(
        outlier_run_id=outlier_run_id,
        algorithm=SSDBCODI_ALGORITHM_NAME,
        scores=outlier_scores,
        diagnostics={
            "execution_order": "scored_with_t_score",
            "alpha": alpha,
            "beta": beta,
            "contamination": contamination,
            "user_overrides": len(outlier_overrides),
            "labeled_outliers": len(labeled_outlier_indices),
        },
    )

    seed_records = tuple(
        SeedRecord(
            point_id=point_ids[index],
            cluster_id=label,
            source=(
                SEED_SOURCE_LABEL if index in manual_seeds else SEED_SOURCE_BOOTSTRAP
            ),
        )
        for index, label in sorted(combined_seeds.items())
    )

    point_scores = tuple(
        PointScores(
            point_id=point_ids[index],
            cluster_id=cluster_id_lookup[index],
            is_outlier=index in outlier_indices,
            r_score=float(core["r_score"][index]),
            l_score=float(core["l_score"][index]),
            sim_score=float(core["sim_score"][index]),
            t_score=float(core["t_score"][index]),
            c_dist=float(core["c_dist"][index]),
            e_max=float(core["e_max"][index]),
            seed_origin_point_id=(
                point_ids[core["seed_origin"][index]]
                if core["seed_origin"][index] is not None
                else None
            ),
        )
        for index in range(len(point_ids))
    )

    return SsdbcodiResult(
        run_id=run_id,
        algorithm=SSDBCODI_ALGORITHM_NAME,
        cluster_result=cluster_result,
        outlier_result=outlier_result,
        point_scores=point_scores,
        seeds=seed_records,
        parameters=parameters,
        diagnostics={
            "provider": PROVIDER_NAME,
            "bootstrap_used": used_bootstrap,
            "manual_seed_count": len(manual_seeds),
            "bootstrap_seed_count": len(bootstrap_seeds),
            "outlier_override_count": len(outlier_overrides),
            "labeled_outlier_count": len(labeled_outlier_indices),
            "execution_order": (
                "kmeans_bootstrap_then_ssdbcodi"
                if used_bootstrap
                else "ssdbcodi_with_user_seeds"
            ),
        },
    )


class SsdbcodiProvider:
    """AnalysisProvider-compatible wrapper for SSDBCODI."""

    name = PROVIDER_NAME

    def __init__(
        self,
        labeling_state: Optional[LabelingState] = None,
        min_pts: int = DEFAULT_MIN_PTS,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self._labeling_state = labeling_state
        self._min_pts = min_pts
        self._alpha = alpha
        self._beta = beta
        self._latest_result: Optional[SsdbcodiResult] = None

    @property
    def latest_result(self) -> Optional[SsdbcodiResult]:
        return self._latest_result

    def run(
        self,
        feature_matrix: FeatureMatrix,
        n_clusters: int = DEFAULT_BOOTSTRAP_K,
        outlier_n_neighbors: int = DEFAULT_MIN_PTS,
        outlier_contamination: float = DEFAULT_CONTAMINATION,
    ) -> AnalysisResult:
        result = run_ssdbcodi(
            feature_matrix=feature_matrix,
            labeling_state=self._labeling_state,
            n_clusters=n_clusters,
            min_pts=self._min_pts if outlier_n_neighbors == DEFAULT_MIN_PTS else outlier_n_neighbors,
            alpha=self._alpha,
            beta=self._beta,
            contamination=outlier_contamination,
        )
        self._latest_result = result
        return AnalysisResult(
            analysis_run_id=result.run_id,
            outlier_result=result.outlier_result,
            cluster_result=result.cluster_result,
            diagnostics=dict(result.diagnostics),
        )


def cluster_counts(result: SsdbcodiResult) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for assignment in result.cluster_result.assignments:
        counts[assignment.cluster_id] = counts.get(assignment.cluster_id, 0) + 1
    return counts


def scores_by_point_id(result: SsdbcodiResult) -> Dict[str, PointScores]:
    return {score.point_id: score for score in result.point_scores}


def _dense_candidate_indices(
    matrix: np.ndarray,
    n_clusters: int,
    min_pts: int,
) -> np.ndarray:
    if matrix.shape[0] <= n_clusters:
        return np.arange(matrix.shape[0])

    distances = pairwise_euclidean(matrix)
    c_dist = core_distances(distances, min(min_pts, matrix.shape[0] - 1))
    q1, q3 = np.percentile(c_dist, [25, 75])
    iqr = q3 - q1
    cutoff = q3 + 1.5 * iqr
    candidates = np.flatnonzero(c_dist <= cutoff)
    if candidates.shape[0] >= n_clusters:
        return candidates
    return np.arange(matrix.shape[0])


def _sorted_seed_payload(
    seeds: Mapping[int, str],
    point_ids: Sequence[str],
) -> List[Tuple[str, str]]:
    return sorted(
        (point_ids[index], label) for index, label in seeds.items()
    )


def _sorted_overrides(
    overrides: Mapping[int, bool],
    point_ids: Sequence[str],
) -> List[Tuple[str, bool]]:
    return sorted(
        (point_ids[index], bool(value)) for index, value in overrides.items()
    )


def _stable_run_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
