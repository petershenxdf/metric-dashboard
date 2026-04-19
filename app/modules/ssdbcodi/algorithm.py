from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SSDBCODI_ALGORITHM_NAME = "ssdbcodi_numpy"


def pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def core_distances(distances: np.ndarray, min_pts: int) -> np.ndarray:
    n_points = distances.shape[0]
    k = max(1, min(min_pts, n_points - 1))
    sorted_distances = np.sort(distances, axis=1)
    return sorted_distances[:, k]


def reachability_matrix(distances: np.ndarray, c_dist: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(distances, c_dist[:, None]), c_dist[None, :])


def compute_local_density_score(
    r_dist: np.ndarray,
    min_pts: int,
) -> np.ndarray:
    n_points = r_dist.shape[0]
    k = max(1, min(min_pts, n_points - 1))
    neighbor_distances = r_dist.copy()
    np.fill_diagonal(neighbor_distances, np.inf)
    neighbor_indices = np.argsort(neighbor_distances, axis=1)[:, :k]
    local_density = np.array(
        [r_dist[index, neighbor_indices[index]].mean() for index in range(n_points)]
    )
    return np.exp(-local_density)


def compute_similarity_score(
    distances: np.ndarray,
    labeled_outlier_indices: Iterable[int] | None = None,
) -> np.ndarray:
    n_points = distances.shape[0]
    outlier_indices = tuple(sorted(set(labeled_outlier_indices or ())))
    if not outlier_indices:
        return np.zeros(n_points, dtype=float)

    outlier_distances = distances[:, outlier_indices]
    nearest_outlier_distance = np.min(outlier_distances, axis=1)
    return np.exp(-nearest_outlier_distance)


def combined_outlier_score(
    r_score: np.ndarray,
    l_score: np.ndarray,
    sim_score: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    gamma = 1.0 - alpha - beta
    return (
        alpha * (1.0 - r_score)
        + beta * (1.0 - l_score)
        + gamma * sim_score
    )


def select_outliers_by_score(
    t_score: np.ndarray,
    contamination: float,
) -> Tuple[int, ...]:
    n_points = t_score.shape[0]
    n_outliers = max(1, int(np.ceil(n_points * contamination)))
    sorted_indices = np.argsort(-t_score)
    return tuple(int(index) for index in sorted_indices[:n_outliers])


def select_outliers_from_candidates(
    t_score: np.ndarray,
    contamination: float,
    candidate_mask: np.ndarray,
) -> Tuple[int, ...]:
    n_points = t_score.shape[0]
    n_outliers = max(1, int(np.ceil(n_points * contamination)))
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.shape[0] == 0:
        return select_outliers_by_score(t_score, contamination)
    ranked = candidate_indices[np.argsort(-t_score[candidate_indices])]
    return tuple(int(index) for index in ranked[:n_outliers])


def _normalize_by_max(matrix: np.ndarray) -> np.ndarray:
    """Scale a non-negative distance matrix into [0, 1] by its global maximum."""
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.zeros_like(matrix)
    peak = float(finite.max())
    if peak <= 0.0:
        return np.zeros_like(matrix)
    return matrix / peak


def assign_classes_by_weighted_distance(
    distances: np.ndarray,
    r_dist: np.ndarray,
    seeds: Mapping[int, str],
    rscore_weight: float,
    excluded_indices: Iterable[int] | None = None,
) -> Tuple[List[Optional[str]], List[Optional[int]]]:
    """Assign each non-excluded point to the class with the smallest combined distance.

    For every class c with seed set S_c the score for point p is

        score(p, c) = w * min_{s in S_c} r_dist_norm[p, s]
                    + (1 - w) * min_{s in S_c} eucl_dist_norm[p, s]

    where w = rscore_weight in [0, 1]. r_dist and the euclidean distance matrix
    are each normalized by their global maximum so the two terms are on a
    comparable [0, 1] scale before the weighted sum. Seed points keep their
    own seed label; the chosen seed_origin is the seed in the winning class
    that gave the minimum combined score.
    """
    if not seeds:
        raise ValueError("at least one labeled seed is required")
    if not 0.0 <= rscore_weight <= 1.0:
        raise ValueError("rscore_weight must be between 0 and 1")

    n_points = r_dist.shape[0]
    excluded = set(int(index) for index in (excluded_indices or ()))

    classes: Dict[str, List[int]] = {}
    for seed_index, label in seeds.items():
        classes.setdefault(label, []).append(int(seed_index))

    r_norm = _normalize_by_max(r_dist)
    e_norm = _normalize_by_max(distances)

    assigned: List[Optional[str]] = [None] * n_points
    origin: List[Optional[int]] = [None] * n_points

    for seed_index, label in seeds.items():
        assigned[seed_index] = label
        origin[seed_index] = int(seed_index)

    for point in range(n_points):
        if point in excluded or assigned[point] is not None:
            continue

        best_label: Optional[str] = None
        best_score = float("inf")
        best_seed: Optional[int] = None

        for label, seed_indices in classes.items():
            seed_array = np.asarray(seed_indices, dtype=int)
            r_values = r_norm[point, seed_array]
            e_values = e_norm[point, seed_array]
            combined = rscore_weight * r_values + (1.0 - rscore_weight) * e_values
            local_best = int(np.argmin(combined))
            local_score = float(combined[local_best])
            if local_score < best_score:
                best_score = local_score
                best_label = label
                best_seed = int(seed_array[local_best])

        assigned[point] = best_label
        origin[point] = best_seed

    return assigned, origin


def validate_inputs(
    values: np.ndarray,
    min_pts: int,
    alpha: float,
    beta: float,
    contamination: float,
    rscore_weight: float,
    labeled_outlier_indices: Iterable[int] | None = None,
) -> None:
    if values.ndim != 2:
        raise ValueError("feature matrix values must be two-dimensional")
    if values.shape[0] == 0:
        raise ValueError("feature matrix must contain at least one point")
    if values.shape[1] == 0:
        raise ValueError("feature matrix must contain at least one feature")
    if not np.isfinite(values).all():
        raise ValueError("feature matrix values must be finite")
    if isinstance(min_pts, bool) or not isinstance(min_pts, int) or min_pts < 1:
        raise ValueError("min_pts must be a positive integer")
    if min_pts >= values.shape[0]:
        raise ValueError("min_pts must be less than the number of points")
    for name, value in (
        ("alpha", alpha),
        ("beta", beta),
        ("contamination", contamination),
        ("rscore_weight", rscore_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
    if alpha < 0 or beta < 0 or alpha + beta > 1.0:
        raise ValueError("alpha and beta must be non-negative and sum to at most 1")
    if contamination <= 0 or contamination >= 0.5:
        raise ValueError("contamination must be greater than 0 and less than 0.5")
    if rscore_weight < 0.0 or rscore_weight > 1.0:
        raise ValueError("rscore_weight must be between 0 and 1")
    if labeled_outlier_indices is not None:
        for index in labeled_outlier_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("labeled outlier indices must be integers")
            if index < 0 or index >= values.shape[0]:
                raise ValueError(f"labeled outlier index out of range: {index}")


def run_ssdbcodi_core(
    values: Sequence[Sequence[float]],
    seeds: Mapping[int, str],
    labeled_outlier_indices: Iterable[int] | None = None,
    min_pts: int = 3,
    alpha: float = 0.4,
    beta: float = 0.3,
    contamination: float = 0.13,
    rscore_weight: float = 0.5,
) -> Dict[str, object]:
    matrix = np.asarray(values, dtype=float)
    labeled_outliers = tuple(sorted(set(labeled_outlier_indices or ())))
    validate_inputs(matrix, min_pts, alpha, beta, contamination, rscore_weight, labeled_outliers)
    if not seeds:
        raise ValueError("seeds must contain at least one labeled point")

    distances = pairwise_euclidean(matrix)
    c_dist = core_distances(distances, min_pts)
    r_dist = reachability_matrix(distances, c_dist)

    n_points = matrix.shape[0]

    # r_score per point: exp(-min rDist to any seed). Seeds get r_score = 1.
    seed_indices = np.asarray(sorted(seeds.keys()), dtype=int)
    seed_r_dist = r_dist[:, seed_indices]
    e_max = np.min(seed_r_dist, axis=1)
    e_max[seed_indices] = 0.0
    r_score = np.exp(-e_max)
    l_score = compute_local_density_score(r_dist, min_pts)
    sim_score = compute_similarity_score(distances, labeled_outliers)
    t_score = combined_outlier_score(r_score, l_score, sim_score, alpha, beta)

    # Outlier candidates: any point that is not a labeled cluster seed.
    seed_set = set(int(index) for index in seeds.keys())
    candidate_mask = np.ones(n_points, dtype=bool)
    for index in seed_set:
        candidate_mask[index] = False
    auto_outliers = set(
        select_outliers_from_candidates(t_score, contamination, candidate_mask)
    )
    outlier_indices_set = (auto_outliers | set(labeled_outliers)) - seed_set
    excluded_for_assignment = outlier_indices_set

    assigned_label, seed_origin = assign_classes_by_weighted_distance(
        distances=distances,
        r_dist=r_dist,
        seeds=seeds,
        rscore_weight=rscore_weight,
        excluded_indices=excluded_for_assignment,
    )

    outlier_indices = tuple(sorted(outlier_indices_set))

    return {
        "assigned_label": tuple(assigned_label),
        "e_max": tuple(float(value) for value in e_max),
        "r_score": tuple(float(value) for value in r_score),
        "l_score": tuple(float(value) for value in l_score),
        "sim_score": tuple(float(value) for value in sim_score),
        "t_score": tuple(float(value) for value in t_score),
        "c_dist": tuple(float(value) for value in c_dist),
        "outlier_indices": outlier_indices,
        "seed_origin": tuple(seed_origin),
        "labeled_outlier_indices": labeled_outliers,
        "min_pts": min_pts,
        "alpha": alpha,
        "beta": beta,
        "contamination": contamination,
        "rscore_weight": rscore_weight,
    }
