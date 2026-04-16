from __future__ import annotations

import heapq
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


def expand_ssdbscan(
    r_dist: np.ndarray,
    seeds: Mapping[int, str],
) -> Tuple[List[Optional[str]], np.ndarray, List[Optional[int]]]:
    """Modified Prim's expansion from labeled seeds, tracking max edge along the path."""
    if not seeds:
        raise ValueError("at least one labeled seed is required")

    n_points = r_dist.shape[0]
    assigned_label: List[Optional[str]] = [None] * n_points
    e_max = np.full(n_points, np.inf, dtype=float)
    seed_origin: List[Optional[int]] = [None] * n_points

    pq: List[Tuple[float, int, int, int, str]] = []
    counter = 0

    for seed_index, label in seeds.items():
        if seed_index < 0 or seed_index >= n_points:
            raise ValueError(f"seed index out of range: {seed_index}")
        assigned_label[seed_index] = label
        e_max[seed_index] = 0.0
        seed_origin[seed_index] = seed_index
        for neighbor in range(n_points):
            if neighbor == seed_index:
                continue
            edge = float(r_dist[seed_index, neighbor])
            heapq.heappush(pq, (edge, counter, seed_index, neighbor, label))
            counter += 1

    while pq:
        weight, _, from_seed, target, label = heapq.heappop(pq)
        if assigned_label[target] is not None:
            continue
        assigned_label[target] = label
        e_max[target] = weight
        seed_origin[target] = from_seed
        for neighbor in range(n_points):
            if assigned_label[neighbor] is not None:
                continue
            edge = max(weight, float(r_dist[target, neighbor]))
            heapq.heappush(pq, (edge, counter, from_seed, neighbor, label))
            counter += 1

    return assigned_label, e_max, seed_origin


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


def validate_inputs(
    values: np.ndarray,
    min_pts: int,
    alpha: float,
    beta: float,
    contamination: float,
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
    for name, value in (("alpha", alpha), ("beta", beta), ("contamination", contamination)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
    if alpha < 0 or beta < 0 or alpha + beta > 1.0:
        raise ValueError("alpha and beta must be non-negative and sum to at most 1")
    if contamination <= 0 or contamination >= 0.5:
        raise ValueError("contamination must be greater than 0 and less than 0.5")
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
) -> Dict[str, object]:
    matrix = np.asarray(values, dtype=float)
    labeled_outliers = tuple(sorted(set(labeled_outlier_indices or ())))
    validate_inputs(matrix, min_pts, alpha, beta, contamination, labeled_outliers)
    if not seeds:
        raise ValueError("seeds must contain at least one labeled point")

    distances = pairwise_euclidean(matrix)
    c_dist = core_distances(distances, min_pts)
    r_dist = reachability_matrix(distances, c_dist)

    assigned_label, e_max, seed_origin = expand_ssdbscan(r_dist, seeds)
    r_score = np.exp(-e_max)
    l_score = compute_local_density_score(r_dist, min_pts)
    sim_score = compute_similarity_score(distances, labeled_outliers)
    t_score = combined_outlier_score(r_score, l_score, sim_score, alpha, beta)

    outlier_indices = select_outliers_by_score(t_score, contamination)

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
    }
