from __future__ import annotations

from math import ceil
from typing import Tuple

import numpy as np


def local_outlier_factor(
    values,
    n_neighbors: int = 5,
    contamination: float = 0.13,
) -> Tuple[Tuple[float, ...], Tuple[bool, ...]]:
    matrix = np.asarray(values, dtype=float)
    _validate_inputs(matrix, n_neighbors, contamination)

    n_points = matrix.shape[0]
    if n_points < 3:
        return tuple(1.0 for _ in range(n_points)), tuple(False for _ in range(n_points))

    k = min(n_neighbors, n_points - 1)
    distances = _pairwise_distances(matrix)
    neighbor_indices = np.argsort(distances, axis=1)[:, 1 : k + 1]
    k_distances = distances[np.arange(n_points), neighbor_indices[:, -1]]

    local_reachability_density = np.zeros(n_points, dtype=float)
    for point_index in range(n_points):
        neighbors = neighbor_indices[point_index]
        reachability = np.maximum(k_distances[neighbors], distances[point_index, neighbors])
        local_reachability_density[point_index] = 1.0 / max(float(np.mean(reachability)), 1e-12)

    scores = np.zeros(n_points, dtype=float)
    for point_index in range(n_points):
        neighbors = neighbor_indices[point_index]
        ratios = local_reachability_density[neighbors] / local_reachability_density[point_index]
        scores[point_index] = float(np.mean(ratios))

    outlier_flags = _select_outliers(scores, contamination)
    return tuple(float(score) for score in scores), tuple(bool(flag) for flag in outlier_flags)


def _validate_inputs(matrix: np.ndarray, n_neighbors: int, contamination: float) -> None:
    if matrix.ndim != 2:
        raise ValueError("feature matrix values must be two-dimensional")

    if matrix.shape[0] == 0:
        raise ValueError("feature matrix must contain at least one point")

    if matrix.shape[1] == 0:
        raise ValueError("feature matrix must contain at least one feature")

    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix values must be finite")

    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")

    if isinstance(contamination, bool) or not isinstance(contamination, (int, float)):
        raise ValueError("contamination must be a number")

    if contamination <= 0 or contamination >= 0.5:
        raise ValueError("contamination must be greater than 0 and less than 0.5")


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _select_outliers(scores: np.ndarray, contamination: float) -> np.ndarray:
    n_points = scores.shape[0]
    outlier_count = max(1, int(ceil(n_points * contamination)))
    selected_indices = np.argsort(scores)[-outlier_count:]
    flags = np.zeros(n_points, dtype=bool)

    for index in selected_indices:
        if scores[index] > 1.0:
            flags[index] = True

    return flags
