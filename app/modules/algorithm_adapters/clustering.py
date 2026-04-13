from __future__ import annotations

from typing import Tuple

import numpy as np


def kmeans(
    values,
    n_clusters: int,
    max_iterations: int = 100,
) -> Tuple[int, ...]:
    matrix = np.asarray(values, dtype=float)
    _validate_inputs(matrix, n_clusters, max_iterations)

    centers = _initial_centers(matrix, n_clusters)
    labels = np.zeros(matrix.shape[0], dtype=int)

    for _ in range(max_iterations):
        previous_labels = labels.copy()
        labels = _assign_labels(matrix, centers)
        centers = _updated_centers(matrix, labels, centers)

        if np.array_equal(labels, previous_labels):
            break

    return tuple(int(label) for label in labels)


def _validate_inputs(matrix: np.ndarray, n_clusters: int, max_iterations: int) -> None:
    if matrix.ndim != 2:
        raise ValueError("feature matrix values must be two-dimensional")

    if matrix.shape[0] == 0:
        raise ValueError("feature matrix must contain at least one point")

    if matrix.shape[1] == 0:
        raise ValueError("feature matrix must contain at least one feature")

    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix values must be finite")

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise ValueError("n_clusters must be an integer")

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")

    if n_clusters > matrix.shape[0]:
        raise ValueError("n_clusters must not exceed the number of non-outlier points")

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")


def _initial_centers(matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    mean = np.mean(matrix, axis=0)
    distances_to_mean = np.sum((matrix - mean) ** 2, axis=1)
    first_index = int(np.argmin(distances_to_mean))
    center_indices = [first_index]

    while len(center_indices) < n_clusters:
        selected_centers = matrix[center_indices]
        distances = _squared_distances_to_centers(matrix, selected_centers)
        nearest_center_distance = np.min(distances, axis=1)
        nearest_center_distance[center_indices] = -1.0
        center_indices.append(int(np.argmax(nearest_center_distance)))

    return matrix[center_indices].copy()


def _assign_labels(matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = _squared_distances_to_centers(matrix, centers)
    return np.argmin(distances, axis=1)


def _updated_centers(matrix: np.ndarray, labels: np.ndarray, previous_centers: np.ndarray) -> np.ndarray:
    centers = previous_centers.copy()

    for cluster_index in range(previous_centers.shape[0]):
        cluster_points = matrix[labels == cluster_index]
        if cluster_points.shape[0] > 0:
            centers[cluster_index] = np.mean(cluster_points, axis=0)

    return centers


def _squared_distances_to_centers(matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)
