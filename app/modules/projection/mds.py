from __future__ import annotations

import numpy as np


def classical_mds(values, n_components: int = 2) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    _validate_matrix(matrix)

    distances_squared = _pairwise_squared_distances(matrix)
    n_points = matrix.shape[0]
    gram = _double_center_distances(distances_squared)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    coordinates = np.zeros((n_points, n_components), dtype=float)
    usable_components = min(n_components, n_points)

    for component in range(usable_components):
        eigenvalue = eigenvalues[component]
        if eigenvalue <= 0:
            continue
        coordinates[:, component] = eigenvectors[:, component] * np.sqrt(eigenvalue)

    return _orient_components(coordinates)


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("feature matrix values must be two-dimensional")

    if matrix.shape[0] == 0:
        raise ValueError("feature matrix must contain at least one point")

    if matrix.shape[1] == 0:
        raise ValueError("feature matrix must contain at least one feature")

    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix values must be finite")


def _pairwise_squared_distances(matrix: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(matrix * matrix, axis=1)
    distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (matrix @ matrix.T)
    )
    return np.maximum(distances, 0.0)


def _double_center_distances(distances_squared: np.ndarray) -> np.ndarray:
    row_means = np.mean(distances_squared, axis=1)
    column_means = np.mean(distances_squared, axis=0)
    total_mean = float(np.mean(distances_squared))
    return -0.5 * (
        distances_squared
        - row_means[:, None]
        - column_means[None, :]
        + total_mean
    )


def _orient_components(coordinates: np.ndarray) -> np.ndarray:
    oriented = coordinates.copy()

    for column_index in range(oriented.shape[1]):
        column = oriented[:, column_index]
        anchor_index = int(np.argmax(np.abs(column)))
        if column[anchor_index] < 0:
            oriented[:, column_index] = -column

    return oriented
