from __future__ import annotations

from math import sqrt

import numpy as np


def classical_mds(values, n_components: int = 2) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    _validate_matrix(matrix)

    distances_squared = _pairwise_squared_distances(matrix)
    n_points = matrix.shape[0]
    gram = _double_center_distances(distances_squared)

    eigenvalues, eigenvectors = _symmetric_eigh(gram)
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
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sum(diff * diff, axis=2)


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


def _symmetric_eigh(matrix: np.ndarray):
    """Small Jacobi eigensolver to avoid platform-specific native linalg issues."""
    values = np.asarray(matrix, dtype=float).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")

    size = values.shape[0]
    vectors = np.eye(size, dtype=float)
    tolerance = 1e-10
    max_iterations = max(1, 100 * size * size)

    for _ in range(max_iterations):
        pivot_row, pivot_col, pivot_value = _largest_off_diagonal(values)
        if pivot_value < tolerance:
            break

        app = values[pivot_row, pivot_row]
        aqq = values[pivot_col, pivot_col]
        apq = values[pivot_row, pivot_col]
        tau = (aqq - app) / (2.0 * apq)
        sign = 1.0 if tau >= 0 else -1.0
        tangent = sign / (abs(tau) + sqrt(1.0 + tau * tau))
        cosine = 1.0 / sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine

        for index in range(size):
            if index in (pivot_row, pivot_col):
                continue
            aip = values[index, pivot_row]
            aiq = values[index, pivot_col]
            values[index, pivot_row] = values[pivot_row, index] = cosine * aip - sine * aiq
            values[index, pivot_col] = values[pivot_col, index] = sine * aip + cosine * aiq

        values[pivot_row, pivot_row] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq
        )
        values[pivot_col, pivot_col] = (
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        values[pivot_row, pivot_col] = values[pivot_col, pivot_row] = 0.0

        for index in range(size):
            vip = vectors[index, pivot_row]
            viq = vectors[index, pivot_col]
            vectors[index, pivot_row] = cosine * vip - sine * viq
            vectors[index, pivot_col] = sine * vip + cosine * viq

    return np.diag(values), vectors


def _largest_off_diagonal(matrix: np.ndarray):
    size = matrix.shape[0]
    pivot_row = 0
    pivot_col = 1 if size > 1 else 0
    pivot_value = 0.0

    for row in range(size):
        for col in range(row + 1, size):
            value = abs(matrix[row, col])
            if value > pivot_value:
                pivot_row = row
                pivot_col = col
                pivot_value = value

    return pivot_row, pivot_col, pivot_value
