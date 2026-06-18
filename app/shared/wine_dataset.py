from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.modules.data_workspace.service import create_dataset, create_feature_matrix
from app.shared.env import repo_root
from app.shared.schemas import Dataset, FeatureMatrix


WINE_DATASET_ID = "wine_mat"
WINE_MAT_FILENAME = "wine.mat"
WINE_FEATURE_NAMES: Tuple[str, ...] = (
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
)


def load_wine_dataset(path: Path | None = None) -> Dataset:
    raw_points = wine_raw_points(path)
    return create_dataset(
        raw_points,
        dataset_id=WINE_DATASET_ID,
        feature_names=WINE_FEATURE_NAMES,
    )


def load_wine_feature_matrix(path: Path | None = None) -> FeatureMatrix:
    return create_feature_matrix(load_wine_dataset(path))


def wine_raw_points(path: Path | None = None) -> List[Dict[str, Any]]:
    matrix, labels = _load_wine_arrays(path)
    points = []
    for index, row in enumerate(matrix):
        label = int(labels[index])
        points.append(
            {
                "point_id": f"wine_{index + 1:03d}",
                "features": [float(value) for value in row],
                "metadata": {
                    "class_label": f"class_{label}",
                    "source_file": WINE_MAT_FILENAME,
                    "row_index": index,
                },
            }
        )
    return points


def wine_mat_path() -> Path:
    return repo_root() / WINE_MAT_FILENAME


def _load_wine_arrays(path: Path | None = None):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to load wine.mat") from exc

    mat_path = Path(path) if path is not None else wine_mat_path()
    if not mat_path.exists():
        raise FileNotFoundError(f"wine.mat not found at {mat_path}")

    payload = loadmat(mat_path)
    if "X" not in payload or "y" not in payload:
        raise ValueError("wine.mat must contain X and y arrays")

    matrix = payload["X"]
    labels = payload["y"].reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("wine.mat X must be a 2D matrix")
    if matrix.shape[1] != len(WINE_FEATURE_NAMES):
        raise ValueError("wine.mat X must have 13 feature columns")
    if matrix.shape[0] != labels.shape[0]:
        raise ValueError("wine.mat X and y must have the same row count")
    return matrix, labels
