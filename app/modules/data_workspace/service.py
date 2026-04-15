from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

from app.shared.schemas import Dataset, FeatureMatrix, Point

DEFAULT_DATASET_ID = "dataset_001"
DEFAULT_POINT_ID_PREFIX = "p"


def create_dataset(
    raw_points: Sequence[Mapping[str, Any]],
    dataset_id: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    point_id_prefix: str = DEFAULT_POINT_ID_PREFIX,
) -> Dataset:
    if isinstance(raw_points, (str, bytes)) or not isinstance(raw_points, Sequence):
        raise ValueError("raw_points must be a sequence of point mappings")

    if not raw_points:
        raise ValueError("raw_points must not be empty")

    first_features = Point(
        point_id="validation_probe",
        features=_extract_features(raw_points[0]),
    ).features
    feature_width = len(first_features)
    names = tuple(_default_feature_names(feature_width) if feature_names is None else feature_names)
    points = []

    for index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, Mapping):
            raise ValueError("each raw point must be a mapping")

        point_id = _extract_point_id(raw_point, index, point_id_prefix)
        metadata = raw_point.get("metadata", {})
        point = Point(point_id=point_id, features=_extract_features(raw_point), metadata=metadata)

        if len(point.features) != feature_width:
            raise ValueError("all points must have the same feature length")

        points.append(point)

    return Dataset(
        dataset_id=DEFAULT_DATASET_ID if dataset_id is None else dataset_id,
        points=tuple(points),
        feature_names=names,
        created_at=_utc_timestamp(),
    )


def create_feature_matrix(dataset: Dataset) -> FeatureMatrix:
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be a Dataset")

    return FeatureMatrix(
        point_ids=tuple(point.point_id for point in dataset.points),
        feature_names=dataset.feature_names,
        values=tuple(point.features for point in dataset.points),
    )


def create_point_id_map(dataset: Dataset) -> Dict[str, Point]:
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be a Dataset")

    return {point.point_id: point for point in dataset.points}


def _extract_features(raw_point: Mapping[str, Any]) -> Sequence[Any]:
    if not isinstance(raw_point, Mapping):
        raise ValueError("each raw point must be a mapping")

    if "features" not in raw_point:
        raise ValueError("each raw point must include features")

    return raw_point["features"]


def _extract_point_id(
    raw_point: Mapping[str, Any],
    index: int,
    point_id_prefix: str,
) -> str:
    point_id = raw_point.get("point_id")
    if point_id is not None:
        return str(point_id)

    return f"{point_id_prefix}{index + 1}"


def _default_feature_names(feature_width: int) -> Sequence[str]:
    return tuple(f"feature_{index + 1}" for index in range(feature_width))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
