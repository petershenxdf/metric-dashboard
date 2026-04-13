from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Mapping, Sequence, Tuple


def clean_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")

    return cleaned


def clean_features(features: object) -> Tuple[float, ...]:
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ValueError("features must be a sequence of numbers")

    if not features:
        raise ValueError("features must not be empty")

    cleaned = []
    for value in features:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("features must contain only numbers")

        number = float(value)
        if not isfinite(number):
            raise ValueError("features must contain only finite numbers")

        cleaned.append(number)

    return tuple(cleaned)


@dataclass(frozen=True)
class Point:
    point_id: str
    features: Tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "features", clean_features(self.features))

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
            return

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "features": list(self.features),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FeatureMatrix:
    point_ids: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    values: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        point_ids = tuple(clean_text(point_id, "point_id") for point_id in self.point_ids)
        feature_names = tuple(clean_text(name, "feature_name") for name in self.feature_names)
        values = tuple(clean_features(row) for row in self.values)

        if not point_ids:
            raise ValueError("point_ids must not be empty")

        if len(set(point_ids)) != len(point_ids):
            raise ValueError("point_ids must be unique")

        if not feature_names:
            raise ValueError("feature_names must not be empty")

        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")

        if len(values) != len(point_ids):
            raise ValueError("values must contain one row per point_id")

        for row in values:
            if len(row) != len(feature_names):
                raise ValueError("each feature row must match feature_names length")

        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "values", values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_ids": list(self.point_ids),
            "feature_names": list(self.feature_names),
            "values": [list(row) for row in self.values],
        }


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    points: Tuple[Point, ...]
    feature_names: Tuple[str, ...]
    created_at: str

    def __post_init__(self) -> None:
        dataset_id = clean_text(self.dataset_id, "dataset_id")
        points = tuple(self.points)
        feature_names = tuple(clean_text(name, "feature_name") for name in self.feature_names)

        if not points:
            raise ValueError("points must not be empty")

        if not all(isinstance(point, Point) for point in points):
            raise ValueError("points must contain Point objects")

        point_ids = [point.point_id for point in points]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("point_id values must be unique")

        if not feature_names:
            raise ValueError("feature_names must not be empty")

        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")

        for point in points:
            if len(point.features) != len(feature_names):
                raise ValueError("each point feature vector must match feature_names length")

        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "created_at", clean_text(self.created_at, "created_at"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "points": [point.to_dict() for point in self.points],
            "feature_names": list(self.feature_names),
            "created_at": self.created_at,
        }


def clean_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")

    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")

    return number


@dataclass(frozen=True)
class ProjectionCoordinate:
    point_id: str
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "x", clean_number(self.x, "x"))
        object.__setattr__(self, "y", clean_number(self.y, "y"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "x": self.x,
            "y": self.y,
        }


@dataclass(frozen=True)
class ProjectionResult:
    projection_id: str
    method: str
    coordinates: Tuple[ProjectionCoordinate, ...]

    def __post_init__(self) -> None:
        projection_id = clean_text(self.projection_id, "projection_id")
        method = clean_text(self.method, "method")
        coordinates = tuple(self.coordinates)

        if not coordinates:
            raise ValueError("coordinates must not be empty")

        if not all(isinstance(coordinate, ProjectionCoordinate) for coordinate in coordinates):
            raise ValueError("coordinates must contain ProjectionCoordinate objects")

        point_ids = [coordinate.point_id for coordinate in coordinates]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("projection coordinates must have unique point_id values")

        object.__setattr__(self, "projection_id", projection_id)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "coordinates", coordinates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "method": self.method,
            "coordinates": [coordinate.to_dict() for coordinate in self.coordinates],
        }
