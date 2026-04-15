from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.shared.schemas import clean_number, clean_text


@dataclass(frozen=True)
class ScatterplotPoint:
    point_id: str
    x: float
    y: float
    screen_x: float
    screen_y: float
    cluster_id: str | None = None
    is_outlier: bool = False
    selected: bool = False
    manual_labels: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    color: str = "#2f6fed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "x", clean_number(self.x, "x"))
        object.__setattr__(self, "y", clean_number(self.y, "y"))
        object.__setattr__(self, "screen_x", clean_number(self.screen_x, "screen_x"))
        object.__setattr__(self, "screen_y", clean_number(self.screen_y, "screen_y"))

        if self.cluster_id is not None:
            object.__setattr__(self, "cluster_id", clean_text(self.cluster_id, "cluster_id"))

        if not isinstance(self.is_outlier, bool):
            raise ValueError("is_outlier must be a boolean")

        if not isinstance(self.selected, bool):
            raise ValueError("selected must be a boolean")

        labels = tuple(self.manual_labels)
        if not all(isinstance(label, Mapping) for label in labels):
            raise ValueError("manual_labels must contain mappings")

        if self.metadata is None or not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        object.__setattr__(self, "manual_labels", tuple(dict(label) for label in labels))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "color", clean_text(self.color, "color"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "x": self.x,
            "y": self.y,
            "screen_x": self.screen_x,
            "screen_y": self.screen_y,
            "cluster_id": self.cluster_id,
            "is_outlier": self.is_outlier,
            "selected": self.selected,
            "manual_labels": [dict(label) for label in self.manual_labels],
            "metadata": dict(self.metadata),
            "color": self.color,
        }


@dataclass(frozen=True)
class ScatterplotRenderPayload:
    render_id: str
    dataset_id: str
    points: Tuple[ScatterplotPoint, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        render_id = clean_text(self.render_id, "render_id")
        dataset_id = clean_text(self.dataset_id, "dataset_id")
        points = tuple(self.points)

        if not points:
            raise ValueError("points must not be empty")

        if not all(isinstance(point, ScatterplotPoint) for point in points):
            raise ValueError("points must contain ScatterplotPoint objects")

        point_ids = [point.point_id for point in points]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("scatterplot points must have unique point_id values")

        if self.diagnostics is None or not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

        object.__setattr__(self, "render_id", render_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "render_id": self.render_id,
            "dataset_id": self.dataset_id,
            "points": [point.to_dict() for point in self.points],
            "diagnostics": dict(self.diagnostics),
        }
