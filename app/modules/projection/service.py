from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Tuple

from app.shared.schemas import FeatureMatrix, ProjectionCoordinate, ProjectionResult

from .mds import classical_mds

MDS_METHOD = "mds"
SVG_WIDTH = 860
SVG_HEIGHT = 520
SVG_PADDING = 56
GROUP_COLORS = ["#2f6fed", "#188038", "#d93025", "#f9ab00", "#9334e6", "#00897b"]
DEFAULT_COLOR = "#2f6fed"


def project_feature_matrix(
    feature_matrix: FeatureMatrix,
    projection_id: str | None = None,
) -> ProjectionResult:
    if not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")

    coordinates = classical_mds(feature_matrix.values, n_components=2)
    projection_coordinates = tuple(
        ProjectionCoordinate(
            point_id=point_id,
            x=float(coordinates[index, 0]),
            y=float(coordinates[index, 1]),
        )
        for index, point_id in enumerate(feature_matrix.point_ids)
    )

    return ProjectionResult(
        projection_id=projection_id or _projection_id(feature_matrix),
        method=MDS_METHOD,
        coordinates=projection_coordinates,
    )


def scaled_projection_points(
    projection: ProjectionResult,
    labels_by_point_id: Mapping[str, str] | None = None,
) -> Tuple[Dict[str, object], ...]:
    labels = dict(labels_by_point_id or {})
    colors = _color_map(labels.values())
    xs = [coordinate.x for coordinate in projection.coordinates]
    ys = [coordinate.y for coordinate in projection.coordinates]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)

    scaled = []
    for coordinate in projection.coordinates:
        label = labels.get(coordinate.point_id, "")
        scaled.append(
            {
                "point_id": coordinate.point_id,
                "label": label,
                "x": coordinate.x,
                "y": coordinate.y,
                "screen_x": SVG_PADDING
                + ((coordinate.x - min_x) / x_span) * (SVG_WIDTH - 2 * SVG_PADDING),
                "screen_y": SVG_HEIGHT
                - SVG_PADDING
                - ((coordinate.y - min_y) / y_span) * (SVG_HEIGHT - 2 * SVG_PADDING),
                "color": colors.get(label, DEFAULT_COLOR),
            }
        )

    return tuple(scaled)


def _projection_id(feature_matrix: FeatureMatrix) -> str:
    payload = {
        "point_ids": list(feature_matrix.point_ids),
        "feature_names": list(feature_matrix.feature_names),
        "values": [list(row) for row in feature_matrix.values],
        "method": MDS_METHOD,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    return f"projection_{MDS_METHOD}_{digest}"


def _color_map(labels) -> Dict[str, str]:
    unique_labels = sorted({label for label in labels if label})
    return {
        label: GROUP_COLORS[index % len(GROUP_COLORS)]
        for index, label in enumerate(unique_labels)
    }
