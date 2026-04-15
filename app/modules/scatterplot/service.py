from __future__ import annotations

import hashlib
import json
from typing import Mapping

from app.modules.algorithm_adapters.schemas import ClusterResult, OutlierResult
from app.modules.algorithm_adapters.service import assignments_by_point_id
from app.modules.labeling.schemas import LabelingState
from app.modules.projection.service import scaled_projection_points
from app.modules.selection.schemas import SelectionContext
from app.shared.schemas import Dataset, ProjectionResult

from .schemas import ScatterplotPoint, ScatterplotRenderPayload


def build_render_payload(
    dataset: Dataset,
    projection: ProjectionResult,
    clusters: ClusterResult,
    outliers: OutlierResult,
    selection_context: SelectionContext | None = None,
    labeling_state: LabelingState | None = None,
    render_id: str | None = None,
) -> ScatterplotRenderPayload:
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be a Dataset")
    if not isinstance(projection, ProjectionResult):
        raise ValueError("projection must be a ProjectionResult")
    if not isinstance(clusters, ClusterResult):
        raise ValueError("clusters must be a ClusterResult")
    if not isinstance(outliers, OutlierResult):
        raise ValueError("outliers must be an OutlierResult")
    if selection_context is not None and not isinstance(selection_context, SelectionContext):
        raise ValueError("selection_context must be a SelectionContext")
    if labeling_state is not None and not isinstance(labeling_state, LabelingState):
        raise ValueError("labeling_state must be a LabelingState")

    dataset_point_ids = tuple(point.point_id for point in dataset.points)
    projection_point_ids = tuple(coordinate.point_id for coordinate in projection.coordinates)
    if set(dataset_point_ids) != set(projection_point_ids):
        raise ValueError("dataset and projection must contain the same point ids")

    cluster_labels = dict(assignments_by_point_id(clusters))
    outlier_ids = set(outliers.outlier_point_ids)
    selected_ids = set(selection_context.selected_point_ids if selection_context is not None else ())
    point_by_id = {point.point_id: point for point in dataset.points}
    manual_labels_by_point_id = _manual_labels_by_point_id(labeling_state)

    points = []
    for scaled in scaled_projection_points(projection, cluster_labels):
        source_point = point_by_id[scaled["point_id"]]
        points.append(
            ScatterplotPoint(
                point_id=scaled["point_id"],
                x=scaled["x"],
                y=scaled["y"],
                screen_x=scaled["screen_x"],
                screen_y=scaled["screen_y"],
                cluster_id=cluster_labels.get(scaled["point_id"]),
                is_outlier=scaled["point_id"] in outlier_ids,
                selected=scaled["point_id"] in selected_ids,
                manual_labels=manual_labels_by_point_id.get(scaled["point_id"], ()),
                metadata=source_point.metadata,
                color=scaled["color"],
            )
        )

    return ScatterplotRenderPayload(
        render_id=render_id or _render_id(dataset, projection, clusters, outliers, selection_context, labeling_state),
        dataset_id=dataset.dataset_id,
        points=tuple(points),
        diagnostics={
            "projection_id": projection.projection_id,
            "cluster_run_id": clusters.cluster_run_id,
            "outlier_run_id": outliers.outlier_run_id,
            "selected_count": len(selected_ids),
            "annotation_count": 0 if labeling_state is None else len(labeling_state.annotations),
        },
    )


def selected_point_ids(render_payload: ScatterplotRenderPayload):
    if not isinstance(render_payload, ScatterplotRenderPayload):
        raise ValueError("render_payload must be a ScatterplotRenderPayload")
    return tuple(point.point_id for point in render_payload.points if point.selected)


def _manual_labels_by_point_id(labeling_state: LabelingState | None) -> Mapping[str, tuple]:
    if labeling_state is None:
        return {}

    labels = {}
    for annotation in labeling_state.annotations:
        display_label = _display_label(annotation)
        label_payload = {
            "annotation_id": annotation.annotation_id,
            "label_type": annotation.label_type,
            "label_value": annotation.label_value,
            "display_label": display_label,
        }
        for point_id in annotation.point_ids:
            labels.setdefault(point_id, []).append(label_payload)

    return {point_id: tuple(point_labels) for point_id, point_labels in labels.items()}


def _display_label(annotation) -> str | bool:
    if annotation.label_type == "outlier" and annotation.label_value is True:
        return "outlier"
    if annotation.label_type == "outlier" and annotation.label_value is False:
        return "not_outlier"
    return annotation.label_value


def _render_id(
    dataset: Dataset,
    projection: ProjectionResult,
    clusters: ClusterResult,
    outliers: OutlierResult,
    selection_context: SelectionContext | None,
    labeling_state: LabelingState | None,
) -> str:
    payload = {
        "dataset_id": dataset.dataset_id,
        "projection_id": projection.projection_id,
        "cluster_run_id": clusters.cluster_run_id,
        "outlier_run_id": outliers.outlier_run_id,
        "selected_point_ids": [] if selection_context is None else list(selection_context.selected_point_ids),
        "annotations": [] if labeling_state is None else [annotation.to_dict() for annotation in labeling_state.annotations],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"scatterplot_{hashlib.sha1(encoded).hexdigest()[:12]}"
