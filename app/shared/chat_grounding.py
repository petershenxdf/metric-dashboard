from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from app.modules.algorithm_adapters.service import DEFAULT_N_CLUSTERS, run_default_analysis
from app.modules.data_workspace.service import create_feature_matrix
from app.modules.labeling.service import get_labeling_state
from app.modules.labeling.state import get_debug_store_for_context as get_labeling_store_for_context
from app.modules.projection.service import project_feature_matrix
from app.modules.scatterplot.service import build_render_payload
from app.modules.selection.service import get_selection_context, get_selection_state, list_selection_groups
from app.modules.selection.state import get_debug_store_for_dataset
from app.shared.effective_analysis import apply_manual_labels_to_analysis
from app.shared.fixtures import (
    DEFAULT_WORKFLOW_DATASET_ID,
    analysis_selection_dataset,
    analysis_selection_initial_selected_point_ids,
)


@dataclass(frozen=True)
class GroundedChatContext:
    dataset: object
    matrix: object
    projection: object
    raw_analysis: object
    provider_analysis: object
    analysis: object
    selection_store: object
    selection_state: object
    selection_context: object
    selection_groups: Tuple[object, ...]
    labeling_state: object
    render_payload: object
    n_clusters: int

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(getattr(self.dataset, "feature_names", ()))

    @property
    def cluster_ids(self) -> Tuple[str, ...]:
        seen = []
        for assignment in self.analysis.cluster_result.assignments:
            cluster_id = assignment.cluster_id
            if cluster_id not in seen:
                seen.append(cluster_id)
        return tuple(seen)

    @property
    def outlier_point_ids(self) -> Tuple[str, ...]:
        return tuple(self.analysis.outlier_result.outlier_point_ids)

    def analysis_context_payload(self) -> Dict[str, Any]:
        point_to_cluster = self._point_to_cluster()
        point_to_features = self._point_to_features()
        outlier_set = set(self.outlier_point_ids)
        return {
            "dataset_id": self.dataset.dataset_id,
            "feature_names": list(self.feature_names),
            "cluster_ids": list(self.cluster_ids),
            "cluster_count": len(self.cluster_ids),
            "cluster_sizes": self._cluster_sizes(),
            "feature_ranges": self._feature_ranges(),
            "label_summary": self._label_summary(),
            "outlier_point_ids": list(self.outlier_point_ids),
            "outlier_count": len(self.outlier_point_ids),
            "point_to_cluster": point_to_cluster,
            "point_catalog": [
                {
                    "point_id": point_id,
                    "cluster_id": point_to_cluster.get(point_id),
                    "is_outlier": point_id in outlier_set,
                    "features": point_to_features.get(point_id, {}),
                }
                for point_id in self.matrix.point_ids
            ],
            "selected_point_ids": list(self.selection_context.selected_point_ids),
            "selected_count": len(self.selection_context.selected_point_ids),
            "selected_point_clusters": [
                {
                    "point_id": point_id,
                    "cluster_id": point_to_cluster.get(point_id),
                    "is_outlier": point_id in outlier_set,
                    "features": point_to_features.get(point_id, {}),
                }
                for point_id in self.selection_context.selected_point_ids
            ],
            "unselected_point_ids": list(self.selection_context.unselected_point_ids),
            "unselected_count": len(self.selection_context.unselected_point_ids),
            "visible_point_count": len(self.render_payload.points),
        }

    def _point_to_cluster(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for assignment in self.analysis.cluster_result.assignments:
            mapping[assignment.point_id] = assignment.cluster_id
        return mapping

    def _point_to_features(self) -> Dict[str, Dict[str, float]]:
        names = tuple(self.matrix.feature_names)
        mapping: Dict[str, Dict[str, float]] = {}
        for point_id, row in zip(self.matrix.point_ids, self.matrix.values):
            mapping[point_id] = {
                name: round(float(value), 4) for name, value in zip(names, row)
            }
        return mapping

    def _cluster_sizes(self) -> Dict[str, int]:
        sizes: Dict[str, int] = {cluster_id: 0 for cluster_id in self.cluster_ids}
        for assignment in self.analysis.cluster_result.assignments:
            if assignment.cluster_id in sizes:
                sizes[assignment.cluster_id] += 1
            else:
                sizes[assignment.cluster_id] = 1
        return sizes

    def _feature_ranges(self) -> Dict[str, Dict[str, float]]:
        names = tuple(self.matrix.feature_names)
        if not names or not self.matrix.values:
            return {}
        ranges: Dict[str, Dict[str, float]] = {}
        for index, name in enumerate(names):
            column = [row[index] for row in self.matrix.values]
            ranges[name] = {
                "min": round(min(column), 4),
                "max": round(max(column), 4),
                "mean": round(sum(column) / len(column), 4),
            }
        return ranges

    def _label_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for annotation in self.labeling_state.annotations:
            label_type = getattr(annotation, "label_type", None)
            label_value = getattr(annotation, "label_value", None)
            if not label_type:
                continue
            key = f"{label_type}:{label_value}" if label_value is not None else label_type
            summary[key] = summary.get(key, 0) + len(getattr(annotation, "point_ids", ()))
        return summary

    def label_context_payload(self) -> Dict[str, Any]:
        annotations = [annotation.to_dict() for annotation in self.labeling_state.annotations]
        return {
            "dataset_id": self.labeling_state.dataset_id,
            "annotation_count": len(annotations),
            "active_annotations": annotations,
            "analysis_context": self.analysis_context_payload(),
        }

    def state_payload(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "projection": self.projection.to_dict(),
            "raw_clusters": self.raw_analysis.cluster_result.to_dict(),
            "raw_outliers": self.raw_analysis.outlier_result.to_dict(),
            "provider_clusters": self.provider_analysis.cluster_result.to_dict(),
            "provider_outliers": self.provider_analysis.outlier_result.to_dict(),
            "clusters": self.analysis.cluster_result.to_dict(),
            "outliers": self.analysis.outlier_result.to_dict(),
            "selection": self.selection_state.to_dict(),
            "selection_context": self.selection_context.to_dict(),
            "selection_groups": [group.to_dict() for group in self.selection_groups],
            "labeling": self.labeling_state.to_dict(),
            "label_context": self.label_context_payload(),
            "analysis_context": self.analysis_context_payload(),
            "render_payload": self.render_payload.to_dict(),
        }


def build_grounded_chat_context(
    dataset_id: str = DEFAULT_WORKFLOW_DATASET_ID,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> GroundedChatContext:
    dataset = analysis_selection_dataset(dataset_id)
    matrix = create_feature_matrix(dataset)
    projection = project_feature_matrix(matrix)
    selection_store = get_debug_store_for_dataset(
        dataset,
        analysis_selection_initial_selected_point_ids(dataset.dataset_id),
    )
    selection_state = get_selection_state(selection_store)
    selection_context = get_selection_context(selection_store)
    selection_groups = tuple(list_selection_groups(selection_store))
    labeling_state = get_labeling_state(get_labeling_store_for_context(selection_context))
    provider_labeling_state = labeling_state if labeling_state.annotations else None
    raw_analysis = run_default_analysis(matrix, n_clusters=n_clusters)
    provider_analysis = run_default_analysis(
        matrix,
        n_clusters=n_clusters,
        labeling_state=provider_labeling_state,
    )
    analysis = apply_manual_labels_to_analysis(dataset, provider_analysis, labeling_state)
    render_payload = build_render_payload(
        dataset=dataset,
        projection=projection,
        clusters=analysis.cluster_result,
        outliers=analysis.outlier_result,
        selection_context=selection_context,
        labeling_state=labeling_state,
    )
    return GroundedChatContext(
        dataset=dataset,
        matrix=matrix,
        projection=projection,
        raw_analysis=raw_analysis,
        provider_analysis=provider_analysis,
        analysis=analysis,
        selection_store=selection_store,
        selection_state=selection_state,
        selection_context=selection_context,
        selection_groups=selection_groups,
        labeling_state=labeling_state,
        render_payload=render_payload,
        n_clusters=n_clusters,
    )
