from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from app.modules.selection.fixtures import initial_selected_point_ids, selection_fixture_dataset

from .schemas import DatasetContext


DEFAULT_FEATURE_NAMES = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)

DEFAULT_CLUSTER_IDS = ("cluster_1", "cluster_2", "cluster_3")


def default_feature_names(dataset_id: str) -> Sequence[str]:
    """Best-effort feature names for building ``DatasetContext``.

    Reads feature names from the selection fixture dataset when
    available; falls back to the canonical Iris-style names used by the
    other module debug pages.
    """

    try:
        dataset = selection_fixture_dataset()
        if dataset and dataset.dataset_id == dataset_id and dataset.feature_names:
            return tuple(dataset.feature_names)
    except Exception:  # pragma: no cover - fixture is deterministic in tests
        pass
    return DEFAULT_FEATURE_NAMES


def demo_dataset_context() -> DatasetContext:
    dataset = selection_fixture_dataset()
    all_point_ids = tuple(point.point_id for point in dataset.points)
    selected = tuple(initial_selected_point_ids())
    unselected = tuple(p for p in all_point_ids if p not in set(selected))
    return DatasetContext(
        dataset_id=dataset.dataset_id,
        feature_names=tuple(dataset.feature_names),
        cluster_ids=DEFAULT_CLUSTER_IDS,
        selection_group_names=("Group A",),
        selected_point_ids=selected,
        unselected_point_ids=unselected,
    )


def example_messages() -> List[Dict[str, str]]:
    return [
        {"intent": "feature_weight", "label": "Feature weight",
         "text": "Make petal_length more important"},
        {"intent": "group_similar", "label": "Group similar",
         "text": "Treat these points as similar"},
        {"intent": "group_dissimilar", "label": "Group dissimilar",
         "text": "Push cluster 1 and cluster 3 apart"},
        {"intent": "merge_clusters", "label": "Merge clusters",
         "text": "Merge clusters 1 and 2"},
        {"intent": "anchor_point", "label": "Anchor point",
         "text": "Treat p42 as a typical example for cluster 2"},
        {"intent": "ignore_cluster", "label": "Ignore cluster",
         "text": "Ignore cluster 3 for now"},
        {"intent": "split_cluster", "label": "Split (Path B)",
         "text": "Split cluster 2 into two"},
        {"intent": "reclassify_outlier", "label": "Reclassify (Path B)",
         "text": "p3 is not an outlier"},
        {"intent": "off_topic", "label": "Off-topic",
         "text": "What's the weather today?"},
        {"intent": "meta_query", "label": "Meta query",
         "text": "How many clusters are there?"},
        {"intent": "ambiguous", "label": "Ambiguous (empty selection)",
         "text": "Push these points apart"},
    ]


def router_category_order() -> Tuple[str, ...]:
    return (
        "on_topic_actionable",
        "on_topic_ambiguous",
        "partial",
        "meta_query",
        "off_topic",
    )
