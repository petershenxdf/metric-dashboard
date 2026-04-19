from __future__ import annotations

from typing import Any, Dict, List

from app.modules.labeling.service import get_labeling_state
from app.modules.labeling.state import get_debug_store_for_context as get_labeling_store_for_context
from app.modules.selection.fixtures import initial_selected_point_ids, selection_fixture_dataset
from app.modules.selection.service import get_selection_context, list_selection_groups
from app.modules.selection.state import get_debug_store_for_dataset


def current_selection_context():
    dataset = selection_fixture_dataset()
    store = get_debug_store_for_dataset(dataset, initial_selected_point_ids())
    return get_selection_context(store)


def current_selection_groups():
    dataset = selection_fixture_dataset()
    store = get_debug_store_for_dataset(dataset, initial_selected_point_ids())
    return list_selection_groups(store)


def current_label_context() -> Dict[str, Any]:
    """Summarized read of labeling state for chatbox context display."""
    context = current_selection_context()
    labeling_store = get_labeling_store_for_context(context)
    state = get_labeling_state(labeling_store)
    annotations = [annotation.to_dict() for annotation in state.annotations]
    return {
        "dataset_id": state.dataset_id,
        "annotation_count": len(annotations),
        "active_annotations": annotations,
    }


def example_messages() -> List[Dict[str, str]]:
    return [
        {"label": "Group similar", "text": "Treat these points as similar"},
        {"label": "Group dissimilar", "text": "Push cluster 1 and cluster 3 apart"},
        {"label": "Merge clusters", "text": "Merge clusters 1 and 2"},
        {"label": "Feature weight", "text": "Make petal_length more important"},
        {"label": "Anchor point", "text": "Treat p42 as a typical example for cluster 2"},
        {"label": "Ignore cluster", "text": "Ignore cluster 5 for now"},
        {"label": "Split (Path B)", "text": "Split cluster 2 into two"},
        {"label": "Reclassify outlier (Path B)", "text": "Reclassify p7 as not an outlier"},
        {"label": "Off-topic", "text": "What's the weather today?"},
        {"label": "Meta query", "text": "How many clusters are there?"},
    ]
