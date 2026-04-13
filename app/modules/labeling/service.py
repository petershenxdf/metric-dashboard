from __future__ import annotations

from typing import Iterable

from app.modules.selection.schemas import SelectionContext
from app.shared.schemas import clean_text

from .schemas import LabelingState, ManualAnnotation, StructuredFeedbackInstruction
from .store import LabelingStore

MANUAL_LABEL_SOURCE = "manual_label"
SELECTED_POINTS_SCOPE = "selected_points"


def create_labeling_store(dataset_id: str) -> LabelingStore:
    return LabelingStore(dataset_id=dataset_id)


def get_labeling_state(store: LabelingStore) -> LabelingState:
    _validate_store(store)
    return LabelingState(dataset_id=store.dataset_id, annotations=tuple(store.annotations))


def list_annotations(store: LabelingStore):
    return get_labeling_state(store).annotations


def clear_annotations(store: LabelingStore) -> LabelingState:
    _validate_store(store)
    store.annotations.clear()
    return get_labeling_state(store)


def create_cluster_annotation(
    store: LabelingStore,
    selection_context: SelectionContext,
    target_label: str,
    point_ids: Iterable[str] | None = None,
) -> ManualAnnotation:
    return _create_annotation(
        store,
        selection_context,
        label_type="cluster",
        label_value=target_label,
        point_ids=point_ids,
    )


def create_new_class_annotation(
    store: LabelingStore,
    selection_context: SelectionContext,
    class_name: str,
    point_ids: Iterable[str] | None = None,
) -> ManualAnnotation:
    return _create_annotation(
        store,
        selection_context,
        label_type="class",
        label_value=class_name,
        point_ids=point_ids,
    )


def create_outlier_annotation(
    store: LabelingStore,
    selection_context: SelectionContext,
    point_ids: Iterable[str] | None = None,
) -> ManualAnnotation:
    return _create_annotation(
        store,
        selection_context,
        label_type="outlier",
        label_value=True,
        point_ids=point_ids,
    )


def create_not_outlier_annotation(
    store: LabelingStore,
    selection_context: SelectionContext,
    point_ids: Iterable[str] | None = None,
) -> ManualAnnotation:
    return _create_annotation(
        store,
        selection_context,
        label_type="outlier",
        label_value=False,
        point_ids=point_ids,
    )


def apply_labeling_action(
    store: LabelingStore,
    selection_context: SelectionContext,
    action: str,
    label_value: str | None = None,
    point_ids: Iterable[str] | None = None,
) -> ManualAnnotation:
    action = clean_text(action, "action")
    handlers = {
        "assign_cluster": lambda: create_cluster_annotation(store, selection_context, label_value or "", point_ids),
        "assign_new_class": lambda: create_new_class_annotation(store, selection_context, label_value or "", point_ids),
        "mark_outlier": lambda: create_outlier_annotation(store, selection_context, point_ids),
        "mark_not_outlier": lambda: create_not_outlier_annotation(store, selection_context, point_ids),
    }
    if action not in handlers:
        raise ValueError(f"unsupported labeling action: {action}")
    return handlers[action]()


def annotation_to_instruction(annotation: ManualAnnotation) -> StructuredFeedbackInstruction:
    if not isinstance(annotation, ManualAnnotation):
        raise ValueError("annotation must be a ManualAnnotation")

    target = {
        "source": annotation.scope,
        "point_ids": list(annotation.point_ids),
    }

    if annotation.label_type == "cluster":
        return StructuredFeedbackInstruction(
            instruction_type="assign_cluster",
            status="actionable",
            source=annotation.source,
            target=target,
            parameters={"target_type": "cluster", "target_label": annotation.label_value},
        )

    if annotation.label_type == "class":
        return StructuredFeedbackInstruction(
            instruction_type="assign_new_class",
            status="actionable",
            source=annotation.source,
            target=target,
            parameters={"target_type": "class", "target_label": annotation.label_value},
        )

    return StructuredFeedbackInstruction(
        instruction_type="is_outlier" if annotation.label_value else "not_outlier",
        status="actionable",
        source=annotation.source,
        target=target,
        parameters={"target_type": "outlier"},
    )


def _create_annotation(
    store: LabelingStore,
    selection_context: SelectionContext,
    label_type: str,
    label_value: str | bool,
    point_ids: Iterable[str] | None = None,
) -> ManualAnnotation:
    _validate_store(store)
    _validate_selection_context(store, selection_context)
    selected_point_ids = _selected_point_ids(selection_context, point_ids)

    annotation = ManualAnnotation(
        annotation_id=store.next_annotation_id(),
        dataset_id=selection_context.dataset_id,
        source=MANUAL_LABEL_SOURCE,
        scope=SELECTED_POINTS_SCOPE,
        point_ids=selected_point_ids,
        label_type=label_type,
        label_value=label_value,
    )
    store.validate_annotation(annotation)
    store.annotations.append(annotation)
    return annotation


def _selected_point_ids(selection_context: SelectionContext, point_ids: Iterable[str] | None):
    selected = tuple(selection_context.selected_point_ids)
    if point_ids is None:
        if not selected:
            raise ValueError("selection must include at least one point")
        return selected

    requested = tuple(clean_text(point_id, "point_id") for point_id in point_ids)
    if not requested:
        raise ValueError("point_ids must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("point_ids must be unique")

    unknown = set(requested) - (set(selection_context.selected_point_ids) | set(selection_context.unselected_point_ids))
    if unknown:
        raise ValueError(f"unknown point id(s): {', '.join(sorted(unknown))}")

    not_selected = set(requested) - set(selection_context.selected_point_ids)
    if not_selected:
        raise ValueError(f"point id(s) are not selected: {', '.join(sorted(not_selected))}")

    return requested


def _validate_selection_context(store: LabelingStore, selection_context: SelectionContext) -> None:
    if not isinstance(selection_context, SelectionContext):
        raise ValueError("selection_context must be a SelectionContext")
    if selection_context.dataset_id != store.dataset_id:
        raise ValueError("selection_context dataset_id must match store dataset_id")


def _validate_store(store: LabelingStore) -> None:
    if not isinstance(store, LabelingStore):
        raise ValueError("store must be a LabelingStore")
