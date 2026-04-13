from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.shared.schemas import clean_text

LABELING_ACTION_TYPES = ("assign_cluster", "assign_new_class", "mark_outlier", "mark_not_outlier")
LABEL_TYPES = ("cluster", "class", "outlier")
ANNOTATION_STATUSES = ("active",)


@dataclass(frozen=True)
class ManualAnnotation:
    annotation_id: str
    dataset_id: str
    source: str
    scope: str
    point_ids: Tuple[str, ...]
    label_type: str
    label_value: str | bool
    status: str = "active"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        annotation_id = clean_text(self.annotation_id, "annotation_id")
        dataset_id = clean_text(self.dataset_id, "dataset_id")
        source = clean_text(self.source, "source")
        scope = clean_text(self.scope, "scope")
        point_ids = _clean_unique_point_ids(self.point_ids, "point_ids")
        label_type = clean_text(self.label_type, "label_type")
        if label_type not in LABEL_TYPES:
            raise ValueError(f"unsupported label_type: {label_type}")

        if label_type == "outlier":
            if not isinstance(self.label_value, bool):
                raise ValueError("outlier label_value must be a boolean")
            label_value = self.label_value
        else:
            label_value = clean_text(self.label_value, "label_value")

        status = clean_text(self.status, "status")
        if status not in ANNOTATION_STATUSES:
            raise ValueError(f"unsupported annotation status: {status}")

        if self.metadata is None or not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        object.__setattr__(self, "annotation_id", annotation_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "label_type", label_type)
        object.__setattr__(self, "label_value", label_value)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "dataset_id": self.dataset_id,
            "source": self.source,
            "scope": self.scope,
            "point_ids": list(self.point_ids),
            "point_count": len(self.point_ids),
            "label_type": self.label_type,
            "label_value": self.label_value,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StructuredFeedbackInstruction:
    instruction_type: str
    status: str
    source: str
    target: Mapping[str, Any]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruction_type", clean_text(self.instruction_type, "instruction_type"))
        object.__setattr__(self, "status", clean_text(self.status, "status"))
        object.__setattr__(self, "source", clean_text(self.source, "source"))
        if self.target is None or not isinstance(self.target, Mapping):
            raise ValueError("target must be a mapping")
        if self.parameters is None or not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        object.__setattr__(self, "target", dict(self.target))
        object.__setattr__(self, "parameters", dict(self.parameters))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_type": self.instruction_type,
            "status": self.status,
            "source": self.source,
            "target": dict(self.target),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class LabelingState:
    dataset_id: str
    annotations: Tuple[ManualAnnotation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        dataset_id = clean_text(self.dataset_id, "dataset_id")
        annotations = tuple(self.annotations)
        if not all(isinstance(annotation, ManualAnnotation) for annotation in annotations):
            raise ValueError("annotations must contain ManualAnnotation objects")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "annotations", annotations)

    @property
    def instructions(self) -> Tuple[StructuredFeedbackInstruction, ...]:
        from .service import annotation_to_instruction

        return tuple(annotation_to_instruction(annotation) for annotation in self.annotations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "annotation_count": len(self.annotations),
            "structured_feedback": [instruction.to_dict() for instruction in self.instructions],
        }


def _clean_unique_point_ids(point_ids: object, field_name: str) -> Tuple[str, ...]:
    if isinstance(point_ids, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of point ids")

    try:
        cleaned = tuple(clean_text(point_id, "point_id") for point_id in point_ids)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of point ids") from exc

    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} must be unique")
    return cleaned
