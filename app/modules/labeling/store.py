from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.shared.schemas import clean_text

from .schemas import ManualAnnotation


@dataclass
class LabelingStore:
    dataset_id: str
    annotations: List[ManualAnnotation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dataset_id = clean_text(self.dataset_id, "dataset_id")
        self.annotations = list(self.annotations)
        for annotation in self.annotations:
            self.validate_annotation(annotation)

    def next_annotation_id(self) -> str:
        return f"annotation_{len(self.annotations) + 1:03d}"

    def validate_annotation(self, annotation: ManualAnnotation) -> ManualAnnotation:
        if not isinstance(annotation, ManualAnnotation):
            raise ValueError("annotation must be a ManualAnnotation")
        if annotation.dataset_id != self.dataset_id:
            raise ValueError("annotation dataset_id must match store dataset_id")
        return annotation
