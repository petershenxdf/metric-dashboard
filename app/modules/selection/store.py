from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set, Tuple

from app.shared.schemas import clean_text

from .schemas import SelectionGroup


@dataclass
class SelectionStore:
    dataset_id: str
    known_point_ids: Tuple[str, ...]
    selected_point_ids: Set[str] = field(default_factory=set)
    selection_groups: Dict[str, SelectionGroup] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_id = clean_text(self.dataset_id, "dataset_id")
        self.known_point_ids = _clean_known_point_ids(self.known_point_ids)
        self.selected_point_ids = set(_clean_point_ids(self.selected_point_ids, allow_empty=True))
        self.validate_known_points(self.selected_point_ids)
        self.selection_groups = dict(self.selection_groups)
        for group in self.selection_groups.values():
            self.validate_selection_group(group)

    def validate_known_points(self, point_ids: Iterable[str]) -> Tuple[str, ...]:
        cleaned = _clean_point_ids(point_ids, allow_empty=True)
        unknown = set(cleaned) - set(self.known_point_ids)
        if unknown:
            raise ValueError(f"selected point id is unknown: {', '.join(sorted(unknown))}")
        return cleaned

    def validate_selection_group(self, group: SelectionGroup) -> SelectionGroup:
        if not isinstance(group, SelectionGroup):
            raise ValueError("selection group must be a SelectionGroup")

        if group.dataset_id != self.dataset_id:
            raise ValueError("selection group dataset_id must match store dataset_id")

        self.validate_known_points(group.point_ids)
        return group

    def ordered_selected_point_ids(self) -> Tuple[str, ...]:
        return tuple(point_id for point_id in self.known_point_ids if point_id in self.selected_point_ids)


def _clean_known_point_ids(point_ids: Iterable[str]) -> Tuple[str, ...]:
    cleaned = _clean_point_ids(point_ids)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("known_point_ids must be unique")
    return cleaned


def _clean_point_ids(point_ids: Iterable[str], allow_empty: bool = False) -> Tuple[str, ...]:
    if isinstance(point_ids, (str, bytes)):
        raise ValueError("point_ids must be a sequence of point ids")

    cleaned = tuple(clean_text(point_id, "point_id") for point_id in point_ids)
    if not cleaned and not allow_empty:
        raise ValueError("point_ids must not be empty")

    if len(set(cleaned)) != len(cleaned):
        raise ValueError("point_ids must be unique")

    return cleaned
