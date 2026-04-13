from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.shared.schemas import clean_text

SELECTION_ACTION_TYPES = ("select", "deselect", "replace", "toggle", "clear")
SELECTION_SOURCES = (
    "api",
    "point_click",
    "lasso",
    "rectangle",
    "manual_list",
    "workflow_fixture",
    "selection_group",
)
SELECTION_MODES = (
    "additive",
    "subtractive",
    "replace",
    "toggle",
    "clear",
    "single",
    "multi",
)


@dataclass(frozen=True)
class SelectionState:
    dataset_id: str
    known_point_ids: Tuple[str, ...]
    selected_point_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        dataset_id = clean_text(self.dataset_id, "dataset_id")
        known = _clean_unique_point_ids(self.known_point_ids, "known_point_ids")
        selected = _clean_unique_point_ids(self.selected_point_ids, "selected_point_ids", allow_empty=True)
        unknown = set(selected) - set(known)

        if unknown:
            raise ValueError(f"selected point id is unknown: {', '.join(sorted(unknown))}")

        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "known_point_ids", known)
        object.__setattr__(self, "selected_point_ids", selected)

    @property
    def unselected_point_ids(self) -> Tuple[str, ...]:
        selected = set(self.selected_point_ids)
        return tuple(point_id for point_id in self.known_point_ids if point_id not in selected)

    def to_context(self) -> "SelectionContext":
        return SelectionContext(
            dataset_id=self.dataset_id,
            selected_point_ids=self.selected_point_ids,
            unselected_point_ids=self.unselected_point_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        unselected = self.unselected_point_ids
        return {
            "dataset_id": self.dataset_id,
            "known_point_ids": list(self.known_point_ids),
            "selected_point_ids": list(self.selected_point_ids),
            "unselected_point_ids": list(unselected),
            "selected_count": len(self.selected_point_ids),
            "unselected_count": len(unselected),
        }


@dataclass(frozen=True)
class SelectionContext:
    dataset_id: str
    selected_point_ids: Tuple[str, ...]
    unselected_point_ids: Tuple[str, ...]
    source: str = "selection"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", clean_text(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "source", clean_text(self.source, "source"))
        object.__setattr__(
            self,
            "selected_point_ids",
            _clean_unique_point_ids(self.selected_point_ids, "selected_point_ids", allow_empty=True),
        )
        object.__setattr__(
            self,
            "unselected_point_ids",
            _clean_unique_point_ids(self.unselected_point_ids, "unselected_point_ids", allow_empty=True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "dataset_id": self.dataset_id,
            "selected_point_ids": list(self.selected_point_ids),
            "unselected_point_ids": list(self.unselected_point_ids),
            "selected_count": len(self.selected_point_ids),
            "unselected_count": len(self.unselected_point_ids),
        }


@dataclass(frozen=True)
class SelectionGroup:
    group_id: str
    group_name: str
    dataset_id: str
    point_ids: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        group_id = clean_text(self.group_id, "group_id")
        group_name = clean_text(self.group_name, "group_name")
        dataset_id = clean_text(self.dataset_id, "dataset_id")
        point_ids = _clean_unique_point_ids(self.point_ids, "point_ids")

        if self.metadata is None or not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "group_name", group_name)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "dataset_id": self.dataset_id,
            "point_ids": list(self.point_ids),
            "point_count": len(self.point_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SelectionAction:
    action: str
    point_ids: Tuple[str, ...] = field(default_factory=tuple)
    source: str = "api"
    mode: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = clean_text(self.action, "action")
        if action not in SELECTION_ACTION_TYPES:
            raise ValueError(f"unsupported selection action: {action}")

        source = clean_text(self.source, "source")
        if source not in SELECTION_SOURCES:
            raise ValueError(f"unsupported selection source: {source}")

        if self.mode is None:
            mode = _default_mode_for_action(action)
        else:
            mode = clean_text(self.mode, "mode")

        if mode not in SELECTION_MODES:
            raise ValueError(f"unsupported selection mode: {mode}")

        point_ids = _clean_unique_point_ids(self.point_ids, "point_ids", allow_empty=(action == "clear"))
        if action != "clear" and not point_ids:
            raise ValueError("point_ids must not be empty for this action")

        if self.metadata is None or not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        object.__setattr__(self, "action", action)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "point_ids": list(self.point_ids),
            "source": self.source,
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SelectionActionResult:
    action: SelectionAction
    state: SelectionState
    context: SelectionContext

    def __post_init__(self) -> None:
        if not isinstance(self.action, SelectionAction):
            raise ValueError("action must be a SelectionAction")

        if not isinstance(self.state, SelectionState):
            raise ValueError("state must be a SelectionState")

        if not isinstance(self.context, SelectionContext):
            raise ValueError("context must be a SelectionContext")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "state": self.state.to_dict(),
            "context": self.context.to_dict(),
        }


def _clean_unique_point_ids(
    point_ids: object,
    field_name: str,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    if isinstance(point_ids, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of point ids")

    if not isinstance(point_ids, tuple):
        try:
            point_ids = tuple(point_ids)
        except TypeError as exc:
            raise ValueError(f"{field_name} must be a sequence of point ids") from exc

    cleaned = tuple(clean_text(point_id, "point_id") for point_id in point_ids)
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")

    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} must be unique")

    return cleaned


def _default_mode_for_action(action: str) -> str:
    return {
        "select": "additive",
        "deselect": "subtractive",
        "replace": "replace",
        "toggle": "toggle",
        "clear": "clear",
    }[action]
