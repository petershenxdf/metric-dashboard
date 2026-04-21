from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from app.shared.schemas import InstructionSnapshot, clean_text

# Intent types are kept aligned with chatbox's definition so suggestion chips,
# the mock keyword router, and downstream adapters all use the same identifiers.
INTENT_TYPES_SHARED = (
    "feature_weight",
    "group_similar",
    "group_dissimilar",
    "merge_clusters",
    "anchor_point",
    "ignore_cluster",
)
INTENT_TYPES_PATH_B_ONLY = ("split_cluster", "reclassify_outlier")
ALL_INTENT_TYPES = INTENT_TYPES_SHARED + INTENT_TYPES_PATH_B_ONLY

ROUTER_CATEGORIES = (
    "on_topic_actionable",
    "on_topic_ambiguous",
    "partial",
    "meta_query",
    "off_topic",
)

GROUP_REF_SOURCES = (
    "selection_group",
    "cluster",
    "outlier_set",
    "selected_points",
    "point_id",
)

DELTA_OPS = ("add", "remove", "modify")

FEATURE_WEIGHT_DIRECTIONS = ("increase", "decrease", "ignore")


@dataclass(frozen=True)
class GroupRef:
    source: str
    ref: str

    def __post_init__(self) -> None:
        source = clean_text(self.source, "source")
        ref = clean_text(self.ref, "ref")
        if source not in GROUP_REF_SOURCES:
            raise ValueError(f"unsupported group ref source: {source}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "ref", ref)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "ref": self.ref}


@dataclass(frozen=True)
class ConstraintEntry:
    """One structured-instruction entry.

    Only a subset of fields is required per intent; unused fields stay as
    ``None`` or empty tuples. The downstream adapters filter by
    ``intent`` and read the relevant fields.
    """

    id: str
    intent: str
    raw_message: str
    feature: Optional[str] = None
    direction: Optional[str] = None
    group_a: Optional[GroupRef] = None
    group_b: Optional[GroupRef] = None
    anchor: Optional[GroupRef] = None
    target_groups: Tuple[GroupRef, ...] = field(default_factory=tuple)
    is_outlier: Optional[bool] = None

    def __post_init__(self) -> None:
        entry_id = clean_text(self.id, "id")
        intent = clean_text(self.intent, "intent")
        raw_message = clean_text(self.raw_message, "raw_message")
        if intent not in ALL_INTENT_TYPES:
            raise ValueError(f"unsupported intent: {intent}")
        feature = clean_text(self.feature, "feature") if self.feature else None
        direction = self.direction
        if direction is not None and direction not in FEATURE_WEIGHT_DIRECTIONS:
            raise ValueError(f"unsupported feature direction: {direction}")
        target_groups = tuple(self.target_groups or ())
        for group in target_groups:
            if not isinstance(group, GroupRef):
                raise ValueError("target_groups must contain GroupRef objects")
        is_outlier = self.is_outlier
        if is_outlier is not None and not isinstance(is_outlier, bool):
            raise ValueError("is_outlier must be a bool")
        object.__setattr__(self, "id", entry_id)
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "raw_message", raw_message)
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "target_groups", target_groups)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "intent": self.intent,
            "raw_message": self.raw_message,
            "feature": self.feature,
            "direction": self.direction,
            "group_a": self.group_a.to_dict() if self.group_a else None,
            "group_b": self.group_b.to_dict() if self.group_b else None,
            "anchor": self.anchor.to_dict() if self.anchor else None,
            "target_groups": [group.to_dict() for group in self.target_groups],
            "is_outlier": self.is_outlier,
        }


@dataclass(frozen=True)
class DeltaOperation:
    op: str
    constraint_id: str
    intent: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        op = clean_text(self.op, "op")
        constraint_id = clean_text(self.constraint_id, "constraint_id")
        if op not in DELTA_OPS:
            raise ValueError(f"unsupported delta op: {op}")
        intent = clean_text(self.intent, "intent") if self.intent else None
        if intent is not None and intent not in ALL_INTENT_TYPES:
            raise ValueError(f"unsupported intent: {intent}")
        payload = dict(self.payload or {})
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "constraint_id", constraint_id)
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "payload", payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "constraint_id": self.constraint_id,
            "intent": self.intent,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class InstructionDelta:
    operations: Tuple[DeltaOperation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ops = tuple(self.operations or ())
        for op in ops:
            if not isinstance(op, DeltaOperation):
                raise ValueError("operations must contain DeltaOperation objects")
        object.__setattr__(self, "operations", ops)

    def to_dict(self) -> Dict[str, Any]:
        return {"operations": [op.to_dict() for op in self.operations]}

    @property
    def is_empty(self) -> bool:
        return not self.operations


@dataclass(frozen=True)
class StructuredInstruction:
    """Full internal state owned by the intent_instruction module."""

    version: int
    constraints: Tuple[ConstraintEntry, ...] = field(default_factory=tuple)
    last_delta: Optional[InstructionDelta] = None
    router_category: Optional[str] = None
    raw_message: Optional[str] = None
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("version must be a non-negative integer")
        constraints = tuple(self.constraints or ())
        for constraint in constraints:
            if not isinstance(constraint, ConstraintEntry):
                raise ValueError("constraints must contain ConstraintEntry objects")
        router_category = (
            clean_text(self.router_category, "router_category")
            if self.router_category
            else None
        )
        if router_category is not None and router_category not in ROUTER_CATEGORIES:
            raise ValueError(f"unsupported router_category: {router_category}")
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "router_category", router_category)

    def snapshot(self) -> InstructionSnapshot:
        return InstructionSnapshot(
            version=self.version,
            constraints=tuple(constraint.to_dict() for constraint in self.constraints),
            last_delta=self.last_delta.to_dict() if self.last_delta else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "last_delta": self.last_delta.to_dict() if self.last_delta else None,
            "router_category": self.router_category,
            "raw_message": self.raw_message,
            "clarification_needed": self.clarification_needed,
            "clarification_question": self.clarification_question,
            "confidence": self.confidence,
            "constraint_count": len(self.constraints),
        }


@dataclass(frozen=True)
class DatasetContext:
    dataset_id: str
    feature_names: Tuple[str, ...] = field(default_factory=tuple)
    cluster_ids: Tuple[str, ...] = field(default_factory=tuple)
    outlier_point_ids: Tuple[str, ...] = field(default_factory=tuple)
    selection_group_names: Tuple[str, ...] = field(default_factory=tuple)
    selection_groups: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    label_annotations: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    analysis_context: Mapping[str, Any] = field(default_factory=dict)
    selected_point_ids: Tuple[str, ...] = field(default_factory=tuple)
    unselected_point_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", clean_text(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "feature_names", tuple(self.feature_names or ()))
        object.__setattr__(self, "cluster_ids", tuple(self.cluster_ids or ()))
        object.__setattr__(self, "outlier_point_ids", tuple(self.outlier_point_ids or ()))
        object.__setattr__(
            self, "selection_group_names", tuple(self.selection_group_names or ())
        )
        object.__setattr__(self, "selection_groups", tuple(dict(group) for group in (self.selection_groups or ())))
        object.__setattr__(self, "label_annotations", tuple(dict(annotation) for annotation in (self.label_annotations or ())))
        object.__setattr__(self, "analysis_context", dict(self.analysis_context or {}))
        object.__setattr__(
            self, "selected_point_ids", tuple(self.selected_point_ids or ())
        )
        object.__setattr__(
            self, "unselected_point_ids", tuple(self.unselected_point_ids or ())
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "feature_names": list(self.feature_names),
            "cluster_ids": list(self.cluster_ids),
            "cluster_count": len(self.cluster_ids),
            "outlier_point_ids": list(self.outlier_point_ids),
            "outlier_count": len(self.outlier_point_ids),
            "selection_group_names": list(self.selection_group_names),
            "selection_groups": [dict(group) for group in self.selection_groups],
            "label_annotations": [dict(annotation) for annotation in self.label_annotations],
            "analysis_context": dict(self.analysis_context),
            "selected_point_ids": list(self.selected_point_ids),
            "unselected_point_ids": list(self.unselected_point_ids),
            "has_selection": bool(self.selected_point_ids),
            "has_named_groups": bool(self.selection_group_names),
        }


@dataclass(frozen=True)
class Turn:
    role: str
    text: str

    def __post_init__(self) -> None:
        role = clean_text(self.role, "role")
        text = clean_text(self.text, "text")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "text", text)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "text": self.text}


@dataclass(frozen=True)
class RouterResult:
    category: str
    confidence: float
    reason: Optional[str] = None
    clarification_question: Optional[str] = None
    reply_text: Optional[str] = None

    def __post_init__(self) -> None:
        category = clean_text(self.category, "category")
        if category not in ROUTER_CATEGORIES:
            raise ValueError(f"unsupported router category: {category}")
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number")
        confidence = float(self.confidence)
        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "reason": self.reason,
            "clarification_question": self.clarification_question,
            "reply_text": self.reply_text,
        }
