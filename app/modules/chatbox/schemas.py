from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from app.shared.schemas import clean_text

CHAT_ROLES = ("user", "assistant")

ROUTER_CATEGORIES = (
    "on_topic_actionable",
    "on_topic_ambiguous",
    "partial",
    "meta_query",
    "off_topic",
)

REFINEMENT_STRATEGIES = ("metric_learning", "direct_ssdbcodi")

DEFAULT_REFINEMENT_STRATEGY = "metric_learning"

DEFAULT_HISTORY_WINDOW = 3

# Intent types emitted by the (mocked) intent provider in Step 7.
# These are the same names used in the process/design docs for Step 8.
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


@dataclass(frozen=True)
class ChatTurn:
    turn_id: str
    role: str
    text: str

    def __post_init__(self) -> None:
        turn_id = clean_text(self.turn_id, "turn_id")
        role = clean_text(self.role, "role")
        if role not in CHAT_ROLES:
            raise ValueError(f"unsupported chat role: {role}")
        text = clean_text(self.text, "text")
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "text", text)

    def to_dict(self) -> Dict[str, Any]:
        return {"turn_id": self.turn_id, "role": self.role, "text": self.text}


@dataclass(frozen=True)
class SuggestionChip:
    text: str
    intent_type: str
    requires_path_b: bool = False

    def __post_init__(self) -> None:
        text = clean_text(self.text, "text")
        intent_type = clean_text(self.intent_type, "intent_type")
        if intent_type not in ALL_INTENT_TYPES:
            raise ValueError(f"unsupported intent_type: {intent_type}")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "intent_type", intent_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "intent_type": self.intent_type,
            "requires_path_b": self.requires_path_b,
        }


@dataclass(frozen=True)
class MockInstructionSnapshot:
    version: int
    constraints: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    last_delta: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("version must be a non-negative integer")
        constraints = tuple(dict(constraint) for constraint in self.constraints)
        last_delta = dict(self.last_delta) if self.last_delta is not None else None
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "last_delta", last_delta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "constraints": [dict(constraint) for constraint in self.constraints],
            "last_delta": dict(self.last_delta) if self.last_delta is not None else None,
            "constraint_count": len(self.constraints),
        }


@dataclass(frozen=True)
class ChatMessagePayload:
    """What the chatbox forwards to the (future) intent provider.

    Chatbox builds this payload but does not mutate any upstream state.
    """

    message: str
    dataset_id: str
    selected_point_ids: Tuple[str, ...]
    unselected_point_ids: Tuple[str, ...]
    selection_groups: Tuple[Mapping[str, Any], ...]
    label_context: Mapping[str, Any]
    history_window: Tuple[Mapping[str, Any], ...]
    strategy: str

    def __post_init__(self) -> None:
        if self.strategy not in REFINEMENT_STRATEGIES:
            raise ValueError(f"unsupported strategy: {self.strategy}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "dataset_id": self.dataset_id,
            "selection_context": {
                "selected_point_ids": list(self.selected_point_ids),
                "unselected_point_ids": list(self.unselected_point_ids),
                "selected_count": len(self.selected_point_ids),
                "unselected_count": len(self.unselected_point_ids),
            },
            "selection_groups": [dict(group) for group in self.selection_groups],
            "label_context": dict(self.label_context),
            "history_window": [dict(turn) for turn in self.history_window],
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class ChatResponse:
    reply: str
    router_category: str
    delta: Optional[Mapping[str, Any]]
    current_instruction_version: int
    requires_followup: bool = False
    followup_question: Optional[str] = None
    intent_type: Optional[str] = None
    provider_label: str = "mock"

    def __post_init__(self) -> None:
        reply = clean_text(self.reply, "reply")
        router_category = clean_text(self.router_category, "router_category")
        if router_category not in ROUTER_CATEGORIES:
            raise ValueError(f"unsupported router_category: {router_category}")
        delta = dict(self.delta) if self.delta is not None else None
        provider_label = clean_text(self.provider_label, "provider_label")
        intent_type = (
            clean_text(self.intent_type, "intent_type") if self.intent_type is not None else None
        )
        if intent_type is not None and intent_type not in ALL_INTENT_TYPES:
            raise ValueError(f"unsupported intent_type: {intent_type}")
        object.__setattr__(self, "reply", reply)
        object.__setattr__(self, "router_category", router_category)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "provider_label", provider_label)
        object.__setattr__(self, "intent_type", intent_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "router_category": self.router_category,
            "delta": dict(self.delta) if self.delta is not None else None,
            "current_instruction_version": self.current_instruction_version,
            "requires_followup": self.requires_followup,
            "followup_question": self.followup_question,
            "intent_type": self.intent_type,
            "provider_label": self.provider_label,
        }


@dataclass(frozen=True)
class ChatboxState:
    dataset_id: str
    strategy: str
    turns: Tuple[ChatTurn, ...] = field(default_factory=tuple)
    instruction_snapshot: MockInstructionSnapshot = field(
        default_factory=lambda: MockInstructionSnapshot(version=0, constraints=())
    )

    def __post_init__(self) -> None:
        dataset_id = clean_text(self.dataset_id, "dataset_id")
        if self.strategy not in REFINEMENT_STRATEGIES:
            raise ValueError(f"unsupported strategy: {self.strategy}")
        if not all(isinstance(turn, ChatTurn) for turn in self.turns):
            raise ValueError("turns must contain ChatTurn objects")
        if not isinstance(self.instruction_snapshot, MockInstructionSnapshot):
            raise ValueError("instruction_snapshot must be a MockInstructionSnapshot")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "turns", tuple(self.turns))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "strategy": self.strategy,
            "turns": [turn.to_dict() for turn in self.turns],
            "turn_count": len(self.turns),
            "instruction_snapshot": self.instruction_snapshot.to_dict(),
        }
