from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from ..schemas import (
    ALL_INTENT_TYPES,
    ChatMessagePayload,
    ChatResponse,
    InstructionSnapshot,
)


_KEYWORDS_OFF_TOPIC = ("weather", "joke", "hello", "how are you", "thanks")
_KEYWORDS_META_QUERY = (
    "how many",
    "what are",
    "which clusters",
    "which points",
    "show me the",
)
_SELECTION_REFERENCES = ("these", "this group", "selected", "those points")

_INTENT_KEYWORDS = (
    # (keyword, intent_type)
    ("merge", "merge_clusters"),
    ("split", "split_cluster"),
    ("not an outlier", "reclassify_outlier"),
    ("reclassify", "reclassify_outlier"),
    ("outlier", "reclassify_outlier"),
    ("weight", "feature_weight"),
    ("important", "feature_weight"),
    ("less important", "feature_weight"),
    ("together", "group_similar"),
    ("similar", "group_similar"),
    ("closer", "group_similar"),
    ("apart", "group_dissimilar"),
    ("separate", "group_dissimilar"),
    ("push", "group_dissimilar"),
    ("anchor", "anchor_point"),
    ("typical example", "anchor_point"),
    ("ignore", "ignore_cluster"),
    ("exclude", "ignore_cluster"),
)


@dataclass
class _MockDatasetState:
    version: int = 0
    constraints: List[Mapping] = field(default_factory=list)
    last_delta: Mapping | None = None
    constraint_counter: int = 0

    def snapshot(self) -> InstructionSnapshot:
        return InstructionSnapshot(
            version=self.version,
            constraints=tuple(self.constraints),
            last_delta=self.last_delta,
        )


class MockIntentProvider:
    """Deterministic keyword-based intent provider used in Step 7.

    Maintains a per-dataset snapshot of a fake ``StructuredInstruction`` so the
    chatbox debug page can display the *current_instruction* panel as described
    in the design doc, even though ``intent_instruction`` (Step 8) does not
    exist yet. This state lives inside the provider, not inside the chatbox
    store — the boundary matches the Step 8 intent_instruction module.
    """

    label = "mock"

    def __init__(self) -> None:
        self._states: Dict[str, _MockDatasetState] = {}

    def current_snapshot(self, dataset_id: str) -> InstructionSnapshot:
        return self._state(dataset_id).snapshot()

    def reset(self, dataset_id: str) -> None:
        self._states[dataset_id] = _MockDatasetState()

    def respond(self, payload: ChatMessagePayload) -> ChatResponse:
        state = self._state(payload.dataset_id)
        text = payload.message.strip().lower()

        router_category = self._route(text, payload)

        if router_category == "off_topic":
            return ChatResponse(
                reply=(
                    "Mock provider classified this as off-topic. I can only help with "
                    "dataset feedback. Try phrases like 'make petal_length more important'."
                ),
                router_category="off_topic",
                delta=None,
                current_instruction_version=state.version,
                provider_label=self.label,
            )

        if router_category == "meta_query":
            return ChatResponse(
                reply=f"Current instruction has {len(state.constraints)} constraints.",
                router_category="meta_query",
                delta=None,
                current_instruction_version=state.version,
                provider_label=self.label,
            )

        if router_category == "on_topic_ambiguous":
            return ChatResponse(
                reply="Please select some points or name a cluster/group first.",
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=state.version,
                requires_followup=True,
                followup_question="Which points or group should I use?",
                provider_label=self.label,
            )

        intent_type = self._match_intent(text)
        if intent_type is None:
            return ChatResponse(
                reply=(
                    f"Mock provider did not match a known keyword for {payload.message!r}. "
                    "Step 7 uses keyword matching; Step 8 swaps in the real "
                    "intent-instruction module boundary. "
                    "Try a suggestion chip, or phrase it with a recognized keyword "
                    "(merge, split, ignore, weight, similar, apart, anchor, outlier)."
                ),
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=state.version,
                requires_followup=True,
                followup_question="Which recognized intent should I apply?",
                provider_label=self.label,
            )

        state.constraint_counter += 1
        constraint_id = f"c{state.constraint_counter}"
        constraint = {"id": constraint_id, "intent": intent_type, "raw_message": payload.message}
        state.constraints.append(constraint)
        state.version += 1
        delta = {"operations": [{"op": "add", "constraint_id": constraint_id, "intent": intent_type}]}
        state.last_delta = delta

        return ChatResponse(
            reply=f"Recorded {intent_type}.",
            router_category="on_topic_actionable",
            delta=delta,
            current_instruction_version=state.version,
            intent_type=intent_type,
            provider_label=self.label,
        )

    def _route(self, text: str, payload: ChatMessagePayload) -> str:
        if not text:
            return "on_topic_ambiguous"

        for keyword in _KEYWORDS_OFF_TOPIC:
            if keyword in text:
                return "off_topic"

        for keyword in _KEYWORDS_META_QUERY:
            if keyword in text:
                return "meta_query"

        references_selection = any(token in text for token in _SELECTION_REFERENCES)
        has_named_group = bool(payload.selection_groups)
        has_selected_points = bool(payload.selected_point_ids)

        if references_selection and not (has_selected_points or has_named_group):
            return "on_topic_ambiguous"

        return "on_topic_actionable"

    def _match_intent(self, text: str) -> str | None:
        # Priority order: longer, more-specific keywords first.
        ordered = sorted(_INTENT_KEYWORDS, key=lambda pair: -len(pair[0]))
        for keyword, intent_type in ordered:
            if keyword in text and intent_type in ALL_INTENT_TYPES:
                return intent_type
        return None

    def _state(self, dataset_id: str) -> _MockDatasetState:
        if dataset_id not in self._states:
            self._states[dataset_id] = _MockDatasetState()
        return self._states[dataset_id]
