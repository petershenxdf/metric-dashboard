from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

from app.modules.chatbox.schemas import ChatMessagePayload, ChatResponse
from app.shared.schemas import InstructionSnapshot

from .extractor import apply_delta, propose_delta
from .providers.base import LlmProvider
from .providers.mock import MockLlmProvider
from .router import classify
from .schemas import (
    ALL_INTENT_TYPES,
    DatasetContext,
    Turn,
)
from .store import IntentInstructionStore


FeatureNameLookup = Callable[[str], Sequence[str]]


@dataclass
class IntentInstructionProvider:
    """Satisfies chatbox's ``IntentProvider`` by wrapping an LLM provider.

    Parameters
    ----------
    llm:
        Router + extractor LLM backend. Defaults to ``MockLlmProvider``.
    feature_name_lookup:
        Callable returning a sequence of feature names for a dataset id.
        Used to build the ``DatasetContext`` passed to the LLM.
    """

    llm: LlmProvider = field(default_factory=MockLlmProvider)
    feature_name_lookup: FeatureNameLookup | None = None
    store: IntentInstructionStore = field(default_factory=IntentInstructionStore)

    @property
    def label(self) -> str:
        return f"intent_instruction({self.llm.label})"

    def current_snapshot(self, dataset_id: str) -> InstructionSnapshot:
        return self.store.get(dataset_id).snapshot()

    def reset(self, dataset_id: str) -> None:
        self.store.reset(dataset_id)

    def respond(self, payload: ChatMessagePayload) -> ChatResponse:
        dataset_id = payload.dataset_id
        context = self._build_context(payload)
        history = _turns_from_payload(payload.history_window)
        snapshot = self.current_snapshot(dataset_id)

        router_result = classify(self.llm, payload.message, context, history)

        if router_result.category == "off_topic":
            return ChatResponse(
                reply=(
                    "Off-topic message. Try phrases like 'make petal_length more important', "
                    "'merge clusters 1 and 2', or 'split cluster 2'."
                ),
                router_category="off_topic",
                delta=None,
                current_instruction_version=snapshot.version,
                provider_label=self.label,
            )

        if router_result.category == "meta_query":
            return ChatResponse(
                reply=_grounded_meta_reply(payload.message, context, snapshot),
                router_category="meta_query",
                delta=None,
                current_instruction_version=snapshot.version,
                provider_label=self.label,
            )

        if router_result.category == "on_topic_ambiguous":
            return ChatResponse(
                reply=router_result.clarification_question
                or "Please clarify which points or cluster you mean.",
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=snapshot.version,
                requires_followup=True,
                followup_question=router_result.clarification_question
                or "Which points or group should I use?",
                provider_label=self.label,
            )

        current_state = self.store.get(dataset_id)
        proposed = propose_delta(
            self.llm,
            payload.message,
            context,
            history,
            current_state,
        )

        if proposed.is_empty:
            return ChatResponse(
                reply=(
                    "Could not extract an intent. Try rephrasing with a recognized "
                    "action such as merge, split, ignore, weight, similar, apart, "
                    "anchor, or outlier."
                ),
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=snapshot.version,
                requires_followup=True,
                followup_question="Which intent should I apply?",
                provider_label=self.label,
            )

        clarification = _clarification_for_incomplete_delta(proposed, context)
        if clarification is not None:
            return ChatResponse(
                reply=clarification,
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=snapshot.version,
                requires_followup=True,
                followup_question=clarification,
                provider_label=self.label,
            )

        next_state, final_delta = apply_delta(
            self.store,
            dataset_id,
            proposed,
            raw_message=payload.message,
            router_category=router_result.category,
            confidence=router_result.confidence,
        )

        intent_type = _primary_intent(final_delta)
        reply = f"Recorded {intent_type}." if intent_type else "Recorded instruction."

        return ChatResponse(
            reply=reply,
            router_category=router_result.category,
            delta=final_delta.to_dict(),
            current_instruction_version=next_state.version,
            intent_type=intent_type,
            provider_label=self.label,
        )

    # ----- helpers -----

    def _build_context(self, payload: ChatMessagePayload) -> DatasetContext:
        label_context = payload.label_context or {}
        analysis_context = _analysis_context_from_label_context(label_context)
        feature_names = tuple(
            self.feature_name_lookup(payload.dataset_id)
            if self.feature_name_lookup is not None
            else tuple(analysis_context.get("feature_names") or ())
        )
        cluster_ids = _cluster_ids_from_analysis_context(analysis_context)
        if not cluster_ids:
            cluster_ids = _cluster_ids_from_label_context(label_context)
        selection_group_names = tuple(
            str(group.get("group_name") or group.get("group_id") or "")
            for group in payload.selection_groups
            if group
        )
        return DatasetContext(
            dataset_id=payload.dataset_id,
            feature_names=feature_names,
            cluster_ids=cluster_ids,
            outlier_point_ids=_outlier_ids_from_analysis_context(analysis_context),
            selection_group_names=tuple(name for name in selection_group_names if name),
            selected_point_ids=tuple(payload.selected_point_ids),
            unselected_point_ids=tuple(payload.unselected_point_ids),
        )


def _turns_from_payload(history_window) -> tuple:
    turns = []
    for entry in history_window or ():
        role = entry.get("role") if isinstance(entry, Mapping) else None
        text = entry.get("text") if isinstance(entry, Mapping) else None
        if role and text:
            turns.append(Turn(role=role, text=text))
    return tuple(turns)


def _cluster_ids_from_label_context(label_context: Mapping[str, Any]) -> tuple:
    seen = []
    for annotation in label_context.get("active_annotations", ()) or ():
        if not isinstance(annotation, Mapping):
            continue
        payload = annotation.get("payload") or {}
        cluster_id = payload.get("cluster_id") or annotation.get("cluster_id")
        if cluster_id and cluster_id not in seen:
            seen.append(cluster_id)
    return tuple(seen)


def _analysis_context_from_label_context(label_context: Mapping[str, Any]) -> Mapping[str, Any]:
    analysis_context = label_context.get("analysis_context") or {}
    if isinstance(analysis_context, Mapping):
        return analysis_context
    return {}


def _cluster_ids_from_analysis_context(analysis_context: Mapping[str, Any]) -> tuple:
    cluster_ids = analysis_context.get("cluster_ids") or ()
    return tuple(str(cluster_id) for cluster_id in cluster_ids if cluster_id)


def _outlier_ids_from_analysis_context(analysis_context: Mapping[str, Any]) -> tuple:
    outlier_point_ids = analysis_context.get("outlier_point_ids") or ()
    return tuple(str(point_id) for point_id in outlier_point_ids if point_id)


def _grounded_meta_reply(
    message: str,
    context: DatasetContext,
    snapshot: InstructionSnapshot,
) -> str:
    text = (message or "").strip().lower()
    if any(token in text for token in ("cluster", "clusters", "class", "classes")):
        if context.cluster_ids:
            joined = ", ".join(context.cluster_ids)
            return f"Current grounded state has {len(context.cluster_ids)} clusters: {joined}."
        return "Current grounded state does not expose any resolved clusters yet."
    if "outlier" in text:
        if context.outlier_point_ids:
            joined = ", ".join(context.outlier_point_ids)
            return f"Current grounded state has {len(context.outlier_point_ids)} outliers: {joined}."
        return "Current grounded state has no visible outliers right now."
    if any(token in text for token in ("feature", "features")):
        if context.feature_names:
            return f"Current dataset features: {', '.join(context.feature_names)}."
        return "No grounded feature list is available right now."
    if any(token in text for token in ("selected", "selection", "which points", "what points")):
        if context.selected_point_ids:
            joined = ", ".join(context.selected_point_ids)
            return f"Current selection has {len(context.selected_point_ids)} points: {joined}."
        return "There is no active selection right now."
    return (
        "Current grounded state has "
        f"{len(context.cluster_ids)} clusters, "
        f"{len(context.selected_point_ids)} selected points, "
        f"{len(context.outlier_point_ids)} outliers, and "
        f"{len(snapshot.constraints)} compiled instruction constraints."
    )


def _clarification_for_incomplete_delta(
    proposed,
    context: DatasetContext,
) -> str | None:
    for op in proposed.operations:
        if op.op != "add" or op.intent is None:
            continue
        payload = dict(op.payload or {})
        if op.intent == "feature_weight" and not _present(payload.get("feature")):
            return "Which feature should I change?"
        if op.intent in ("group_similar", "group_dissimilar"):
            if not _group_ref_present(payload.get("group_a"), context) or not _group_ref_present(payload.get("group_b"), context):
                return "I need two grounded groups or clusters. Which points, saved group, or cluster should I use?"
        if op.intent == "merge_clusters" and len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 2:
            return "Which two grounded clusters should I merge?"
        if op.intent == "split_cluster" and len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 1:
            return "Which grounded cluster should I split?"
        if op.intent == "ignore_cluster" and len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 1:
            return "Which grounded cluster should I ignore?"
        if op.intent == "anchor_point":
            if not _group_ref_present(payload.get("anchor"), context) or len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 1:
                return "Which point should be the anchor, and which grounded cluster should it represent?"
        if op.intent == "reclassify_outlier" and not _group_ref_present(payload.get("anchor"), context):
            return "Which point should I reclassify as an outlier or non-outlier?"
    return None


def _present(value: Any) -> bool:
    return bool(value) and value != "unspecified"


def _group_ref_present(value: Any, context: DatasetContext) -> bool:
    if not isinstance(value, Mapping):
        return False
    source = value.get("source")
    ref = value.get("ref")
    if not source or not ref:
        return False
    if source == "selected_points":
        return ref == "current" and bool(context.selected_point_ids)
    if source == "cluster":
        return ref in context.cluster_ids
    if source == "selection_group":
        return ref in context.selection_group_names
    if source == "point_id":
        return ref in set((*context.selected_point_ids, *context.unselected_point_ids))
    if source == "outlier_set":
        return ref in context.outlier_point_ids
    return False


def _resolved_groups(
    items: Any,
    context: DatasetContext,
    allowed_sources: set[str] | None = None,
) -> tuple:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return ()
    groups = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        source = item.get("source")
        if allowed_sources is not None and source not in allowed_sources:
            continue
        if _group_ref_present(item, context):
            groups.append(item)
    return tuple(groups)


def _primary_intent(delta) -> str | None:
    for op in delta.operations:
        if op.op == "add" and op.intent in ALL_INTENT_TYPES:
            return op.intent
    return None
