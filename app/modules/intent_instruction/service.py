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
    traces: Dict[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"intent_instruction({self.llm.label})"

    def current_snapshot(self, dataset_id: str) -> InstructionSnapshot:
        return self.store.get(dataset_id).snapshot()

    def reset(self, dataset_id: str) -> None:
        self.store.reset(dataset_id)
        self.traces.pop(dataset_id, None)

    def trace_for(self, dataset_id: str) -> Mapping[str, Any]:
        return self.traces.get(dataset_id, {})

    def respond(self, payload: ChatMessagePayload) -> ChatResponse:
        dataset_id = payload.dataset_id
        context = self._build_context(payload)
        history = _turns_from_payload(payload.history_window)
        snapshot = self.current_snapshot(dataset_id)
        trace: Dict[str, Any] = {
            "message": payload.message,
            "dataset_context": context.to_dict(),
            "history": [turn.to_dict() for turn in history],
            "memory_context": dict(payload.memory_context or {}),
            "instruction_snapshot": snapshot.to_dict(),
        }

        router_result = classify(
            self.llm,
            payload.message,
            context,
            history,
            memory_context=payload.memory_context,
        )
        trace["router_result"] = router_result.to_dict()

        if router_result.category == "off_topic":
            reply = router_result.reply_text or (
                "Off-topic message. Try phrases like 'make petal_length more important', "
                "'merge clusters 1 and 2', or 'split cluster 2'."
            )
            response = ChatResponse(
                reply=reply,
                router_category="off_topic",
                delta=None,
                current_instruction_version=snapshot.version,
                provider_label=self.label,
            )
            trace["reply_source"] = (
                "route_reply_text" if router_result.reply_text else "off_topic_default_reply"
            )
            trace["response"] = response.to_dict()
            self.traces[dataset_id] = trace
            return response

        if router_result.category == "meta_query":
            reply = router_result.reply_text or _grounded_meta_summary(context, snapshot)
            response = ChatResponse(
                reply=reply,
                router_category="meta_query",
                delta=None,
                current_instruction_version=snapshot.version,
                provider_label=self.label,
            )
            trace["reply_source"] = (
                "route_reply_text"
                if router_result.reply_text
                else "meta_query_grounded_summary"
            )
            trace["response"] = response.to_dict()
            self.traces[dataset_id] = trace
            return response

        current_state = self.store.get(dataset_id)
        proposed = propose_delta(
            self.llm,
            payload.message,
            context,
            history,
            current_state,
            memory_context=payload.memory_context,
        )
        trace["proposed_delta"] = proposed.to_dict()

        if router_result.category == "on_topic_ambiguous":
            clarification = (
                _clarification_for_incomplete_delta(proposed, context)
                or router_result.clarification_question
                or "Please clarify which points or cluster you mean."
            )
            response = ChatResponse(
                reply=clarification,
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=snapshot.version,
                requires_followup=True,
                followup_question=clarification,
                provider_label=self.label,
            )
            trace["reply_source"] = "clarification_from_draft_or_router"
            trace["response"] = response.to_dict()
            self.traces[dataset_id] = trace
            return response

        if proposed.is_empty:
            followup = router_result.clarification_question or "Which intent should I apply?"
            response = ChatResponse(
                reply=router_result.clarification_question or (
                    "Could not extract an intent. Try rephrasing with a recognized "
                    "action such as merge, split, ignore, weight, similar, apart, "
                    "anchor, or outlier."
                ),
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=snapshot.version,
                requires_followup=True,
                followup_question=followup,
                provider_label=self.label,
            )
            trace["reply_source"] = (
                "router_clarification_after_empty_extract"
                if router_result.clarification_question
                else "empty_extract_default_reply"
            )
            trace["response"] = response.to_dict()
            self.traces[dataset_id] = trace
            return response

        proposed = _synthesize_delta_from_context(proposed, context, payload.message)
        clarification = _clarification_for_incomplete_delta(proposed, context)
        if clarification is not None:
            response = ChatResponse(
                reply=clarification,
                router_category="on_topic_ambiguous",
                delta=None,
                current_instruction_version=snapshot.version,
                requires_followup=True,
                followup_question=clarification,
                provider_label=self.label,
            )
            trace["reply_source"] = "required_slot_gate_clarification"
            trace["response"] = response.to_dict()
            self.traces[dataset_id] = trace
            return response

        next_state, final_delta = apply_delta(
            self.store,
            dataset_id,
            proposed,
            raw_message=payload.message,
            router_category=router_result.category,
            confidence=router_result.confidence,
        )

        intent_type = _primary_intent(final_delta)
        reply = _action_reply(final_delta, context)

        response = ChatResponse(
            reply=reply,
            router_category=router_result.category,
            delta=final_delta.to_dict(),
            current_instruction_version=next_state.version,
            intent_type=intent_type,
            provider_label=self.label,
        )
        trace["reply_source"] = "deterministic_action_confirmation"
        trace["final_delta"] = final_delta.to_dict()
        trace["response"] = response.to_dict()
        trace["next_instruction"] = next_state.to_dict()
        self.traces[dataset_id] = trace
        return response

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
            selection_groups=tuple(dict(group) for group in payload.selection_groups),
            label_annotations=tuple(
                dict(annotation)
                for annotation in (label_context.get("active_annotations", ()) or ())
                if isinstance(annotation, Mapping)
            ),
            analysis_context=analysis_context,
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


def _grounded_meta_summary(
    context: DatasetContext,
    snapshot: InstructionSnapshot,
) -> str:
    """Deterministic fallback when the LLM did not draft a reply_text for a
    meta_query. Always leads with cluster information so paraphrases like
    "how many classes / categories / groups" receive a useful answer.
    """
    parts: list[str] = []
    if context.cluster_ids:
        cluster_sizes = _cluster_sizes(context)
        if cluster_sizes:
            joined = ", ".join(
                f"{cluster_id} ({cluster_sizes[cluster_id]} points)"
                if cluster_id in cluster_sizes
                else cluster_id
                for cluster_id in context.cluster_ids
            )
        else:
            joined = ", ".join(context.cluster_ids)
        parts.append(f"{len(context.cluster_ids)} clusters: {joined}")
    else:
        parts.append("no resolved clusters yet")

    if context.feature_names:
        parts.append(f"features: {', '.join(context.feature_names)}")

    if context.selected_point_ids:
        selected_by_cluster = _selected_by_cluster(context)
        if selected_by_cluster:
            cluster_breakdown = ", ".join(
                f"{cluster_id}: {', '.join(points)}"
                for cluster_id, points in selected_by_cluster.items()
            )
            parts.append(
                f"{len(context.selected_point_ids)} selected points ({cluster_breakdown})"
            )
        else:
            parts.append(
                f"{len(context.selected_point_ids)} selected points: "
                + ", ".join(context.selected_point_ids)
            )
    else:
        parts.append("no active selection")

    if context.outlier_point_ids:
        parts.append(f"{len(context.outlier_point_ids)} outliers")

    return "Current grounded state has " + "; ".join(parts) + "."


def _cluster_sizes(context: DatasetContext) -> Mapping[str, int]:
    raw = context.analysis_context.get("cluster_sizes") if isinstance(context.analysis_context, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    sizes: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            sizes[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return sizes


def _selected_by_cluster(context: DatasetContext) -> Dict[str, list]:
    raw = (
        context.analysis_context.get("selected_point_clusters")
        if isinstance(context.analysis_context, Mapping)
        else None
    )
    if not isinstance(raw, Sequence):
        return {}
    breakdown: Dict[str, list] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        point_id = entry.get("point_id")
        cluster_id = entry.get("cluster_id")
        if not point_id or not cluster_id:
            continue
        breakdown.setdefault(str(cluster_id), []).append(str(point_id))
    return breakdown


_COLLECTIVE_RECLASSIFY_TOKENS = (
    "no outlier", "no outliers",
    "none of", "none are", "none is",
    "all of them", "all of these", "all of those",
    "both of", "all outliers",
    "these points", "those points",
    "they are", "they're", "they aren't", "they arent",
    "these are", "those are",
    "these aren't", "those aren't",
)


def _is_collective_reclassify_message(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return any(token in text for token in _COLLECTIVE_RECLASSIFY_TOKENS)


def _synthesize_delta_from_context(
    proposed,
    context: DatasetContext,
    message: str = "",
):
    """Fill missing cluster targets when the LLM chose merge_clusters but couldn't
    ground the target_groups, and the current selection spans known clusters.

    Also expands reclassify_outlier with a multi-point selected_points anchor into
    one op per target point so phrases like "there should be no outlier" with
    multiple points selected resolve without clarification.

    Only fires for merge_clusters with an empty target_groups — NOT for group_similar.
    group_similar with a missing group_b is kept as-is so the clarification path can
    ask the user a context-aware follow-up.
    """
    from .schemas import DeltaOperation, InstructionDelta

    repaired = []
    changed = False
    for op in proposed.operations:
        if op.op == "add" and op.intent == "merge_clusters":
            payload = dict(op.payload or {})
            target_groups = payload.get("target_groups") or []
            grounded = _resolved_groups(target_groups, context, allowed_sources={"cluster"})
            if len(grounded) < 2:
                clusters = _clusters_of_selection(context)
                if len(clusters) >= 2:
                    repaired.append(
                        DeltaOperation(
                            op="add",
                            constraint_id=op.constraint_id,
                            intent="merge_clusters",
                            payload={
                                "target_groups": [
                                    {"source": "cluster", "ref": cid}
                                    for cid in clusters
                                ]
                            },
                        )
                    )
                    changed = True
                    continue

        if op.op == "add" and op.intent == "reclassify_outlier":
            payload = dict(op.payload or {})
            anchor = payload.get("anchor")
            is_outlier = payload.get("is_outlier")
            if (
                isinstance(anchor, Mapping)
                and anchor.get("source") == "selected_points"
                and anchor.get("ref") == "current"
                and len(context.selected_point_ids) >= 2
                and isinstance(is_outlier, bool)
                and _is_collective_reclassify_message(message)
            ):
                if is_outlier is False:
                    targets = _selected_outlier_point_ids(context) or list(context.selected_point_ids)
                else:
                    targets = list(context.selected_point_ids)
                for pid in targets:
                    repaired.append(
                        DeltaOperation(
                            op="add",
                            constraint_id=op.constraint_id,
                            intent="reclassify_outlier",
                            payload={
                                "anchor": {"source": "point_id", "ref": pid},
                                "is_outlier": is_outlier,
                            },
                        )
                    )
                changed = True
                continue

        repaired.append(op)

    if not changed:
        return proposed
    return InstructionDelta(operations=tuple(repaired))


def _selected_outlier_point_ids(context: DatasetContext) -> list:
    raw = (
        context.analysis_context.get("selected_point_clusters")
        if isinstance(context.analysis_context, Mapping)
        else None
    )
    if not isinstance(raw, Sequence):
        return []
    point_ids = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("is_outlier") and entry.get("point_id"):
            point_ids.append(str(entry["point_id"]))
    return point_ids


def _clusters_of_selection(context: DatasetContext) -> list:
    """Return the distinct cluster ids that contain the currently selected points."""
    raw = (
        context.analysis_context.get("selected_point_clusters")
        if isinstance(context.analysis_context, Mapping)
        else None
    )
    if not isinstance(raw, Sequence):
        return []
    seen = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        cluster_id = entry.get("cluster_id")
        if cluster_id and cluster_id in context.cluster_ids and cluster_id not in seen:
            seen.append(str(cluster_id))
    return seen


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
                return _group_pair_clarification(op.intent, payload, context)
        if op.intent == "merge_clusters" and len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 2:
            return "Which two grounded clusters should I merge?"
        if op.intent == "split_cluster" and len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 1:
            return "Which grounded cluster should I split?"
        if op.intent == "ignore_cluster" and len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 1:
            return "Which grounded cluster should I ignore?"
        if op.intent == "anchor_point":
            if not _group_ref_present(payload.get("anchor"), context) or len(_resolved_groups(payload.get("target_groups"), context, allowed_sources={"cluster"})) < 1:
                return "Which point should be the anchor, and which grounded cluster should it represent?"
        if op.intent == "reclassify_outlier" and not _reclassify_anchor_present(payload.get("anchor"), context):
            return _reclassify_outlier_clarification(context)
    return None


def _group_pair_clarification(
    intent: str,
    payload: Mapping[str, Any],
    context: DatasetContext,
) -> str:
    group_a = payload.get("group_a")
    group_b = payload.get("group_b")
    a_present = _group_ref_present(group_a, context)
    b_present = _group_ref_present(group_b, context)

    # group_a is the current selection, group_b is missing
    if (
        a_present
        and not b_present
        and isinstance(group_a, Mapping)
        and group_a.get("source") == "selected_points"
    ):
        clusters = _clusters_of_selection(context)
        selected_ids = list(context.selected_point_ids[:4])
        preview = ", ".join(selected_ids)
        if clusters:
            cluster_preview = " and ".join(clusters)
            if intent == "group_similar":
                return (
                    f"Your selected points ({preview}) span {cluster_preview}. "
                    f"Which cluster or group should they be pulled toward?"
                )
            else:
                return (
                    f"Your selected points ({preview}) span {cluster_preview}. "
                    f"Which cluster or group should they be pushed away from?"
                )
        return (
            f"I see {len(context.selected_point_ids)} selected points. "
            "Which cluster or group should they be paired with?"
        )

    return "I need two grounded groups or clusters. Which points, saved group, or cluster should I use?"


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


def _reclassify_anchor_present(value: Any, context: DatasetContext) -> bool:
    if not isinstance(value, Mapping):
        return False
    source = value.get("source")
    ref = value.get("ref")
    if source == "selected_points" and ref == "current":
        return len(context.selected_point_ids) == 1
    if source == "point_id":
        return ref in set((*context.selected_point_ids, *context.unselected_point_ids))
    return False


def _reclassify_outlier_clarification(context: DatasetContext) -> str:
    if not context.selected_point_ids:
        return "Which point should I reclassify as an outlier or non-outlier?"
    selected_points = ", ".join(context.selected_point_ids)
    if len(context.selected_point_ids) == 1:
        return (
            f"I can use the current selected point {selected_points}. "
            "Should I mark it as an outlier or not an outlier?"
        )
    return (
        f"I can use the current selection, but it has {len(context.selected_point_ids)} points: "
        f"{selected_points}. Which point should I reclassify as an outlier or non-outlier?"
    )


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


def _action_reply(delta, context: DatasetContext) -> str:
    add_ops = [op for op in delta.operations if op.op == "add" and op.intent in ALL_INTENT_TYPES]
    if not add_ops:
        return "Recorded instruction."

    if len(add_ops) > 1 and all(op.intent == "reclassify_outlier" for op in add_ops):
        point_ids = [
            op.payload.get("anchor", {}).get("ref")
            for op in add_ops
            if isinstance(op.payload.get("anchor"), Mapping)
        ]
        point_ids = [point_id for point_id in point_ids if point_id]
        is_outlier = bool(add_ops[0].payload.get("is_outlier"))
        status_text = "outliers" if is_outlier else "not outliers"
        if point_ids:
            return f"Okay, I recorded {', '.join(point_ids)} as {status_text}."
        return f"Okay, I recorded {len(add_ops)} outlier updates."

    op = add_ops[0]
    payload = dict(op.payload or {})
    intent = op.intent
    if intent == "merge_clusters":
        labels = _describe_group_refs(payload.get("target_groups"), context, allowed_sources={"cluster"})
        if labels:
            return f"Okay, I recorded a merge for {', '.join(labels)}."
    if intent == "split_cluster":
        labels = _describe_group_refs(payload.get("target_groups"), context, allowed_sources={"cluster"})
        if labels:
            return f"Okay, I recorded a split request for {labels[0]}."
    if intent == "ignore_cluster":
        labels = _describe_group_refs(payload.get("target_groups"), context, allowed_sources={"cluster"})
        if labels:
            return f"Okay, I recorded an ignore request for {', '.join(labels)}."
    if intent == "group_similar":
        return (
            "Okay, I recorded a similarity request"
            f" between {_describe_group_ref(payload.get('group_a'), context)}"
            f" and {_describe_group_ref(payload.get('group_b'), context)}."
        )
    if intent == "group_dissimilar":
        return (
            "Okay, I recorded a separation request"
            f" between {_describe_group_ref(payload.get('group_a'), context)}"
            f" and {_describe_group_ref(payload.get('group_b'), context)}."
        )
    if intent == "feature_weight":
        feature = payload.get("feature") or "that feature"
        direction = payload.get("direction") or "change"
        if direction == "increase":
            return f"Okay, I recorded a higher weight for {feature}."
        if direction == "decrease":
            return f"Okay, I recorded a lower weight for {feature}."
        if direction == "ignore":
            return f"Okay, I recorded that {feature} should be ignored."
    if intent == "anchor_point":
        anchor = _describe_group_ref(payload.get("anchor"), context)
        targets = _describe_group_refs(payload.get("target_groups"), context, allowed_sources={"cluster"})
        if targets:
            return f"Okay, I recorded {anchor} as an anchor for {targets[0]}."
    if intent == "reclassify_outlier":
        anchor = _describe_group_ref(payload.get("anchor"), context)
        status = "an outlier" if payload.get("is_outlier") else "not an outlier"
        return f"Okay, I recorded {anchor} as {status}."
    return f"Recorded {intent}."


def _describe_group_refs(
    items: Any,
    context: DatasetContext,
    allowed_sources: set[str] | None = None,
) -> list[str]:
    refs = _resolved_groups(items, context, allowed_sources=allowed_sources)
    return [_describe_group_ref(item, context) for item in refs]


def _describe_group_ref(value: Any, context: DatasetContext) -> str:
    if not isinstance(value, Mapping):
        return "the requested target"
    source = value.get("source")
    ref = value.get("ref")
    if source == "selected_points" and ref == "current":
        if context.selected_point_ids:
            preview = ", ".join(context.selected_point_ids[:4])
            if len(context.selected_point_ids) > 4:
                preview += ", ..."
            return f"the current selection ({preview})"
        return "the current selection"
    if source == "cluster" and ref:
        return str(ref)
    if source == "selection_group" and ref:
        return f"saved group {ref}"
    if source == "point_id" and ref:
        return str(ref)
    if source == "outlier_set" and ref:
        return f"outlier {ref}"
    return "the requested target"
