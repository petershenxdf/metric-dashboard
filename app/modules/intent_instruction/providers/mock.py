from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from ..schemas import (
    ALL_INTENT_TYPES,
    DatasetContext,
    DeltaOperation,
    GroupRef,
    InstructionDelta,
    RouterResult,
    StructuredInstruction,
    Turn,
)


_KEYWORDS_OFF_TOPIC = ("weather", "joke", "hello", "how are you", "thanks", "bye")

_KEYWORDS_META_QUERY = (
    "how many",
    "what are",
    "what clusters",
    "what classes",
    "which classes",
    "which clusters",
    "which points",
    "show me the",
    "list clusters",
    "list classes",
    "list features",
)

_SELECTION_REFERENCES = (
    "these points",
    "this group",
    "these",
    "selected",
    "those points",
    "this selection",
)

# Keyword → intent type. Order matters: longer / more specific phrases must come
# first so "not an outlier" is matched before "outlier".
_INTENT_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("not an outlier", "reclassify_outlier"),
    ("typical example", "anchor_point"),
    ("less important", "feature_weight"),
    ("more important", "feature_weight"),
    ("reclassify", "reclassify_outlier"),
    ("outlier", "reclassify_outlier"),
    ("important", "feature_weight"),
    ("together", "group_similar"),
    ("similar", "group_similar"),
    ("closer", "group_similar"),
    ("apart", "group_dissimilar"),
    ("separate", "group_dissimilar"),
    ("push", "group_dissimilar"),
    ("anchor", "anchor_point"),
    ("merge", "merge_clusters"),
    ("combine", "merge_clusters"),
    ("split", "split_cluster"),
    ("break", "split_cluster"),
    ("weight", "feature_weight"),
    ("ignore", "ignore_cluster"),
    ("exclude", "ignore_cluster"),
)


@dataclass
class MockLlmProvider:
    """Deterministic keyword-driven router + extractor.

    This is the only ``LlmProvider`` that ships with Step 8. Tests and the
    debug page rely on its determinism; real LLM providers (qwen, Claude,
    OpenAI) slot into the same protocol when added later.
    """

    label: str = "mock_llm"

    def route(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        memory_context: Mapping[str, object] | None = None,
    ) -> RouterResult:
        text = (message or "").strip().lower()
        if not text:
            return RouterResult(
                category="on_topic_ambiguous",
                confidence=0.5,
                reason="empty message",
                clarification_question="What would you like to do?",
            )

        for keyword in _KEYWORDS_OFF_TOPIC:
            if keyword in text:
                return RouterResult(
                    category="off_topic",
                    confidence=0.9,
                    reason=f"matched off-topic keyword '{keyword}'",
                )

        for keyword in _KEYWORDS_META_QUERY:
            if keyword in text:
                return RouterResult(
                    category="meta_query",
                    confidence=0.85,
                    reason=f"matched meta-query keyword '{keyword}'",
                )

        references_selection = any(token in text for token in _SELECTION_REFERENCES)
        if references_selection and not (
            context.selected_point_ids or context.selection_group_names
        ):
            return RouterResult(
                category="on_topic_ambiguous",
                confidence=0.75,
                reason="selection reference with no active selection",
                clarification_question="Which points should I use?",
            )

        intent = _match_intent(text)
        if intent is None:
            return RouterResult(
                category="on_topic_ambiguous",
                confidence=0.55,
                reason="no keyword match",
                clarification_question=(
                    "Try phrases like 'make petal_length more important', "
                    "'merge clusters 1 and 2', or 'split cluster 2'."
                ),
            )

        return RouterResult(
            category="on_topic_actionable",
            confidence=0.9,
            reason=f"matched intent keyword for '{intent}'",
        )

    def extract(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        current_instruction: StructuredInstruction,
        memory_context: Mapping[str, object] | None = None,
    ) -> InstructionDelta:
        text = (message or "").strip().lower()
        intent = _match_intent(text)
        if intent is None:
            return InstructionDelta(operations=())

        payload = _build_payload(intent, text, context)
        return InstructionDelta(
            operations=(
                DeltaOperation(
                    op="add",
                    constraint_id="pending",
                    intent=intent,
                    payload=payload,
                ),
            )
        )

    def freeform_reply(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        response: Mapping[str, Any],
        provider_trace: Mapping[str, Any],
        memory_context: Mapping[str, object] | None = None,
    ) -> str:
        reply = str(response.get("reply") or "").strip()
        router_category = str(response.get("router_category") or "").strip()
        if router_category == "meta_query" and reply:
            return reply
        if response.get("requires_followup") and response.get("followup_question"):
            return str(response["followup_question"])

        proposed_delta = provider_trace.get("proposed_delta")
        operations = proposed_delta.get("operations", ()) if isinstance(proposed_delta, Mapping) else ()
        first = operations[0] if isinstance(operations, Sequence) and operations else {}
        intent = first.get("intent") if isinstance(first, Mapping) else None
        payload = dict(first.get("payload") or {}) if isinstance(first, Mapping) else {}

        if intent == "merge_clusters":
            targets = payload.get("target_groups") or []
            labels = [
                item.get("ref")
                for item in targets
                if isinstance(item, Mapping) and item.get("ref")
            ]
            if labels:
                return f"I read that as a request to merge {', '.join(labels)}."
        if intent == "split_cluster":
            targets = payload.get("target_groups") or []
            if targets and isinstance(targets[0], Mapping) and targets[0].get("ref"):
                return f"I read that as a request to split {targets[0]['ref']}."
        if intent == "group_similar":
            return "I read that as a request to pull two grounded groups closer together."
        if intent == "group_dissimilar":
            return "I read that as a request to push two grounded groups farther apart."
        if intent == "feature_weight":
            feature = payload.get("feature") or "that feature"
            direction = payload.get("direction") or "change"
            return f"I read that as a request to {direction} the influence of {feature}."
        if intent == "ignore_cluster":
            targets = payload.get("target_groups") or []
            labels = [
                item.get("ref")
                for item in targets
                if isinstance(item, Mapping) and item.get("ref")
            ]
            if labels:
                return f"I read that as a request to ignore {', '.join(labels)}."
        if intent == "anchor_point":
            return "I read that as an anchor-point request."
        if intent == "reclassify_outlier":
            anchor = payload.get("anchor") or {}
            ref = anchor.get("ref") if isinstance(anchor, Mapping) else None
            status = "an outlier" if payload.get("is_outlier") else "not an outlier"
            if ref:
                return f"I read that as a request to mark {ref} as {status}."

        if reply:
            return reply
        return "I understood the grounded context, but I do not have a more specific direct reply."


def _match_intent(text: str) -> str | None:
    for keyword, intent in _INTENT_KEYWORDS:
        if keyword in text and intent in ALL_INTENT_TYPES:
            return intent
    return None


def _build_payload(intent: str, text: str, context: DatasetContext) -> dict:
    if intent == "feature_weight":
        feature = _match_feature(text, context.feature_names)
        direction = "decrease" if "less" in text else "increase"
        if "ignore" in text:
            direction = "ignore"
        return {"feature": feature or "unspecified", "direction": direction}

    if intent in ("group_similar", "group_dissimilar"):
        group_a, group_b = _match_two_group_refs(text, context)
        return {
            "group_a": group_a.to_dict() if group_a else None,
            "group_b": group_b.to_dict() if group_b else None,
        }

    if intent == "merge_clusters":
        clusters = _match_cluster_refs(text, context)
        return {
            "target_groups": [cluster.to_dict() for cluster in clusters],
        }

    if intent == "anchor_point":
        anchor = _match_anchor(text, context)
        target = _match_cluster_refs(text, context)
        return {
            "anchor": anchor.to_dict() if anchor else None,
            "target_groups": [cluster.to_dict() for cluster in target],
        }

    if intent == "ignore_cluster":
        clusters = _match_cluster_refs(text, context)
        return {"target_groups": [cluster.to_dict() for cluster in clusters]}

    if intent == "split_cluster":
        clusters = _match_cluster_refs(text, context)
        return {"target_groups": [cluster.to_dict() for cluster in clusters]}

    if intent == "reclassify_outlier":
        point_ref = _match_point_ref(text, context)
        is_outlier = "not" not in text
        return {
            "anchor": point_ref.to_dict() if point_ref else None,
            "is_outlier": is_outlier,
        }

    return {}


def _match_feature(text: str, feature_names: Sequence[str]) -> str | None:
    for name in feature_names:
        if name.lower() in text:
            return name
    return None


def _match_cluster_refs(text: str, context: DatasetContext) -> Tuple[GroupRef, ...]:
    # Collect explicit "cluster(s) N" numeric suffixes, including trailing
    # enumerations like "clusters 1 and 2" or "cluster 1, 2 or 3".
    numeric_suffixes: set[str] = set()
    enum_pattern = re.compile(
        r"clusters?\s+(\d+(?:\s*(?:,|and|or|&)\s*\d+)*)"
    )
    for match in enum_pattern.finditer(text):
        numeric_suffixes.update(re.findall(r"\d+", match.group(1)))
    refs = []
    seen = set()
    for cluster_id in context.cluster_ids:
        lower_id = cluster_id.lower()
        suffix = cluster_id.split("_")[-1]
        matches = (
            lower_id in text
            or lower_id.replace("cluster_", "cluster ") in text
            or (suffix and suffix in numeric_suffixes)
        )
        if matches and cluster_id not in seen:
            refs.append(GroupRef(source="cluster", ref=cluster_id))
            seen.add(cluster_id)
    return tuple(refs)


def _match_two_group_refs(
    text: str, context: DatasetContext
) -> Tuple[GroupRef | None, GroupRef | None]:
    named_groups = _match_selection_group_refs(text, context)
    if len(named_groups) >= 2:
        return named_groups[0], named_groups[1]

    clusters = _match_cluster_refs(text, context)
    if len(clusters) >= 2:
        return clusters[0], clusters[1]

    group_a = GroupRef(source="selected_points", ref="current") if context.selected_point_ids else None
    if named_groups:
        return group_a, named_groups[0]

    group_b = clusters[0] if clusters else None
    return group_a, group_b


def _match_selection_group_refs(text: str, context: DatasetContext) -> Tuple[GroupRef, ...]:
    refs = []
    seen = set()
    for group_name in context.selection_group_names:
        if group_name.lower() in text and group_name not in seen:
            refs.append(GroupRef(source="selection_group", ref=group_name))
            seen.add(group_name)
    return tuple(refs)


def _match_anchor(text: str, context: DatasetContext) -> GroupRef | None:
    point = _match_point_ref(text, context)
    if point is not None:
        return point
    if context.selected_point_ids:
        return GroupRef(source="selected_points", ref="current")
    return None


def _match_point_ref(text: str, context: DatasetContext) -> GroupRef | None:
    tokens = text.replace(",", " ").split()
    point_ids_lower = {pid.lower(): pid for pid in (*context.selected_point_ids, *context.unselected_point_ids)}
    for token in tokens:
        token = token.strip(".;:!?")
        if token in point_ids_lower:
            return GroupRef(source="point_id", ref=point_ids_lower[token])
    return None
