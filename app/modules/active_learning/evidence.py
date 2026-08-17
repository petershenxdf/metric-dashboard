from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from functools import lru_cache
from math import sqrt
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from app.modules.rule_panel.schemas import RuleCard, RuleSet
from app.shared.schemas import AnalysisResult, ProjectionResult

from .data import PreparedDataset, display_condition
from .schemas import (
    ActiveLearningRound,
    CategoryEvidenceCard,
    EvidenceBullet,
    LabelEvent,
)


EVIDENCE_POLICY_VERSION = "category_evidence_v2"

CATEGORY_EXPLANATIONS = {
    "label_priority": (
        "This category identifies the current labeling question that is most "
        "useful to answer first."
    ),
    "boundary_review": (
        "This category checks whether the current dividing line separates "
        "records that people would genuinely consider different."
    ),
    "overlap_merge_signal": (
        "This category checks whether two group descriptions may be describing "
        "the same real-world type."
    ),
    "split_or_new_cluster_signal": (
        "This category checks whether one part of a group may represent a "
        "meaningfully different type."
    ),
    "anomaly_label_review": (
        "This category checks whether a record is truly unusual, merely rare, "
        "or possibly affected by a data problem."
    ),
    "exception_relabel_review": (
        "This category checks whether a record has the wrong label or whether "
        "the current group description is incomplete."
    ),
    "feature_label_strategy": (
        "This category checks whether a field used by the system also helps "
        "people distinguish meaningful types."
    ),
    "rule_confidence_audit": (
        "This category asks whether the current plain-language description of "
        "a group also makes sense to people."
    ),
}

CATEGORY_DIMENSIONS = {
    "label_priority": (
        "priority_question",
        "priority_example",
        "priority_coverage",
        "priority_history",
    ),
    "boundary_review": (
        "boundary_other_group_nearby",
        "boundary_own_group_edge",
        "boundary_rule_line",
        "boundary_mixed_neighborhood",
        "boundary_projection_agreement",
    ),
    "overlap_merge_signal": (
        "overlap_dual_description",
        "overlap_mixed_neighbors",
        "overlap_group_resemblance",
        "overlap_human_labels",
        "overlap_outside_separation",
    ),
    "split_or_new_cluster_signal": (
        "split_separated_from_group",
        "split_consistent_pocket",
        "split_existing_group_resemblance",
        "split_human_labels",
        "split_round_stability",
    ),
    "anomaly_label_review": (
        "anomaly_within_group",
        "anomaly_isolated_or_pattern",
        "anomaly_data_quality",
        "anomaly_round_change",
        "anomaly_confirmed_examples",
    ),
    "exception_relabel_review": (
        "exception_rule_mismatch",
        "exception_disagreement_size",
        "exception_neighbor_support",
        "exception_other_group_resemblance",
        "exception_rule_scope",
    ),
    "feature_label_strategy": (
        "feature_repeated_use",
        "feature_dividing_line",
        "feature_whole_record_agreement",
        "feature_human_label_pattern",
        "feature_shortcut_risk",
    ),
    "rule_confidence_audit": (
        "audit_rule_scope",
        "audit_current_consistency",
        "audit_human_label_agreement",
        "audit_exception_pattern",
        "audit_round_stability",
    ),
}

_DIMENSION_QUESTIONS = {
    "priority_question": "Which unanswered labeling question should come first?",
    "priority_example": "Why is this record a useful example of that question?",
    "priority_coverage": "Can this one label help answer more than one question?",
    "priority_history": "Is this a first check or a justified recheck?",
    "boundary_other_group_nearby": "Are records from another group very similar to this one?",
    "boundary_own_group_edge": "Is this record near the edge of its current group?",
    "boundary_rule_line": "Is one important field close to the current dividing line?",
    "boundary_mixed_neighborhood": "Do its most similar records fall into different groups?",
    "boundary_projection_agreement": "Does the 2D plot agree with the full record?",
    "overlap_dual_description": "Do two group descriptions both include this record?",
    "overlap_mixed_neighbors": "Do its most similar records come from both groups?",
    "overlap_group_resemblance": "Does it look like a typical member of both groups?",
    "overlap_human_labels": "What do nearby human labels say about the shared area?",
    "overlap_outside_separation": "Are the two groups still different outside the shared area?",
    "split_separated_from_group": "Does this record stand apart from typical members of its group?",
    "split_consistent_pocket": "Is it alone, or are there several similar records beside it?",
    "split_existing_group_resemblance": "Does it already look like one of the existing groups?",
    "split_human_labels": "Do nearby human labels suggest one shared real-world type?",
    "split_round_stability": "Has this separation remained visible after earlier labels?",
    "anomaly_within_group": "How unusual is this record within its current group?",
    "anomaly_isolated_or_pattern": "Is it alone, or part of a small repeated pattern?",
    "anomaly_data_quality": "Could missing or extreme values explain the difference?",
    "anomaly_round_change": "Did its unusual status change after earlier labels?",
    "anomaly_confirmed_examples": "How does it compare with human-confirmed examples?",
    "exception_rule_mismatch": "Which part of the current group description does it not fit?",
    "exception_disagreement_size": "Is the mismatch slight or clear?",
    "exception_neighbor_support": "Which answer do its most similar records support?",
    "exception_other_group_resemblance": "Does it look more like a typical member of another group?",
    "exception_rule_scope": "Is this one exceptional record, or does the rule fail on many similar records?",
    "feature_repeated_use": "Why is this field being checked?",
    "feature_dividing_line": "Is this record close to the field's current dividing line?",
    "feature_whole_record_agreement": "Does this field tell the same story as the full record?",
    "feature_human_label_pattern": "Do human labels collected so far follow the same pattern?",
    "feature_shortcut_risk": "Could this field be a convenient but misleading shortcut?",
    "audit_rule_scope": "How many records in this group does the rule include?",
    "audit_current_consistency": "Does the rule usually agree with the groups shown on the dashboard?",
    "audit_human_label_agreement": "Do the human labels collected so far agree with the rule?",
    "audit_exception_pattern": "Are there records the rule handles poorly, and do they share a pattern?",
    "audit_round_stability": "Has this rule stayed similar after earlier labeling rounds?",
}


def build_category_evidence_cards(
    plan: Mapping[str, Any],
    *,
    prepared: PreparedDataset,
    analysis: AnalysisResult,
    rule_set: RuleSet,
    projection: ProjectionResult,
    active_events: Sequence[LabelEvent],
    parent_round: ActiveLearningRound | None,
    label_vocabulary: Mapping[str, str],
) -> Tuple[Mapping[str, Any], ...]:
    context = _evidence_context(
        prepared,
        analysis,
        rule_set,
        projection,
        active_events,
        parent_round,
        label_vocabulary,
    )
    return _build_cards_from_context(plan, context=context)


def build_evidence_cards_for_plans(
    plans: Mapping[str, Mapping[str, Any]],
    *,
    prepared: PreparedDataset,
    analysis: AnalysisResult,
    rule_set: RuleSet,
    projection: ProjectionResult,
    active_events: Sequence[LabelEvent],
    parent_round: ActiveLearningRound | None,
    label_vocabulary: Mapping[str, str],
) -> Dict[str, Tuple[Mapping[str, Any], ...]]:
    context = _evidence_context(
        prepared,
        analysis,
        rule_set,
        projection,
        active_events,
        parent_round,
        label_vocabulary,
    )
    return {
        category: _build_cards_from_context(plan, context=context)
        for category, plan in plans.items()
    }


def _build_cards_from_context(
    plan: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    category = str(plan.get("focus_category", "label_priority"))
    delegated = str(
        plan.get("history_context", {}).get("delegated_category", "")
    )
    evidence_category = (
        delegated
        if category == "label_priority" and delegated in CATEGORY_DIMENSIONS
        else category
    )
    cards = []
    for point_id in plan.get("recommended_point_ids", ()):
        point_context = _point_context(
            str(point_id),
            plan=plan,
            context=context,
        )
        bullets = []
        if category == "label_priority":
            bullets.extend(
                _priority_bullets(
                    point_context,
                    delegated_category=evidence_category,
                )
            )
        bullets.extend(
            _category_bullets(
                evidence_category,
                point_context,
            )
        )
        bullets = [
            _connect_bullet_to_point(
                bullet,
                context=point_context,
            )
            for bullet in bullets
        ]
        card = CategoryEvidenceCard(
            point_id=str(point_id),
            category=category,
            evidence_category=evidence_category,
            category_explanation=CATEGORY_EXPLANATIONS[category],
            evidence_policy_version=EVIDENCE_POLICY_VERSION,
            evidence_bullets=tuple(bullets),
            comparison_targets=tuple(
                _comparison_targets(point_context, evidence_category)
            ),
            round_context=_round_context(point_context),
        )
        cards.append(card.to_dict())
    return tuple(cards)


def _evidence_context(
    prepared: PreparedDataset,
    analysis: AnalysisResult,
    rule_set: RuleSet,
    projection: ProjectionResult,
    active_events: Sequence[LabelEvent],
    parent_round: ActiveLearningRound | None,
    label_vocabulary: Mapping[str, str],
) -> Dict[str, Any]:
    point_ids = tuple(prepared.feature_matrix.point_ids)
    matrix = np.asarray(prepared.feature_matrix.values, dtype=float)
    distance_matrix, neighbor_order_indices = _cached_full_feature_geometry(
        prepared.version.dataset_version_id,
        prepared.version.preprocessing_version,
        point_ids,
        prepared.feature_matrix.values,
    )
    index_by_id = {point_id: index for index, point_id in enumerate(point_ids)}
    assignments = {
        item.point_id: item.cluster_id
        for item in analysis.cluster_result.assignments
    }
    outliers = {
        item.point_id: item for item in analysis.outlier_result.scores
    }
    raw_by_id = {
        str(item["point_id"]): dict(item) for item in prepared.raw_records
    }
    projection_by_id = {
        item.point_id: np.asarray((item.x, item.y), dtype=float)
        for item in projection.coordinates
    }
    semantic_labels: Dict[str, Any] = {}
    outlier_labels: Dict[str, bool] = {}
    for event in active_events:
        if event.status != "active":
            continue
        if event.label_dimension == "semantic_class":
            semantic_labels[event.point_id] = event.label_value
        elif event.label_dimension == "outlier_status":
            outlier_labels[event.point_id] = bool(event.label_value)
    parent_assignments: Dict[str, str] = {}
    parent_outliers: Dict[str, bool] = {}
    parent_rules: Tuple[Mapping[str, Any], ...] = ()
    if parent_round is not None:
        parent_assignments = {
            item["point_id"]: item["cluster_id"]
            for item in parent_round.analysis["cluster_result"]["assignments"]
        }
        parent_outliers = {
            item["point_id"]: bool(item["is_outlier"])
            for item in parent_round.analysis["outlier_result"]["scores"]
        }
        parent_rules = tuple(parent_round.rule_set.get("rules", ()))

    members_by_group: Dict[str, list[str]] = {}
    for point_id, group_id in assignments.items():
        members_by_group.setdefault(group_id, []).append(point_id)
    centers = {}
    center_distances = {}
    group_radii = {}
    for group_id, member_ids in members_by_group.items():
        indices = [index_by_id[point_id] for point_id in member_ids]
        center = np.mean(matrix[indices], axis=0)
        distances = np.linalg.norm(matrix[indices] - center, axis=1)
        centers[group_id] = center
        center_distances[group_id] = {
            point_id: float(distance)
            for point_id, distance in zip(member_ids, distances)
        }
        group_radii[group_id] = float(np.mean(distances)) if len(distances) else 0.0

    transform_by_model = {
        str(item.get("model_feature")): dict(item)
        for item in prepared.version.transformation_map
    }
    source_by_model = {
        name: str(transform_by_model.get(name, {}).get("source_feature", name))
        for name in prepared.feature_matrix.feature_names
    }
    return {
        "prepared": prepared,
        "analysis": analysis,
        "rule_set": rule_set,
        "projection": projection,
        "point_ids": point_ids,
        "matrix": matrix,
        "distance_matrix": distance_matrix,
        "neighbor_order_indices": neighbor_order_indices,
        "index_by_id": index_by_id,
        "assignments": assignments,
        "outliers": outliers,
        "raw_by_id": raw_by_id,
        "projection_by_id": projection_by_id,
        "semantic_labels": semantic_labels,
        "outlier_labels": outlier_labels,
        "label_vocabulary": dict(label_vocabulary),
        "parent_round": parent_round,
        "parent_assignments": parent_assignments,
        "parent_outliers": parent_outliers,
        "parent_rules": parent_rules,
        "members_by_group": members_by_group,
        "centers": centers,
        "center_distances": center_distances,
        "group_radii": group_radii,
        "transform_by_model": transform_by_model,
        "source_by_model": source_by_model,
        "rules_by_id": {rule.rule_id: rule for rule in rule_set.rules},
    }


def _point_context(
    point_id: str,
    *,
    plan: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    matrix = context["matrix"]
    index = context["index_by_id"][point_id]
    vector = matrix[index]
    assignments = context["assignments"]
    group_id = assignments.get(point_id, "")
    distances = context["distance_matrix"][index]
    ordered_indices = context["neighbor_order_indices"][index]
    neighbor_count = min(
        len(ordered_indices),
        max(0, min(12, max(5, round(sqrt(len(context["point_ids"])))))),
    )
    neighbor_ids = tuple(
        context["point_ids"][candidate_index]
        for candidate_index in ordered_indices[:neighbor_count]
    )
    neighbor_groups = tuple(assignments.get(item, "") for item in neighbor_ids)
    other_neighbor_ids = tuple(
        item
        for item in neighbor_ids
        if assignments.get(item, "") != group_id
    )
    other_ratio = (
        len(other_neighbor_ids) / len(neighbor_ids) if neighbor_ids else 0.0
    )
    own_distances = context["center_distances"].get(group_id, {})
    point_center_distance = float(own_distances.get(point_id, 0.0))
    ordered_center_distances = sorted(own_distances.values())
    edge_percentile = (
        sum(value <= point_center_distance for value in ordered_center_distances)
        / len(ordered_center_distances)
        if ordered_center_distances
        else 0.0
    )
    group_center_distances = {
        candidate_group: float(np.linalg.norm(vector - center))
        for candidate_group, center in context["centers"].items()
    }
    ordered_groups = sorted(
        group_center_distances,
        key=lambda candidate_group: (
            group_center_distances[candidate_group],
            candidate_group,
        ),
    )
    nearest_group = ordered_groups[0] if ordered_groups else ""
    second_group = ordered_groups[1] if len(ordered_groups) > 1 else ""
    group_distance_ratio = None
    if nearest_group and second_group:
        first_distance = group_center_distances[nearest_group]
        second_distance = group_center_distances[second_group]
        group_distance_ratio = second_distance / max(first_distance, 1e-12)

    target_rules = tuple(
        context["rules_by_id"][rule_id]
        for rule_id in plan.get("target_rule_ids", ())
        if rule_id in context["rules_by_id"]
    )
    related_rules = tuple(
        rule
        for rule in context["rule_set"].rules
        if point_id in rule.matched_point_ids
        or point_id in rule.exception_point_ids
    )
    boundary = _closest_boundary(
        vector,
        target_rules or related_rules,
        context,
    )
    projection_overlap = _projection_neighbor_overlap(
        point_id,
        neighbor_ids,
        context,
    )
    profile = next(
        (
            dict(item)
            for item in plan.get("point_profiles", ())
            if item.get("point_id") == point_id
        ),
        {"point_id": point_id},
    )
    raw = context["raw_by_id"][point_id]
    missing_features = tuple(
        name
        for name, value in raw.get("raw_features", {}).items()
        if value is None or str(value).strip() == ""
    )
    extreme_features = tuple(
        _plain_feature(context["source_by_model"].get(name, name))
        for name, value in zip(
            context["prepared"].feature_matrix.feature_names,
            vector,
        )
        if abs(float(value)) >= 3.0
    )
    labeled_neighbor_ids = tuple(
        item for item in neighbor_ids if item in context["semantic_labels"]
    )
    labeled_neighbor_values = tuple(
        context["semantic_labels"][item] for item in labeled_neighbor_ids
    )
    outlier = context["outliers"].get(point_id)
    outlier_scores = sorted(
        float(item.score) for item in context["outliers"].values()
    )
    outlier_percentile = 0.0
    if outlier is not None and outlier_scores:
        outlier_percentile = (
            sum(value <= float(outlier.score) for value in outlier_scores)
            / len(outlier_scores)
        )
    return {
        **dict(context),
        "plan": dict(plan),
        "point_id": point_id,
        "vector": vector,
        "group_id": group_id,
        "distances": distances,
        "neighbor_ids": neighbor_ids,
        "neighbor_groups": neighbor_groups,
        "other_neighbor_ids": other_neighbor_ids,
        "other_ratio": other_ratio,
        "edge_percentile": edge_percentile,
        "point_center_distance": point_center_distance,
        "group_center_distances": group_center_distances,
        "nearest_group": nearest_group,
        "second_group": second_group,
        "group_distance_ratio": group_distance_ratio,
        "target_rules": target_rules,
        "related_rules": related_rules,
        "boundary": boundary,
        "projection_overlap": projection_overlap,
        "profile": profile,
        "raw_record": raw,
        "missing_features": missing_features,
        "extreme_features": tuple(dict.fromkeys(extreme_features)),
        "labeled_neighbor_ids": labeled_neighbor_ids,
        "labeled_neighbor_values": labeled_neighbor_values,
        "outlier": outlier,
        "outlier_percentile": outlier_percentile,
    }


@lru_cache(maxsize=2)
def _cached_full_feature_geometry(
    dataset_version_id: str,
    preprocessing_version: str,
    point_ids: Tuple[str, ...],
    values: Tuple[Tuple[float, ...], ...],
) -> Tuple[np.ndarray, Tuple[Tuple[int, ...], ...]]:
    del dataset_version_id, preprocessing_version
    matrix = np.asarray(values, dtype=float)
    squared_norms = np.sum(matrix * matrix, axis=1)
    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (matrix @ matrix.T)
    )
    distances = np.sqrt(
        np.maximum(squared_distances, 0.0),
        dtype=float,
    )
    point_id_array = np.asarray(point_ids, dtype=object)
    ordered = []
    for index in range(len(point_ids)):
        row_order = np.lexsort((point_id_array, distances[index]))
        ordered.append(
            tuple(
                int(candidate_index)
                for candidate_index in row_order
                if int(candidate_index) != index
            )
        )
    distances.setflags(write=False)
    return distances, tuple(ordered)


def _closest_boundary(
    vector: np.ndarray,
    rules: Sequence[RuleCard],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    feature_index = {
        name: index
        for index, name in enumerate(
            context["prepared"].feature_matrix.feature_names
        )
    }
    candidates = []
    categorical_conditions = []
    for rule in rules:
        for condition in rule.conditions:
            index = feature_index.get(condition.feature)
            if index is None:
                continue
            transform = context["transform_by_model"].get(condition.feature, {})
            display = display_condition(
                condition.feature,
                condition.operator,
                condition.threshold,
                context["prepared"].version.transformation_map,
            )
            item = {
                "rule_id": rule.rule_id,
                "target_id": rule.target_id,
                "model_feature": condition.feature,
                "source_feature": display.get(
                    "source_feature",
                    condition.feature,
                ),
                "kind": transform.get("kind", "numeric"),
                "operator": condition.operator,
                "threshold": float(condition.threshold),
                "point_value": float(vector[index]),
                "display_text": display.get("display_text", ""),
            }
            if item["kind"] == "categorical":
                categorical_conditions.append(item)
                continue
            item["normalized_margin"] = abs(
                item["point_value"] - item["threshold"]
            )
            candidates.append(item)
    if candidates:
        return dict(
            sorted(
                candidates,
                key=lambda item: (
                    item["normalized_margin"],
                    item["rule_id"],
                    item["model_feature"],
                ),
            )[0]
        )
    if categorical_conditions:
        return {
            **dict(categorical_conditions[0]),
            "categorical_only": True,
        }
    return {}


def _projection_neighbor_overlap(
    point_id: str,
    feature_neighbor_ids: Sequence[str],
    context: Mapping[str, Any],
) -> float | None:
    point = context["projection_by_id"].get(point_id)
    if point is None or not feature_neighbor_ids:
        return None
    projection_distances = []
    for candidate_id, coordinate in context["projection_by_id"].items():
        if candidate_id == point_id:
            continue
        projection_distances.append(
            (
                float(np.linalg.norm(coordinate - point)),
                candidate_id,
            )
        )
    projection_neighbors = {
        point_id
        for _, point_id in sorted(projection_distances)[
            : len(feature_neighbor_ids)
        ]
    }
    return len(projection_neighbors & set(feature_neighbor_ids)) / len(
        feature_neighbor_ids
    )


def _priority_bullets(
    context: Mapping[str, Any],
    *,
    delegated_category: str,
) -> Tuple[EvidenceBullet, ...]:
    plan = context["plan"]
    point_id = context["point_id"]
    covered = tuple(context["profile"].get("covered_categories", ()))
    ranking = next(
        (
            item
            for item in plan.get("candidate_rankings", ())
            if item.get("point_id") == point_id
        ),
        {},
    )
    recheck_reason = str(ranking.get("recheck_reason", ""))
    diff = dict(plan.get("previous_plan_diff", {}))
    delegated_name = _plain_category_name(delegated_category)
    return (
        _bullet(
            point_id,
            "priority_question",
            "yes",
            f"The most useful open question is about {delegated_name}.",
            (
                "The current evidence makes this the clearest unresolved "
                "labeling question in this round."
            ),
            (
                "Starting here gives the next round a better chance to resolve "
                "an uncertainty that is still active."
            ),
            {
                "delegated_category": delegated_category,
                "meta_score_components": dict(
                    plan.get("history_context", {}).get(
                        "meta_score_components",
                        {},
                    )
                ),
            },
        ),
        _bullet(
            point_id,
            "priority_example",
            "yes",
            "This point is one of the clearest available examples.",
            (
                "Compared with the other available records, it shows the "
                "current question especially clearly."
            ),
            (
                "A clear example is easier to compare and is less likely to "
                "consume a label without clarifying the current question."
            ),
            {
                "candidate_rank": ranking.get("candidate_rank"),
                "selection_reason": ranking.get("selection_reason", ""),
            },
        ),
        _bullet(
            point_id,
            "priority_coverage",
            "yes" if len(covered) > 1 else "no",
            (
                "One label can inform more than one current question."
                if len(covered) > 1
                else "This label mainly addresses one focused question."
            ),
            (
                "The point also appears in other current checks."
                if len(covered) > 1
                else "The reasons for showing it mainly concern this one question."
            ),
            (
                "Its label can reduce several uncertainties at once."
                if len(covered) > 1
                else "A focused check can still be useful when the question is important."
            ),
            {"covered_categories": list(covered)},
        ),
        _bullet(
            point_id,
            "priority_history",
            "partly" if recheck_reason else "yes",
            (
                "This point is being checked again after the analysis changed."
                if recheck_reason
                else "This is a new check in the current labeling batch."
            ),
            (
                _plain_recheck_reason(recheck_reason)
                if recheck_reason
                else (
                    "It was not retained from the previous recommended batch."
                    if point_id in set(diff.get("added_point_ids", ()))
                    else "It has no unresolved recheck reason in the current plan."
                )
            ),
            (
                "A recheck is useful only when new evidence has changed the "
                "question around the point."
                if recheck_reason
                else "New checks help the session cover records it has not already emphasized."
            ),
            {
                "recheck_reason": recheck_reason,
                "added_this_round": point_id
                in set(diff.get("added_point_ids", ())),
            },
        ),
    )


def _category_bullets(
    category: str,
    context: Mapping[str, Any],
) -> Tuple[EvidenceBullet, ...]:
    builders = {
        "boundary_review": _boundary_bullets,
        "overlap_merge_signal": _overlap_bullets,
        "split_or_new_cluster_signal": _split_bullets,
        "anomaly_label_review": _anomaly_bullets,
        "exception_relabel_review": _exception_bullets,
        "feature_label_strategy": _feature_bullets,
        "rule_confidence_audit": _audit_bullets,
    }
    builder = builders.get(category)
    return builder(context) if builder is not None else ()


def _connect_bullet_to_point(
    bullet: EvidenceBullet,
    *,
    context: Mapping[str, Any],
) -> EvidenceBullet:
    """Make every category-level finding explain why this record is the test."""

    dimension_id = bullet.dimension_id
    point_connection = bullet.plain_fact
    labeling_value = bullet.why_it_matters

    if bullet.status == "insufficient":
        point_connection = (
            f"For this record, {bullet.plain_fact[:1].lower()}"
            f"{bullet.plain_fact[1:]}"
        )
        labeling_value = (
            "This unanswered check should not be used as a reason by itself. "
            "A human label now may supply the missing reference in a later round."
        )

    if dimension_id.startswith("feature_"):
        feature = _plain_feature(
            bullet.technical_details.get("source_feature")
            or bullet.technical_details.get("feature")
            or context.get("boundary", {}).get("source_feature")
            or "this field"
        )
        if dimension_id == "feature_repeated_use":
            point_connection = (
                f"This record is included because its {feature} value is part "
                "of the group description being checked."
            )
            labeling_value = (
                "If the record's human type agrees with the whole record but "
                f"not with the {feature} clue, that field may be a shortcut "
                "rather than a meaningful distinction."
            )
        elif dimension_id == "feature_human_label_pattern":
            point_connection = (
                "This record does not yet provide a human check for the field's "
                "current pattern."
            )
            labeling_value = (
                f"Its label would show whether the {feature} clue continues to "
                "work on a new case, or fails when a person judges the full record."
            )
        elif dimension_id == "feature_shortcut_risk":
            point_connection = (
                f"This record is a concrete test of whether {feature} reflects "
                "its full meaning instead of only an easy system pattern."
            )
            labeling_value = (
                "Agreement with the full record supports using the field as a "
                "clue; disagreement means it should not drive future labels."
            )

    if not dimension_id.startswith("audit_"):
        labeling_value = _labeling_value_for_dimension(
            bullet,
            context=context,
            fallback=labeling_value,
        )
    else:
        point_connection, labeling_value = _audit_point_connection(
            bullet,
            context=context,
        )

    return replace(
        bullet,
        point_connection=point_connection,
        labeling_value=labeling_value,
    )


def _labeling_value_for_dimension(
    bullet: EvidenceBullet,
    *,
    context: Mapping[str, Any],
    fallback: str,
) -> str:
    dimension_id = bullet.dimension_id
    status = bullet.status
    supports = status in {"yes", "partly"}

    if status == "insufficient":
        if dimension_id in {
            "overlap_human_labels",
            "split_human_labels",
            "anomaly_confirmed_examples",
            "feature_human_label_pattern",
        }:
            return (
                "Labeling this record would add a human reference for this "
                "question instead of asking the system to guess."
            )
        if dimension_id in {
            "split_round_stability",
            "anomaly_round_change",
            "priority_history",
        }:
            return (
                "Its label would create a baseline that can be checked again "
                "after the next round."
            )
        return (
            "This unanswered check is not a reason by itself. A human label "
            "may provide the missing reference for a later round."
        )

    if dimension_id == "priority_question":
        return (
            "Labeling this record now addresses the open question ranked ahead "
            "of the other available checks."
        )
    if dimension_id == "priority_example":
        return (
            "Its human label will test a concrete example of that question; "
            "the answer can either support the current concern or rule it out."
        )
    if dimension_id == "priority_coverage":
        covered = tuple(
            context.get("profile", {}).get("covered_categories", ())
        )
        return (
            "One human label can inform several open questions at once."
            if len(covered) > 1
            else (
                "This human label answers one focused question rather than "
                "being treated as evidence for unrelated concerns."
            )
        )
    if dimension_id == "priority_history":
        return (
            "A human label can test whether the new reason for revisiting this "
            "record is meaningful."
            if context.get("profile", {}).get("recheck_reason")
            else "Its label adds a new case instead of repeating an earlier check."
        )

    if dimension_id == "boundary_other_group_nearby":
        return (
            "Its human label can show whether similar records on opposite sides "
            "of the current grouping are genuinely different."
            if supports
            else (
                "This answer does not make the record a boundary case by itself; "
                "its recommendation must be supported by another check below."
            )
        )
    if dimension_id == "boundary_own_group_edge":
        return (
            "Its human label can help decide whether the current group should "
            "end before or after records like this one."
            if supports
            else (
                "This answer does not support labeling it as an edge case; "
                "look to the dividing-line or neighbor checks for the reason."
            )
        )
    if dimension_id == "boundary_rule_line":
        return (
            "Its human label can show whether the current dividing line falls "
            "at a real-world difference or cuts through one type."
            if supports
            else (
                "This field is not the reason to label the record now; another "
                "boundary check must explain the recommendation."
            )
        )
    if dimension_id == "boundary_mixed_neighborhood":
        return (
            "Its human label can clarify which side similar nearby records "
            "should follow."
            if supports
            else (
                "Nearby records already tell a consistent story, so this check "
                "does not add a boundary concern."
            )
        )
    if dimension_id == "boundary_projection_agreement":
        return (
            "This answer tells the user whether the plot is a useful comparison "
            "guide; the human label should still come from the full record."
        )

    if dimension_id == "overlap_dual_description":
        return (
            "Its human label can show whether the shared description represents "
            "one real-world type or hides a meaningful difference."
            if supports
            else (
                "This record does not support the shared-description concern by "
                "itself; another overlap check must justify attention."
            )
        )
    if dimension_id == "overlap_mixed_neighbors":
        return (
            "Its human label can show whether similar records from both groups "
            "should share one type."
            if supports
            else "Its neighbors do not currently provide a reason to combine types."
        )
    if dimension_id == "overlap_group_resemblance":
        return (
            "Its human label can reveal whether looking typical of both groups "
            "has one meaning or two."
            if supports
            else (
                "Its stronger resemblance to one group argues against treating "
                "this record as shared evidence."
            )
        )
    if dimension_id == "overlap_human_labels":
        return (
            "Labeling this record adds another human judgment to the shared area "
            "and can confirm or challenge the pattern already seen there."
        )
    if dimension_id == "overlap_outside_separation":
        return (
            "Its human label tests the shared area only; it should not be used "
            "to claim that the two entire groups have the same meaning."
        )

    if dimension_id == "split_separated_from_group":
        return (
            "Its human label can show whether this separated area still has the "
            "same meaning as the current group."
            if supports
            else (
                "Its closeness to typical members favors the existing type; a "
                "human label can confirm that no separate meaning is needed."
            )
        )
    if dimension_id == "split_consistent_pocket":
        return (
            "If this record and its similar neighbors receive the same human "
            "type, the small area may have a consistent meaning."
            if supports
            else (
                "A human label can show whether this is merely one unusual case "
                "rather than a repeated type."
            )
        )
    if dimension_id == "split_existing_group_resemblance":
        return (
            "Its human label can decide whether an existing type already fits "
            "or whether the record has a different real-world meaning."
        )
    if dimension_id == "split_human_labels":
        return (
            "Its human label can confirm or challenge the type suggested by "
            "nearby reviewed records."
        )
    if dimension_id == "split_round_stability":
        return (
            "Its label can test whether a separation that survived earlier "
            "rounds also has a stable human meaning."
        )

    if dimension_id == "anomaly_within_group":
        return (
            "A human label can decide whether the difference is genuinely "
            "unusual or still a normal variation of the same type."
        )
    if dimension_id == "anomaly_isolated_or_pattern":
        return (
            "Its human label can distinguish a rare but repeated type from a "
            "single unusual case or possible data problem."
        )
    if dimension_id == "anomaly_data_quality":
        return (
            "Check the source record first; if the data is valid, the human "
            "label can decide whether the difference has real-world meaning."
        )
    if dimension_id == "anomaly_round_change":
        return (
            "Its human label can show whether the changed unusual status is a "
            "real improvement or an unstable system reaction."
        )
    if dimension_id == "anomaly_confirmed_examples":
        return (
            "Comparing its human label with reviewed examples can show whether "
            "it follows a known normal or unusual pattern."
        )

    if dimension_id == "exception_rule_mismatch":
        return (
            "Its human label can show whether this record is assigned the wrong "
            "type or the simple group description is incomplete."
        )
    if dimension_id == "exception_disagreement_size":
        return (
            "Its human label can show whether a small mismatch is harmless or "
            "whether a clear mismatch exposes a weak description."
        )
    if dimension_id == "exception_neighbor_support":
        return (
            "Its human label can confirm which answer from the similar records "
            "matches the record's real-world meaning."
        )
    if dimension_id == "exception_other_group_resemblance":
        return (
            "Its human label can decide whether the other existing type is a "
            "better fit or whether neither description is adequate."
        )
    if dimension_id == "exception_rule_scope":
        return (
            "Its human label helps distinguish one exceptional record from a "
            "repeated kind of case that the rule handles poorly."
        )

    if dimension_id == "feature_repeated_use":
        return (
            "Its human label can test whether a frequently used field also "
            "reflects a difference that matters to people."
        )
    if dimension_id == "feature_dividing_line":
        return (
            "Its human label can show whether this field's dividing line matches "
            "a real-world distinction in a difficult case."
        )
    if dimension_id == "feature_whole_record_agreement":
        return (
            "Its human label can show whether the field deserves to remain a "
            "clue or should be overruled by the full record."
        )
    if dimension_id == "feature_human_label_pattern":
        return (
            "Its human label adds a new test of whether this field follows the "
            "same pattern people use."
        )
    if dimension_id == "feature_shortcut_risk":
        return (
            "Its human label can show whether the field captures real meaning "
            "or only an easy pattern in the current data."
        )
    return fallback


def _audit_point_connection(
    bullet: EvidenceBullet,
    *,
    context: Mapping[str, Any],
) -> tuple[str, str]:
    rules = context["target_rules"] or context["related_rules"]
    rule = rules[0] if rules else None
    if rule is None:
        return (
            "There is no group description that can be tied to this record yet.",
            "Do not treat this missing check as a reason to change the record's label.",
        )

    point_id = context["point_id"]
    is_exception = point_id in set(rule.exception_point_ids)
    is_match = point_id in set(rule.matched_point_ids)
    if is_exception:
        role = (
            "This record is one of the cases where the rule and the current "
            "grouping disagree."
        )
    elif is_match:
        role = (
            "This record is one of the cases the rule includes, so it directly "
            "tests the kind of record the rule claims to describe."
        )
    else:
        role = (
            "This record is linked to the rule as a nearby comparison case, "
            "rather than as a clear match."
        )

    if bullet.dimension_id == "audit_rule_scope":
        return (
            role,
            (
                "If a person gives it the type expected by the rule, that "
                "supports using the description for similar records. If not, "
                "the rule may be including too many kinds of records."
            ),
        )
    if bullet.dimension_id == "audit_current_consistency":
        return (
            role,
            (
                "Its human label can show whether agreement between the rule "
                "and the groups shown on the dashboard also reflects a "
                "distinction that matters to people."
            ),
        )
    if bullet.dimension_id == "audit_human_label_agreement":
        human_label = context["semantic_labels"].get(point_id)
        if human_label is None:
            connection = (
                "This record has no confirmed human type yet, so the rule's "
                "answer for this case has never been checked by a person."
            )
        else:
            connection = (
                "This record already has a confirmed human type and can be used "
                "to compare the rule with that judgment."
            )
        return (
            connection,
            (
                "If its human label agrees, that supports this part of the "
                "description; if it disagrees, it identifies a specific case "
                "the rule cannot explain."
            ),
        )
    if bullet.dimension_id == "audit_exception_pattern":
        exception_count = int(
            bullet.technical_details.get("exception_count", 0)
        )
        if exception_count == 0:
            return (
                role,
                (
                    "A matching human label would support this clean result; "
                    "a disagreement would reveal the first case where a "
                    "person's label says the rule handles the record poorly."
                ),
            )
        return (
            role,
            (
                "Labeling this record helps distinguish a one-off difficult "
                "case from a repeated kind of case that the rule handles poorly."
            ),
        )
    if bullet.dimension_id == "audit_round_stability":
        if context["parent_round"] is None:
            connection = (
                "This is the first round, so this record can only establish a "
                "human reference for later rounds."
            )
        else:
            connection = (
                "This same record can now be compared with how the group "
                "description behaved before the latest labels."
            )
        return (
            connection,
            (
                "Its label creates a stable reference for judging whether later "
                "rule changes improve the description or merely move it around."
            ),
        )
    return role, bullet.why_it_matters


def _boundary_bullets(context: Mapping[str, Any]) -> Tuple[EvidenceBullet, ...]:
    other_ratio = float(context["other_ratio"])
    if not context["neighbor_ids"]:
        nearby = _insufficient(
            context,
            "boundary_other_group_nearby",
            "There are not enough records to compare nearby groups.",
        )
    elif other_ratio >= 0.5:
        nearby = _bullet(
            context["point_id"],
            "boundary_other_group_nearby",
            "yes",
            "It is close to another group.",
            "Many of the records most similar to this one currently belong to another group.",
            "Its label can show whether those nearby records are genuinely different.",
            _neighbor_details(context),
        )
    elif other_ratio >= 0.25:
        nearby = _bullet(
            context["point_id"],
            "boundary_other_group_nearby",
            "partly",
            "Some nearby records belong to another group.",
            "Its closest records are mostly from the current group, but another group is also present.",
            "This makes it useful for checking whether the dividing line needs a closer look.",
            _neighbor_details(context),
        )
    else:
        nearby = _bullet(
            context["point_id"],
            "boundary_other_group_nearby",
            "no",
            "It is not especially close to another group.",
            "Most of the records that resemble it are currently in the same group.",
            "The recommendation therefore depends more on the rule line or another boundary clue.",
            _neighbor_details(context),
        )

    edge = _edge_bullet(
        context,
        dimension_id="boundary_own_group_edge",
        yes_headline="It sits near the edge of its current group.",
        partly_headline="It is moving toward the edge of its current group.",
        no_headline="It looks fairly typical of its current group.",
        why=(
            "Checking an edge case helps a person decide where the current "
            "group should end."
        ),
    )
    boundary = _boundary_line_bullet(
        context,
        dimension_id="boundary_rule_line",
        why=(
            "A human label can test whether this simple dividing line matches "
            "a real-world distinction."
        ),
    )
    mixed = _mixed_neighbor_bullet(
        context,
        dimension_id="boundary_mixed_neighborhood",
        why=(
            "A mixed neighborhood is worth checking because the area does not "
            "clearly support only one side."
        ),
    )
    projection = _projection_bullet(context)
    return nearby, edge, boundary, mixed, projection


def _overlap_bullets(context: Mapping[str, Any]) -> Tuple[EvidenceBullet, ...]:
    matched_target_rules = [
        rule
        for rule in context["target_rules"]
        if context["point_id"] in rule.matched_point_ids
    ]
    if len(matched_target_rules) >= 2:
        dual = _bullet(
            context["point_id"],
            "overlap_dual_description",
            "yes",
            "It fits both current group descriptions.",
            "Important parts of two group descriptions include this record.",
            "Its label can show whether the descriptions refer to one type or overlap only in this area.",
            {
                "matched_target_rule_ids": [
                    rule.rule_id for rule in matched_target_rules
                ]
            },
        )
    elif len(matched_target_rules) == 1:
        dual = _bullet(
            context["point_id"],
            "overlap_dual_description",
            "partly",
            "Only one description clearly includes this point.",
            "The second group is linked through nearby records rather than a direct rule match.",
            "This is weaker overlap evidence, so the human comparison matters more than the rule alone.",
            {
                "matched_target_rule_ids": [
                    rule.rule_id for rule in matched_target_rules
                ]
            },
        )
    else:
        dual = _bullet(
            context["point_id"],
            "overlap_dual_description",
            "no",
            "Neither target description directly includes this point.",
            "The point was linked to the overlap question through its surrounding region.",
            "The explanation must rely on nearby examples rather than claiming a direct shared rule.",
            {"matched_target_rule_ids": []},
        )
    mixed = _mixed_neighbor_bullet(
        context,
        dimension_id="overlap_mixed_neighbors",
        why=(
            "Its label can show whether the shared neighborhood represents one "
            "real-world type or two."
        ),
    )
    resemblance = _group_resemblance_bullet(
        context,
        dimension_id="overlap_group_resemblance",
        overlap_wording=True,
    )
    labels = _human_neighbor_labels_bullet(
        context,
        dimension_id="overlap_human_labels",
        same_headline="Existing human labels in this area mostly agree.",
        mixed_headline="Existing human labels in this area disagree.",
        why=(
            "Human labels are the strongest evidence for whether the shared "
            "area has one meaning or several."
        ),
    )
    separation = _outside_group_separation_bullet(context)
    return dual, mixed, resemblance, labels, separation


def _split_bullets(context: Mapping[str, Any]) -> Tuple[EvidenceBullet, ...]:
    separated = _edge_bullet(
        context,
        dimension_id="split_separated_from_group",
        yes_headline="It is separated from typical members of its current group.",
        partly_headline="It is less typical than many members of its current group.",
        no_headline="It remains close to typical members of its current group.",
        why=(
            "A human label can show whether this distant area still has the "
            "same real-world meaning."
        ),
    )
    same_group_neighbors = sum(
        group_id == context["group_id"]
        for group_id in context["neighbor_groups"]
    )
    if same_group_neighbors >= 3:
        pocket = _bullet(
            context["point_id"],
            "split_consistent_pocket",
            "no",
            "It is part of a small, consistent pocket.",
            "Several nearby records resemble this one rather than leaving it isolated.",
            "A shared human type would make this area more meaningful than a single unusual record.",
            {
                **_neighbor_details(context),
                "same_group_neighbor_count": same_group_neighbors,
            },
        )
    elif same_group_neighbors:
        pocket = _bullet(
            context["point_id"],
            "split_consistent_pocket",
            "partly",
            "It has a few similar companions.",
            "Some nearby records support this pattern, but the pocket is still small.",
            "More labels are needed before treating the area as a stable type.",
            {
                **_neighbor_details(context),
                "same_group_neighbor_count": same_group_neighbors,
            },
        )
    else:
        pocket = _bullet(
            context["point_id"],
            "split_consistent_pocket",
            "yes",
            "It currently looks isolated.",
            "The nearest records do not form a clear same-group pocket around it.",
            "An isolated case is weaker evidence for a separate type and may instead be unusual.",
            {
                **_neighbor_details(context),
                "same_group_neighbor_count": same_group_neighbors,
            },
        )
    resemblance = _group_resemblance_bullet(
        context,
        dimension_id="split_existing_group_resemblance",
        overlap_wording=False,
    )
    labels = _human_neighbor_labels_bullet(
        context,
        dimension_id="split_human_labels",
        same_headline="Nearby human labels support a shared type.",
        mixed_headline="Nearby human labels do not yet support one shared type.",
        why=(
            "A consistent human label can connect this area to an existing "
            "type or justify checking it separately."
        ),
    )
    stability = _round_stability_bullet(context)
    return separated, pocket, resemblance, labels, stability


def _anomaly_bullets(context: Mapping[str, Any]) -> Tuple[EvidenceBullet, ...]:
    flagged = bool(context["outlier"] and context["outlier"].is_outlier)
    percentile = float(context["outlier_percentile"])
    if flagged or percentile >= 0.8:
        unusual = _bullet(
            context["point_id"],
            "anomaly_within_group",
            "yes",
            "It looks unusual within its current group.",
            "Its overall pattern differs more from the group than most members do.",
            "A human check can separate a meaningful rare case from an ordinary member.",
            {
                "is_currently_flagged_unusual": flagged,
                "outlier_score_percentile": round(percentile, 6),
            },
        )
    elif percentile >= 0.5:
        unusual = _bullet(
            context["point_id"],
            "anomaly_within_group",
            "partly",
            "It looks somewhat unusual, but not extreme.",
            "It differs from many group members without standing completely apart.",
            "This is a useful borderline case for deciding how much difference should count as unusual.",
            {
                "is_currently_flagged_unusual": flagged,
                "outlier_score_percentile": round(percentile, 6),
            },
        )
    else:
        unusual = _bullet(
            context["point_id"],
            "anomaly_within_group",
            "no",
            "It does not look strongly unusual overall.",
            "Most of the current evidence places it within the ordinary range of the group.",
            "The recommendation may instead be testing a specific anomaly rule or a recent status change.",
            {
                "is_currently_flagged_unusual": flagged,
                "outlier_score_percentile": round(percentile, 6),
            },
        )
    similar_count = sum(
        context["assignments"].get(point_id) == context["group_id"]
        for point_id in context["neighbor_ids"][:5]
    )
    if similar_count >= 3:
        pattern = _bullet(
            context["point_id"],
            "anomaly_isolated_or_pattern",
            "yes",
            "It is part of a rare pattern rather than completely alone.",
            "Several of its closest records show a related pattern.",
            "This makes a rare but valid type more plausible than a one-off error.",
            {
                "same_group_neighbors_among_five": similar_count,
                **_neighbor_details(context),
            },
        )
    elif similar_count:
        pattern = _bullet(
            context["point_id"],
            "anomaly_isolated_or_pattern",
            "partly",
            "It has only a few similar neighbors.",
            "Some records resemble it, but the local pattern is weak.",
            "A human label can help decide whether those few examples belong together.",
            {
                "same_group_neighbors_among_five": similar_count,
                **_neighbor_details(context),
            },
        )
    else:
        pattern = _bullet(
            context["point_id"],
            "anomaly_isolated_or_pattern",
            "yes",
            "It currently looks isolated.",
            "Very few nearby records show a similar pattern.",
            "This increases the need to check the source record for a meaningful exception or data problem.",
            {
                "same_group_neighbors_among_five": similar_count,
                **_neighbor_details(context),
            },
        )
    quality = _data_quality_bullet(context)
    change = _unusual_round_change_bullet(context)
    examples = _confirmed_outlier_examples_bullet(context)
    return unusual, pattern, quality, change, examples


def _exception_bullets(context: Mapping[str, Any]) -> Tuple[EvidenceBullet, ...]:
    exception_rules = [
        rule
        for rule in context["related_rules"]
        if context["point_id"] in rule.exception_point_ids
    ]
    boundary = context["boundary"]
    feature = _plain_feature(boundary.get("source_feature", "a key field"))
    if exception_rules:
        mismatch = _bullet(
            context["point_id"],
            "exception_rule_mismatch",
            "yes",
            "It does not fit the usual description of its assigned group.",
            (
                f"The current description based on {feature} includes this "
                "record, but the analysis assigns it differently."
            ),
            "A human label can show whether the record or the simple description needs reconsideration.",
            {
                "exception_rule_ids": [
                    rule.rule_id for rule in exception_rules
                ],
                "feature": boundary.get("source_feature"),
            },
        )
    else:
        mismatch = _bullet(
            context["point_id"],
            "exception_rule_mismatch",
            "partly",
            "Its connection to a rule exception is indirect.",
            "Nearby or related records break the current description, although this point is not a direct exception.",
            "The user should compare it with the direct exceptions before changing its label.",
            {"exception_rule_ids": []},
        )
    disagreement = _disagreement_size_bullet(context)
    neighbor = _neighbor_suggestion_bullet(context)
    resemblance = _group_resemblance_bullet(
        context,
        dimension_id="exception_other_group_resemblance",
        overlap_wording=False,
    )
    scope = _exception_scope_bullet(context, exception_rules)
    return mismatch, disagreement, neighbor, resemblance, scope


def _feature_bullets(context: Mapping[str, Any]) -> Tuple[EvidenceBullet, ...]:
    boundary = context["boundary"]
    feature = str(boundary.get("source_feature", "")).strip()
    usage_by_source: Dict[str, int] = {}
    for model_feature, count in context["rule_set"].diagnostics.get(
        "feature_usage",
        {},
    ).items():
        source = context["source_by_model"].get(model_feature, model_feature)
        usage_by_source[source] = usage_by_source.get(source, 0) + int(count)
    if not feature and usage_by_source:
        feature = sorted(
            usage_by_source,
            key=lambda name: (-usage_by_source[name], name),
        )[0]
    plain_feature = _plain_feature(feature or "a key field")
    usage = int(usage_by_source.get(feature, 0))
    max_usage = max(usage_by_source.values(), default=0)
    repeated = _bullet(
        context["point_id"],
        "feature_repeated_use",
        "yes" if usage >= 2 else "partly" if usage else "insufficient",
        (
            f"{plain_feature.title()} appears repeatedly in the current descriptions."
            if usage >= 2
            else f"{plain_feature.title()} appears in a current description."
            if usage
            else "There is not enough rule evidence to identify one repeated field."
        ),
        (
            "The system uses this field in several current group descriptions."
            if usage >= 2
            else "The field is relevant to this point, but it is not repeatedly used."
            if usage
            else "No current rule provides a reliable feature-use pattern."
        ),
        (
            "A human label can test whether the repeated field also reflects a "
            "difference that matters to people."
        ),
        {
            "source_feature": feature,
            "feature_rule_use_count": usage,
            "maximum_feature_rule_use_count": max_usage,
        },
    )
    dividing = _boundary_line_bullet(
        context,
        dimension_id="feature_dividing_line",
        why=(
            "A record near the dividing line is useful for testing whether this "
            "field remains meaningful in a difficult case."
        ),
    )
    target_groups = [
        rule.target_id
        for rule in context["target_rules"]
        if rule.target_kind == "cluster"
    ]
    suggested_group = target_groups[0] if target_groups else ""
    nearest_group = context["nearest_group"]
    if suggested_group and nearest_group:
        agrees = suggested_group == nearest_group
        whole = _bullet(
            context["point_id"],
            "feature_whole_record_agreement",
            "yes" if agrees else "no",
            (
                "This field and the whole record point in the same direction."
                if agrees
                else "This field and the whole record point in different directions."
            ),
            (
                "The group suggested by the feature also best matches the record's overall pattern."
                if agrees
                else "The feature-based description points to a different group from the record's overall pattern."
            ),
            (
                "Agreement makes the field a more useful clue; disagreement "
                "warns against using it as the answer by itself."
            ),
            {
                "feature_rule_target_group": suggested_group,
                "nearest_group_by_full_record": nearest_group,
            },
        )
    else:
        whole = _insufficient(
            context,
            "feature_whole_record_agreement",
            "There is not enough group evidence to compare this field with the whole record.",
        )
    label_pattern = _feature_label_pattern_bullet(context, boundary)
    shortcut = _feature_shortcut_bullet(
        context,
        feature=feature,
        usage=usage,
        maximum_usage=max_usage,
    )
    return repeated, dividing, whole, label_pattern, shortcut


def _audit_bullets(context: Mapping[str, Any]) -> Tuple[EvidenceBullet, ...]:
    rules = context["target_rules"] or context["related_rules"]
    rule = rules[0] if rules else None
    if rule is None:
        return tuple(
            _insufficient(
                context,
                dimension_id,
                "There is no current rule available for this check.",
            )
            for dimension_id in CATEGORY_DIMENSIONS["rule_confidence_audit"]
        )
    target_size = int(rule.diagnostics.get("target_total", 0))
    scope_ratio = (
        int(rule.diagnostics.get("target_matches", 0)) / target_size
        if target_size
        else 0.0
    )
    if scope_ratio >= 0.7:
        scope_status = "yes"
        scope_headline = "The rule describes most of this group."
        scope_fact = "The description reaches a large share of the current group."
    elif scope_ratio >= 0.3:
        scope_status = "partly"
        scope_headline = "The rule describes a substantial part of this group."
        scope_fact = "The description is useful for part of the group but does not cover everyone."
    else:
        scope_status = "no"
        scope_headline = "The rule describes only a small part of this group."
        scope_fact = "The description applies to a limited set of current members."
    scope = _bullet(
        context["point_id"],
        "audit_rule_scope",
        scope_status,
        scope_headline,
        scope_fact,
        "The wider the description is used, the more important it is to verify it with people.",
        {
            "rule_id": rule.rule_id,
            "target_total": target_size,
            "target_matches": int(rule.diagnostics.get("target_matches", 0)),
            "coverage": rule.coverage,
        },
    )
    if rule.purity >= 0.9:
        consistency_status = "yes"
        consistency_headline = (
            "The rule usually agrees with the groups shown on the dashboard."
        )
        consistency_fact = (
            "Only a small share of the records included by this rule are shown "
            "in a different group."
        )
    elif rule.purity >= 0.75:
        consistency_status = "partly"
        consistency_headline = (
            "The rule sometimes disagrees with the groups shown on the dashboard."
        )
        consistency_fact = (
            "The description agrees for many records but gives a different "
            "answer for several others."
        )
    else:
        consistency_status = "no"
        consistency_headline = (
            "The rule often disagrees with the groups shown on the dashboard."
        )
        consistency_fact = (
            "Many records included by the rule are shown in a different group."
        )
    consistency = _bullet(
        context["point_id"],
        "audit_current_consistency",
        consistency_status,
        consistency_headline,
        consistency_fact,
        "Computer agreement alone is not enough, but frequent disagreement is a clear reason to be cautious.",
        {
            "rule_id": rule.rule_id,
            "purity": rule.purity,
            "exception_count": len(rule.exception_point_ids),
        },
    )
    human = _rule_human_agreement_bullet(context, rule)
    exceptions = _audit_exception_pattern_bullet(context, rule)
    stability = _rule_stability_bullet(context, rule)
    return scope, consistency, human, exceptions, stability


def _edge_bullet(
    context: Mapping[str, Any],
    *,
    dimension_id: str,
    yes_headline: str,
    partly_headline: str,
    no_headline: str,
    why: str,
) -> EvidenceBullet:
    percentile = float(context["edge_percentile"])
    if len(context["members_by_group"].get(context["group_id"], ())) < 3:
        return _insufficient(
            context,
            dimension_id,
            "There are too few group members to judge whether this point is at the edge.",
        )
    if percentile >= 0.8:
        status = "yes"
        headline = yes_headline
        fact = "It is less typical of the current group than most members."
    elif percentile >= 0.6:
        status = "partly"
        headline = partly_headline
        fact = "It is not in the group center, but it is not one of the most distant members."
    else:
        status = "no"
        headline = no_headline
        fact = "Its overall pattern remains fairly typical of the current group."
    return _bullet(
        context["point_id"],
        dimension_id,
        status,
        headline,
        fact,
        why,
        {
            "current_group_id": context["group_id"],
            "distance_from_group_center": round(
                float(context["point_center_distance"]),
                6,
            ),
            "within_group_distance_percentile": round(percentile, 6),
        },
    )


def _boundary_line_bullet(
    context: Mapping[str, Any],
    *,
    dimension_id: str,
    why: str,
) -> EvidenceBullet:
    boundary = context["boundary"]
    if not boundary:
        return _insufficient(
            context,
            dimension_id,
            "No current rule provides a dividing line for this point.",
        )
    feature = _plain_feature(boundary.get("source_feature", "a key field"))
    if boundary.get("categorical_only"):
        return _bullet(
            context["point_id"],
            dimension_id,
            "insufficient",
            "The relevant rule uses a category rather than a gradual dividing line.",
            f"The current description checks which category {feature} belongs to.",
            "There is no meaningful numeric distance from a categorical condition.",
            {
                "feature": boundary.get("source_feature"),
                "kind": "categorical",
                "display_text": boundary.get("display_text"),
            },
        )
    margin = float(boundary.get("normalized_margin", 999999.0))
    if margin <= 0.1:
        status = "yes"
        headline = f"It is very close to the dividing line based on {feature}."
        fact = "A small difference in this field would place the record on the other side of the rule."
    elif margin <= 0.25:
        status = "partly"
        headline = f"It is fairly close to the dividing line based on {feature}."
        fact = "The field is on one side of the rule, but not far from the difficult area."
    else:
        status = "no"
        headline = f"It is not especially close to the dividing line based on {feature}."
        fact = "The rule places this field clearly on one side of its dividing line."
    return _bullet(
        context["point_id"],
        dimension_id,
        status,
        headline,
        fact,
        why,
        {
            "rule_id": boundary.get("rule_id"),
            "source_feature": boundary.get("source_feature"),
            "normalized_margin_in_iqr_units": round(margin, 6),
            "display_condition": boundary.get("display_text"),
        },
    )


def _mixed_neighbor_bullet(
    context: Mapping[str, Any],
    *,
    dimension_id: str,
    why: str,
) -> EvidenceBullet:
    if not context["neighbor_ids"]:
        return _insufficient(
            context,
            dimension_id,
            "There are not enough nearby records for this comparison.",
        )
    other_ratio = float(context["other_ratio"])
    represented_groups = len(set(context["neighbor_groups"]))
    if represented_groups >= 2 and 0.25 <= other_ratio <= 0.75:
        status = "yes"
        headline = "Nearby records are divided between groups."
        fact = "The records most similar to this one are split across more than one current group."
    elif represented_groups >= 2:
        status = "partly"
        headline = "More than one group appears nearby, but one is dominant."
        fact = "A second group is present among similar records without making the area evenly mixed."
    else:
        status = "no"
        headline = "Nearby records mostly agree on one group."
        fact = "The closest records are currently placed together rather than divided."
    return _bullet(
        context["point_id"],
        dimension_id,
        status,
        headline,
        fact,
        why,
        _neighbor_details(context),
    )


def _projection_bullet(context: Mapping[str, Any]) -> EvidenceBullet:
    overlap = context["projection_overlap"]
    if overlap is None:
        return _insufficient(
            context,
            "boundary_projection_agreement",
            "The 2D view cannot be compared with the full record space.",
        )
    if overlap >= 0.6:
        status = "yes"
        headline = "The 2D view tells a similar neighborhood story."
        fact = "Most records shown nearby in the plot are also similar in the full set of fields."
        why = "The plot is a reasonable navigation aid for this point, although the original fields remain the evidence."
    elif overlap >= 0.35:
        status = "partly"
        headline = "The 2D view preserves only part of the neighborhood."
        fact = "Some visually nearby records are truly similar, while others are close only in the projection."
        why = "Use the linked comparisons and original fields instead of relying on plot position alone."
    else:
        status = "no"
        headline = "The 2D view is misleading around this point."
        fact = "Most visual neighbors are not the records most similar in the full set of fields."
        why = "The label decision should rely on the evidence bullets and comparison records, not visual distance."
    return _bullet(
        context["point_id"],
        "boundary_projection_agreement",
        status,
        headline,
        fact,
        why,
        {"local_neighbor_overlap": round(float(overlap), 6)},
    )


def _group_resemblance_bullet(
    context: Mapping[str, Any],
    *,
    dimension_id: str,
    overlap_wording: bool,
) -> EvidenceBullet:
    ratio = context["group_distance_ratio"]
    if ratio is None or not context["second_group"]:
        return _insufficient(
            context,
            dimension_id,
            "There are not enough current groups for a meaningful comparison.",
        )
    nearest = context["nearest_group"]
    second = context["second_group"]
    if ratio <= 1.25:
        status = "yes"
        headline = (
            "It resembles typical examples from two groups."
            if overlap_wording
            else "It resembles more than one existing group."
        )
        fact = "Its overall pattern is almost equally close to the two nearest group examples."
        why = (
            "A human label can show whether this similarity has one real-world meaning."
            if overlap_wording
            else "The label can show whether this area belongs to an existing type rather than a new one."
        )
    elif ratio <= 1.75:
        status = "partly"
        headline = "It has a preferred group, but another group is still plausible."
        fact = "The whole record is closer to one typical group while retaining some similarity to another."
        why = "Comparing both typical examples can reveal which differences matter to a person."
    else:
        status = "no"
        headline = "It resembles one group much more than the others."
        fact = "Its overall pattern has one clearly closest current group."
        why = (
            "This weakens the case that the point truly sits between two group meanings."
            if overlap_wording
            else "This suggests an existing group may already provide a suitable type."
        )
    return _bullet(
        context["point_id"],
        dimension_id,
        status,
        headline,
        fact,
        why,
        {
            "nearest_group_id": nearest,
            "second_group_id": second,
            "nearest_group_distance": round(
                context["group_center_distances"][nearest],
                6,
            ),
            "second_group_distance": round(
                context["group_center_distances"][second],
                6,
            ),
            "distance_ratio": round(float(ratio), 6),
        },
    )


def _human_neighbor_labels_bullet(
    context: Mapping[str, Any],
    *,
    dimension_id: str,
    same_headline: str,
    mixed_headline: str,
    why: str,
) -> EvidenceBullet:
    labels = context["labeled_neighbor_values"]
    if len(labels) < 2:
        return _insufficient(
            context,
            dimension_id,
            "Too few nearby records have confirmed human labels.",
            why=(
                "The system cannot use missing human evidence as support for a "
                "group decision."
            ),
            technical={
                "labeled_neighbor_ids": list(context["labeled_neighbor_ids"]),
            },
        )
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        status = "yes"
        headline = same_headline
        fact = "The nearby records already checked by people use the same real-world type."
    else:
        status = "partly"
        headline = mixed_headline
        fact = "People have used more than one real-world type for nearby records."
    return _bullet(
        context["point_id"],
        dimension_id,
        status,
        headline,
        fact,
        why,
        {
            "labeled_neighbor_ids": list(context["labeled_neighbor_ids"]),
            "human_label_ids": list(labels),
        },
    )


def _outside_group_separation_bullet(
    context: Mapping[str, Any],
) -> EvidenceBullet:
    nearest = context["nearest_group"]
    second = context["second_group"]
    if not nearest or not second:
        return _insufficient(
            context,
            "overlap_outside_separation",
            "There are not enough groups to compare their typical regions.",
        )
    center_distance = float(
        np.linalg.norm(
            context["centers"][nearest] - context["centers"][second]
        )
    )
    average_radius = (
        float(context["group_radii"].get(nearest, 0.0))
        + float(context["group_radii"].get(second, 0.0))
    ) / 2.0
    separation = center_distance / max(average_radius, 1e-12)
    if separation >= 2.0:
        status = "yes"
        headline = "The groups remain clearly different away from this shared area."
        fact = "Their typical members are well separated even though this point lies in an ambiguous region."
        why = "This suggests a local overlap rather than evidence that the entire groups mean the same thing."
    elif separation >= 1.0:
        status = "partly"
        headline = "The groups are only moderately different outside this area."
        fact = "Their typical regions have some separation, but the distinction is not strong."
        why = "Human labels from both the shared and typical regions are needed before drawing a wider conclusion."
    else:
        status = "no"
        headline = "The groups also look similar outside this shared area."
        fact = "Even their typical regions are close in the full set of fields."
        why = "This makes it more important to check whether people see a meaningful difference at all."
    return _bullet(
        context["point_id"],
        "overlap_outside_separation",
        status,
        headline,
        fact,
        why,
        {
            "first_group_id": nearest,
            "second_group_id": second,
            "center_distance": round(center_distance, 6),
            "average_group_radius": round(average_radius, 6),
            "separation_ratio": round(separation, 6),
        },
    )


def _round_stability_bullet(context: Mapping[str, Any]) -> EvidenceBullet:
    parent = context["parent_round"]
    point_id = context["point_id"]
    if parent is None:
        return _insufficient(
            context,
            "split_round_stability",
            "This is the first round, so the separation has no history yet.",
            why="More than one round is needed before calling a separated area stable.",
        )
    previous_plan = parent.recommendation_plans.get(
        "split_or_new_cluster_signal",
        {},
    )
    if point_id in set(previous_plan.get("recommended_point_ids", ())):
        status = "yes"
        headline = "This separated area has remained important across rounds."
        fact = "The same point was also selected for this question in the previous round."
        why = "Repeated evidence is more credible than a pattern that appears only once."
    elif context["parent_assignments"].get(point_id) != context["group_id"]:
        status = "partly"
        headline = "The point changed groups after the previous labels."
        fact = "Its current separation may reflect a genuine shift, but it is not yet stable."
        why = "A human check can confirm whether the new placement makes real-world sense."
    else:
        status = "partly"
        headline = "This separation is new in the current round."
        fact = "The point was not a recommended split example in the previous round."
        why = "A new signal deserves checking, but it should not yet be treated as persistent."
    return _bullet(
        point_id,
        "split_round_stability",
        status,
        headline,
        fact,
        why,
        {
            "previously_recommended_for_split": point_id
            in set(previous_plan.get("recommended_point_ids", ())),
            "previous_group_id": context["parent_assignments"].get(point_id),
            "current_group_id": context["group_id"],
        },
    )


def _data_quality_bullet(context: Mapping[str, Any]) -> EvidenceBullet:
    missing = context["missing_features"]
    extreme = context["extreme_features"]
    if missing or extreme:
        status = "partly"
        headline = "Part of the difference may come from data quality."
        facts = []
        if missing:
            facts.append("some original fields are missing")
        if extreme:
            facts.append("some fields are far outside their usual range")
        fact = "The record has " + " and ".join(facts) + "."
        why = "Check the source record before treating this pattern as a meaningful unusual case."
    else:
        status = "no"
        headline = "There is no obvious missing or implausible field value."
        fact = "The available fields are present and none is far outside its usual range."
        why = "This does not prove the record is correct, but the current data offers no simple quality explanation."
    return _bullet(
        context["point_id"],
        "anomaly_data_quality",
        status,
        headline,
        fact,
        why,
        {
            "missing_features": list(missing),
            "extreme_features": list(extreme),
        },
    )


def _unusual_round_change_bullet(context: Mapping[str, Any]) -> EvidenceBullet:
    parent = context["parent_round"]
    point_id = context["point_id"]
    current = bool(context["outlier"] and context["outlier"].is_outlier)
    if parent is None:
        return _insufficient(
            context,
            "anomaly_round_change",
            "This is the first round, so there is no earlier unusual status to compare.",
        )
    previous = context["parent_outliers"].get(point_id)
    if previous is None:
        return _insufficient(
            context,
            "anomaly_round_change",
            "No earlier unusual status is available for this record.",
        )
    if previous != current:
        headline = "Its unusual status changed after the previous labels."
        fact = (
            "The latest analysis now treats it as unusual."
            if current
            else "The latest analysis no longer treats it as unusual."
        )
        why = "A human check can confirm whether that change matches the record's real-world meaning."
        status = "yes"
    else:
        headline = "Its unusual status has remained stable."
        fact = (
            "The analysis has continued to treat it as unusual."
            if current
            else "The analysis has continued to treat it as an ordinary record."
        )
        why = "A stable result is useful context, but it still needs human confirmation when the case matters."
        status = "no"
    return _bullet(
        point_id,
        "anomaly_round_change",
        status,
        headline,
        fact,
        why,
        {
            "previous_is_unusual": previous,
            "current_is_unusual": current,
        },
    )


def _confirmed_outlier_examples_bullet(
    context: Mapping[str, Any],
) -> EvidenceBullet:
    labeled = [
        point_id
        for point_id in context["neighbor_ids"]
        if point_id in context["outlier_labels"]
    ]
    if not labeled:
        return _insufficient(
            context,
            "anomaly_confirmed_examples",
            "No nearby record has a human-confirmed unusual or normal label.",
            why="The current check cannot borrow confidence from examples that people have not reviewed.",
        )
    values = [context["outlier_labels"][point_id] for point_id in labeled]
    if all(values):
        status = "yes"
        headline = "Nearby confirmed examples are also unusual."
        fact = "People have marked the closest reviewed examples as genuinely unusual."
    elif not any(values):
        status = "no"
        headline = "Nearby confirmed examples are normal."
        fact = "People have marked the closest reviewed examples as ordinary records."
    else:
        status = "partly"
        headline = "Nearby confirmed examples include both unusual and normal records."
        fact = "Human decisions in this area are mixed."
    return _bullet(
        context["point_id"],
        "anomaly_confirmed_examples",
        status,
        headline,
        fact,
        "Compare the point with these reviewed examples before deciding whether rarity is meaningful.",
        {
            "confirmed_example_ids": labeled,
            "confirmed_outlier_values": values,
        },
    )


def _disagreement_size_bullet(context: Mapping[str, Any]) -> EvidenceBullet:
    boundary = context["boundary"]
    if not boundary:
        return _insufficient(
            context,
            "exception_disagreement_size",
            "No rule distance is available to judge the size of the disagreement.",
        )
    if boundary.get("categorical_only"):
        return _bullet(
            context["point_id"],
            "exception_disagreement_size",
            "partly",
            "The disagreement concerns a category rather than a gradual boundary.",
            "The record either matches or does not match the category condition, so there is no meaningful near/far distance.",
            "The user should compare the full record instead of treating this as a small numeric miss.",
            {
                "kind": "categorical",
                "display_condition": boundary.get("display_text"),
            },
        )
    margin = float(boundary.get("normalized_margin", 999999.0))
    feature = _plain_feature(boundary.get("source_feature", "a key field"))
    if margin <= 0.1:
        status = "partly"
        headline = "It lies only just outside the expected description."
        fact = f"The disagreement based on {feature} is close to the current dividing line."
        why = "A small miss may be a normal boundary case rather than a wrong label."
    elif margin <= 0.5:
        status = "yes"
        headline = "The disagreement is noticeable."
        fact = f"The record differs clearly from the expected side of the description based on {feature}."
        why = "This is strong enough to compare the current label with other group examples."
    else:
        status = "yes"
        headline = "The disagreement is substantial."
        fact = f"The record sits far from the expected description based on {feature}."
        why = "A large mismatch is a strong reason to check both the label and the rule."
    return _bullet(
        context["point_id"],
        "exception_disagreement_size",
        status,
        headline,
        fact,
        why,
        {
            "source_feature": boundary.get("source_feature"),
            "normalized_margin_in_iqr_units": round(margin, 6),
            "display_condition": boundary.get("display_text"),
        },
    )


def _neighbor_suggestion_bullet(context: Mapping[str, Any]) -> EvidenceBullet:
    labels = context["labeled_neighbor_values"]
    if len(labels) >= 2:
        unique = set(labels)
        if len(unique) == 1:
            status = "yes"
            headline = "Nearby human labels point to one real-world type."
            fact = "The closest reviewed records use the same human type."
        else:
            status = "partly"
            headline = "Nearby human labels point in different directions."
            fact = "The closest reviewed records do not share one human type."
        why = "These nearby decisions are more useful than the rule alone when checking this exception."
        technical = {
            "labeled_neighbor_ids": list(context["labeled_neighbor_ids"]),
            "human_label_ids": list(labels),
        }
    else:
        status = "partly" if len(set(context["neighbor_groups"])) > 1 else "no"
        headline = (
            "Nearby system groups point in different directions."
            if status == "partly"
            else "Nearby records mostly remain in the current group."
        )
        fact = (
            "Similar records are currently split across groups, but too few have human labels."
            if status == "partly"
            else "Similar records are placed together, but this has not been confirmed by people."
        )
        why = "Treat this as model context only until nearby human labels are available."
        technical = _neighbor_details(context)
    return _bullet(
        context["point_id"],
        "exception_neighbor_support",
        status,
        headline,
        fact,
        why,
        technical,
    )


def _exception_scope_bullet(
    context: Mapping[str, Any],
    exception_rules: Sequence[RuleCard],
) -> EvidenceBullet:
    rules = tuple(exception_rules) or context["target_rules"]
    if not rules:
        return _insufficient(
            context,
            "exception_rule_scope",
            "No current rule is available to compare other exceptions.",
        )
    exception_count = sum(len(rule.exception_point_ids) for rule in rules)
    support_count = sum(rule.support_count for rule in rules)
    rate = exception_count / support_count if support_count else 0.0
    if rate >= 0.2:
        status = "yes"
        headline = "The rule has many similar exceptions."
        fact = "This point is part of a wider pattern of records the simple description gets wrong."
        why = "The rule may be too simple, so the point should not be blamed automatically."
    elif rate >= 0.05:
        status = "partly"
        headline = "The rule has several exceptions."
        fact = "This is not the only record that disagrees with the current description."
        why = "Both the point label and the rule deserve checking."
    else:
        status = "no"
        headline = "This appears to be an isolated rule exception."
        fact = "Very few records break the same description."
        why = "An isolated mismatch makes the individual record especially important to inspect."
    return _bullet(
        context["point_id"],
        "exception_rule_scope",
        status,
        headline,
        fact,
        why,
        {
            "rule_ids": [rule.rule_id for rule in rules],
            "exception_count": exception_count,
            "support_count": support_count,
            "exception_rate": round(rate, 6),
        },
    )


def _feature_label_pattern_bullet(
    context: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> EvidenceBullet:
    if not boundary or boundary.get("categorical_only"):
        return _insufficient(
            context,
            "feature_human_label_pattern",
            "There is no numeric feature line to compare with existing human labels.",
        )
    model_feature = boundary.get("model_feature")
    feature_names = context["prepared"].feature_matrix.feature_names
    if model_feature not in feature_names:
        return _insufficient(
            context,
            "feature_human_label_pattern",
            "The relevant feature cannot be matched to the current model data.",
        )
    feature_index = feature_names.index(model_feature)
    threshold = float(boundary["threshold"])
    labeled_sides: Dict[str, list[Any]] = {"left": [], "right": []}
    for point_id, label in context["semantic_labels"].items():
        index = context["index_by_id"].get(point_id)
        if index is None:
            continue
        side = (
            "left"
            if float(context["matrix"][index, feature_index]) <= threshold
            else "right"
        )
        labeled_sides[side].append(label)
    total = sum(len(values) for values in labeled_sides.values())
    if total < 3 or not all(labeled_sides.values()):
        return _insufficient(
            context,
            "feature_human_label_pattern",
            "Too few human-labeled records appear on both sides of this feature line.",
            technical={"human_labels_by_side": labeled_sides},
        )
    left_labels = set(labeled_sides["left"])
    right_labels = set(labeled_sides["right"])
    if left_labels.isdisjoint(right_labels):
        status = "yes"
        headline = "Existing human labels follow this feature pattern."
        fact = "Reviewed records on opposite sides of the line use different real-world types."
        why = "This supports the field as a useful clue, although the full record should still decide the label."
    elif left_labels == right_labels:
        status = "no"
        headline = "Existing human labels do not follow this feature pattern."
        fact = "People use the same real-world types on both sides of the feature line."
        why = "The field may be a convenient computer shortcut rather than a meaningful distinction."
    else:
        status = "partly"
        headline = "Existing human labels only partly follow this feature pattern."
        fact = "Some human types cross the feature line while others stay mainly on one side."
        why = "The field may help in combination with other information, but not by itself."
    return _bullet(
        context["point_id"],
        "feature_human_label_pattern",
        status,
        headline,
        fact,
        why,
        {"human_labels_by_side": labeled_sides},
    )


def _feature_shortcut_bullet(
    context: Mapping[str, Any],
    *,
    feature: str,
    usage: int,
    maximum_usage: int,
) -> EvidenceBullet:
    spec = next(
        (
            item
            for item in context["prepared"].version.feature_specs
            if item.name == feature
        ),
        None,
    )
    missing_count = spec.missing_count if spec is not None else 0
    dominance = usage / max(
        sum(
            context["rule_set"].diagnostics.get("feature_usage", {}).values()
        ),
        1,
    )
    if missing_count or dominance >= 0.5:
        status = "partly"
        headline = "This field may be acting as a shortcut."
        reasons = []
        if missing_count:
            reasons.append("some records are missing this field")
        if dominance >= 0.5:
            reasons.append("the current rules rely on it unusually often")
        fact = "There is a caution sign: " + " and ".join(reasons) + "."
        why = "Use the complete record and human meaning rather than letting this field decide alone."
    else:
        status = "no"
        headline = "There is no obvious shortcut warning for this field."
        fact = "The field is not dominated by missing values and does not account for most rule use."
        why = "It can still be misleading, so agreement with human labels remains the important test."
    return _bullet(
        context["point_id"],
        "feature_shortcut_risk",
        status,
        headline,
        fact,
        why,
        {
            "source_feature": feature,
            "missing_count": missing_count,
            "feature_rule_use_count": usage,
            "maximum_feature_rule_use_count": maximum_usage,
            "rule_use_share": round(dominance, 6),
        },
    )


def _rule_human_agreement_bullet(
    context: Mapping[str, Any],
    rule: RuleCard,
) -> EvidenceBullet:
    target_group_labels = [
        context["semantic_labels"][point_id]
        for point_id in context["members_by_group"].get(rule.target_id, ())
        if point_id in context["semantic_labels"]
    ]
    covered_labels = [
        context["semantic_labels"][point_id]
        for point_id in rule.matched_point_ids
        if point_id in context["semantic_labels"]
    ]
    if len(target_group_labels) < 2 or len(covered_labels) < 2:
        return _insufficient(
            context,
            "audit_human_label_agreement",
            "Too few records covered by this rule have confirmed human labels.",
            why="Computer agreement cannot replace missing human evidence.",
            technical={
                "target_group_human_label_count": len(target_group_labels),
                "covered_human_label_count": len(covered_labels),
            },
        )
    dominant = max(
        set(target_group_labels),
        key=lambda label: (
            target_group_labels.count(label),
            str(label),
        ),
    )
    agreement = sum(label == dominant for label in covered_labels) / len(
        covered_labels
    )
    if agreement >= 0.8:
        status = "yes"
        headline = "The rule agrees with most existing human labels."
        fact = "Reviewed records covered by the description usually share the main human type for this group."
        why = "This makes the rule more credible as an explanation, not as an automatic label."
    elif agreement >= 0.5:
        status = "partly"
        headline = "The rule agrees with some human labels but not consistently."
        fact = "Reviewed records covered by the description use a mixture of human types."
        why = "More targeted labels are needed before the description can be trusted."
    else:
        status = "no"
        headline = "The rule often disagrees with existing human labels."
        fact = "Most reviewed records covered by the description do not use the main human type for this group."
        why = "The rule may explain the computer result without matching real-world meaning."
    return _bullet(
        context["point_id"],
        "audit_human_label_agreement",
        status,
        headline,
        fact,
        why,
        {
            "dominant_human_label_id": dominant,
            "covered_human_label_count": len(covered_labels),
            "human_label_agreement": round(agreement, 6),
        },
    )


def _audit_exception_pattern_bullet(
    context: Mapping[str, Any],
    rule: RuleCard,
) -> EvidenceBullet:
    count = len(rule.exception_point_ids)
    if count == 0:
        status = "yes"
        headline = "No current record clearly breaks this rule."
        fact = "The description has no exceptions in the current analysis."
        why = "This is encouraging, but human labels are still needed to test real-world meaning."
    else:
        exception_margins = []
        feature_index = {
            name: index
            for index, name in enumerate(
                context["prepared"].feature_matrix.feature_names
            )
        }
        for point_id in rule.exception_point_ids:
            index = context["index_by_id"].get(point_id)
            if index is None:
                continue
            for condition in rule.conditions:
                column = feature_index.get(condition.feature)
                if column is None:
                    continue
                transform = context["transform_by_model"].get(
                    condition.feature,
                    {},
                )
                if transform.get("kind") == "categorical":
                    continue
                exception_margins.append(
                    abs(
                        float(context["matrix"][index, column])
                        - float(condition.threshold)
                    )
                )
        median_margin = (
            float(np.median(exception_margins))
            if exception_margins
            else None
        )
        if median_margin is not None and median_margin <= 0.25:
            status = "partly"
            headline = "Most rule exceptions sit near a dividing line."
            fact = "The records that break the description are mainly borderline cases."
            why = "The rule may need a softer edge rather than a completely different explanation."
        else:
            status = "no"
            headline = "The rule has clear exceptions, not only borderline cases."
            fact = "Some records break the description well away from its dividing lines."
            why = "These records are important tests of whether the rule is too simple."
    return _bullet(
        context["point_id"],
        "audit_exception_pattern",
        status,
        headline,
        fact,
        why,
        {
            "rule_id": rule.rule_id,
            "exception_count": count,
            "median_exception_margin": (
                round(median_margin, 6)
                if count and median_margin is not None
                else None
            ),
        },
    )


def _rule_stability_bullet(
    context: Mapping[str, Any],
    rule: RuleCard,
) -> EvidenceBullet:
    if context["parent_round"] is None:
        return _insufficient(
            context,
            "audit_round_stability",
            "This is the first round, so the rule has no history yet.",
            why="A rule needs more than one round before it can be called stable.",
        )
    current_fingerprint = _rule_fingerprint(rule.to_dict())
    comparable = [
        item
        for item in context["parent_rules"]
        if item.get("target_kind") == rule.target_kind
        and item.get("target_id") == rule.target_id
    ]
    previous_fingerprints = {
        _rule_fingerprint(item) for item in comparable
    }
    if current_fingerprint in previous_fingerprints:
        status = "yes"
        headline = "The same rule description remained across rounds."
        fact = "Its fields, directions, and dividing values match a rule from the previous round."
        why = "A stable description is easier to trust than one that changes after every label."
    elif comparable:
        status = "partly"
        headline = "The group still has a rule, but the description changed."
        fact = "The fields or dividing values differ from the previous round."
        why = "The rule should be treated cautiously until the description becomes more stable."
    else:
        status = "no"
        headline = "This rule is new in the current round."
        fact = "No comparable description for the same target existed in the previous round."
        why = (
            "A new rule needs human checks before it is used as a reliable "
            "group description."
        )
    return _bullet(
        context["point_id"],
        "audit_round_stability",
        status,
        headline,
        fact,
        why,
        {
            "rule_id": rule.rule_id,
            "rule_fingerprint": current_fingerprint,
            "comparable_previous_rule_count": len(comparable),
        },
    )


def _comparison_targets(
    context: Mapping[str, Any],
    category: str,
) -> Tuple[Mapping[str, Any], ...]:
    point_id = context["point_id"]
    distances = context["distances"]
    index_by_id = context["index_by_id"]
    assignments = context["assignments"]
    semantic_labels = context["semantic_labels"]
    outlier_labels = context["outlier_labels"]
    group_id = context["group_id"]
    selected = []

    def add_target(candidate_id: str, relation: str, source: str) -> None:
        if not candidate_id or candidate_id == point_id:
            return
        if candidate_id in {item["point_id"] for item in selected}:
            return
        selected.append(
            {
                "point_id": candidate_id,
                "relation": relation,
                "source": source,
                "current_group_id": assignments.get(candidate_id, ""),
                "human_label": _display_human_label(
                    semantic_labels.get(candidate_id),
                    context["label_vocabulary"],
                ),
                "confirmed_unusual": outlier_labels.get(candidate_id),
                "features_to_compare": _features_to_compare(
                    point_id,
                    candidate_id,
                    context,
                ),
            }
        )

    ordered_ids = sorted(
        (
            candidate_id
            for candidate_id in context["point_ids"]
            if candidate_id != point_id
        ),
        key=lambda candidate_id: (
            float(distances[index_by_id[candidate_id]]),
            candidate_id,
        ),
    )
    if category == "anomaly_label_review":
        reviewed = [
            candidate_id
            for candidate_id in ordered_ids
            if candidate_id in outlier_labels
        ]
        unusual = next(
            (
                candidate_id
                for candidate_id in reviewed
                if outlier_labels[candidate_id]
            ),
            "",
        )
        normal = next(
            (
                candidate_id
                for candidate_id in reviewed
                if not outlier_labels[candidate_id]
            ),
            "",
        )
        add_target(unusual, "human-confirmed unusual example", "human_label")
        add_target(normal, "human-confirmed normal example", "human_label")
    else:
        reviewed = [
            candidate_id
            for candidate_id in ordered_ids
            if candidate_id in semantic_labels
        ]
        same = next(
            (
                candidate_id
                for candidate_id in reviewed
                if assignments.get(candidate_id) == group_id
            ),
            "",
        )
        other = next(
            (
                candidate_id
                for candidate_id in reviewed
                if assignments.get(candidate_id) != group_id
            ),
            "",
        )
        add_target(same, "nearby human-labeled example from the current group", "human_label")
        add_target(other, "nearby human-labeled example from another group", "human_label")

    if len(selected) < 2:
        same = next(
            (
                candidate_id
                for candidate_id in ordered_ids
                if assignments.get(candidate_id) == group_id
            ),
            "",
        )
        other = next(
            (
                candidate_id
                for candidate_id in ordered_ids
                if assignments.get(candidate_id) != group_id
            ),
            "",
        )
        add_target(
            same,
            "typical system example from the current group",
            "model_example",
        )
        add_target(
            other,
            "nearby system example from another group",
            "model_example",
        )
    return tuple(selected[:2])


def _features_to_compare(
    first_id: str,
    second_id: str,
    context: Mapping[str, Any],
) -> list[str]:
    first = context["matrix"][context["index_by_id"][first_id]]
    second = context["matrix"][context["index_by_id"][second_id]]
    differences: Dict[str, float] = {}
    for model_feature, difference in zip(
        context["prepared"].feature_matrix.feature_names,
        np.abs(first - second),
    ):
        source = context["source_by_model"].get(model_feature, model_feature)
        differences[source] = max(
            differences.get(source, 0.0),
            float(difference),
        )
    return [
        name
        for name in sorted(
            differences,
            key=lambda name: (-differences[name], name),
        )[:2]
    ]


def _round_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    point_id = context["point_id"]
    parent = context["parent_round"]
    return {
        "baseline": parent is None,
        "previous_group_id": context["parent_assignments"].get(point_id),
        "current_group_id": context["group_id"],
        "group_changed": (
            parent is not None
            and context["parent_assignments"].get(point_id)
            != context["group_id"]
        ),
        "previous_unusual": context["parent_outliers"].get(point_id),
        "current_unusual": bool(
            context["outlier"] and context["outlier"].is_outlier
        ),
    }


def _bullet(
    point_id: str,
    dimension_id: str,
    status: str,
    headline: str,
    plain_fact: str,
    why_it_matters: str,
    technical_details: Mapping[str, Any],
) -> EvidenceBullet:
    fact_payload = {
        "point_id": point_id,
        "dimension_id": dimension_id,
        "status": status,
        "technical_details": technical_details,
        "policy": EVIDENCE_POLICY_VERSION,
    }
    fact_id = "fact_" + hashlib.sha1(
        json.dumps(
            fact_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:14]
    return EvidenceBullet(
        dimension_id=dimension_id,
        question=_DIMENSION_QUESTIONS[dimension_id],
        status=status,
        headline=headline,
        plain_fact=plain_fact,
        why_it_matters=why_it_matters,
        point_connection=plain_fact,
        labeling_value=why_it_matters,
        evidence_fact_ids=(fact_id,),
        technical_details=dict(technical_details),
    )


def _insufficient(
    context: Mapping[str, Any],
    dimension_id: str,
    fact: str,
    *,
    why: str = (
        "This part cannot be judged reliably in the current round, so the "
        "system will not invent a conclusion."
    ),
    technical: Mapping[str, Any] | None = None,
) -> EvidenceBullet:
    return _bullet(
        context["point_id"],
        dimension_id,
        "insufficient",
        "Not enough information yet.",
        fact,
        why,
        dict(technical or {}),
    )


def _neighbor_details(context: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "neighbor_count": len(context["neighbor_ids"]),
        "neighbor_point_ids": list(context["neighbor_ids"]),
        "neighbor_group_ids": list(context["neighbor_groups"]),
        "other_group_neighbor_count": len(context["other_neighbor_ids"]),
        "other_group_neighbor_share": round(
            float(context["other_ratio"]),
            6,
        ),
        "computed_in": "preprocessed_full_feature_space",
    }


def _plain_category_name(category: str) -> str:
    return {
        "boundary_review": "the current dividing line",
        "overlap_merge_signal": "an area shared by two group descriptions",
        "split_or_new_cluster_signal": "a separated part of a group",
        "anomaly_label_review": "whether a record is truly unusual",
        "exception_relabel_review": "a record that does not fit its group description",
        "feature_label_strategy": "whether a field matches human meaning",
        "rule_confidence_audit": (
            "whether a group description agrees with human judgment"
        ),
    }.get(category, "the most important unresolved case")


def _plain_recheck_reason(reason: str) -> str:
    return {
        "cluster_changed_after_label": (
            "The latest analysis moved the record to a different group."
        ),
        "outlier_status_changed_after_label": (
            "The latest analysis changed whether it treats the record as unusual."
        ),
        "current_rule_conflicts_with_existing_label": (
            "A current group description conflicts with an existing human label."
        ),
    }.get(reason, "New evidence made the point worth checking again.")


def _display_human_label(
    label_id: Any,
    vocabulary: Mapping[str, str],
) -> str:
    if label_id is None:
        return ""
    return str(vocabulary.get(str(label_id), label_id))


def _plain_feature(value: Any) -> str:
    return " ".join(str(value or "a key field").replace("_", " ").split())


def _rule_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {
        "target_kind": payload.get("target_kind"),
        "target_id": payload.get("target_id"),
        "conditions": [
            {
                "feature": condition.get("feature"),
                "operator": condition.get("operator"),
                "threshold": round(float(condition.get("threshold", 0.0)), 9),
            }
            for condition in payload.get("conditions", ())
        ],
        "matched_point_ids": sorted(payload.get("matched_point_ids", ())),
    }
    return hashlib.sha1(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
