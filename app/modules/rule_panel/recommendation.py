from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Dict, Mapping, Tuple

from app.modules.algorithm_adapters.schemas import AnalysisResult
from app.shared.schemas import FeatureMatrix

from .schemas import (
    RECOMMENDATION_CATEGORIES,
    RecommendationPlan,
    RuleSet,
)


_TOP_PAIR_LIMIT = 12
_TOP_LABEL_POINT_LIMIT = 12
_RECOMMENDATION_PLAN_VERSION = "deterministic_recommendation_points_v1"


def build_recommendation_plan(
    rule_set: RuleSet,
    *,
    analysis_result: AnalysisResult | None = None,
    feature_matrix: FeatureMatrix | None = None,
    focus_category: str | None = None,
    guidance_metrics: Mapping[str, Any] | None = None,
    candidate_pool_limit: int | None = None,
) -> Dict[str, Any]:
    if not isinstance(rule_set, RuleSet):
        raise ValueError("rule_set must be a RuleSet")
    if analysis_result is not None and not isinstance(analysis_result, AnalysisResult):
        raise ValueError("analysis_result must be an AnalysisResult")
    if feature_matrix is not None and not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")
    category = _validate_focus_category(focus_category) or "label_priority"
    metrics = (
        dict(guidance_metrics)
        if isinstance(guidance_metrics, Mapping)
        else build_rule_guidance_metrics(
            rule_set,
            analysis_result=analysis_result,
            feature_matrix=feature_matrix,
        )
    )
    target_rule_ids = _target_rule_ids_for_category(rule_set, category, metrics)
    has_typical_case = _category_has_typical_case(rule_set, category, metrics)
    pool_limit = (
        _candidate_pool_limit_for_category(category)
        if candidate_pool_limit is None
        else candidate_pool_limit
    )
    if isinstance(pool_limit, bool) or not isinstance(pool_limit, int) or pool_limit < 1:
        raise ValueError("candidate_pool_limit must be a positive integer")
    candidate_pool = tuple(
        _candidate_point_ids_for_category(
            rule_set,
            category,
            metrics,
            target_rule_ids,
            limit=pool_limit,
        )
        if has_typical_case
        else ()
    )
    recommended = tuple(candidate_pool[: _recommended_limit_for_category(category)])
    profiles_by_id = _profiles_by_point(metrics)
    point_profiles = tuple(
        dict(profiles_by_id[point_id])
        for point_id in recommended
        if point_id in profiles_by_id
    )
    ranking_features = tuple(
        _ranking_feature_row(point_id, index, category, metrics, profiles_by_id.get(point_id))
        for index, point_id in enumerate(recommended)
    )
    plan_basis = {
        "dataset_id": rule_set.dataset_id,
        "rule_set_id": rule_set.rule_set_id,
        "analysis_run_id": analysis_result.analysis_run_id if analysis_result is not None else "",
        "category": category,
        "target_rule_ids": target_rule_ids,
        "candidate_pool": candidate_pool,
        "recommended": recommended,
        "plan_version": _RECOMMENDATION_PLAN_VERSION,
    }
    plan_hash = hashlib.sha1(
        json.dumps(plan_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return RecommendationPlan(
        plan_id=f"recplan_{plan_hash}",
        plan_version=_RECOMMENDATION_PLAN_VERSION,
        dataset_id=rule_set.dataset_id,
        analysis_run_id=analysis_result.analysis_run_id if analysis_result is not None else None,
        rule_set_id=rule_set.rule_set_id,
        focus_category=category,
        recommendation_kind=_recommendation_kind_for_category(category),
        has_typical_case=has_typical_case,
        candidate_pool_point_ids=candidate_pool,
        recommended_point_ids=recommended,
        highlighted_point_ids=recommended,
        target_rule_ids=target_rule_ids,
        ranking_method=_ranking_method_for_category(category),
        ranking_features=ranking_features,
        evidence_rows=_evidence_for_category(rule_set, category, metrics, target_rule_ids),
        point_profiles=point_profiles,
        label_questions=_plan_label_questions(category, target_rule_ids, recommended),
        label_options=_plan_label_options_for_category(category),
        expected_label_outcomes=_expected_outcomes_for_category(category),
        uncertainty_notes=(_risk_note_for_category(category, metrics),),
        immutable_fields=(
            "focus_category",
            "target_rule_ids",
            "candidate_pool_point_ids",
            "recommended_point_ids",
            "highlighted_point_ids",
        ),
        llm_role="translate_only_do_not_select_points",
        not_selected_summary=_not_selected_summary(category, candidate_pool, recommended),
        diagnostics={
            "source": "deterministic_rule_panel",
            "no_typical_case_reason": _no_case_recommendation(category) if not has_typical_case else "",
        },
    ).to_dict()


def build_rule_guidance_metrics(
    rule_set: RuleSet,
    *,
    analysis_result: AnalysisResult | None = None,
    feature_matrix: FeatureMatrix | None = None,
) -> Dict[str, Any]:
    if not isinstance(rule_set, RuleSet):
        raise ValueError("rule_set must be a RuleSet")
    if analysis_result is not None and not isinstance(analysis_result, AnalysisResult):
        raise ValueError("analysis_result must be an AnalysisResult")
    if feature_matrix is not None and not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")

    rule_metrics = [_rule_metric(rule) for rule in rule_set.rules]
    pair_metrics = _top_pair_metrics(rule_set.rules)
    label_candidates = _label_candidates(rule_set, pair_metrics)
    point_profiles = _label_candidate_point_profiles(
        rule_set,
        label_candidates,
        analysis_result=analysis_result,
        feature_matrix=feature_matrix,
    )
    feature_usage = rule_set.diagnostics.get("feature_usage", {})
    if not isinstance(feature_usage, Mapping):
        feature_usage = {}

    return {
        "rule_metrics": rule_metrics,
        "pair_metrics": pair_metrics,
        "label_candidate_groups": label_candidates,
        "label_candidate_point_profiles": point_profiles,
        "feature_usage": dict(feature_usage),
        "total_rule_count": len(rule_set.rules),
        "cluster_rule_count": sum(1 for rule in rule_set.rules if rule.target_kind == "cluster"),
        "anomaly_rule_count": sum(1 for rule in rule_set.rules if rule.target_kind == "anomaly"),
        "exception_point_count": len(
            {
                point_id
                for rule in rule_set.rules
                for point_id in rule.exception_point_ids
            }
        ),
    }


def _rule_metric(rule) -> Dict[str, Any]:
    exception_count = len(rule.exception_point_ids)
    exception_rate = exception_count / rule.support_count if rule.support_count else 0.0
    return {
        "rule_id": rule.rule_id,
        "target_kind": rule.target_kind,
        "target_id": rule.target_id,
        "support_count": rule.support_count,
        "coverage": rule.coverage,
        "purity": rule.purity,
        "exception_count": exception_count,
        "exception_rate": round(exception_rate, 6),
        "condition_count": len(rule.conditions),
        "condition_features": sorted({condition.feature for condition in rule.conditions}),
        "matched_preview_point_ids": list(rule.matched_point_ids[:_TOP_LABEL_POINT_LIMIT]),
        "exception_point_ids": list(rule.exception_point_ids[:_TOP_LABEL_POINT_LIMIT]),
        "rule_confidence_score": round(rule.purity * rule.coverage, 6),
        "quality_warnings": list(rule.diagnostics.get("quality_warnings", ())),
    }


def _top_pair_metrics(rules: Tuple[Any, ...]) -> Tuple[Mapping[str, Any], ...]:
    scored = []
    for left, right in combinations(rules, 2):
        left_points = set(left.matched_point_ids)
        right_points = set(right.matched_point_ids)
        intersection = sorted(left_points & right_points)
        union_count = len(left_points | right_points)
        intersection_count = len(intersection)
        left_support = len(left_points)
        right_support = len(right_points)
        jaccard = intersection_count / union_count if union_count else 0.0
        overlap_share_left = intersection_count / left_support if left_support else 0.0
        overlap_share_right = intersection_count / right_support if right_support else 0.0
        shared_features = sorted(
            {condition.feature for condition in left.conditions}
            & {condition.feature for condition in right.conditions}
        )
        boundary_gaps = _boundary_gaps(left, right, shared_features)
        relation = _pair_relation(
            left,
            right,
            intersection_count=intersection_count,
            shared_features=shared_features,
            boundary_gaps=boundary_gaps,
        )
        priority_score = _pair_priority(
            relation,
            jaccard,
            overlap_share_left,
            overlap_share_right,
            boundary_gaps,
        )
        metric = {
            "rule_ids": [left.rule_id, right.rule_id],
            "target_ids": [left.target_id, right.target_id],
            "target_kinds": [left.target_kind, right.target_kind],
            "relation": relation,
            "intersection_count": intersection_count,
            "union_count": union_count,
            "jaccard_overlap": round(jaccard, 6),
            "overlap_share_a": round(overlap_share_left, 6),
            "overlap_share_b": round(overlap_share_right, 6),
            "point_ids": intersection[:_TOP_LABEL_POINT_LIMIT],
            "shared_features": shared_features,
            "boundary_gaps": boundary_gaps,
            "priority_score": round(priority_score, 6),
        }
        scored.append((priority_score, metric))

    scored.sort(
        key=lambda item: (
            -item[0],
            -item[1]["intersection_count"],
            item[1]["rule_ids"][0],
            item[1]["rule_ids"][1],
        )
    )
    return tuple(metric for _, metric in scored[:_TOP_PAIR_LIMIT])


def _label_candidates(rule_set: RuleSet, pair_metrics: Tuple[Mapping[str, Any], ...]) -> Tuple[Mapping[str, Any], ...]:
    candidates = []
    seen_groups = set()

    for metric in pair_metrics:
        point_ids = tuple(metric.get("point_ids", ()))
        if not point_ids:
            continue
        key = ("pair", tuple(metric["rule_ids"]), point_ids)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        candidates.append(
            {
                "candidate_kind": "overlap_points",
                "priority": "high" if metric["intersection_count"] >= 3 else "medium",
                "rule_ids": list(metric["rule_ids"]),
                "point_ids": list(point_ids[:6]),
                "reason": (
                    f"{metric['intersection_count']} point(s) are shared by the pair; "
                    f"Jaccard={metric['jaccard_overlap']:.2f}."
                ),
            }
        )

    for rule in rule_set.rules:
        if not rule.exception_point_ids:
            continue
        point_ids = tuple(rule.exception_point_ids[:6])
        key = ("exception", rule.rule_id, point_ids)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        candidates.append(
            {
                "candidate_kind": "exception_points",
                "priority": "high",
                "rule_ids": [rule.rule_id],
                "point_ids": list(point_ids),
                "reason": (
                    f"{len(rule.exception_point_ids)} exception point(s) are matched by {rule.rule_id} "
                    f"but disagree with target {rule.target_id}."
                ),
            }
        )

    if not candidates and rule_set.rules:
        strongest_rule = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count))
        candidates.append(
            {
                "candidate_kind": "representative_points",
                "priority": "medium",
                "rule_ids": [strongest_rule.rule_id],
                "point_ids": list(strongest_rule.matched_point_ids[:6]),
                "reason": (
                    f"{strongest_rule.rule_id} has support={strongest_rule.support_count}, "
                    f"purity={strongest_rule.purity:.2f}, coverage={strongest_rule.coverage:.2f}."
                ),
            }
        )

    return tuple(candidates[:_TOP_LABEL_POINT_LIMIT])


def _label_candidate_point_profiles(
    rule_set: RuleSet,
    label_candidates: Tuple[Mapping[str, Any], ...],
    *,
    analysis_result: AnalysisResult | None,
    feature_matrix: FeatureMatrix | None,
) -> Tuple[Mapping[str, Any], ...]:
    ordered_point_ids = []
    for candidate in label_candidates:
        for point_id in candidate.get("point_ids", ()):
            if point_id not in ordered_point_ids:
                ordered_point_ids.append(point_id)
    for rule in rule_set.rules:
        for point_id in (*rule.exception_point_ids, *rule.matched_point_ids):
            if point_id not in ordered_point_ids:
                ordered_point_ids.append(point_id)

    assignments = {}
    outlier_scores = {}
    if analysis_result is not None:
        assignments = {
            assignment.point_id: assignment.cluster_id
            for assignment in analysis_result.cluster_result.assignments
        }
        outlier_scores = {
            score.point_id: {
                "score": round(score.score, 6),
                "is_outlier": score.is_outlier,
            }
            for score in analysis_result.outlier_result.scores
        }

    feature_lookup = {}
    feature_names: Tuple[str, ...] = ()
    if feature_matrix is not None:
        feature_names = tuple(feature_matrix.feature_names)
        feature_lookup = {
            point_id: dict(zip(feature_names, row))
            for point_id, row in zip(feature_matrix.point_ids, feature_matrix.values)
        }

    profiles = []
    for point_id in ordered_point_ids:
        related_rules = [
            rule
            for rule in rule_set.rules
            if point_id in rule.matched_point_ids or point_id in rule.exception_point_ids
        ]
        condition_features = tuple(
            dict.fromkeys(
                condition.feature
                for rule in related_rules
                for condition in rule.conditions
            )
        )
        values = feature_lookup.get(point_id, {})
        raw_feature_values = {
            feature: round(float(values[feature]), 6)
            for feature in condition_features
            if feature in values
        }
        threshold_margins = []
        for rule in related_rules[:4]:
            for condition in rule.conditions:
                if condition.feature not in values:
                    continue
                value = float(values[condition.feature])
                margin = value - condition.threshold
                satisfied = value <= condition.threshold if condition.operator == "<=" else value > condition.threshold
                threshold_margins.append(
                    {
                        "rule_id": rule.rule_id,
                        "feature": condition.feature,
                        "operator": condition.operator,
                        "threshold": condition.threshold,
                        "point_value": round(value, 6),
                        "signed_margin": round(margin, 6),
                        "absolute_margin": round(abs(margin), 6),
                        "condition_satisfied": satisfied,
                    }
                )
        profiles.append(
            {
                "point_id": point_id,
                "current_cluster_id": assignments.get(point_id),
                "outlier": outlier_scores.get(point_id),
                "related_rule_ids": [rule.rule_id for rule in related_rules[:4]],
                "related_targets": [
                    {
                        "rule_id": rule.rule_id,
                        "target_kind": rule.target_kind,
                        "target_id": rule.target_id,
                    }
                    for rule in related_rules[:4]
                ],
                "raw_feature_values": raw_feature_values,
                "threshold_margins": threshold_margins[:8],
            }
        )
    return tuple(profiles)


def _pair_relation(
    left,
    right,
    *,
    intersection_count: int,
    shared_features: Tuple[str, ...],
    boundary_gaps: Tuple[Mapping[str, Any], ...],
) -> str:
    same_target = left.target_kind == right.target_kind and left.target_id == right.target_id
    cluster_pair = left.target_kind == right.target_kind == "cluster"
    cluster_anomaly_pair = {left.target_kind, right.target_kind} == {"cluster", "anomaly"}

    if intersection_count:
        if same_target:
            return "same_target_overlap"
        if cluster_pair:
            return "cross_cluster_overlap"
        if cluster_anomaly_pair:
            return "cluster_anomaly_overlap"
        return "cross_target_overlap"

    if same_target:
        return "same_target_disjoint_regions"
    if cluster_pair and shared_features and any(item["gap"] == 0 for item in boundary_gaps):
        return "adjacent_cluster_boundary"
    if cluster_pair:
        return "separate_cluster_rules"
    if cluster_anomaly_pair:
        return "cluster_anomaly_separate"
    return "separate_rules"


def _pair_priority(
    relation: str,
    jaccard: float,
    overlap_share_left: float,
    overlap_share_right: float,
    boundary_gaps: Tuple[Mapping[str, Any], ...],
) -> float:
    priority = max(jaccard, overlap_share_left, overlap_share_right)
    if relation in {"cluster_anomaly_overlap", "cross_cluster_overlap", "cross_target_overlap"}:
        priority += 0.5
    if relation in {"same_target_disjoint_regions", "adjacent_cluster_boundary"}:
        priority += 0.25
    if any(item["gap"] == 0 for item in boundary_gaps):
        priority += 0.1
    return priority


def _condition_intervals(rule) -> Dict[str, Tuple[float | None, float | None]]:
    intervals: Dict[str, Tuple[float | None, float | None]] = {}
    for condition in rule.conditions:
        lower, upper = intervals.get(condition.feature, (None, None))
        if condition.operator == ">":
            lower = condition.threshold if lower is None else max(lower, condition.threshold)
        else:
            upper = condition.threshold if upper is None else min(upper, condition.threshold)
        intervals[condition.feature] = (lower, upper)
    return intervals


def _boundary_gaps(left, right, shared_features: Tuple[str, ...]) -> Tuple[Mapping[str, Any], ...]:
    left_intervals = _condition_intervals(left)
    right_intervals = _condition_intervals(right)
    gaps = []
    for feature in shared_features:
        left_lower, left_upper = left_intervals.get(feature, (None, None))
        right_lower, right_upper = right_intervals.get(feature, (None, None))
        gap = _interval_gap(left_lower, left_upper, right_lower, right_upper)
        gaps.append(
            {
                "feature": feature,
                "rule_a_interval": _interval_payload(left_lower, left_upper),
                "rule_b_interval": _interval_payload(right_lower, right_upper),
                "gap": gap,
            }
        )
    return tuple(gaps)


def _interval_gap(
    left_lower: float | None,
    left_upper: float | None,
    right_lower: float | None,
    right_upper: float | None,
) -> float | None:
    if left_upper is not None and right_lower is not None and left_upper < right_lower:
        return round(right_lower - left_upper, 6)
    if right_upper is not None and left_lower is not None and right_upper < left_lower:
        return round(left_lower - right_upper, 6)
    return 0.0


def _interval_payload(lower: float | None, upper: float | None) -> Mapping[str, Any]:
    return {
        "lower_exclusive": lower,
        "upper_inclusive": upper,
    }


def _category_has_typical_case(rule_set: RuleSet, category: str, metrics: Mapping[str, Any]) -> bool:
    if not rule_set.rules:
        return False
    pair = _best_pair(metrics, category)
    if category == "overlap_merge_signal":
        return pair is not None and pair.get("intersection_count", 0) > 0
    if category == "boundary_review":
        return pair is not None and pair.get("relation") in {
            "adjacent_cluster_boundary",
            "cross_cluster_overlap",
            "cluster_anomaly_overlap",
        }
    if category == "split_or_new_cluster_signal":
        return pair is not None and pair.get("relation") in {
            "same_target_disjoint_regions",
            "adjacent_cluster_boundary",
            "separate_cluster_rules",
        }
    if category == "anomaly_label_review":
        return metrics.get("anomaly_rule_count", 0) > 0
    if category == "exception_relabel_review":
        return metrics.get("exception_point_count", 0) > 0
    if category == "feature_label_strategy":
        feature_usage = metrics.get("feature_usage", {})
        return isinstance(feature_usage, Mapping) and bool(feature_usage)
    return category in {"label_priority", "rule_confidence_audit"}


def _no_case_recommendation(category: str) -> str:
    return {
        "label_priority": "No generated rule is available, so there is no useful next point for this category yet.",
        "boundary_review": "No clear boundary example is available right now; use this category after two rules form a visible boundary.",
        "overlap_merge_signal": "No two rules currently share matched points, so there is no typical merge example to label.",
        "split_or_new_cluster_signal": "No separated rule region is strong enough for a split/new-cluster example right now.",
        "anomaly_label_review": "No anomaly rule is available, so there is no anomaly-specific point to review in this category.",
        "exception_relabel_review": "No exception point is available, so there is no relabel case to inspect here.",
        "feature_label_strategy": "No raw feature threshold dominates the current rule cards, so this category has no clear checklist example.",
        "rule_confidence_audit": "No rule is available to audit yet.",
    }[category]


def _target_rule_ids_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
) -> Tuple[str, ...]:
    if not rule_set.rules:
        return ()
    pair = _best_pair(metrics, category)
    if pair is not None:
        return tuple(pair["rule_ids"])
    if category == "anomaly_label_review":
        anomaly_rules = [rule.rule_id for rule in rule_set.rules if rule.target_kind == "anomaly"]
        return tuple(anomaly_rules[:3])
    if category == "exception_relabel_review":
        exception_rules = [rule.rule_id for rule in rule_set.rules if rule.exception_point_ids]
        if exception_rules:
            return tuple(exception_rules[:3])
    strongest_rule = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count))
    return (strongest_rule.rule_id,)


def _best_pair(metrics: Mapping[str, Any], category: str) -> Mapping[str, Any] | None:
    pair_metrics = tuple(metrics.get("pair_metrics", ()))
    if not pair_metrics:
        return None
    if category == "overlap_merge_signal":
        for metric in pair_metrics:
            if metric["intersection_count"] > 0:
                return metric
        return None
    preferred = {
        "label_priority": {
            "cluster_anomaly_overlap",
            "cross_cluster_overlap",
            "cross_target_overlap",
            "same_target_overlap",
            "adjacent_cluster_boundary",
        },
        "boundary_review": {"adjacent_cluster_boundary", "cross_cluster_overlap", "cluster_anomaly_overlap"},
        "overlap_merge_signal": {
            "cross_cluster_overlap",
            "cluster_anomaly_overlap",
            "cross_target_overlap",
            "same_target_overlap",
        },
        "split_or_new_cluster_signal": {"same_target_disjoint_regions", "adjacent_cluster_boundary", "separate_cluster_rules"},
    }.get(category)
    if preferred is None:
        return None
    for metric in pair_metrics:
        if metric["relation"] in preferred:
            return metric
    return pair_metrics[0]


def _candidate_point_ids_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
    *,
    limit: int | None = None,
) -> list[str]:
    if limit is None:
        limit = _recommended_limit_for_category(category)
    pair = _best_pair(metrics, category)
    rules_by_id = {rule.rule_id: rule for rule in rule_set.rules}
    target_rules = [rules_by_id[rule_id] for rule_id in target_rule_ids if rule_id in rules_by_id]
    profiles = _profiles_by_point(metrics)

    if category == "overlap_merge_signal":
        return _unique_point_ids(pair.get("point_ids", ()) if pair is not None else (), limit=limit)

    if category == "exception_relabel_review":
        source_rules = target_rules or list(rule_set.rules)
        return _unique_point_ids(
            (point_id for rule in source_rules for point_id in rule.exception_point_ids),
            limit=limit,
        )

    if category == "anomaly_label_review":
        anomaly_rules = target_rules or [rule for rule in rule_set.rules if rule.target_kind == "anomaly"]
        return _rank_point_ids(
            (point_id for rule in anomaly_rules for point_id in (*rule.exception_point_ids, *rule.matched_point_ids)),
            profiles,
            mode="outlier_first",
            limit=limit,
        )

    if category == "boundary_review":
        boundary_points = tuple(
            point_id
            for rule in target_rules
            for point_id in (*rule.exception_point_ids, *rule.matched_point_ids)
        )
        return _rank_point_ids(boundary_points, profiles, mode="closest_cutoff", limit=limit)

    if category == "split_or_new_cluster_signal":
        separated_groups = [(*rule.exception_point_ids, *rule.matched_point_ids) for rule in target_rules]
        return _unique_point_ids(_interleave_point_groups(separated_groups), limit=limit)

    if category == "feature_label_strategy":
        dominant_feature = _dominant_feature(metrics)
        feature_points = tuple(
            point_id
            for rule in (target_rules or list(rule_set.rules))
            if dominant_feature is None or any(condition.feature == dominant_feature for condition in rule.conditions)
            for point_id in (*rule.exception_point_ids, *rule.matched_point_ids)
        )
        return _rank_point_ids(
            feature_points,
            profiles,
            mode="dominant_feature_cutoff",
            dominant_feature=dominant_feature,
            limit=limit,
        )

    if category == "rule_confidence_audit":
        audit_rules = target_rules or sorted(
            rule_set.rules,
            key=lambda rule: (-rule.purity, -rule.coverage, -rule.support_count, rule.rule_id),
        )
        return _unique_point_ids(
            (point_id for rule in audit_rules for point_id in (*rule.exception_point_ids, *rule.matched_point_ids)),
            limit=limit,
        )

    candidates = tuple(metrics.get("label_candidate_groups", ()))
    if candidates:
        point_ids = list(candidates[0].get("point_ids", ()))[:6]
        if point_ids:
            return _unique_point_ids(point_ids, limit=limit)
    seed_rule = rules_by_id.get(target_rule_ids[0]) if target_rule_ids else (rule_set.rules[0] if rule_set.rules else None)
    if seed_rule is None:
        return []
    return _unique_point_ids((*seed_rule.exception_point_ids, *seed_rule.matched_point_ids), limit=limit)


def _recommendation_kind_for_category(category: str) -> str:
    return {
        "label_priority": "highest_value_next_labels",
        "boundary_review": "near_cutoff_boundary_labels",
        "overlap_merge_signal": "shared_rule_membership_labels",
        "split_or_new_cluster_signal": "separated_region_labels",
        "anomaly_label_review": "outlier_rule_confirmation_labels",
        "exception_relabel_review": "rule_exception_relabel_labels",
        "feature_label_strategy": "raw_feature_cutoff_checklist_labels",
        "rule_confidence_audit": "surrogate_rule_validation_labels",
    }[category]


def _ranking_method_for_category(category: str) -> str:
    return {
        "label_priority": "Select the highest-ranked deterministic candidate group; use representative rule points only if no stronger candidate group exists.",
        "boundary_review": "Rank points from the linked rules by closeness to the raw-feature cutoff, so labels test both sides of the boundary.",
        "overlap_merge_signal": "Use only points shared by the selected rule pair; do not infer merge evidence from non-overlapping rules.",
        "split_or_new_cluster_signal": "Interleave points from separated rule regions so the user can compare whether they share one human label.",
        "anomaly_label_review": "Rank anomaly-rule points by outlier status and outlier score, then by closeness to rule cutoffs.",
        "exception_relabel_review": "Use rule exception points first because they are direct contradictions between a rule and the current assignment.",
        "feature_label_strategy": "Rank points near the most repeated raw-feature cutoff so the user can test whether that cutoff is meaningful.",
        "rule_confidence_audit": "Use matched and exception points from the strongest rule to test whether a numerically clean rule also matches human labels.",
    }[category]


def _ranking_feature_row(
    point_id: str,
    index: int,
    category: str,
    metrics: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    outlier = profile.get("outlier") if isinstance(profile, Mapping) else None
    outlier_score = outlier.get("score") if isinstance(outlier, Mapping) else None
    closest_margin = _closest_margin(profile)
    dominant_feature = _dominant_feature(metrics)
    components = {
        "reason_code": _recommendation_kind_for_category(category),
        "closest_cutoff_distance": None if closest_margin == 999999.0 else round(closest_margin, 6),
        "outlier_score": outlier_score,
        "current_cluster_id": profile.get("current_cluster_id") if isinstance(profile, Mapping) else None,
        "dominant_feature": dominant_feature,
        "related_rule_count": len(profile.get("related_rule_ids", ())) if isinstance(profile, Mapping) else 0,
    }
    row = {
        "rank": index + 1,
        "candidate_rank": index + 1,
        "point_id": point_id,
        "category": category,
        "reason_code": _recommendation_kind_for_category(category),
        "selection_reason": _selection_reason_for_category(category, profile, components),
        "ranking_score_components": components,
        "closest_cutoff_distance": components["closest_cutoff_distance"],
        "outlier_score": components["outlier_score"],
        "current_cluster_id": components["current_cluster_id"],
        "dominant_feature": components["dominant_feature"],
    }
    if isinstance(profile, Mapping):
        row["related_rule_ids"] = list(profile.get("related_rule_ids", ()))
    return row


def _selection_reason_for_category(
    category: str,
    profile: Mapping[str, Any] | None,
    components: Mapping[str, Any],
) -> str:
    point_id = str(profile.get("point_id")) if isinstance(profile, Mapping) and profile.get("point_id") else "This point"
    distance = components.get("closest_cutoff_distance")
    distance_text = (
        "one of the closest points to a rule cutoff"
        if distance is None
        else f"close to a rule cutoff, with distance {distance}"
    )
    cluster_text = components.get("current_cluster_id") or "its current group"
    outlier_score = components.get("outlier_score")
    outlier_text = f" and outlier score {outlier_score}" if outlier_score is not None else ""
    return {
        "label_priority": f"{point_id} is in {cluster_text}{outlier_text}; labeling it checks the strongest next rule-guided question.",
        "boundary_review": f"{point_id} is {distance_text}, so its human label tests whether the boundary is meaningful.",
        "overlap_merge_signal": f"{point_id} is shared by the selected rule pair, so its label can support or weaken merge review.",
        "split_or_new_cluster_signal": f"{point_id} represents one separated rule region, so its label helps test whether a split or new group is needed.",
        "anomaly_label_review": f"{point_id} is selected from anomaly-rule evidence{outlier_text}; its label checks true anomaly versus normal member.",
        "exception_relabel_review": f"{point_id} is selected from rule exception evidence, where the rule and current assignment disagree.",
        "feature_label_strategy": f"{point_id} is {distance_text} on a frequently used raw feature, so its label tests whether that cutoff is useful.",
        "rule_confidence_audit": f"{point_id} is covered by a high-confidence surrogate rule, so its label audits whether the rule is meaningful to a human.",
    }[category]


def _not_selected_summary(
    category: str,
    candidate_pool: Tuple[str, ...],
    recommended: Tuple[str, ...],
) -> Mapping[str, Any]:
    not_selected = tuple(point_id for point_id in candidate_pool if point_id not in set(recommended))
    if not not_selected:
        return {
            "count": 0,
            "point_ids": [],
            "reason": "Every candidate point is included in the current recommended set.",
        }
    return {
        "count": len(not_selected),
        "point_ids": list(not_selected),
        "reason": (
            f"These points remain in the {category} candidate pool, but the current view shows the highest-ranked "
            "recommended points first to keep the next labeling task small and focused."
        ),
    }


def _plan_label_questions(
    category: str,
    target_rule_ids: Tuple[str, ...],
    recommended_point_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    if not recommended_point_ids:
        return ()
    return (
        {
            "point_ids": recommended_point_ids,
            "rule_ids": target_rule_ids,
            "question": {
                "label_priority": "Which real-world type best describes these records after a human checks the source fields?",
                "boundary_review": "Do records on opposite sides of this dividing line need the same human label or different labels?",
                "overlap_merge_signal": "Do the records shared by these rules describe one real-world type or distinct types?",
                "split_or_new_cluster_signal": "Do these separated regions describe an existing group or a new one?",
                "anomaly_label_review": "Are these records truly unusual, or rare but valid members of a real-world type?",
                "exception_relabel_review": "Do these rule exceptions need a different label, or is the rule simply too broad?",
                "feature_label_strategy": "Does the repeated source-field condition match the labels a human would assign?",
                "rule_confidence_audit": "Does this confident surrogate rule agree with human labels?",
            }[category],
        },
    )


def _plan_label_options_for_category(category: str) -> Tuple[str, ...]:
    return {
        "label_priority": ("existing type", "different type", "truly unusual", "uncertain"),
        "boundary_review": ("same type", "different types", "boundary case", "uncertain"),
        "overlap_merge_signal": ("same type", "different types", "needs more examples", "uncertain"),
        "split_or_new_cluster_signal": ("existing type", "new type", "same as paired region", "uncertain"),
        "anomaly_label_review": ("truly unusual", "rare but valid", "boundary case", "uncertain"),
        "exception_relabel_review": ("current type fits", "different type", "rule is incomplete", "uncertain"),
        "feature_label_strategy": ("cutoff matches human label", "cutoff does not match", "needs more examples", "uncertain"),
        "rule_confidence_audit": ("rule agrees with human label", "rule disagrees", "rule only partly agrees", "uncertain"),
    }[category]


def _candidate_pool_limit_for_category(category: str) -> int:
    return {
        "label_priority": 12,
        "boundary_review": 12,
        "overlap_merge_signal": 12,
        "split_or_new_cluster_signal": 12,
        "anomaly_label_review": 12,
        "exception_relabel_review": 12,
        "feature_label_strategy": 12,
        "rule_confidence_audit": 12,
    }[category]


def _recommended_limit_for_category(category: str) -> int:
    return {
        "label_priority": 4,
        "boundary_review": 4,
        "overlap_merge_signal": 6,
        "split_or_new_cluster_signal": 4,
        "anomaly_label_review": 4,
        "exception_relabel_review": 6,
        "feature_label_strategy": 4,
        "rule_confidence_audit": 4,
    }[category]


def _unique_point_ids(point_ids, *, limit: int) -> list[str]:
    ordered = []
    for point_id in point_ids:
        point_text = str(point_id)
        if not point_text or point_text in ordered:
            continue
        ordered.append(point_text)
        if len(ordered) >= limit:
            break
    return ordered


def _rank_point_ids(
    point_ids,
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
    limit: int,
    dominant_feature: str | None = None,
) -> list[str]:
    unique_ids = _unique_point_ids(point_ids, limit=max(limit * 4, limit))
    scored = []
    for index, point_id in enumerate(unique_ids):
        profile = profiles.get(point_id)
        scored.append((_point_rank_key(profile, mode=mode, dominant_feature=dominant_feature, original_index=index), point_id))
    scored.sort(key=lambda item: item[0])
    return [point_id for _, point_id in scored[:limit]]


def _point_rank_key(
    profile: Mapping[str, Any] | None,
    *,
    mode: str,
    dominant_feature: str | None,
    original_index: int,
) -> tuple:
    if profile is None:
        return (1, 999999.0, original_index)
    if mode == "outlier_first":
        outlier = profile.get("outlier")
        score = float(outlier.get("score", 0.0)) if isinstance(outlier, Mapping) else 0.0
        is_outlier = 1 if isinstance(outlier, Mapping) and outlier.get("is_outlier") else 0
        return (0, -is_outlier, -score, _closest_margin(profile), original_index)
    if mode == "dominant_feature_cutoff":
        return (0, _closest_margin(profile, feature=dominant_feature), original_index)
    return (0, _closest_margin(profile), original_index)


def _closest_margin(profile: Mapping[str, Any] | None, *, feature: str | None = None) -> float:
    if not profile:
        return 999999.0
    margins = profile.get("threshold_margins")
    if not isinstance(margins, (list, tuple)):
        return 999999.0
    values = []
    for margin in margins:
        if not isinstance(margin, Mapping):
            continue
        if feature is not None and margin.get("feature") != feature:
            continue
        value = margin.get("absolute_margin")
        if value is not None:
            values.append(float(value))
    return min(values) if values else 999999.0


def _dominant_feature(metrics: Mapping[str, Any]) -> str | None:
    usage = metrics.get("feature_usage")
    if not isinstance(usage, Mapping) or not usage:
        return None
    return sorted(usage.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _interleave_point_groups(groups) -> list[str]:
    materialized = [list(group) for group in groups if group]
    interleaved = []
    max_length = max((len(group) for group in materialized), default=0)
    for index in range(max_length):
        for group in materialized:
            if index < len(group):
                interleaved.append(group[index])
    return interleaved


def _profiles_by_point(metrics: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(profile.get("point_id")): profile
        for profile in metrics.get("label_candidate_point_profiles", ())
        if isinstance(profile, Mapping) and profile.get("point_id")
    }


def _evidence_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    evidence = []
    pair = _best_pair(metrics, category)
    if pair is not None:
        evidence.append(
            {
                "rule_ids": list(pair["rule_ids"]),
                "target_ids": list(pair["target_ids"]),
                "relation": pair["relation"],
                "intersection_count": pair["intersection_count"],
                "jaccard_overlap": pair["jaccard_overlap"],
                "point_ids": list(pair.get("point_ids", ())[:6]),
            }
        )

    rules_by_id = {rule.rule_id: rule for rule in rule_set.rules}
    for rule_id in target_rule_ids[:3]:
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        entry = {
            "rule_id": rule.rule_id,
            "target_kind": rule.target_kind,
            "target_id": rule.target_id,
            "support_count": rule.support_count,
            "coverage": rule.coverage,
            "purity": rule.purity,
            "exception_count": len(rule.exception_point_ids),
        }
        if rule.conditions:
            condition = rule.conditions[0]
            entry.update(
                {
                    "feature": condition.feature,
                    "operator": condition.operator,
                    "threshold": condition.threshold,
                }
            )
        evidence.append(entry)
    return tuple(evidence)


def _expected_outcomes_for_category(category: str) -> Tuple[Mapping[str, str], ...]:
    return {
        "label_priority": (
            {"label_result": "labels agree with the current rule target", "decision_implication": "use the rule as an anchor for later review"},
            {"label_result": "labels disagree or are mixed", "decision_implication": "switch to boundary, exception, or split review before changing state"},
        ),
        "boundary_review": (
            {"label_result": "both sides get the same label", "decision_implication": "boundary may be artificial and merge/shared-boundary review becomes plausible"},
            {"label_result": "each side gets a different label", "decision_implication": "boundary is likely meaningful and should be preserved"},
        ),
        "overlap_merge_signal": (
            {"label_result": "overlap labels agree", "decision_implication": "merge or shared-boundary review becomes plausible"},
            {"label_result": "overlap labels are mixed", "decision_implication": "keep separate targets or investigate a new cluster"},
        ),
        "split_or_new_cluster_signal": (
            {"label_result": "regions get different labels", "decision_implication": "split or new-cluster review becomes plausible"},
            {"label_result": "regions get the same label", "decision_implication": "current target may be coherent despite multiple rules"},
        ),
        "anomaly_label_review": (
            {"label_result": "points are true anomalies", "decision_implication": "retain or strengthen the anomaly rule explanation"},
            {"label_result": "points are normal members", "decision_implication": "review outlier flags or mark them normal"},
        ),
        "exception_relabel_review": (
            {"label_result": "exceptions share one label", "decision_implication": "use them as relabel or boundary-fix evidence"},
            {"label_result": "exceptions are mixed", "decision_implication": "collect more nearby labels before changing structure"},
        ),
        "feature_label_strategy": (
            {"label_result": "labels align with raw thresholds", "decision_implication": "use these thresholds in the labeling checklist"},
            {"label_result": "labels do not align with raw thresholds", "decision_implication": "treat the surrogate feature story as weak"},
        ),
        "rule_confidence_audit": (
            {"label_result": "labels agree with high-confidence rules", "decision_implication": "rules are usable as guidance for follow-up checks"},
            {"label_result": "labels disagree with high-confidence rules", "decision_implication": "rule fidelity to SSDBCODI is not enough; user semantics require refinement"},
        ),
    }[category]


def _risk_note_for_category(category: str, metrics: Mapping[str, Any]) -> str:
    if category == "overlap_merge_signal" and _best_pair(metrics, category) is None:
        return "There is no sample-level overlap, so this action must not be treated as merge evidence yet."
    if category == "exception_relabel_review" and not metrics.get("exception_point_count", 0):
        return "No exception points are present, so the action relies on representative points rather than direct contradictions."
    if category == "anomaly_label_review" and not metrics.get("anomaly_rule_count", 0):
        return "No anomaly rule is present, so anomaly review should wait for outlier evidence."
    return "The recommendation is grounded in surrogate rules, but semantic correctness still depends on user labels."


def _validate_focus_category(focus_category: str | None) -> str | None:
    if focus_category in (None, ""):
        return None
    if not isinstance(focus_category, str):
        raise ValueError("focus_category must be a string")
    cleaned = focus_category.strip()
    if cleaned not in RECOMMENDATION_CATEGORIES:
        raise ValueError(f"focus_category must be one of: {', '.join(RECOMMENDATION_CATEGORIES)}")
    return cleaned

__all__ = [
    "RECOMMENDATION_CATEGORIES",
    "build_recommendation_plan",
    "build_rule_guidance_metrics",
]
