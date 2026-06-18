from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Tuple

from app.modules.algorithm_adapters.schemas import AnalysisResult
from app.modules.algorithm_adapters.service import assignments_by_point_id
from app.shared.schemas import FeatureMatrix

from .decision_tree_rules import build_surrogate_rules
from .schemas import RuleCard, RuleInterpretation, RuleSet, TreeConfig

DEFAULT_TREE_CONFIG = TreeConfig(max_depth=3, min_samples_leaf=1)
ANOMALY_TARGET_ID = "current_outliers"
NORMAL_TARGET_ID = "normal_points"


def generate_rule_set(
    feature_matrix: FeatureMatrix,
    analysis_result: AnalysisResult,
    *,
    dataset_id: str,
    config: TreeConfig = DEFAULT_TREE_CONFIG,
) -> RuleSet:
    if not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")
    if not isinstance(analysis_result, AnalysisResult):
        raise ValueError("analysis_result must be an AnalysisResult")

    cluster_labels = assignments_by_point_id(analysis_result.cluster_result)
    _validate_known_points(feature_matrix, cluster_labels, "cluster assignment")
    cluster_rules = build_surrogate_rules(
        feature_matrix,
        cluster_labels,
        target_kind="cluster",
        config=config,
    )

    anomaly_labels = {
        score.point_id: ANOMALY_TARGET_ID if score.is_outlier else NORMAL_TARGET_ID
        for score in analysis_result.outlier_result.scores
    }
    _validate_known_points(feature_matrix, anomaly_labels, "outlier score")
    anomaly_rules: Tuple[RuleCard, ...] = ()
    if any(label == ANOMALY_TARGET_ID for label in anomaly_labels.values()):
        anomaly_rules = build_surrogate_rules(
            feature_matrix,
            anomaly_labels,
            target_kind="anomaly",
            config=config,
            target_ids=(ANOMALY_TARGET_ID,),
        )

    rules = (*cluster_rules, *anomaly_rules)
    payload = {
        "dataset_id": dataset_id,
        "analysis_run_id": analysis_result.analysis_run_id,
        "config": config.to_dict(),
        "rule_fingerprint": [rule.to_dict() for rule in rules],
    }

    return RuleSet(
        rule_set_id=_stable_id("rules", payload),
        dataset_id=dataset_id,
        source_analysis_run_id=analysis_result.analysis_run_id,
        model={
            "algorithm": "decision_tree_surrogate",
            **config.to_dict(),
            "source_of_truth": "ssdbcodi",
            "role": "explanation_only",
        },
        rules=rules,
        diagnostics={
            "cluster_rule_count": len(cluster_rules),
            "anomaly_rule_count": len(anomaly_rules),
            "source_point_count": len(feature_matrix.point_ids),
            "source_feature_count": len(feature_matrix.feature_names),
            "raw_feature_names": list(feature_matrix.feature_names),
            "feature_usage": _feature_usage(rules),
            "quality_warning_counts": _quality_warning_counts(rules),
            "source_cluster_run_id": analysis_result.cluster_result.cluster_run_id,
            "source_outlier_run_id": analysis_result.outlier_result.outlier_run_id,
            "decision_tree_boundary": "rules explain SSDBCODI output; they do not replace clustering or outlier detection",
        },
    )


def interpret_rule_set_preview(rule_set: RuleSet) -> RuleInterpretation:
    """Return a deterministic interpretation preview for Step 8.6.

    Step 8.7 can replace this with a DeepSeek-backed provider while preserving
    the output schema and evidence contract.
    """

    if not rule_set.rules:
        return RuleInterpretation(
            interpretation_id=_stable_id("interp", {"rule_set_id": rule_set.rule_set_id, "empty": True}),
            rule_set_id=rule_set.rule_set_id,
            categories=("rule_confidence_audit",),
            target_rule_ids=(),
            summary="No rule cards were generated for the current SSDBCODI output.",
            evidence=(),
            recommendation="Run SSDBCODI and generate rule cards before asking for label guidance.",
            category_explanation="Rule confidence audit checks whether generated rules are reliable enough to guide labels.",
            decision_rationale="There are no rule paths, matched points, or target regions to compare, so any label advice would be ungrounded.",
            label_targets=(),
            suspicion_reasons=(),
            point_label_guidance=(),
            label_outcomes=(
                {
                    "label_result": "Rules are generated after SSDBCODI runs",
                    "decision_implication": "The system can then rank points, boundaries, and anomaly candidates for labeling.",
                },
            ),
            quantitative_findings=(),
            suggested_label_actions=(),
            confidence=1.0,
            warnings=("empty_rule_set",),
        )

    cluster_rules = tuple(rule for rule in rule_set.rules if rule.target_kind == "cluster")
    anomaly_rules = tuple(rule for rule in rule_set.rules if rule.target_kind == "anomaly")
    strongest_rule = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count))
    categories = ["label_priority", "feature_label_strategy", "rule_confidence_audit"]
    if anomaly_rules:
        categories.append("anomaly_label_review")
    if any(rule.exception_point_ids for rule in rule_set.rules):
        categories.append("exception_relabel_review")

    summary_parts = []
    if cluster_rules:
        summary_parts.append(f"{len(cluster_rules)} cluster rule cards explain SSDBCODI cluster assignments")
    if anomaly_rules:
        summary_parts.append(f"{len(anomaly_rules)} anomaly rule cards explain current SSDBCODI outlier flags")
    summary_parts.append(
        f"strongest rule is {strongest_rule.rule_id} with purity {strongest_rule.purity:.2f} and coverage {strongest_rule.coverage:.2f}"
    )
    evidence = _evidence_from_rule(strongest_rule)
    preview_points = list(strongest_rule.matched_point_ids[:5])

    return RuleInterpretation(
        interpretation_id=_stable_id(
            "interp",
            {
                "rule_set_id": rule_set.rule_set_id,
                "target_rule_ids": [strongest_rule.rule_id],
                "categories": categories,
            },
        ),
        rule_set_id=rule_set.rule_set_id,
        categories=tuple(categories),
        target_rule_ids=(strongest_rule.rule_id,),
        summary="; ".join(summary_parts) + ".",
        evidence=evidence,
        recommendation=(
            f"Use {strongest_rule.rule_id} as the first labeling checkpoint: label a small sample of its "
            f"{strongest_rule.support_count} matched points before trusting the surrogate rule."
        ),
        category_explanation="Label priority ranks the next points whose human labels would most reduce uncertainty.",
        decision_rationale=(
            f"{strongest_rule.rule_id} is the safest initial checkpoint because it combines support "
            f"{strongest_rule.support_count}, purity {strongest_rule.purity:.2f}, and coverage "
            f"{strongest_rule.coverage:.2f}. The LLM should use this stable region as a baseline before "
            "suggesting higher-cost merge, split, or anomaly decisions."
        ),
        label_targets=(
            {
                "priority": "medium",
                "point_ids": preview_points,
                "rule_ids": [strongest_rule.rule_id],
                "label_question": f"Do these wine samples really belong to {strongest_rule.target_id}?",
                "why_label_these_points": (
                    f"They are the first matched points from the strongest rule, which has support "
                    f"{strongest_rule.support_count}, purity {strongest_rule.purity:.2f}, and coverage {strongest_rule.coverage:.2f}."
                ),
            },
        ),
        suspicion_reasons=(
            {
                "point_ids": preview_points,
                "rule_ids": [strongest_rule.rule_id],
                "suspicious_signal": "High surrogate confidence still only means agreement with SSDBCODI.",
                "rule_based_reason": (
                    f"{strongest_rule.rule_id} is strong enough to test first, but it is not yet validated by human labels."
                ),
                "point_based_reason": "These points are representative matched samples, so a disagreement would question the rule's semantic value.",
            },
        ),
        point_label_guidance=(
            {
                "point_ids": preview_points,
                "rule_ids": [strongest_rule.rule_id],
                "suggested_label_frame": f"Label each point as {strongest_rule.target_id}, another visible cluster, or uncertain/anomaly.",
                "how_to_label": "Use the raw feature conditions shown in the rule card as the checklist, then record the user's semantic label.",
                "decision_impact": "Agreement makes the rule a useful anchor; disagreement means the next step should be boundary or exception review.",
                "llm_analysis_note": "Mock preview only; provider_kind=deepseek asks DeepSeek V4 Pro to analyze point-level feature values.",
            },
        ),
        label_outcomes=(
            {
                "label_result": "sample labels agree with the rule target",
                "decision_implication": "treat the rule as a stable reference region for later boundary or anomaly review",
                "rule_ids": [strongest_rule.rule_id],
            },
            {
                "label_result": "sample labels disagree with the rule target",
                "decision_implication": "audit the target assignment before using this rule for refinement",
                "rule_ids": [strongest_rule.rule_id],
            },
        ),
        quantitative_findings=(
            {
                "metric": "strongest_rule_support",
                "value": strongest_rule.support_count,
                "rule_ids": [strongest_rule.rule_id],
                "interpretation": "largest high-purity rule to inspect first",
            },
            {
                "metric": "strongest_rule_purity",
                "value": strongest_rule.purity,
                "rule_ids": [strongest_rule.rule_id],
                "interpretation": "fraction of matched points already aligned with the SSDBCODI target",
            },
        ),
        suggested_label_actions=(
            {
                "action_type": "inspect_points",
                "priority": "medium",
                "rule_ids": [strongest_rule.rule_id],
                "point_ids": preview_points,
                "reason": "preview sample from the strongest rule before manual refinement",
                "hypothesis": "The strongest rule describes a coherent region that can anchor later refinement decisions.",
                "why_this_action": "A small label sample checks semantic agreement before the system treats the rule as reliable guidance.",
                "expected_outcomes": [
                    {
                        "label_result": "labels agree",
                        "decision_implication": "use this rule as a trusted reference region",
                    },
                    {
                        "label_result": "labels disagree",
                        "decision_implication": "downgrade rule confidence and inspect nearby boundaries",
                    },
                ],
                "risk_note": "High purity in the surrogate still reflects SSDBCODI agreement, not human semantic truth.",
            },
        ),
        confidence=0.72,
        warnings=_interpretation_warnings(rule_set),
    )


def _validate_known_points(
    feature_matrix: FeatureMatrix,
    labels_by_point_id: Mapping[str, str],
    label_name: str,
) -> None:
    unknown = sorted(set(labels_by_point_id) - set(feature_matrix.point_ids))
    if unknown:
        raise ValueError(f"{label_name} references unknown point id(s): {', '.join(unknown)}")


def _evidence_from_rule(rule: RuleCard) -> Tuple[Mapping[str, object], ...]:
    if not rule.conditions:
        return (
            {
                "rule_id": rule.rule_id,
                "target_kind": rule.target_kind,
                "target_id": rule.target_id,
                "support_count": rule.support_count,
                "purity": rule.purity,
                "coverage": rule.coverage,
            },
        )

    evidence = []
    for condition in rule.conditions:
        evidence.append(
            {
                "rule_id": rule.rule_id,
                "target_kind": rule.target_kind,
                "target_id": rule.target_id,
                "feature": condition.feature,
                "operator": condition.operator,
                "threshold": condition.threshold,
            }
        )
    return tuple(evidence)


def _interpretation_warnings(rule_set: RuleSet) -> Tuple[str, ...]:
    warnings = []
    if any(rule.purity < 0.8 for rule in rule_set.rules):
        warnings.append("some_rules_have_low_purity")
    if any(not rule.conditions for rule in rule_set.rules):
        warnings.append("some_rules_are_broad")
    if not any(rule.target_kind == "anomaly" for rule in rule_set.rules):
        warnings.append("no_anomaly_rule_generated")
    return tuple(warnings)


def _feature_usage(rules: Tuple[RuleCard, ...]) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for rule in rules:
        for condition in rule.conditions:
            usage[condition.feature] = usage.get(condition.feature, 0) + 1
    return dict(sorted(usage.items(), key=lambda item: (-item[1], item[0])))


def _quality_warning_counts(rules: Tuple[RuleCard, ...]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rule in rules:
        for warning in rule.diagnostics.get("quality_warnings", ()):
            counts[warning] = counts.get(warning, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
