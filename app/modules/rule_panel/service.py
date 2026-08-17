from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Tuple

from app.modules.algorithm_adapters.schemas import AnalysisResult
from app.modules.algorithm_adapters.service import assignments_by_point_id
from app.shared.schemas import FeatureMatrix

from .decision_tree_rules import build_surrogate_rules
from .schemas import RuleCard, RuleSet, TreeConfig

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


def _validate_known_points(
    feature_matrix: FeatureMatrix,
    labels_by_point_id: Mapping[str, str],
    label_name: str,
) -> None:
    unknown = sorted(set(labels_by_point_id) - set(feature_matrix.point_ids))
    if unknown:
        raise ValueError(f"{label_name} references unknown point id(s): {', '.join(unknown)}")


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
