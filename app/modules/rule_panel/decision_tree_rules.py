from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from app.shared.schemas import FeatureMatrix

from .schemas import RuleCard, RuleCondition, TreeConfig


@dataclass(frozen=True)
class _PathCondition:
    feature_index: int
    feature: str
    operator: str
    threshold: float

    def as_rule_condition(self) -> RuleCondition:
        return RuleCondition(
            feature=self.feature,
            operator=self.operator,
            threshold=self.threshold,
        )


@dataclass(frozen=True)
class _Leaf:
    samples: Tuple[int, ...]
    conditions: Tuple[_PathCondition, ...]
    depth: int
    leaf_id: str


def build_surrogate_rules(
    feature_matrix: FeatureMatrix,
    labels_by_point_id: Mapping[str, str],
    *,
    target_kind: str,
    config: TreeConfig,
    target_ids: Iterable[str] | None = None,
) -> Tuple[RuleCard, ...]:
    """Fit a tiny deterministic CART-style surrogate and return leaf rules.

    The labels are fixed upstream outputs. This function only extracts rules
    that approximate those labels; it is never an analysis provider.
    """

    sample_indices = tuple(
        index
        for index, point_id in enumerate(feature_matrix.point_ids)
        if point_id in labels_by_point_id
    )
    if not sample_indices:
        return ()

    labels = {index: labels_by_point_id[feature_matrix.point_ids[index]] for index in sample_indices}
    leaves = _build_leaves(feature_matrix, labels, sample_indices, (), 0, config, "leaf")
    allowed_targets = set(target_ids) if target_ids is not None else None
    rules = []
    sequence_by_target: Dict[str, int] = {}

    for leaf in leaves:
        target_id = _majority_label(labels, leaf.samples)
        if allowed_targets is not None and target_id not in allowed_targets:
            continue

        sequence_by_target[target_id] = sequence_by_target.get(target_id, 0) + 1
        matched = _matching_indices(feature_matrix, sample_indices, leaf.conditions)
        target_total = sum(1 for label in labels.values() if label == target_id)
        target_matches = tuple(index for index in matched if labels[index] == target_id)
        exceptions = tuple(index for index in matched if labels[index] != target_id)
        support_count = len(matched)
        coverage = len(target_matches) / target_total if target_total else 0.0
        purity = len(target_matches) / support_count if support_count else 0.0
        rule_conditions = tuple(condition.as_rule_condition() for condition in leaf.conditions)
        rule_id = f"rule_{target_kind}_{_safe_id(target_id)}_{sequence_by_target[target_id]:03d}"
        warnings = _quality_warnings(rule_conditions, support_count, purity, config)

        rules.append(
            RuleCard(
                rule_id=rule_id,
                target_kind=target_kind,
                target_id=target_id,
                conditions=rule_conditions,
                support_count=support_count,
                coverage=round(coverage, 6),
                purity=round(purity, 6),
                matched_point_ids=tuple(feature_matrix.point_ids[index] for index in matched),
                exception_point_ids=tuple(feature_matrix.point_ids[index] for index in exceptions),
                diagnostics={
                    "tree_depth": leaf.depth,
                    "leaf_id": leaf.leaf_id,
                    "target_total": target_total,
                    "target_matches": len(target_matches),
                    "quality_warnings": warnings,
                },
            )
        )

    return tuple(sorted(rules, key=lambda rule: (rule.target_kind, rule.target_id, rule.rule_id)))


def _build_leaves(
    feature_matrix: FeatureMatrix,
    labels: Mapping[int, str],
    sample_indices: Tuple[int, ...],
    conditions: Tuple[_PathCondition, ...],
    depth: int,
    config: TreeConfig,
    leaf_prefix: str,
) -> Tuple[_Leaf, ...]:
    if _should_stop(labels, sample_indices, depth, config):
        return (_Leaf(samples=sample_indices, conditions=conditions, depth=depth, leaf_id=leaf_prefix),)

    split = _best_split(feature_matrix, labels, sample_indices, config)
    if split is None:
        return (_Leaf(samples=sample_indices, conditions=conditions, depth=depth, leaf_id=leaf_prefix),)

    feature_index, threshold, left, right = split
    feature = feature_matrix.feature_names[feature_index]
    left_condition = _PathCondition(feature_index, feature, "<=", threshold)
    right_condition = _PathCondition(feature_index, feature, ">", threshold)

    return (
        *_build_leaves(
            feature_matrix,
            labels,
            left,
            (*conditions, left_condition),
            depth + 1,
            config,
            f"{leaf_prefix}_l",
        ),
        *_build_leaves(
            feature_matrix,
            labels,
            right,
            (*conditions, right_condition),
            depth + 1,
            config,
            f"{leaf_prefix}_r",
        ),
    )


def _should_stop(
    labels: Mapping[int, str],
    sample_indices: Tuple[int, ...],
    depth: int,
    config: TreeConfig,
) -> bool:
    if depth >= config.max_depth:
        return True
    if len(sample_indices) < config.min_samples_leaf * 2:
        return True
    return len({labels[index] for index in sample_indices}) <= 1


def _best_split(
    feature_matrix: FeatureMatrix,
    labels: Mapping[int, str],
    sample_indices: Tuple[int, ...],
    config: TreeConfig,
) -> Tuple[int, float, Tuple[int, ...], Tuple[int, ...]] | None:
    best: Tuple[float, int, float, Tuple[int, ...], Tuple[int, ...]] | None = None
    n_features = len(feature_matrix.feature_names)

    for feature_index in range(n_features):
        values = sorted({feature_matrix.values[index][feature_index] for index in sample_indices})
        if len(values) < 2:
            continue

        thresholds = tuple((values[index] + values[index + 1]) / 2 for index in range(len(values) - 1))
        for threshold in thresholds:
            left = tuple(
                index for index in sample_indices if feature_matrix.values[index][feature_index] <= threshold
            )
            right = tuple(index for index in sample_indices if index not in set(left))
            if len(left) < config.min_samples_leaf or len(right) < config.min_samples_leaf:
                continue

            impurity = _weighted_gini(labels, left, right)
            candidate = (impurity, feature_index, threshold, left, right)
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    if best is None:
        return None

    _, feature_index, threshold, left, right = best
    return feature_index, threshold, left, right


def _weighted_gini(
    labels: Mapping[int, str],
    left: Tuple[int, ...],
    right: Tuple[int, ...],
) -> float:
    total = len(left) + len(right)
    return (len(left) / total) * _gini(labels, left) + (len(right) / total) * _gini(labels, right)


def _gini(labels: Mapping[int, str], sample_indices: Tuple[int, ...]) -> float:
    counts: Dict[str, int] = {}
    for index in sample_indices:
        label = labels[index]
        counts[label] = counts.get(label, 0) + 1
    total = len(sample_indices)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def _majority_label(labels: Mapping[int, str], sample_indices: Tuple[int, ...]) -> str:
    counts: Dict[str, int] = {}
    for index in sample_indices:
        label = labels[index]
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _matching_indices(
    feature_matrix: FeatureMatrix,
    sample_indices: Sequence[int],
    conditions: Tuple[_PathCondition, ...],
) -> Tuple[int, ...]:
    matched = []
    for index in sample_indices:
        if all(_condition_matches(feature_matrix.values[index], condition) for condition in conditions):
            matched.append(index)
    return tuple(matched)


def _condition_matches(row: Sequence[float], condition: _PathCondition) -> bool:
    value = row[condition.feature_index]
    if condition.operator == "<=":
        return value <= condition.threshold
    return value > condition.threshold


def _quality_warnings(
    conditions: Tuple[RuleCondition, ...],
    support_count: int,
    purity: float,
    config: TreeConfig,
) -> Tuple[str, ...]:
    warnings = []
    if not conditions:
        warnings.append("broad_rule")
    if support_count < config.min_samples_leaf:
        warnings.append("low_support")
    if purity < config.min_purity_warning:
        warnings.append("low_purity")
    return tuple(warnings)


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_") or "target"
