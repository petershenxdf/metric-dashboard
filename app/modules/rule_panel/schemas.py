from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.shared.schemas import clean_number, clean_text


RULE_INTERPRETATION_CATEGORIES = (
    "label_priority",
    "boundary_review",
    "overlap_merge_signal",
    "split_or_new_cluster_signal",
    "anomaly_label_review",
    "exception_relabel_review",
    "feature_label_strategy",
    "rule_confidence_audit",
)


@dataclass(frozen=True)
class TreeConfig:
    max_depth: int = 3
    min_samples_leaf: int = 1
    min_purity_warning: float = 0.8

    def __post_init__(self) -> None:
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
            raise ValueError("max_depth must be an integer")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        if isinstance(self.min_samples_leaf, bool) or not isinstance(self.min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be an integer")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")

        min_purity = clean_number(self.min_purity_warning, "min_purity_warning")
        if min_purity <= 0 or min_purity > 1:
            raise ValueError("min_purity_warning must be in (0, 1]")
        object.__setattr__(self, "min_purity_warning", min_purity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_purity_warning": self.min_purity_warning,
        }


@dataclass(frozen=True)
class RuleCondition:
    feature: str
    operator: str
    threshold: float

    def __post_init__(self) -> None:
        operator = clean_text(self.operator, "operator")
        if operator not in ("<=", ">"):
            raise ValueError("operator must be <= or >")

        object.__setattr__(self, "feature", clean_text(self.feature, "feature"))
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "threshold", clean_number(self.threshold, "threshold"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "operator": self.operator,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class RuleCard:
    rule_id: str
    target_kind: str
    target_id: str
    conditions: Tuple[RuleCondition, ...]
    support_count: int
    coverage: float
    purity: float
    matched_point_ids: Tuple[str, ...]
    exception_point_ids: Tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target_kind = clean_text(self.target_kind, "target_kind")
        if target_kind not in ("cluster", "anomaly"):
            raise ValueError("target_kind must be cluster or anomaly")

        conditions = tuple(self.conditions)
        if not all(isinstance(condition, RuleCondition) for condition in conditions):
            raise ValueError("conditions must contain RuleCondition objects")

        if isinstance(self.support_count, bool) or not isinstance(self.support_count, int):
            raise ValueError("support_count must be an integer")
        if self.support_count < 0:
            raise ValueError("support_count must be non-negative")

        coverage = clean_number(self.coverage, "coverage")
        purity = clean_number(self.purity, "purity")
        if coverage < 0 or coverage > 1:
            raise ValueError("coverage must be between 0 and 1")
        if purity < 0 or purity > 1:
            raise ValueError("purity must be between 0 and 1")

        matched = tuple(clean_text(point_id, "point_id") for point_id in self.matched_point_ids)
        exceptions = tuple(clean_text(point_id, "point_id") for point_id in self.exception_point_ids)
        if len(set(matched)) != len(matched):
            raise ValueError("matched_point_ids must be unique")
        if len(set(exceptions)) != len(exceptions):
            raise ValueError("exception_point_ids must be unique")

        object.__setattr__(self, "rule_id", clean_text(self.rule_id, "rule_id"))
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(self, "target_id", clean_text(self.target_id, "target_id"))
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "purity", purity)
        object.__setattr__(self, "matched_point_ids", matched)
        object.__setattr__(self, "exception_point_ids", exceptions)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "support_count": self.support_count,
            "coverage": self.coverage,
            "purity": self.purity,
            "matched_point_ids": list(self.matched_point_ids),
            "exception_point_ids": list(self.exception_point_ids),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class RuleSet:
    rule_set_id: str
    dataset_id: str
    source_analysis_run_id: str
    model: Mapping[str, Any]
    rules: Tuple[RuleCard, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if not all(isinstance(rule, RuleCard) for rule in rules):
            raise ValueError("rules must contain RuleCard objects")

        object.__setattr__(self, "rule_set_id", clean_text(self.rule_set_id, "rule_set_id"))
        object.__setattr__(self, "dataset_id", clean_text(self.dataset_id, "dataset_id"))
        object.__setattr__(
            self,
            "source_analysis_run_id",
            clean_text(self.source_analysis_run_id, "source_analysis_run_id"),
        )
        object.__setattr__(self, "model", dict(self.model or {}))
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_set_id": self.rule_set_id,
            "dataset_id": self.dataset_id,
            "source_analysis_run_id": self.source_analysis_run_id,
            "model": dict(self.model),
            "rules": [rule.to_dict() for rule in self.rules],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class RuleInterpretation:
    interpretation_id: str
    rule_set_id: str
    categories: Tuple[str, ...]
    target_rule_ids: Tuple[str, ...]
    summary: str
    evidence: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    recommendation: str | None = None
    category_explanation: str | None = None
    decision_rationale: str | None = None
    label_targets: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    suspicion_reasons: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    point_label_guidance: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    label_outcomes: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    quantitative_findings: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    suggested_label_actions: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    confidence: float = 1.0
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    provider_label: str = "mock_rule_interpreter"

    def __post_init__(self) -> None:
        categories = tuple(clean_text(category, "category") for category in self.categories)
        unknown = sorted(set(categories) - set(RULE_INTERPRETATION_CATEGORIES))
        if unknown:
            raise ValueError(f"unknown interpretation categories: {', '.join(unknown)}")

        target_rule_ids = tuple(clean_text(rule_id, "rule_id") for rule_id in self.target_rule_ids)
        evidence = tuple(dict(item) for item in self.evidence)
        label_targets = tuple(dict(item) for item in self.label_targets)
        suspicion_reasons = tuple(dict(item) for item in self.suspicion_reasons)
        point_label_guidance = tuple(dict(item) for item in self.point_label_guidance)
        label_outcomes = tuple(dict(item) for item in self.label_outcomes)
        quantitative_findings = tuple(dict(item) for item in self.quantitative_findings)
        suggested_label_actions = tuple(dict(item) for item in self.suggested_label_actions)
        confidence = clean_number(self.confidence, "confidence")
        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        summary = clean_text(self.summary, "summary")
        recommendation = summary
        if self.recommendation is not None:
            recommendation = clean_text(self.recommendation, "recommendation")
        category_explanation = summary
        if self.category_explanation is not None:
            category_explanation = clean_text(self.category_explanation, "category_explanation")
        decision_rationale = summary
        if self.decision_rationale is not None:
            decision_rationale = clean_text(self.decision_rationale, "decision_rationale")

        object.__setattr__(self, "interpretation_id", clean_text(self.interpretation_id, "interpretation_id"))
        object.__setattr__(self, "rule_set_id", clean_text(self.rule_set_id, "rule_set_id"))
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "target_rule_ids", target_rule_ids)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "recommendation", recommendation)
        object.__setattr__(self, "category_explanation", category_explanation)
        object.__setattr__(self, "decision_rationale", decision_rationale)
        object.__setattr__(self, "label_targets", label_targets)
        object.__setattr__(self, "suspicion_reasons", suspicion_reasons)
        object.__setattr__(self, "point_label_guidance", point_label_guidance)
        object.__setattr__(self, "label_outcomes", label_outcomes)
        object.__setattr__(self, "quantitative_findings", quantitative_findings)
        object.__setattr__(self, "suggested_label_actions", suggested_label_actions)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "provider_label", clean_text(self.provider_label, "provider_label"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interpretation_id": self.interpretation_id,
            "rule_set_id": self.rule_set_id,
            "categories": list(self.categories),
            "target_rule_ids": list(self.target_rule_ids),
            "summary": self.summary,
            "evidence": [dict(item) for item in self.evidence],
            "recommendation": self.recommendation,
            "category_explanation": self.category_explanation,
            "decision_rationale": self.decision_rationale,
            "label_targets": [dict(item) for item in self.label_targets],
            "suspicion_reasons": [dict(item) for item in self.suspicion_reasons],
            "point_label_guidance": [dict(item) for item in self.point_label_guidance],
            "label_outcomes": [dict(item) for item in self.label_outcomes],
            "quantitative_findings": [dict(item) for item in self.quantitative_findings],
            "suggested_label_actions": [dict(item) for item in self.suggested_label_actions],
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "provider_label": self.provider_label,
        }
