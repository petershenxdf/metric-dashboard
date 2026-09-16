from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.shared.schemas import clean_number, clean_text


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
