from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.shared.schemas import clean_number, clean_text


RECOMMENDATION_CATEGORIES = (
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


def _clean_unique_text_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    cleaned = tuple(clean_text(item, field_name) for item in value)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} must contain unique values")
    return cleaned


def _validate_ranking_features(features: Tuple[Mapping[str, Any], ...], recommended_point_ids: Tuple[str, ...]) -> None:
    feature_points = []
    for row in features:
        point_id = row.get("point_id")
        if not isinstance(point_id, str) or not point_id.strip():
            raise ValueError("ranking_features entries must include point_id")
        feature_points.append(point_id)
        rank = row.get("candidate_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("ranking_features candidate_rank must be a positive integer")
        selection_reason = row.get("selection_reason")
        if not isinstance(selection_reason, str) or not selection_reason.strip():
            raise ValueError("ranking_features selection_reason must be a non-empty string")
        components = row.get("ranking_score_components")
        if not isinstance(components, Mapping):
            raise ValueError("ranking_features ranking_score_components must be an object")
    if tuple(feature_points) != recommended_point_ids:
        raise ValueError("ranking_features point order must match recommended_point_ids")


def _validate_point_profiles(profiles: Tuple[Mapping[str, Any], ...], recommended_point_ids: Tuple[str, ...]) -> None:
    allowed = set(recommended_point_ids)
    profile_point_ids = []
    for profile in profiles:
        point_id = profile.get("point_id")
        if not isinstance(point_id, str) or not point_id.strip():
            raise ValueError("point_profiles entries must include point_id")
        if point_id not in allowed:
            raise ValueError(f"point_profiles contains a point outside recommended_point_ids: {point_id}")
        profile_point_ids.append(point_id)
    if tuple(profile_point_ids) != recommended_point_ids:
        raise ValueError("point_profiles point order must match recommended_point_ids")


@dataclass(frozen=True)
class RecommendationPlan:
    plan_id: str
    plan_version: str
    dataset_id: str
    rule_set_id: str
    focus_category: str
    recommendation_kind: str
    has_typical_case: bool
    candidate_pool_point_ids: Tuple[str, ...]
    recommended_point_ids: Tuple[str, ...]
    highlighted_point_ids: Tuple[str, ...]
    target_rule_ids: Tuple[str, ...]
    ranking_method: str
    ranking_features: Tuple[Mapping[str, Any], ...]
    point_profiles: Tuple[Mapping[str, Any], ...]
    analysis_run_id: str | None = None
    evidence_rows: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    label_questions: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    label_options: Tuple[str, ...] = field(default_factory=tuple)
    expected_label_outcomes: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    uncertainty_notes: Tuple[str, ...] = field(default_factory=tuple)
    immutable_fields: Tuple[str, ...] = field(default_factory=tuple)
    llm_role: str = "translate_only_do_not_select_points"
    not_selected_summary: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plan_id = clean_text(self.plan_id, "plan_id")
        if not plan_id.startswith("recplan_"):
            raise ValueError("plan_id must start with recplan_")
        focus_category = clean_text(self.focus_category, "focus_category")
        if focus_category not in RECOMMENDATION_CATEGORIES:
            raise ValueError(f"focus_category must be one of: {', '.join(RECOMMENDATION_CATEGORIES)}")
        if not isinstance(self.has_typical_case, bool):
            raise ValueError("has_typical_case must be a boolean")

        candidate_pool = _clean_unique_text_tuple(self.candidate_pool_point_ids, "candidate_pool_point_ids")
        recommended = _clean_unique_text_tuple(self.recommended_point_ids, "recommended_point_ids")
        highlighted = _clean_unique_text_tuple(self.highlighted_point_ids, "highlighted_point_ids")
        target_rule_ids = _clean_unique_text_tuple(self.target_rule_ids, "target_rule_ids")
        missing_recommended = sorted(set(recommended) - set(candidate_pool))
        if missing_recommended:
            raise ValueError(f"recommended_point_ids must be contained in candidate_pool_point_ids: {', '.join(missing_recommended)}")
        missing_highlighted = sorted(set(highlighted) - set(recommended))
        if missing_highlighted:
            raise ValueError(f"highlighted_point_ids must be contained in recommended_point_ids: {', '.join(missing_highlighted)}")
        if not self.has_typical_case and (candidate_pool or recommended or highlighted):
            raise ValueError("plans without a typical case must not contain candidate, recommended, or highlighted points")

        ranking_features = tuple(dict(item) for item in self.ranking_features)
        _validate_ranking_features(ranking_features, recommended)
        point_profiles = tuple(dict(item) for item in self.point_profiles)
        _validate_point_profiles(point_profiles, recommended)

        analysis_run_id = None
        if self.analysis_run_id is not None:
            analysis_run_id = clean_text(self.analysis_run_id, "analysis_run_id")

        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "plan_version", clean_text(self.plan_version, "plan_version"))
        object.__setattr__(self, "dataset_id", clean_text(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "rule_set_id", clean_text(self.rule_set_id, "rule_set_id"))
        object.__setattr__(self, "focus_category", focus_category)
        object.__setattr__(self, "recommendation_kind", clean_text(self.recommendation_kind, "recommendation_kind"))
        object.__setattr__(self, "candidate_pool_point_ids", candidate_pool)
        object.__setattr__(self, "recommended_point_ids", recommended)
        object.__setattr__(self, "highlighted_point_ids", highlighted)
        object.__setattr__(self, "target_rule_ids", target_rule_ids)
        object.__setattr__(self, "ranking_method", clean_text(self.ranking_method, "ranking_method"))
        object.__setattr__(self, "ranking_features", ranking_features)
        object.__setattr__(self, "point_profiles", point_profiles)
        object.__setattr__(self, "analysis_run_id", analysis_run_id)
        object.__setattr__(self, "evidence_rows", tuple(dict(item) for item in self.evidence_rows))
        object.__setattr__(self, "label_questions", tuple(dict(item) for item in self.label_questions))
        object.__setattr__(self, "label_options", _clean_unique_text_tuple(self.label_options, "label_options"))
        object.__setattr__(self, "expected_label_outcomes", tuple(dict(item) for item in self.expected_label_outcomes))
        object.__setattr__(self, "uncertainty_notes", tuple(clean_text(note, "uncertainty_note") for note in self.uncertainty_notes))
        object.__setattr__(self, "immutable_fields", _clean_unique_text_tuple(self.immutable_fields, "immutable_fields"))
        object.__setattr__(self, "llm_role", clean_text(self.llm_role, "llm_role"))
        object.__setattr__(self, "not_selected_summary", dict(self.not_selected_summary or {}))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "dataset_id": self.dataset_id,
            "analysis_run_id": self.analysis_run_id,
            "rule_set_id": self.rule_set_id,
            "focus_category": self.focus_category,
            "recommendation_kind": self.recommendation_kind,
            "has_typical_case": self.has_typical_case,
            "candidate_pool_point_ids": list(self.candidate_pool_point_ids),
            "recommended_point_ids": list(self.recommended_point_ids),
            "highlighted_point_ids": list(self.highlighted_point_ids),
            "candidate_pool_count": len(self.candidate_pool_point_ids),
            "recommended_point_count": len(self.recommended_point_ids),
            "highlighted_point_count": len(self.highlighted_point_ids),
            "ranking_method": self.ranking_method,
            "ranking_features": [dict(item) for item in self.ranking_features],
            "target_rule_ids": list(self.target_rule_ids),
            "evidence_rows": [dict(item) for item in self.evidence_rows],
            "point_profiles": [dict(item) for item in self.point_profiles],
            "label_questions": [dict(item) for item in self.label_questions],
            "label_options": list(self.label_options),
            "expected_label_outcomes": [dict(item) for item in self.expected_label_outcomes],
            "uncertainty_notes": list(self.uncertainty_notes),
            "immutable_fields": list(self.immutable_fields),
            "llm_role": self.llm_role,
            "not_selected_summary": dict(self.not_selected_summary),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecommendationPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("recommendation plan payload must be an object")
        return cls(
            plan_id=payload.get("plan_id"),
            plan_version=payload.get("plan_version"),
            dataset_id=payload.get("dataset_id"),
            analysis_run_id=payload.get("analysis_run_id"),
            rule_set_id=payload.get("rule_set_id"),
            focus_category=payload.get("focus_category"),
            recommendation_kind=payload.get("recommendation_kind"),
            has_typical_case=payload.get("has_typical_case"),
            candidate_pool_point_ids=tuple(payload.get("candidate_pool_point_ids", ())),
            recommended_point_ids=tuple(payload.get("recommended_point_ids", ())),
            highlighted_point_ids=tuple(payload.get("highlighted_point_ids", ())),
            target_rule_ids=tuple(payload.get("target_rule_ids", ())),
            ranking_method=payload.get("ranking_method"),
            ranking_features=tuple(payload.get("ranking_features", ())),
            evidence_rows=tuple(payload.get("evidence_rows", ())),
            point_profiles=tuple(payload.get("point_profiles", ())),
            label_questions=tuple(payload.get("label_questions", ())),
            label_options=tuple(payload.get("label_options", ())),
            expected_label_outcomes=tuple(payload.get("expected_label_outcomes", ())),
            uncertainty_notes=tuple(payload.get("uncertainty_notes", ())),
            immutable_fields=tuple(payload.get("immutable_fields", ())),
            llm_role=payload.get("llm_role", "translate_only_do_not_select_points"),
            not_selected_summary=payload.get("not_selected_summary", {}),
            diagnostics=payload.get("diagnostics", {}),
        )
