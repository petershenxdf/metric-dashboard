from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.modules.rule_panel.schemas import RECOMMENDATION_CATEGORIES
from app.shared.schemas import clean_text


FEATURE_KINDS = ("numeric", "categorical")
ROUND_STATUSES = (
    "computing",
    "ready_for_labeling",
    "labels_committed",
    "failed",
    "stopped",
)
SESSION_STATUSES = ("active", "computing", "stopped", "failed")
LABEL_DIMENSIONS = ("semantic_class", "outlier_status", "uncertain")
LABEL_EVENT_STATUSES = ("active", "superseded", "retracted")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    missing_count: int = 0
    categories: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = clean_text(self.name, "feature name")
        kind = clean_text(self.kind, "feature kind")
        if kind not in FEATURE_KINDS:
            raise ValueError(f"feature kind must be one of: {', '.join(FEATURE_KINDS)}")
        if isinstance(self.missing_count, bool) or not isinstance(self.missing_count, int):
            raise ValueError("missing_count must be an integer")
        if self.missing_count < 0:
            raise ValueError("missing_count must be non-negative")
        categories = tuple(str(value) for value in self.categories)
        if kind == "numeric" and categories:
            raise ValueError("numeric features must not define categories")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "categories", categories)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "missing_count": self.missing_count,
            "categories": list(self.categories),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSpec":
        return cls(
            name=payload.get("name"),
            kind=payload.get("kind"),
            missing_count=int(payload.get("missing_count", 0)),
            categories=tuple(payload.get("categories", ())),
        )


@dataclass(frozen=True)
class DatasetVersion:
    dataset_version_id: str
    dataset_id: str
    fingerprint: str
    content_fingerprint: str
    entity_name: str
    source_format: str
    point_ids: Tuple[str, ...]
    feature_specs: Tuple[FeatureSpec, ...]
    metadata_columns: Tuple[str, ...]
    ground_truth_columns: Tuple[str, ...]
    model_feature_names: Tuple[str, ...]
    transformation_map: Tuple[Mapping[str, Any], ...]
    preprocessing_version: str
    preprocessing_config: Mapping[str, Any]
    created_at: str
    raw_artifact_path: str = ""
    matrix_artifact_path: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_version_id",
            "dataset_id",
            "fingerprint",
            "content_fingerprint",
            "entity_name",
            "source_format",
            "preprocessing_version",
            "created_at",
        ):
            object.__setattr__(self, field_name, clean_text(getattr(self, field_name), field_name))
        point_ids = tuple(clean_text(value, "point_id") for value in self.point_ids)
        if not point_ids or len(set(point_ids)) != len(point_ids):
            raise ValueError("point_ids must be non-empty and unique")
        feature_specs = tuple(self.feature_specs)
        if not feature_specs or not all(isinstance(item, FeatureSpec) for item in feature_specs):
            raise ValueError("feature_specs must contain FeatureSpec objects")
        model_names = tuple(clean_text(value, "model feature name") for value in self.model_feature_names)
        if not model_names or len(set(model_names)) != len(model_names):
            raise ValueError("model_feature_names must be non-empty and unique")
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "feature_specs", feature_specs)
        object.__setattr__(
            self,
            "metadata_columns",
            tuple(clean_text(value, "metadata column") for value in self.metadata_columns),
        )
        object.__setattr__(
            self,
            "ground_truth_columns",
            tuple(clean_text(value, "ground truth column") for value in self.ground_truth_columns),
        )
        object.__setattr__(self, "model_feature_names", model_names)
        object.__setattr__(
            self,
            "transformation_map",
            tuple(dict(item) for item in self.transformation_map),
        )
        object.__setattr__(
            self,
            "preprocessing_config",
            dict(self.preprocessing_config),
        )

    @property
    def point_count(self) -> int:
        return len(self.point_ids)

    @property
    def feature_count(self) -> int:
        return len(self.feature_specs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_version_id": self.dataset_version_id,
            "dataset_id": self.dataset_id,
            "fingerprint": self.fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "entity_name": self.entity_name,
            "source_format": self.source_format,
            "point_ids": list(self.point_ids),
            "point_count": self.point_count,
            "feature_specs": [item.to_dict() for item in self.feature_specs],
            "feature_count": self.feature_count,
            "metadata_columns": list(self.metadata_columns),
            "ground_truth_columns": list(self.ground_truth_columns),
            "model_feature_names": list(self.model_feature_names),
            "transformation_map": [dict(item) for item in self.transformation_map],
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_config": dict(self.preprocessing_config),
            "created_at": self.created_at,
            "raw_artifact_path": self.raw_artifact_path,
            "matrix_artifact_path": self.matrix_artifact_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetVersion":
        return cls(
            dataset_version_id=payload.get("dataset_version_id"),
            dataset_id=payload.get("dataset_id"),
            fingerprint=payload.get("fingerprint"),
            content_fingerprint=payload.get(
                "content_fingerprint",
                payload.get("fingerprint"),
            ),
            entity_name=payload.get("entity_name", "record"),
            source_format=payload.get("source_format", "json"),
            point_ids=tuple(payload.get("point_ids", ())),
            feature_specs=tuple(
                FeatureSpec.from_dict(item) for item in payload.get("feature_specs", ())
            ),
            metadata_columns=tuple(payload.get("metadata_columns", ())),
            ground_truth_columns=tuple(payload.get("ground_truth_columns", ())),
            model_feature_names=tuple(payload.get("model_feature_names", ())),
            transformation_map=tuple(payload.get("transformation_map", ())),
            preprocessing_version=payload.get("preprocessing_version", "mixed_tabular_v1"),
            preprocessing_config=payload.get("preprocessing_config", {}),
            created_at=payload.get("created_at"),
            raw_artifact_path=str(payload.get("raw_artifact_path", "")),
            matrix_artifact_path=str(payload.get("matrix_artifact_path", "")),
        )


@dataclass(frozen=True)
class SessionConfig:
    n_clusters: int = 3
    max_depth: int = 3
    min_samples_leaf: int = 1
    batch_size: int = 4
    label_budget: int | None = None
    max_points: int = 2000

    def __post_init__(self) -> None:
        for field_name in (
            "n_clusters",
            "max_depth",
            "min_samples_leaf",
            "batch_size",
            "max_points",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.label_budget is not None:
            if (
                isinstance(self.label_budget, bool)
                or not isinstance(self.label_budget, int)
                or self.label_budget < 1
            ):
                raise ValueError("label_budget must be a positive integer or null")

    @property
    def candidate_pool_size(self) -> int:
        return max(12, self.batch_size * 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "batch_size": self.batch_size,
            "candidate_pool_size": self.candidate_pool_size,
            "label_budget": self.label_budget,
            "max_points": self.max_points,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SessionConfig":
        payload = dict(payload or {})
        return cls(
            n_clusters=int(payload.get("n_clusters", 3)),
            max_depth=int(payload.get("max_depth", 3)),
            min_samples_leaf=int(payload.get("min_samples_leaf", 1)),
            batch_size=int(payload.get("batch_size", 4)),
            label_budget=(
                None
                if payload.get("label_budget") in (None, "")
                else int(payload.get("label_budget"))
            ),
            max_points=int(payload.get("max_points", 2000)),
        )


@dataclass(frozen=True)
class ActiveLearningSession:
    session_id: str
    dataset_version_id: str
    status: str
    config: SessionConfig
    label_vocabulary: Mapping[str, str]
    current_round_id: str | None
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", clean_text(self.session_id, "session_id"))
        object.__setattr__(
            self,
            "dataset_version_id",
            clean_text(self.dataset_version_id, "dataset_version_id"),
        )
        status = clean_text(self.status, "session status")
        if status not in SESSION_STATUSES:
            raise ValueError(f"session status must be one of: {', '.join(SESSION_STATUSES)}")
        if not isinstance(self.config, SessionConfig):
            raise ValueError("config must be a SessionConfig")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "label_vocabulary", dict(self.label_vocabulary))
        object.__setattr__(self, "created_at", clean_text(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", clean_text(self.updated_at, "updated_at"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "dataset_version_id": self.dataset_version_id,
            "status": self.status,
            "config": self.config.to_dict(),
            "label_vocabulary": dict(self.label_vocabulary),
            "current_round_id": self.current_round_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ActiveLearningRound:
    round_id: str
    session_id: str
    round_index: int
    parent_round_id: str | None
    label_revision: int
    status: str
    analysis: Mapping[str, Any]
    rule_set: Mapping[str, Any]
    display_rule_set: Mapping[str, Any]
    projection: Mapping[str, Any]
    recommendation_plans: Mapping[str, Any]
    delta: Mapping[str, Any]
    cluster_lineage: Mapping[str, str]
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", clean_text(self.round_id, "round_id"))
        object.__setattr__(self, "session_id", clean_text(self.session_id, "session_id"))
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int):
            raise ValueError("round_index must be an integer")
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if isinstance(self.label_revision, bool) or not isinstance(self.label_revision, int):
            raise ValueError("label_revision must be an integer")
        if self.label_revision < 0:
            raise ValueError("label_revision must be non-negative")
        status = clean_text(self.status, "round status")
        if status not in ROUND_STATUSES:
            raise ValueError(f"round status must be one of: {', '.join(ROUND_STATUSES)}")
        object.__setattr__(self, "status", status)
        for field_name in (
            "analysis",
            "rule_set",
            "display_rule_set",
            "projection",
            "recommendation_plans",
            "delta",
            "cluster_lineage",
        ):
            object.__setattr__(self, field_name, dict(getattr(self, field_name)))
        object.__setattr__(self, "created_at", clean_text(self.created_at, "created_at"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "session_id": self.session_id,
            "round_index": self.round_index,
            "parent_round_id": self.parent_round_id,
            "label_revision": self.label_revision,
            "status": self.status,
            "analysis": dict(self.analysis),
            "rule_set": dict(self.rule_set),
            "display_rule_set": dict(self.display_rule_set),
            "projection": dict(self.projection),
            "recommendation_plans": {
                key: dict(value) for key, value in self.recommendation_plans.items()
            },
            "delta": dict(self.delta),
            "cluster_lineage": dict(self.cluster_lineage),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class LabelEvent:
    event_id: str
    session_id: str
    round_id: str
    point_id: str
    label_dimension: str
    label_value: Any
    status: str
    supersedes_event_id: str | None
    provenance: Mapping[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        for field_name in ("event_id", "session_id", "round_id", "point_id"):
            object.__setattr__(
                self,
                field_name,
                clean_text(getattr(self, field_name), field_name),
            )
        dimension = clean_text(self.label_dimension, "label dimension")
        if dimension not in LABEL_DIMENSIONS:
            raise ValueError(
                f"label dimension must be one of: {', '.join(LABEL_DIMENSIONS)}"
            )
        status = clean_text(self.status, "label event status")
        if status not in LABEL_EVENT_STATUSES:
            raise ValueError(
                f"label event status must be one of: {', '.join(LABEL_EVENT_STATUSES)}"
            )
        object.__setattr__(self, "label_dimension", dimension)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "created_at", clean_text(self.created_at, "created_at"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "round_id": self.round_id,
            "point_id": self.point_id,
            "label_dimension": self.label_dimension,
            "label_value": self.label_value,
            "status": self.status,
            "supersedes_event_id": self.supersedes_event_id,
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
        }


EVIDENCE_STATUSES = ("yes", "partly", "no", "insufficient")


@dataclass(frozen=True)
class EvidenceBullet:
    dimension_id: str
    question: str
    status: str
    headline: str
    plain_fact: str
    why_it_matters: str
    point_connection: str
    labeling_value: str
    evidence_fact_ids: Tuple[str, ...]
    technical_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "dimension_id",
            "question",
            "headline",
            "plain_fact",
            "why_it_matters",
            "point_connection",
            "labeling_value",
        ):
            object.__setattr__(
                self,
                field_name,
                clean_text(getattr(self, field_name), field_name),
            )
        status = clean_text(self.status, "evidence status")
        if status not in EVIDENCE_STATUSES:
            raise ValueError(
                f"evidence status must be one of: {', '.join(EVIDENCE_STATUSES)}"
            )
        fact_ids = _unique_text_tuple(
            self.evidence_fact_ids,
            "evidence_fact_ids",
        )
        if not fact_ids:
            raise ValueError("evidence_fact_ids must not be empty")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_fact_ids", fact_ids)
        object.__setattr__(
            self,
            "technical_details",
            dict(self.technical_details or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_id": self.dimension_id,
            "question": self.question,
            "status": self.status,
            "headline": self.headline,
            "plain_fact": self.plain_fact,
            "why_it_matters": self.why_it_matters,
            "point_connection": self.point_connection,
            "labeling_value": self.labeling_value,
            "evidence_fact_ids": list(self.evidence_fact_ids),
            "technical_details": dict(self.technical_details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceBullet":
        return cls(
            dimension_id=payload.get("dimension_id"),
            question=payload.get("question"),
            status=payload.get("status"),
            headline=payload.get("headline"),
            plain_fact=payload.get("plain_fact"),
            why_it_matters=payload.get(
                "why_it_matters",
                payload.get("labeling_value"),
            ),
            point_connection=payload.get(
                "point_connection",
                payload.get("plain_fact"),
            ),
            labeling_value=payload.get(
                "labeling_value",
                payload.get("why_it_matters"),
            ),
            evidence_fact_ids=tuple(payload.get("evidence_fact_ids", ())),
            technical_details=payload.get("technical_details", {}),
        )


@dataclass(frozen=True)
class CategoryEvidenceCard:
    point_id: str
    category: str
    evidence_category: str
    category_explanation: str
    evidence_policy_version: str
    evidence_bullets: Tuple[EvidenceBullet, ...]
    comparison_targets: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    round_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        category = clean_text(self.category, "category")
        evidence_category = clean_text(
            self.evidence_category,
            "evidence_category",
        )
        if category not in RECOMMENDATION_CATEGORIES:
            raise ValueError("unknown evidence card category")
        if evidence_category not in RECOMMENDATION_CATEGORIES:
            raise ValueError("unknown evidence card evidence_category")
        bullets = tuple(self.evidence_bullets)
        if not all(isinstance(item, EvidenceBullet) for item in bullets):
            raise ValueError("evidence_bullets must contain EvidenceBullet objects")
        dimension_ids = tuple(item.dimension_id for item in bullets)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("evidence_bullets must contain unique dimensions")
        target_ids = tuple(
            clean_text(item.get("point_id"), "comparison target point_id")
            for item in self.comparison_targets
        )
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("comparison_targets must contain unique point IDs")
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "evidence_category", evidence_category)
        object.__setattr__(
            self,
            "category_explanation",
            clean_text(self.category_explanation, "category_explanation"),
        )
        object.__setattr__(
            self,
            "evidence_policy_version",
            clean_text(self.evidence_policy_version, "evidence_policy_version"),
        )
        object.__setattr__(self, "evidence_bullets", bullets)
        object.__setattr__(
            self,
            "comparison_targets",
            tuple(dict(item) for item in self.comparison_targets),
        )
        object.__setattr__(self, "round_context", dict(self.round_context or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "category": self.category,
            "evidence_category": self.evidence_category,
            "category_explanation": self.category_explanation,
            "evidence_policy_version": self.evidence_policy_version,
            "evidence_bullets": [
                item.to_dict() for item in self.evidence_bullets
            ],
            "comparison_targets": [
                dict(item) for item in self.comparison_targets
            ],
            "round_context": dict(self.round_context),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CategoryEvidenceCard":
        return cls(
            point_id=payload.get("point_id"),
            category=payload.get("category"),
            evidence_category=payload.get("evidence_category"),
            category_explanation=payload.get("category_explanation"),
            evidence_policy_version=payload.get("evidence_policy_version"),
            evidence_bullets=tuple(
                EvidenceBullet.from_dict(item)
                for item in payload.get("evidence_bullets", ())
            ),
            comparison_targets=tuple(payload.get("comparison_targets", ())),
            round_context=payload.get("round_context", {}),
        )


@dataclass(frozen=True)
class RecommendationPlanV2:
    plan_id: str
    plan_version: str
    session_id: str
    round_id: str
    dataset_version_id: str
    preprocessing_version: str
    label_revision: int
    focus_category: str
    has_typical_case: bool
    candidate_pool_point_ids: Tuple[str, ...]
    recommended_point_ids: Tuple[str, ...]
    highlighted_point_ids: Tuple[str, ...]
    target_rule_ids: Tuple[str, ...]
    candidate_rankings: Tuple[Mapping[str, Any], ...]
    candidate_point_profiles: Tuple[Mapping[str, Any], ...]
    point_profiles: Tuple[Mapping[str, Any], ...]
    excluded_points: Tuple[Mapping[str, Any], ...]
    deferred_points: Tuple[Mapping[str, Any], ...]
    previous_plan_id: str | None
    previous_plan_diff: Mapping[str, Any]
    history_context: Mapping[str, Any]
    label_options: Tuple[str, ...]
    expected_label_outcomes: Tuple[Mapping[str, Any], ...]
    stop_reason: str = ""
    source_plan: Mapping[str, Any] = field(default_factory=dict)
    category_explanation: str = ""
    evidence_policy_version: str = ""
    category_evidence_cards: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        for field_name in (
            "plan_id",
            "plan_version",
            "session_id",
            "round_id",
            "dataset_version_id",
            "preprocessing_version",
            "focus_category",
        ):
            object.__setattr__(
                self,
                field_name,
                clean_text(getattr(self, field_name), field_name),
            )
        if isinstance(self.label_revision, bool) or not isinstance(self.label_revision, int):
            raise ValueError("label_revision must be an integer")
        if self.label_revision < 0:
            raise ValueError("label_revision must be non-negative")
        if not isinstance(self.has_typical_case, bool):
            raise ValueError("has_typical_case must be a boolean")
        candidate_ids = _unique_text_tuple(
            self.candidate_pool_point_ids,
            "candidate_pool_point_ids",
        )
        recommended_ids = _unique_text_tuple(
            self.recommended_point_ids,
            "recommended_point_ids",
        )
        highlighted_ids = _unique_text_tuple(
            self.highlighted_point_ids,
            "highlighted_point_ids",
        )
        if not set(recommended_ids).issubset(candidate_ids):
            raise ValueError("recommended_point_ids must be contained in the candidate pool")
        if highlighted_ids != recommended_ids:
            raise ValueError("highlighted_point_ids must exactly match recommended_point_ids")
        if self.has_typical_case != bool(recommended_ids):
            raise ValueError("has_typical_case must match whether recommendations exist")
        rankings = tuple(dict(item) for item in self.candidate_rankings)
        ranking_ids = _mapping_point_ids(rankings, "candidate_rankings")
        if set(ranking_ids) != set(candidate_ids):
            raise ValueError("candidate_rankings must cover every candidate exactly once")
        candidate_profiles = tuple(dict(item) for item in self.candidate_point_profiles)
        if _mapping_point_ids(
            candidate_profiles,
            "candidate_point_profiles",
        ) != candidate_ids:
            raise ValueError(
                "candidate_point_profiles must follow candidate_pool_point_ids order"
            )
        point_profiles = tuple(dict(item) for item in self.point_profiles)
        if _mapping_point_ids(point_profiles, "point_profiles") != recommended_ids:
            raise ValueError("point_profiles must follow recommended_point_ids order")
        deferred = tuple(dict(item) for item in self.deferred_points)
        deferred_ids = _mapping_point_ids(deferred, "deferred_points")
        expected_deferred = tuple(
            point_id for point_id in candidate_ids if point_id not in set(recommended_ids)
        )
        if set(deferred_ids) != set(expected_deferred):
            raise ValueError("deferred_points must cover non-recommended candidates")
        object.__setattr__(self, "candidate_pool_point_ids", candidate_ids)
        object.__setattr__(self, "recommended_point_ids", recommended_ids)
        object.__setattr__(self, "highlighted_point_ids", highlighted_ids)
        object.__setattr__(
            self,
            "target_rule_ids",
            _unique_text_tuple(self.target_rule_ids, "target_rule_ids"),
        )
        object.__setattr__(self, "candidate_rankings", rankings)
        object.__setattr__(self, "candidate_point_profiles", candidate_profiles)
        object.__setattr__(self, "point_profiles", point_profiles)
        object.__setattr__(
            self,
            "excluded_points",
            tuple(dict(item) for item in self.excluded_points),
        )
        object.__setattr__(self, "deferred_points", deferred)
        object.__setattr__(self, "previous_plan_diff", dict(self.previous_plan_diff))
        object.__setattr__(self, "history_context", dict(self.history_context))
        object.__setattr__(
            self,
            "label_options",
            tuple(clean_text(value, "label option") for value in self.label_options),
        )
        object.__setattr__(
            self,
            "expected_label_outcomes",
            tuple(dict(item) for item in self.expected_label_outcomes),
        )
        object.__setattr__(self, "source_plan", dict(self.source_plan))
        cards = tuple(dict(item) for item in self.category_evidence_cards)
        card_ids = _mapping_point_ids(cards, "category_evidence_cards")
        if cards and card_ids != recommended_ids:
            raise ValueError(
                "category_evidence_cards must follow recommended_point_ids order"
            )
        for card in cards:
            parsed = CategoryEvidenceCard.from_dict(card)
            if parsed.category != self.focus_category:
                raise ValueError(
                    "category_evidence_cards must match focus_category"
                )
        object.__setattr__(
            self,
            "category_explanation",
            str(self.category_explanation or "").strip(),
        )
        object.__setattr__(
            self,
            "evidence_policy_version",
            str(self.evidence_policy_version or "").strip(),
        )
        object.__setattr__(self, "category_evidence_cards", cards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "session_id": self.session_id,
            "round_id": self.round_id,
            "dataset_version_id": self.dataset_version_id,
            "preprocessing_version": self.preprocessing_version,
            "label_revision": self.label_revision,
            "focus_category": self.focus_category,
            "has_typical_case": self.has_typical_case,
            "candidate_pool_point_ids": list(self.candidate_pool_point_ids),
            "recommended_point_ids": list(self.recommended_point_ids),
            "highlighted_point_ids": list(self.highlighted_point_ids),
            "candidate_pool_count": len(self.candidate_pool_point_ids),
            "recommended_point_count": len(self.recommended_point_ids),
            "highlighted_point_count": len(self.highlighted_point_ids),
            "target_rule_ids": list(self.target_rule_ids),
            "candidate_rankings": [dict(item) for item in self.candidate_rankings],
            "ranking_features": [
                dict(item)
                for item in self.candidate_rankings
                if item.get("point_id") in set(self.recommended_point_ids)
            ],
            "candidate_point_profiles": [
                dict(item) for item in self.candidate_point_profiles
            ],
            "point_profiles": [dict(item) for item in self.point_profiles],
            "excluded_points": [dict(item) for item in self.excluded_points],
            "deferred_points": [dict(item) for item in self.deferred_points],
            "previous_plan_id": self.previous_plan_id,
            "previous_plan_diff": dict(self.previous_plan_diff),
            "history_context": dict(self.history_context),
            "label_options": list(self.label_options),
            "expected_label_outcomes": [dict(item) for item in self.expected_label_outcomes],
            "stop_reason": self.stop_reason,
            "source_plan": dict(self.source_plan),
            "category_explanation": self.category_explanation,
            "evidence_policy_version": self.evidence_policy_version,
            "category_evidence_cards": [
                dict(item) for item in self.category_evidence_cards
            ],
            "llm_role": "translate_only_do_not_select_points",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecommendationPlanV2":
        return cls(
            plan_id=payload.get("plan_id"),
            plan_version=payload.get("plan_version"),
            session_id=payload.get("session_id"),
            round_id=payload.get("round_id"),
            dataset_version_id=payload.get("dataset_version_id"),
            preprocessing_version=payload.get("preprocessing_version"),
            label_revision=int(payload.get("label_revision", 0)),
            focus_category=payload.get("focus_category"),
            has_typical_case=bool(payload.get("has_typical_case")),
            candidate_pool_point_ids=tuple(
                payload.get("candidate_pool_point_ids", ())
            ),
            recommended_point_ids=tuple(payload.get("recommended_point_ids", ())),
            highlighted_point_ids=tuple(payload.get("highlighted_point_ids", ())),
            target_rule_ids=tuple(payload.get("target_rule_ids", ())),
            candidate_rankings=tuple(payload.get("candidate_rankings", ())),
            candidate_point_profiles=tuple(
                payload.get("candidate_point_profiles", ())
            ),
            point_profiles=tuple(payload.get("point_profiles", ())),
            excluded_points=tuple(payload.get("excluded_points", ())),
            deferred_points=tuple(payload.get("deferred_points", ())),
            previous_plan_id=payload.get("previous_plan_id"),
            previous_plan_diff=payload.get("previous_plan_diff", {}),
            history_context=payload.get("history_context", {}),
            label_options=tuple(payload.get("label_options", ())),
            expected_label_outcomes=tuple(
                payload.get("expected_label_outcomes", ())
            ),
            stop_reason=str(payload.get("stop_reason", "")),
            source_plan=payload.get("source_plan", {}),
            category_explanation=str(payload.get("category_explanation", "")),
            evidence_policy_version=str(
                payload.get("evidence_policy_version", "")
            ),
            category_evidence_cards=tuple(
                payload.get("category_evidence_cards", ())
            ),
        )


@dataclass(frozen=True)
class TranslationPacket:
    task: str
    plan: Mapping[str, Any]
    target_rules: Tuple[Mapping[str, Any], ...]
    point_profiles: Tuple[Mapping[str, Any], ...]
    category_evidence_cards: Tuple[Mapping[str, Any], ...]
    label_options: Tuple[str, ...]
    round_delta: Mapping[str, Any]
    previous_label_events: Tuple[Mapping[str, Any], ...]
    entity_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", clean_text(self.task, "translation task"))
        object.__setattr__(self, "plan", dict(self.plan))
        object.__setattr__(
            self,
            "target_rules",
            tuple(dict(item) for item in self.target_rules),
        )
        object.__setattr__(
            self,
            "point_profiles",
            tuple(dict(item) for item in self.point_profiles),
        )
        cards = tuple(dict(item) for item in self.category_evidence_cards)
        expected_ids = tuple(self.plan.get("recommended_point_ids", ()))
        if _mapping_point_ids(cards, "category_evidence_cards") != expected_ids:
            raise ValueError(
                "translation category_evidence_cards must cover recommended points in order"
            )
        for card in cards:
            CategoryEvidenceCard.from_dict(card)
        object.__setattr__(self, "category_evidence_cards", cards)
        object.__setattr__(
            self,
            "label_options",
            tuple(clean_text(value, "label option") for value in self.label_options),
        )
        object.__setattr__(self, "round_delta", dict(self.round_delta))
        object.__setattr__(
            self,
            "previous_label_events",
            tuple(dict(item) for item in self.previous_label_events),
        )
        object.__setattr__(self, "entity_name", clean_text(self.entity_name, "entity_name"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "plan": dict(self.plan),
            "target_rules": [dict(item) for item in self.target_rules],
            "point_profiles": [dict(item) for item in self.point_profiles],
            "category_evidence_cards": [
                dict(item) for item in self.category_evidence_cards
            ],
            "label_options": list(self.label_options),
            "round_delta": dict(self.round_delta),
            "previous_label_events": [
                dict(item) for item in self.previous_label_events
            ],
            "entity_name": self.entity_name,
        }


@dataclass(frozen=True)
class PointGuidance:
    point_id: str
    why_selected: str
    what_changed_since_last_round: str
    how_to_label: str
    possible_outcomes: Tuple[str, ...]
    evidence_bullets: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    comparison_targets: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in (
            "point_id",
            "why_selected",
            "what_changed_since_last_round",
            "how_to_label",
        ):
            object.__setattr__(
                self,
                field_name,
                clean_text(getattr(self, field_name), field_name),
            )
        outcomes = tuple(
            clean_text(value, "possible outcome") for value in self.possible_outcomes
        )
        if not outcomes:
            raise ValueError("possible_outcomes must not be empty")
        object.__setattr__(self, "possible_outcomes", outcomes)
        object.__setattr__(
            self,
            "evidence_bullets",
            tuple(dict(item) for item in self.evidence_bullets),
        )
        object.__setattr__(
            self,
            "comparison_targets",
            tuple(dict(item) for item in self.comparison_targets),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "why_selected": self.why_selected,
            "what_changed_since_last_round": self.what_changed_since_last_round,
            "how_to_label": self.how_to_label,
            "possible_outcomes": list(self.possible_outcomes),
            "evidence_bullets": [
                dict(item) for item in self.evidence_bullets
            ],
            "comparison_targets": [
                dict(item) for item in self.comparison_targets
            ],
        }


def _unique_text_tuple(values: Tuple[str, ...], field_name: str) -> Tuple[str, ...]:
    cleaned = tuple(clean_text(value, field_name) for value in values)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} must contain unique values")
    return cleaned


def _mapping_point_ids(
    values: Tuple[Mapping[str, Any], ...],
    field_name: str,
) -> Tuple[str, ...]:
    point_ids = tuple(
        clean_text(item.get("point_id"), f"{field_name} point_id")
        for item in values
    )
    if len(set(point_ids)) != len(point_ids):
        raise ValueError(f"{field_name} must contain each point exactly once")
    return point_ids
