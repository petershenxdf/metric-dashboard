from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

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
    min_pts: int = 3
    review_percentile_threshold: float = 75.0

    def __post_init__(self) -> None:
        for field_name in (
            "n_clusters",
            "max_depth",
            "min_samples_leaf",
            "batch_size",
            "max_points",
            "min_pts",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if not 0 <= self.review_percentile_threshold <= 100:
            raise ValueError("review_percentile_threshold must be between 0 and 100")
        if self.label_budget is not None:
            if (
                isinstance(self.label_budget, bool)
                or not isinstance(self.label_budget, int)
                or self.label_budget < 1
            ):
                raise ValueError("label_budget must be a positive integer or null")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "batch_size": self.batch_size,
            "label_budget": self.label_budget,
            "max_points": self.max_points,
            "min_pts": self.min_pts,
            "review_percentile_threshold": self.review_percentile_threshold,
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
            min_pts=int(payload.get("min_pts", 3)),
            review_percentile_threshold=float(payload.get("review_percentile_threshold", 75)),
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
    review: Mapping[str, Any]
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
            "review",
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
            "review": dict(self.review),
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
