from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.shared.schemas import clean_number, clean_text


@dataclass(frozen=True)
class ClusterAssignment:
    point_id: str
    cluster_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "cluster_id", clean_text(self.cluster_id, "cluster_id"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "cluster_id": self.cluster_id,
        }


@dataclass(frozen=True)
class ClusterResult:
    cluster_run_id: str
    algorithm: str
    n_clusters: int
    assignments: Tuple[ClusterAssignment, ...]
    excluded_outlier_point_ids: Tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cluster_run_id = clean_text(self.cluster_run_id, "cluster_run_id")
        algorithm = clean_text(self.algorithm, "algorithm")
        n_clusters = _clean_positive_int(self.n_clusters, "n_clusters")
        assignments = tuple(self.assignments)
        excluded = tuple(clean_text(point_id, "point_id") for point_id in self.excluded_outlier_point_ids)

        if not all(isinstance(assignment, ClusterAssignment) for assignment in assignments):
            raise ValueError("assignments must contain ClusterAssignment objects")

        assigned_point_ids = [assignment.point_id for assignment in assignments]
        if len(set(assigned_point_ids)) != len(assigned_point_ids):
            raise ValueError("cluster assignments must have unique point_id values")

        if len(set(excluded)) != len(excluded):
            raise ValueError("excluded_outlier_point_ids must be unique")

        if self.diagnostics is None or not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

        object.__setattr__(self, "cluster_run_id", cluster_run_id)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "n_clusters", n_clusters)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "excluded_outlier_point_ids", excluded)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_run_id": self.cluster_run_id,
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "excluded_outlier_point_ids": list(self.excluded_outlier_point_ids),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class OutlierScore:
    point_id: str
    score: float
    is_outlier: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "score", clean_number(self.score, "score"))

        if not isinstance(self.is_outlier, bool):
            raise ValueError("is_outlier must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "score": self.score,
            "is_outlier": self.is_outlier,
        }


@dataclass(frozen=True)
class OutlierResult:
    outlier_run_id: str
    algorithm: str
    scores: Tuple[OutlierScore, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        outlier_run_id = clean_text(self.outlier_run_id, "outlier_run_id")
        algorithm = clean_text(self.algorithm, "algorithm")
        scores = tuple(self.scores)

        if not scores:
            raise ValueError("scores must not be empty")

        if not all(isinstance(score, OutlierScore) for score in scores):
            raise ValueError("scores must contain OutlierScore objects")

        point_ids = [score.point_id for score in scores]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("outlier scores must have unique point_id values")

        if self.diagnostics is None or not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

        object.__setattr__(self, "outlier_run_id", outlier_run_id)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def outlier_point_ids(self) -> Tuple[str, ...]:
        return tuple(score.point_id for score in self.scores if score.is_outlier)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outlier_run_id": self.outlier_run_id,
            "algorithm": self.algorithm,
            "scores": [score.to_dict() for score in self.scores],
            "outlier_point_ids": list(self.outlier_point_ids),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class AnalysisResult:
    analysis_run_id: str
    outlier_result: OutlierResult
    cluster_result: ClusterResult
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_run_id", clean_text(self.analysis_run_id, "analysis_run_id"))

        if not isinstance(self.outlier_result, OutlierResult):
            raise ValueError("outlier_result must be an OutlierResult")

        if not isinstance(self.cluster_result, ClusterResult):
            raise ValueError("cluster_result must be a ClusterResult")

        if self.diagnostics is None or not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_run_id": self.analysis_run_id,
            "outlier_result": self.outlier_result.to_dict(),
            "cluster_result": self.cluster_result.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _clean_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")

    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")

    return value
