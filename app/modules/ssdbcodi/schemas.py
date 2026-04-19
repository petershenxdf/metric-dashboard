from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from app.modules.algorithm_adapters.schemas import ClusterResult, OutlierResult
from app.shared.schemas import clean_number, clean_text


@dataclass(frozen=True)
class PointScores:
    point_id: str
    cluster_id: str
    is_outlier: bool
    r_score: float
    l_score: float
    sim_score: float
    t_score: float
    c_dist: float
    e_max: float
    seed_origin_point_id: str | None = None
    is_reliable_normal: bool = False
    is_uncertain: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "cluster_id", clean_text(self.cluster_id, "cluster_id"))
        if not isinstance(self.is_outlier, bool):
            raise ValueError("is_outlier must be a boolean")
        if not isinstance(self.is_reliable_normal, bool):
            raise ValueError("is_reliable_normal must be a boolean")
        if not isinstance(self.is_uncertain, bool):
            raise ValueError("is_uncertain must be a boolean")
        for name in ("r_score", "l_score", "sim_score", "t_score", "c_dist", "e_max"):
            object.__setattr__(self, name, clean_number(getattr(self, name), name))
        if self.seed_origin_point_id is not None:
            object.__setattr__(
                self,
                "seed_origin_point_id",
                clean_text(self.seed_origin_point_id, "seed_origin_point_id"),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "cluster_id": self.cluster_id,
            "is_outlier": self.is_outlier,
            "r_score": self.r_score,
            "l_score": self.l_score,
            "sim_score": self.sim_score,
            "t_score": self.t_score,
            "c_dist": self.c_dist,
            "e_max": self.e_max,
            "seed_origin_point_id": self.seed_origin_point_id,
            "is_reliable_normal": self.is_reliable_normal,
            "is_uncertain": self.is_uncertain,
        }


@dataclass(frozen=True)
class SeedRecord:
    point_id: str
    cluster_id: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", clean_text(self.point_id, "point_id"))
        object.__setattr__(self, "cluster_id", clean_text(self.cluster_id, "cluster_id"))
        object.__setattr__(self, "source", clean_text(self.source, "source"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "cluster_id": self.cluster_id,
            "source": self.source,
        }


@dataclass(frozen=True)
class SsdbcodiResult:
    run_id: str
    algorithm: str
    cluster_result: ClusterResult
    outlier_result: OutlierResult
    point_scores: Tuple[PointScores, ...]
    seeds: Tuple[SeedRecord, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", clean_text(self.run_id, "run_id"))
        object.__setattr__(self, "algorithm", clean_text(self.algorithm, "algorithm"))

        if not isinstance(self.cluster_result, ClusterResult):
            raise ValueError("cluster_result must be a ClusterResult")
        if not isinstance(self.outlier_result, OutlierResult):
            raise ValueError("outlier_result must be an OutlierResult")

        point_scores = tuple(self.point_scores)
        if not all(isinstance(score, PointScores) for score in point_scores):
            raise ValueError("point_scores must contain PointScores objects")
        point_ids = [score.point_id for score in point_scores]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("point_scores must have unique point_id values")

        seeds = tuple(self.seeds)
        if not all(isinstance(seed, SeedRecord) for seed in seeds):
            raise ValueError("seeds must contain SeedRecord objects")

        for field_name in ("parameters", "diagnostics"):
            value = getattr(self, field_name)
            if value is None or not isinstance(value, Mapping):
                raise ValueError(f"{field_name} must be a mapping")
            object.__setattr__(self, field_name, dict(value))

        object.__setattr__(self, "point_scores", point_scores)
        object.__setattr__(self, "seeds", seeds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "cluster_result": self.cluster_result.to_dict(),
            "outlier_result": self.outlier_result.to_dict(),
            "point_scores": [score.to_dict() for score in self.point_scores],
            "seeds": [seed.to_dict() for seed in self.seeds],
            "parameters": dict(self.parameters),
            "diagnostics": dict(self.diagnostics),
        }
