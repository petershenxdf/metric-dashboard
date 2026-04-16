from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Mapping, Optional

from app.shared.schemas import clean_text

from .schemas import SsdbcodiResult

DEFAULT_HISTORY_LIMIT = 20


@dataclass
class SsdbcodiStore:
    dataset_id: str
    history_limit: int = DEFAULT_HISTORY_LIMIT
    latest_result: Optional[SsdbcodiResult] = None
    history: Deque[SsdbcodiResult] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.dataset_id = clean_text(self.dataset_id, "dataset_id")
        if isinstance(self.history_limit, bool) or not isinstance(self.history_limit, int):
            raise ValueError("history_limit must be an integer")
        if self.history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        self.history = deque(self.history, maxlen=self.history_limit)
        if self.latest_result is not None and not isinstance(self.latest_result, SsdbcodiResult):
            raise ValueError("latest_result must be a SsdbcodiResult")

    def record_result(self, result: SsdbcodiResult) -> SsdbcodiResult:
        if not isinstance(result, SsdbcodiResult):
            raise ValueError("result must be a SsdbcodiResult")
        self.latest_result = result
        self.history.append(result)
        return result

    def reset(self) -> None:
        self.latest_result = None
        self.history.clear()

    def history_summary(self) -> List[Mapping[str, object]]:
        return [
            {
                "run_id": result.run_id,
                "bootstrap_used": bool(result.parameters.get("bootstrap_used")),
                "manual_seed_count": result.diagnostics.get("manual_seed_count", 0),
                "outlier_count": len(result.outlier_result.outlier_point_ids),
                "cluster_count": result.cluster_result.n_clusters,
            }
            for result in self.history
        ]


_debug_stores_by_dataset: dict[str, SsdbcodiStore] = {}


def get_debug_store(dataset_id: str) -> SsdbcodiStore:
    cleaned = clean_text(dataset_id, "dataset_id")
    if cleaned not in _debug_stores_by_dataset:
        _debug_stores_by_dataset[cleaned] = SsdbcodiStore(dataset_id=cleaned)
    return _debug_stores_by_dataset[cleaned]


def reset_debug_store(dataset_id: str) -> SsdbcodiStore:
    cleaned = clean_text(dataset_id, "dataset_id")
    _debug_stores_by_dataset[cleaned] = SsdbcodiStore(dataset_id=cleaned)
    return _debug_stores_by_dataset[cleaned]
