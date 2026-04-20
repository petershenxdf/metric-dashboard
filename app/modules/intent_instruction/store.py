from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .schemas import StructuredInstruction


@dataclass
class IntentInstructionStore:
    """Per-dataset instruction state owned by ``IntentInstructionProvider``."""

    _states: Dict[str, StructuredInstruction] = field(default_factory=dict)
    _counters: Dict[str, int] = field(default_factory=dict)

    def get(self, dataset_id: str) -> StructuredInstruction:
        if dataset_id not in self._states:
            self._states[dataset_id] = StructuredInstruction(version=0, constraints=())
            self._counters[dataset_id] = 0
        return self._states[dataset_id]

    def set(self, dataset_id: str, state: StructuredInstruction) -> None:
        self._states[dataset_id] = state
        if dataset_id not in self._counters:
            self._counters[dataset_id] = 0

    def reset(self, dataset_id: str) -> None:
        self._states[dataset_id] = StructuredInstruction(version=0, constraints=())
        self._counters[dataset_id] = 0

    def next_constraint_id(self, dataset_id: str) -> str:
        self._counters[dataset_id] = self._counters.get(dataset_id, 0) + 1
        return f"c{self._counters[dataset_id]}"
