from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .schemas import ChatTurn


@dataclass
class ChatboxStore:
    dataset_id: str
    turns: List[ChatTurn] = field(default_factory=list)
    _turn_counter: int = 0

    def next_turn_id(self) -> str:
        self._turn_counter += 1
        return f"turn_{self._turn_counter:03d}"
