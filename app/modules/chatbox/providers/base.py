from __future__ import annotations

from typing import Protocol

from ..schemas import ChatMessagePayload, ChatResponse, InstructionSnapshot


class IntentProvider(Protocol):
    """Protocol the chatbox forwards messages through.

    In Step 7 this is satisfied by ``MockIntentProvider``. In Step 8 the real
    ``intent_instruction`` module will satisfy the same shape.
    """

    label: str

    def respond(self, payload: ChatMessagePayload) -> ChatResponse: ...

    def current_snapshot(self, dataset_id: str) -> InstructionSnapshot: ...

    def reset(self, dataset_id: str) -> None: ...
