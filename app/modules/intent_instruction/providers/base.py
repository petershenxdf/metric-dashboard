from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from ..schemas import (
    DatasetContext,
    InstructionDelta,
    RouterResult,
    StructuredInstruction,
    Turn,
)


class LlmProvider(Protocol):
    """Router + extractor slot used by ``IntentInstructionProvider``.

    Step 8 ships only ``MockLlmProvider``. Real LLM backends (local qwen,
    Claude, OpenAI) slot into this protocol when added later; nothing in
    ``service.py`` depends on a particular implementation.
    """

    label: str

    def route(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        memory_context: Mapping[str, object] | None = None,
    ) -> RouterResult: ...

    def extract(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        current_instruction: StructuredInstruction,
        memory_context: Mapping[str, object] | None = None,
    ) -> InstructionDelta: ...
