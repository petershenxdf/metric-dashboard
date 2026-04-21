from __future__ import annotations

from typing import Mapping, Sequence

from .providers.base import LlmProvider
from .schemas import DatasetContext, RouterResult, Turn


def classify(
    llm: LlmProvider,
    message: str,
    context: DatasetContext,
    history: Sequence[Turn] = (),
    memory_context: Mapping[str, object] | None = None,
) -> RouterResult:
    """Thin service wrapper over ``LlmProvider.route``.

    Kept as its own module so future providers can be unit-tested for
    routing behavior independently from extraction.
    """

    return llm.route(message, context, history, memory_context=memory_context)
