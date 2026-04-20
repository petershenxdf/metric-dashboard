from __future__ import annotations

from typing import Sequence

from .providers.base import LlmProvider
from .schemas import DatasetContext, RouterResult, Turn


def classify(
    llm: LlmProvider,
    message: str,
    context: DatasetContext,
    history: Sequence[Turn] = (),
) -> RouterResult:
    """Thin service wrapper over ``LlmProvider.route``.

    Kept as its own module so future providers can be unit-tested for
    routing behavior independently from extraction.
    """

    return llm.route(message, context, history)
