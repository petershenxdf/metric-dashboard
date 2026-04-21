from __future__ import annotations

from typing import Mapping, Sequence, Tuple

from .providers.base import LlmProvider
from .schemas import (
    ConstraintEntry,
    DatasetContext,
    DeltaOperation,
    GroupRef,
    InstructionDelta,
    StructuredInstruction,
    Turn,
)
from .store import IntentInstructionStore


def propose_delta(
    llm: LlmProvider,
    message: str,
    context: DatasetContext,
    history: Sequence[Turn],
    current_instruction: StructuredInstruction,
    memory_context: Mapping[str, object] | None = None,
) -> InstructionDelta:
    return llm.extract(
        message,
        context,
        history,
        current_instruction,
        memory_context=memory_context,
    )


def apply_delta(
    store: IntentInstructionStore,
    dataset_id: str,
    proposed: InstructionDelta,
    raw_message: str,
    router_category: str,
    confidence: float,
) -> Tuple[StructuredInstruction, InstructionDelta]:
    """Apply a proposed delta to the store's state for ``dataset_id``.

    The service layer owns ID allocation: the LLM may emit ``constraint_id
    = "pending"`` for add operations, and we rewrite those to real IDs
    here. The returned ``InstructionDelta`` reflects the final IDs.
    """

    current = store.get(dataset_id)
    constraints = list(current.constraints)
    final_ops = []

    for op in proposed.operations:
        if op.op == "add":
            if op.intent is None:
                raise ValueError("add op must carry an intent")
            constraint_id = (
                op.constraint_id
                if op.constraint_id and op.constraint_id != "pending"
                else store.next_constraint_id(dataset_id)
            )
            entry = _constraint_from_payload(
                constraint_id=constraint_id,
                intent=op.intent,
                raw_message=raw_message,
                payload=op.payload,
            )
            constraints.append(entry)
            final_ops.append(
                DeltaOperation(
                    op="add",
                    constraint_id=constraint_id,
                    intent=op.intent,
                    payload=dict(op.payload),
                )
            )
        elif op.op == "remove":
            constraints = [c for c in constraints if c.id != op.constraint_id]
            final_ops.append(op)
        elif op.op == "modify":
            replaced = []
            for c in constraints:
                if c.id == op.constraint_id and op.intent:
                    replaced.append(
                        _constraint_from_payload(
                            constraint_id=c.id,
                            intent=op.intent,
                            raw_message=raw_message,
                            payload=op.payload,
                        )
                    )
                else:
                    replaced.append(c)
            constraints = replaced
            final_ops.append(op)
        else:
            raise ValueError(f"unsupported delta op: {op.op}")

    final_delta = InstructionDelta(operations=tuple(final_ops))
    next_state = StructuredInstruction(
        version=current.version + 1,
        constraints=tuple(constraints),
        last_delta=final_delta,
        router_category=router_category,
        raw_message=raw_message,
        clarification_needed=False,
        clarification_question=None,
        confidence=confidence,
    )
    store.set(dataset_id, next_state)
    return next_state, final_delta


def _constraint_from_payload(
    constraint_id: str,
    intent: str,
    raw_message: str,
    payload,
) -> ConstraintEntry:
    payload = dict(payload or {})

    def _group(key):
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, GroupRef):
            return value
        if isinstance(value, dict):
            return GroupRef(source=value.get("source", ""), ref=value.get("ref", ""))
        raise ValueError(f"{key} must be a GroupRef or mapping")

    target_groups = tuple(
        GroupRef(source=item["source"], ref=item["ref"])
        if isinstance(item, dict)
        else item
        for item in payload.get("target_groups", ())
    )

    return ConstraintEntry(
        id=constraint_id,
        intent=intent,
        raw_message=raw_message,
        feature=payload.get("feature") or None,
        direction=payload.get("direction") or None,
        group_a=_group("group_a"),
        group_b=_group("group_b"),
        anchor=_group("anchor"),
        target_groups=target_groups,
        is_outlier=payload.get("is_outlier"),
    )
