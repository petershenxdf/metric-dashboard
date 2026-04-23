from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.modules.chatbox.schemas import ChatTurn, ChatboxState


RESPONSE_MODES = ("processed", "raw")
DEFAULT_RESPONSE_MODE = "processed"


def normalize_response_mode(value: Any) -> str:
    text = str(value or DEFAULT_RESPONSE_MODE).strip().lower()
    if text not in RESPONSE_MODES:
        raise ValueError(f"response_mode must be one of: {', '.join(RESPONSE_MODES)}")
    return text


def build_display_chat_state(
    base_state: ChatboxState,
    interactions: Sequence[Mapping[str, Any]],
    response_mode: str,
) -> ChatboxState:
    response_mode = normalize_response_mode(response_mode)
    if response_mode == "processed" or not interactions:
        return base_state

    display_turns: list[ChatTurn] = []
    interaction_count = min(len(interactions), len(base_state.turns) // 2)
    for index in range(interaction_count):
        user_turn = base_state.turns[index * 2]
        assistant_turn = base_state.turns[index * 2 + 1]
        interaction = interactions[index]
        display_turns.append(user_turn)
        display_turns.append(
            ChatTurn(
                turn_id=assistant_turn.turn_id,
                role=assistant_turn.role,
                text=_display_reply_for_interaction(interaction, response_mode),
            )
        )

    display_turns.extend(base_state.turns[interaction_count * 2 :])
    return ChatboxState(
        dataset_id=base_state.dataset_id,
        turns=tuple(display_turns),
        instruction_snapshot=base_state.instruction_snapshot,
    )


def build_display_response(
    response: Mapping[str, Any],
    provider_trace: Mapping[str, Any],
    response_mode: str,
) -> dict[str, Any]:
    response_mode = normalize_response_mode(response_mode)
    payload = dict(response or {})
    processed_reply = str(payload.get("reply") or "")
    raw_reply = _raw_reply_from_trace(provider_trace)
    payload["processed_reply"] = processed_reply
    payload["raw_reply"] = raw_reply
    payload["response_mode"] = response_mode
    payload["reply"] = (
        processed_reply
        if response_mode == "processed"
        else raw_reply or processed_reply
    )
    payload["raw_reply_available"] = bool(raw_reply)
    return payload


def last_display_response(
    interactions: Sequence[Mapping[str, Any]],
    response_mode: str,
) -> dict[str, Any] | None:
    if not interactions:
        return None
    interaction = interactions[-1]
    response = interaction.get("response")
    if not isinstance(response, Mapping):
        return None
    provider_trace = interaction.get("provider_trace")
    return build_display_response(
        response,
        provider_trace if isinstance(provider_trace, Mapping) else {},
        response_mode,
    )


def _display_reply_for_interaction(
    interaction: Mapping[str, Any],
    response_mode: str,
) -> str:
    response = interaction.get("response")
    provider_trace = interaction.get("provider_trace")
    if not isinstance(response, Mapping):
        return "No response recorded."

    display_response = build_display_response(
        response,
        provider_trace if isinstance(provider_trace, Mapping) else {},
        response_mode,
    )
    reply = str(display_response.get("reply") or "").strip()
    return reply or "No response recorded."


def _raw_reply_from_trace(provider_trace: Mapping[str, Any]) -> str | None:
    llm_trace = provider_trace.get("llm_trace")
    if isinstance(llm_trace, Mapping):
        reply_trace = llm_trace.get("reply")
        if isinstance(reply_trace, Mapping):
            raw_response = reply_trace.get("raw_response")
            if isinstance(raw_response, str) and raw_response.strip():
                return raw_response.strip()

            result = reply_trace.get("result")
            if isinstance(result, Mapping):
                reply_text = result.get("reply")
                if isinstance(reply_text, str) and reply_text.strip():
                    return reply_text.strip()

            error_text = reply_trace.get("error")
            if isinstance(error_text, str) and error_text.strip():
                return error_text.strip()

    freeform_reply = provider_trace.get("freeform_reply")
    if isinstance(freeform_reply, str) and freeform_reply.strip():
        return freeform_reply.strip()

    return None
