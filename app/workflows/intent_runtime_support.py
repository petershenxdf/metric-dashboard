from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping
from uuid import uuid4

from app.modules.chatbox.service import create_chatbox_store
from app.modules.intent_instruction.providers.mock import MockLlmProvider
from app.modules.intent_instruction.providers.ollama import OllamaLlmProvider
from app.modules.intent_instruction.service import IntentInstructionProvider
from app.modules.intent_instruction.store import IntentInstructionStore


SUPPORTED_PROVIDER_KINDS = ("ollama", "mock")
DEFAULT_PROVIDER_KIND = "ollama"
DEFAULT_MODEL_NAME = "qwen2.5:14b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class IntentRuntimeConfig:
    provider_kind: str = DEFAULT_PROVIDER_KIND
    model_name: str = DEFAULT_MODEL_NAME
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.1
    timeout_seconds: int = 45
    max_output_tokens: int = 800
    allow_mock_fallback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_kind": self.provider_kind,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "allow_mock_fallback": self.allow_mock_fallback,
        }

    def merged(self, payload: Mapping[str, Any] | None = None) -> "IntentRuntimeConfig":
        payload = payload or {}
        provider_kind = str(payload.get("provider_kind", self.provider_kind)).strip().lower()
        if provider_kind not in SUPPORTED_PROVIDER_KINDS:
            raise ValueError(f"provider_kind must be one of: {', '.join(SUPPORTED_PROVIDER_KINDS)}")

        model_name = str(payload.get("model_name", self.model_name)).strip() or self.model_name
        base_url = str(payload.get("base_url", self.base_url)).strip() or self.base_url
        temperature = float(payload.get("temperature", self.temperature))
        timeout_seconds = max(1, int(payload.get("timeout_seconds", self.timeout_seconds)))
        max_output_tokens = max(1, int(payload.get("max_output_tokens", self.max_output_tokens)))
        allow_mock_fallback = _coerce_bool(
            payload.get("allow_mock_fallback", self.allow_mock_fallback),
            default=self.allow_mock_fallback,
        )

        return IntentRuntimeConfig(
            provider_kind=provider_kind,
            model_name=model_name,
            base_url=base_url.rstrip("/"),
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            allow_mock_fallback=allow_mock_fallback,
        )


@dataclass
class IntentRuntimeSession:
    dataset_id: str
    session_id: str
    config: IntentRuntimeConfig
    chat_store: object
    instruction_store: IntentInstructionStore
    provider: IntentInstructionProvider
    last_response: Mapping[str, Any] | None = None
    interactions: list[Dict[str, Any]] = field(default_factory=list)
    event_log: list[Dict[str, Any]] = field(default_factory=list)


class IntentRuntimeManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, IntentRuntimeSession] = {}

    def session_for(
        self,
        dataset_id: str,
        config_updates: Mapping[str, Any] | None = None,
    ) -> IntentRuntimeSession:
        if dataset_id not in self._sessions:
            config = IntentRuntimeConfig().merged(config_updates)
            self._sessions[dataset_id] = self._create_session(dataset_id, config)
            self.record_event(dataset_id, "session_started", {"config": config.to_dict()})
        elif config_updates:
            self.apply_config(dataset_id, config_updates)
        return self._sessions[dataset_id]

    def apply_config(self, dataset_id: str, config_updates: Mapping[str, Any]) -> IntentRuntimeSession:
        session = self._sessions[dataset_id]
        new_config = session.config.merged(config_updates)
        if new_config == session.config:
            return session
        session.config = new_config
        session.provider = _build_provider(new_config, session.instruction_store)
        self.record_event(dataset_id, "runtime_config_updated", {"config": new_config.to_dict()})
        return session

    def remember_response(
        self,
        dataset_id: str,
        forwarded_payload,
        response,
    ) -> None:
        session = self._sessions[dataset_id]
        trace = session.provider.trace_for(dataset_id)
        event = {
            "event_type": "message_exchange",
            "timestamp": _now_iso(),
            "forwarded_payload": forwarded_payload.to_dict(),
            "response": response.to_dict(),
            "provider_trace": dict(trace),
        }
        session.last_response = response.to_dict()
        session.interactions.append(event)
        session.event_log.append(
            {
                "event_type": "message_exchange",
                "timestamp": event["timestamp"],
                "router_category": response.router_category,
                "intent_type": response.intent_type,
                "requires_followup": response.requires_followup,
            }
        )

    def remember_manual_event(self, dataset_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        self.record_event(dataset_id, event_type, payload)

    def clear_chat(self, dataset_id: str) -> None:
        session = self._sessions[dataset_id]
        session.last_response = None
        self.record_event(dataset_id, "chat_cleared", {})

    def reset(self, dataset_id: str, config_updates: Mapping[str, Any] | None = None) -> IntentRuntimeSession:
        existing = self._sessions.get(dataset_id)
        config = (
            existing.config.merged(config_updates)
            if existing is not None
            else IntentRuntimeConfig().merged(config_updates)
        )
        self._sessions[dataset_id] = self._create_session(dataset_id, config)
        self.record_event(dataset_id, "session_started", {"config": config.to_dict()})
        return self._sessions[dataset_id]

    def provider_diagnostics(self, dataset_id: str) -> Dict[str, Any]:
        session = self._sessions[dataset_id]
        llm = session.provider.llm
        diagnostics = {
            "provider_label": session.provider.label,
            "provider_kind": session.config.provider_kind,
            "runtime_config": session.config.to_dict(),
        }
        if hasattr(llm, "diagnostics"):
            diagnostics["llm_diagnostics"] = llm.diagnostics()
        if hasattr(llm, "health"):
            diagnostics["health"] = llm.health()
        return diagnostics

    def storage_payload(self, dataset_id: str) -> Dict[str, Any]:
        session = self._sessions[dataset_id]
        session_dir = _session_dir(dataset_id, session.session_id)
        files = {
            "runtime_config": str(session_dir / "runtime_config.json"),
            "chat_state": str(session_dir / "chat_state.json"),
            "current_instruction": str(session_dir / "current_instruction.json"),
            "grounded_state": str(session_dir / "grounded_state.json"),
            "memory_state": str(session_dir / "memory_state.json"),
            "runtime_diagnostics": str(session_dir / "runtime_diagnostics.json"),
            "last_route_prompt": str(session_dir / "last_route_prompt.txt"),
            "last_extract_prompt": str(session_dir / "last_extract_prompt.txt"),
            "interactions": str(session_dir / "interactions.json"),
            "event_log": str(session_dir / "event_log.json"),
            "session_manifest": str(session_dir / "session_manifest.json"),
        }
        return {
            "session_id": session.session_id,
            "artifact_dir": str(session_dir),
            "files": files,
        }

    def persist_runtime_snapshot(
        self,
        dataset_id: str,
        runtime_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        session = self._sessions[dataset_id]
        session_dir = _session_dir(dataset_id, session.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        provider_diagnostics = self.provider_diagnostics(dataset_id)
        storage = self.storage_payload(dataset_id)

        _write_json(session_dir / "runtime_config.json", session.config.to_dict())
        _write_json(session_dir / "chat_state.json", runtime_payload.get("chat_state", {}))
        _write_json(session_dir / "current_instruction.json", runtime_payload.get("current_instruction", {}))
        _write_json(session_dir / "grounded_state.json", runtime_payload.get("state", {}))
        _write_json(session_dir / "memory_state.json", runtime_payload.get("memory_state", {}))
        _write_json(session_dir / "runtime_diagnostics.json", provider_diagnostics)
        llm_diagnostics = provider_diagnostics.get("llm_diagnostics", {})
        last_route = llm_diagnostics.get("last_route", {}) if isinstance(llm_diagnostics, Mapping) else {}
        last_extract = llm_diagnostics.get("last_extract", {}) if isinstance(llm_diagnostics, Mapping) else {}
        _write_text(session_dir / "last_route_prompt.txt", last_route.get("prompt_text") if isinstance(last_route, Mapping) else "")
        _write_text(session_dir / "last_extract_prompt.txt", last_extract.get("prompt_text") if isinstance(last_extract, Mapping) else "")
        _write_json(session_dir / "interactions.json", {"interactions": list(session.interactions)})
        _write_json(session_dir / "event_log.json", {"events": list(session.event_log)})
        _write_json(
            session_dir / "session_manifest.json",
            {
                "dataset_id": dataset_id,
                "session_id": session.session_id,
                "saved_at": _now_iso(),
                "storage": storage,
                "runtime_config": session.config.to_dict(),
                "provider_diagnostics": provider_diagnostics,
            },
        )
        return storage

    def record_event(self, dataset_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        session = self._sessions[dataset_id]
        session.event_log.append(
            {
                "event_type": event_type,
                "timestamp": _now_iso(),
                "payload": dict(payload),
            }
        )

    def _create_session(self, dataset_id: str, config: IntentRuntimeConfig) -> IntentRuntimeSession:
        instruction_store = IntentInstructionStore()
        return IntentRuntimeSession(
            dataset_id=dataset_id,
            session_id=_new_session_id(),
            config=config,
            chat_store=create_chatbox_store(dataset_id),
            instruction_store=instruction_store,
            provider=_build_provider(config, instruction_store),
        )


def build_memory_state(
    runtime_payload: Mapping[str, Any],
    session: IntentRuntimeSession,
) -> Dict[str, Any]:
    chat_state = runtime_payload.get("chat_state", {})
    current_instruction = runtime_payload.get("current_instruction", {})
    turns = list(chat_state.get("turns", ()))
    interactions = list(session.interactions)
    last_trace = interactions[-1].get("provider_trace", {}) if interactions else {}
    last_response = session.last_response or {}
    draft_state = _draft_state_from_trace(last_trace, last_response, turns)
    off_topic_turns = [
        interaction
        for interaction in interactions
        if interaction.get("response", {}).get("router_category") == "off_topic"
    ]
    relevant_turns = [
        {
            "message": interaction.get("forwarded_payload", {}).get("message"),
            "router_category": interaction.get("response", {}).get("router_category"),
            "intent_type": interaction.get("response", {}).get("intent_type"),
            "reply": interaction.get("response", {}).get("reply"),
            "reply_source": interaction.get("provider_trace", {}).get("reply_source"),
        }
        for interaction in interactions
        if interaction.get("response", {}).get("router_category") != "off_topic"
    ][-4:]
    confirmed_facts = [
        {
            "fact_id": f"confirmed_{index + 1:03d}",
            "kind": constraint.get("intent"),
            "value": constraint,
            "status": "confirmed",
            "provenance": {
                "raw_message": constraint.get("raw_message"),
                "source_turn_id": _turn_id_for_message(turns, constraint.get("raw_message")),
            },
        }
        for index, constraint in enumerate(current_instruction.get("constraints", ()))
    ]
    tentative_facts = []
    proposed_delta = last_trace.get("proposed_delta", {})
    if draft_state and proposed_delta:
        tentative_facts.append(
            {
                "fact_id": "tentative_001",
                "kind": draft_state.get("candidate_intent"),
                "value": proposed_delta,
                "status": "tentative",
                "provenance": {
                    "raw_message": last_trace.get("message"),
                    "source_turn_id": _turn_id_for_message(turns, last_trace.get("message")),
                },
            }
        )
    summary = _summary_text(runtime_payload, session, draft_state)
    return {
        "turn_count": len(turns),
        "summary": summary,
        "recent_turns": turns[-6:],
        "relevant_turns": relevant_turns,
        "confirmed_facts": confirmed_facts,
        "tentative_facts": tentative_facts,
        "irrelevant_turns": [
            {
                "message": interaction.get("forwarded_payload", {}).get("message"),
                "reply": interaction.get("response", {}).get("reply"),
            }
            for interaction in off_topic_turns
        ],
        "draft_state": draft_state,
        "interaction_count": len(interactions),
    }


def _summary_text(
    runtime_payload: Mapping[str, Any],
    session: IntentRuntimeSession,
    draft_state: Mapping[str, Any] | None,
) -> str:
    state = runtime_payload.get("state", {})
    analysis_context = state.get("analysis_context", {})
    current_instruction = runtime_payload.get("current_instruction", {})
    latest_message = (
        session.interactions[-1].get("forwarded_payload", {}).get("message")
        if session.interactions
        else "No user message yet."
    )
    if draft_state and draft_state.get("pending_clarification"):
        return (
            f"Latest user message: {latest_message}. "
            f"Active draft intent: {draft_state.get('candidate_intent')}. "
            f"Missing slots: {', '.join(draft_state.get('unresolved_slots', [])) or 'none'}. "
            f"Visible clusters: {analysis_context.get('cluster_count', 0)}. "
            f"Current compiled constraints: {current_instruction.get('constraint_count', 0)}."
        )
    return (
        f"Latest user message: {latest_message}. "
        f"Visible clusters: {analysis_context.get('cluster_count', 0)}. "
        f"Selected points: {analysis_context.get('selected_count', 0)}. "
        f"Current compiled constraints: {current_instruction.get('constraint_count', 0)}."
    )


def build_memory_context(
    runtime_payload: Mapping[str, Any],
    session: IntentRuntimeSession,
) -> Dict[str, Any]:
    memory_state = build_memory_state(runtime_payload, session)
    return {
        "summary": memory_state.get("summary"),
        "draft_state": memory_state.get("draft_state"),
        "confirmed_facts": memory_state.get("confirmed_facts", []),
        "tentative_facts": memory_state.get("tentative_facts", []),
        "relevant_turns": memory_state.get("relevant_turns", []),
        "irrelevant_turns": memory_state.get("irrelevant_turns", []),
        "recent_turns": memory_state.get("recent_turns", []),
    }


def _build_provider(
    config: IntentRuntimeConfig,
    store: IntentInstructionStore,
) -> IntentInstructionProvider:
    if config.provider_kind == "mock":
        llm = MockLlmProvider()
    else:
        llm = OllamaLlmProvider(
            model_name=config.model_name,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            allow_mock_fallback=config.allow_mock_fallback,
        )
    return IntentInstructionProvider(llm=llm, store=store)


def _draft_state_from_trace(
    trace: Mapping[str, Any],
    response: Mapping[str, Any],
    turns: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    proposed_delta = trace.get("proposed_delta", {})
    operations = proposed_delta.get("operations", ()) if isinstance(proposed_delta, Mapping) else ()
    if not operations:
        return {
            "candidate_intent": None,
            "pending_clarification": None,
            "proposed_delta": None,
            "grounded_refs": [],
            "unresolved_slots": [],
            "source_turn_id": None,
        }

    first = operations[0] if isinstance(operations[0], Mapping) else {}
    payload = dict(first.get("payload") or {})
    candidate_intent = first.get("intent")
    grounded_refs = _collect_grounded_refs(payload)
    unresolved_slots = _missing_slots(candidate_intent, payload)
    return {
        "candidate_intent": candidate_intent,
        "pending_clarification": response.get("followup_question"),
        "proposed_delta": proposed_delta,
        "grounded_refs": grounded_refs,
        "unresolved_slots": unresolved_slots,
        "source_turn_id": _turn_id_for_message(turns, trace.get("message")),
    }


def _collect_grounded_refs(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    refs = []
    for key in ("group_a", "group_b", "anchor"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            refs.append(dict(value))
    for item in payload.get("target_groups", ()) or ():
        if isinstance(item, Mapping):
            refs.append(dict(item))
    return refs


def _missing_slots(intent: Any, payload: Mapping[str, Any]) -> list[str]:
    if intent == "reclassify_outlier" and not payload.get("anchor"):
        return ["anchor_point"]
    if intent in {"group_similar", "group_dissimilar"}:
        missing = []
        if not payload.get("group_a"):
            missing.append("group_a")
        if not payload.get("group_b"):
            missing.append("group_b")
        return missing
    if intent == "merge_clusters":
        groups = payload.get("target_groups", ()) or ()
        if len(groups) < 2:
            return ["target_groups"]
    if intent in {"split_cluster", "ignore_cluster"}:
        groups = payload.get("target_groups", ()) or ()
        if len(groups) < 1:
            return ["target_groups"]
    if intent == "anchor_point":
        missing = []
        if not payload.get("anchor"):
            missing.append("anchor")
        if not (payload.get("target_groups", ()) or ()):
            missing.append("target_groups")
        return missing
    return []


def _turn_id_for_message(turns: list[Mapping[str, Any]], message: Any) -> str | None:
    if not message:
        return None
    for turn in reversed(turns):
        if turn.get("role") == "user" and turn.get("text") == message:
            return turn.get("turn_id")
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _session_dir(dataset_id: str, session_id: str) -> Path:
    return _repo_root() / "runtime_data" / "intent_runtime_validation" / dataset_id / session_id


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_text(path: Path, text: Any) -> None:
    path.write_text("" if text is None else str(text), encoding="utf-8")


def _new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"session_{timestamp}_{uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)
