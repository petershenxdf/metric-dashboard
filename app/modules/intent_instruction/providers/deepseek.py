from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence
from urllib import error, request

from app.shared.env import env_bool, env_float, env_int, env_text

from ..schemas import DatasetContext, InstructionDelta, RouterResult, StructuredInstruction, Turn
from .mock import MockLlmProvider
from .ollama import (
    _EXTRACT_PROMPT_PATH,
    _REPLY_PROMPT_PATH,
    _ROUTE_PROMPT_PATH,
    _backfill_delta_from_draft,
    _extract_prompt,
    _instruction_delta_from_payload,
    _optional_text,
    _reply_prompt,
    _route_prompt,
    _router_result_from_payload,
    _trim,
    _upgrade_offtopic_to_meta_query,
    _upgrade_router_with_draft,
)


DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"
DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
DEEPSEEK_MODEL_OPTIONS = (DEEPSEEK_FLASH_MODEL, DEEPSEEK_PRO_MODEL)
_DEFAULT_MODEL_NAME = DEEPSEEK_PRO_MODEL
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_MAX_OUTPUT_TOKENS = 800
_DEFAULT_ALLOW_MOCK_FALLBACK = True


class DeepSeekResponseContentError(ValueError):
    def __init__(self, message: str, response_metadata: Mapping[str, Any]):
        super().__init__(message)
        self.response_metadata = dict(response_metadata)


def _default_model_name() -> str:
    return env_text("METRIC_DASHBOARD_LLM_MODEL", _DEFAULT_MODEL_NAME)


def _default_base_url() -> str:
    return env_text("METRIC_DASHBOARD_DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL)


def _default_api_key() -> str:
    return env_text("METRIC_DASHBOARD_DEEPSEEK_API_KEY", "")


def _default_timeout_seconds() -> int:
    return env_int("METRIC_DASHBOARD_LLM_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)


def _default_temperature() -> float:
    return env_float("METRIC_DASHBOARD_LLM_TEMPERATURE", _DEFAULT_TEMPERATURE)


def _default_max_output_tokens() -> int:
    return env_int("METRIC_DASHBOARD_LLM_MAX_OUTPUT_TOKENS", _DEFAULT_MAX_OUTPUT_TOKENS)


def _default_allow_mock_fallback() -> bool:
    return env_bool(
        "METRIC_DASHBOARD_LLM_ALLOW_MOCK_FALLBACK",
        _DEFAULT_ALLOW_MOCK_FALLBACK,
    )


@dataclass
class DeepSeekLlmProvider:
    model_name: str = field(default_factory=_default_model_name)
    base_url: str = field(default_factory=_default_base_url)
    api_key: str = field(default_factory=_default_api_key)
    timeout_seconds: int = field(default_factory=_default_timeout_seconds)
    temperature: float = field(default_factory=_default_temperature)
    max_output_tokens: int = field(default_factory=_default_max_output_tokens)
    allow_mock_fallback: bool = field(default_factory=_default_allow_mock_fallback)
    fallback: MockLlmProvider = field(default_factory=MockLlmProvider)
    _last_route: Dict[str, Any] = field(default_factory=dict)
    _last_extract: Dict[str, Any] = field(default_factory=dict)
    _last_reply: Dict[str, Any] = field(default_factory=dict)
    _last_health: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"deepseek:{self.model_name}"

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "provider_kind": "deepseek",
            "label": self.label,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "allow_mock_fallback": self.allow_mock_fallback,
            "api_key_configured": bool(self.api_key),
            "thinking": {"type": "disabled"},
            "prompt_template_files": {
                "route": str(_ROUTE_PROMPT_PATH),
                "extract": str(_EXTRACT_PROMPT_PATH),
                "reply": str(_REPLY_PROMPT_PATH),
            },
            "last_route": dict(self._last_route),
            "last_extract": dict(self._last_extract),
            "last_reply": dict(self._last_reply),
            "last_health": dict(self._last_health),
        }

    def health(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._last_health and not force_refresh:
            return dict(self._last_health)

        if not self.api_key:
            health = {
                "available": False,
                "provider_kind": "deepseek",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "model_present": False,
                "available_models": [],
                "error": "METRIC_DASHBOARD_DEEPSEEK_API_KEY is not configured",
            }
            self._last_health = health
            return health

        try:
            payload = self._request_json(f"{self.base_url.rstrip('/')}/models")
            models = payload.get("data", ()) if isinstance(payload, Mapping) else ()
            available_models = [
                str(model.get("id"))
                for model in models
                if isinstance(model, Mapping) and model.get("id")
            ]
            health = {
                "available": True,
                "provider_kind": "deepseek",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "model_present": self.model_name in available_models,
                "available_models": available_models,
            }
        except Exception as exc:  # pragma: no cover - depends on network/API status
            health = {
                "available": False,
                "provider_kind": "deepseek",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "model_present": False,
                "available_models": [],
                "error": str(exc),
            }
        self._last_health = health
        return health

    def route(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        memory_context: Mapping[str, object] | None = None,
    ) -> RouterResult:
        memory_context = memory_context or {}
        prompt = _route_prompt(message, context, history, memory_context)
        try:
            payload, raw_response = self._generate_json(prompt)
            result = _router_result_from_payload(payload)
            result = _upgrade_offtopic_to_meta_query(result, message)
            result = _upgrade_router_with_draft(result, message, context, memory_context)
            self._last_route = {
                "used_fallback": False,
                "message": message,
                "prompt_template_path": str(_ROUTE_PROMPT_PATH),
                "prompt_text": prompt,
                "prompt_char_count": len(prompt),
                "raw_response": _trim(raw_response),
                "result": result.to_dict(),
            }
            return result
        except Exception as exc:
            result = self._fallback_route(message, context, history, exc)
            result = _upgrade_offtopic_to_meta_query(result, message)
            result = _upgrade_router_with_draft(result, message, context, memory_context)
            self._last_route = {
                "used_fallback": True,
                "message": message,
                "prompt_template_path": str(_ROUTE_PROMPT_PATH),
                "prompt_text": prompt,
                "prompt_char_count": len(prompt),
                "error": str(exc),
                "result": result.to_dict(),
            }
            return result

    def extract(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        current_instruction: StructuredInstruction,
        memory_context: Mapping[str, object] | None = None,
    ) -> InstructionDelta:
        memory_context = memory_context or {}
        prompt = _extract_prompt(
            message,
            context,
            history,
            current_instruction,
            memory_context,
        )
        try:
            payload, raw_response = self._generate_json(prompt)
            delta = _instruction_delta_from_payload(payload, context)
            delta = _backfill_delta_from_draft(message, delta, context, memory_context)
            self._last_extract = {
                "used_fallback": False,
                "message": message,
                "prompt_template_path": str(_EXTRACT_PROMPT_PATH),
                "prompt_text": prompt,
                "prompt_char_count": len(prompt),
                "raw_response": _trim(raw_response),
                "delta": delta.to_dict(),
            }
            return delta
        except Exception as exc:
            delta = self._fallback_extract(message, context, history, current_instruction, exc)
            delta = _backfill_delta_from_draft(message, delta, context, memory_context)
            self._last_extract = {
                "used_fallback": True,
                "message": message,
                "prompt_template_path": str(_EXTRACT_PROMPT_PATH),
                "prompt_text": prompt,
                "prompt_char_count": len(prompt),
                "error": str(exc),
                "delta": delta.to_dict(),
            }
            return delta

    def freeform_reply(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        response: Mapping[str, Any],
        provider_trace: Mapping[str, Any],
        memory_context: Mapping[str, object] | None = None,
    ) -> str:
        memory_context = memory_context or {}
        prompt = _reply_prompt(
            message,
            context,
            history,
            response,
            provider_trace,
            memory_context,
        )
        try:
            reply_text = self._generate_text(prompt)
            cleaned_reply = _optional_text(reply_text) or str(response.get("reply") or "")
            self._last_reply = {
                "used_fallback": False,
                "message": message,
                "prompt_template_path": str(_REPLY_PROMPT_PATH),
                "prompt_text": prompt,
                "prompt_char_count": len(prompt),
                "raw_response": _trim(reply_text),
                "result": {"reply": cleaned_reply},
            }
            return cleaned_reply
        except Exception as exc:
            reply_text = self._fallback_freeform_reply(response, provider_trace, exc)
            self._last_reply = {
                "used_fallback": True,
                "message": message,
                "prompt_template_path": str(_REPLY_PROMPT_PATH),
                "prompt_text": prompt,
                "prompt_char_count": len(prompt),
                "error": str(exc),
                "result": {"reply": reply_text},
            }
            return reply_text

    def _generate_json(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        thinking: Mapping[str, str] | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[Mapping[str, Any], str]:
        payload = self._chat_completion(
            prompt,
            response_format={"type": "json_object"},
            max_tokens=max_tokens or self.max_output_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        raw_response = _message_content(payload)
        parsed = json.loads(raw_response)
        if not isinstance(parsed, Mapping):
            raise ValueError("deepseek JSON response must be an object")
        return parsed, raw_response

    def _generate_text(self, prompt: str) -> str:
        payload = self._chat_completion(
            prompt,
            response_format=None,
            max_tokens=min(self.max_output_tokens, 300),
            thinking=None,
            reasoning_effort=None,
        )
        raw_response = _message_content(payload)
        if not raw_response.strip():
            raise ValueError("deepseek response did not contain reply text")
        return raw_response.strip()

    def _chat_completion(
        self,
        prompt: str,
        response_format: Mapping[str, str] | None,
        max_tokens: int,
        *,
        thinking: Mapping[str, str] | None,
        reasoning_effort: str | None,
    ) -> Mapping[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "thinking": dict(thinking or {"type": "disabled"}),
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            body["response_format"] = dict(response_format)
        return self._post_json(f"{self.base_url.rstrip('/')}/chat/completions", body)

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.api_key:
            raise RuntimeError("METRIC_DASHBOARD_DEEPSEEK_API_KEY is not configured")
        request_body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._urlopen(req) as response:
                decoded = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"deepseek request failed: {exc}") from exc
        parsed = json.loads(decoded)
        if not isinstance(parsed, Mapping):
            raise ValueError("deepseek endpoint did not return a JSON object")
        return parsed

    def _request_json(self, url: str) -> Mapping[str, Any]:
        if not self.api_key:
            raise RuntimeError("METRIC_DASHBOARD_DEEPSEEK_API_KEY is not configured")
        req = request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with self._urlopen(req) as response:
                decoded = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"deepseek request failed: {exc}") from exc
        parsed = json.loads(decoded)
        if not isinstance(parsed, Mapping):
            raise ValueError("deepseek endpoint did not return a JSON object")
        return parsed

    def _urlopen(self, req: request.Request):
        # The local dev environment may define a catch-all proxy for sandboxed
        # tooling. DeepSeek should use a direct HTTPS connection unless a future
        # provider config explicitly adds proxy support.
        opener = request.build_opener(request.ProxyHandler({}))
        return opener.open(req, timeout=self.timeout_seconds)

    def _fallback_route(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        exc: Exception,
    ) -> RouterResult:
        if self.allow_mock_fallback:
            return self.fallback.route(message, context, history)
        return RouterResult(
            category="on_topic_ambiguous",
            confidence=0.0,
            reason=str(exc),
            clarification_question="I could not parse the DeepSeek response. Please try again.",
        )

    def _fallback_extract(
        self,
        message: str,
        context: DatasetContext,
        history: Sequence[Turn],
        current_instruction: StructuredInstruction,
        exc: Exception,
    ) -> InstructionDelta:
        if self.allow_mock_fallback:
            return self.fallback.extract(message, context, history, current_instruction)
        return InstructionDelta(operations=())

    def _fallback_freeform_reply(
        self,
        response: Mapping[str, Any],
        provider_trace: Mapping[str, Any],
        exc: Exception,
    ) -> str:
        if self.allow_mock_fallback:
            return self.fallback.freeform_reply(
                str(provider_trace.get("message") or ""),
                DatasetContext(dataset_id="fallback"),
                (),
                response,
                provider_trace,
                {},
            )
        reply_text = _optional_text(response.get("reply"))
        if reply_text is not None:
            return reply_text
        return f"I could not generate a DeepSeek direct reply: {exc}"


def _message_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise DeepSeekResponseContentError(
            "deepseek response did not contain choices",
            _response_metadata(payload),
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise DeepSeekResponseContentError(
            "deepseek response choice must be an object",
            _response_metadata(payload),
        )
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise DeepSeekResponseContentError(
            _content_error_message(payload, "deepseek response did not contain a message"),
            _response_metadata(payload),
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekResponseContentError(
            _content_error_message(payload, "deepseek response did not contain message content"),
            _response_metadata(payload),
        )
    return content


def _content_error_message(payload: Mapping[str, Any], prefix: str) -> str:
    metadata = _response_metadata(payload)
    parts = [prefix]
    finish_reason = metadata.get("finish_reason")
    if finish_reason is not None:
        parts.append(f"finish_reason={finish_reason}")
    total_tokens = metadata.get("usage", {}).get("total_tokens") if isinstance(metadata.get("usage"), Mapping) else None
    if total_tokens is not None:
        parts.append(f"total_tokens={total_tokens}")
    message_keys = metadata.get("message_keys")
    if message_keys:
        parts.append(f"message_keys={','.join(message_keys)}")
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} ({'; '.join(parts[1:])})"


def _response_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices else {}
    if not isinstance(first, Mapping):
        first = {}
    message = first.get("message")
    if not isinstance(message, Mapping):
        message = {}
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    usage = payload.get("usage")
    return {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "finish_reason": first.get("finish_reason"),
        "message_keys": sorted(str(key) for key in message.keys()),
        "has_content": isinstance(content, str) and bool(content.strip()),
        "content_length": len(content) if isinstance(content, str) else 0,
        "has_reasoning_content": isinstance(reasoning_content, str) and bool(reasoning_content.strip()),
        "reasoning_content_length": len(reasoning_content) if isinstance(reasoning_content, str) else 0,
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


__all__ = [
    "DEEPSEEK_FLASH_MODEL",
    "DEEPSEEK_MODEL_OPTIONS",
    "DEEPSEEK_PRO_MODEL",
    "DeepSeekResponseContentError",
    "DeepSeekLlmProvider",
]
