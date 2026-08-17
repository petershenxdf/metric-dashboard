from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence
from urllib import error, request

from .env import env_int, env_text


DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_OUTPUT_TOKENS = 6000


class DeepSeekResponseContentError(ValueError):
    def __init__(self, message: str, response_metadata: Mapping[str, Any]):
        super().__init__(message)
        self.response_metadata = dict(response_metadata)


def _default_base_url() -> str:
    return env_text(
        "METRIC_DASHBOARD_DEEPSEEK_BASE_URL",
        DEFAULT_DEEPSEEK_BASE_URL,
    )


def _default_api_key() -> str:
    return env_text("METRIC_DASHBOARD_DEEPSEEK_API_KEY", "")


def _default_timeout_seconds() -> int:
    return env_int(
        "METRIC_DASHBOARD_LLM_TIMEOUT_SECONDS",
        DEFAULT_TIMEOUT_SECONDS,
    )


@dataclass
class DeepSeekClient:
    """Minimal DeepSeek JSON client for active-learning explanations."""

    model_name: str = DEEPSEEK_PRO_MODEL
    base_url: str = field(default_factory=_default_base_url)
    api_key: str = field(default_factory=_default_api_key)
    timeout_seconds: int = field(default_factory=_default_timeout_seconds)
    temperature: float = 0.0
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
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
            "api_key_configured": bool(self.api_key),
            "last_health": dict(self._last_health),
        }

    def health(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._last_health and not force_refresh:
            return dict(self._last_health)
        if not self.api_key:
            result = {
                "available": False,
                "provider_kind": "deepseek",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "model_present": False,
                "available_models": [],
                "error": "METRIC_DASHBOARD_DEEPSEEK_API_KEY is not configured",
            }
            self._last_health = result
            return result

        try:
            payload = self._request_json(f"{self.base_url.rstrip('/')}/models")
            models = payload.get("data", ()) if isinstance(payload, Mapping) else ()
            available_models = [
                str(model.get("id"))
                for model in models
                if isinstance(model, Mapping) and model.get("id")
            ]
            result = {
                "available": True,
                "provider_kind": "deepseek",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "model_present": self.model_name in available_models,
                "available_models": available_models,
            }
        except Exception as exc:  # pragma: no cover - depends on network/API status
            result = {
                "available": False,
                "provider_kind": "deepseek",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "model_present": False,
                "available_models": [],
                "error": str(exc),
            }
        self._last_health = result
        return result

    def generate_json_with_metadata(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        thinking: Mapping[str, str] | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> tuple[Mapping[str, Any], str, Mapping[str, Any]]:
        payload = self._chat_completion(
            prompt,
            response_format={"type": "json_object"},
            max_tokens=max_tokens or self.max_output_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        raw_response = _message_content(payload)
        metadata = _response_metadata(payload)
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise DeepSeekResponseContentError(
                f"deepseek response content was not valid JSON: {exc}",
                metadata,
            ) from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("deepseek JSON response must be an object")
        return parsed, raw_response, metadata

    def _chat_completion(
        self,
        prompt: str,
        response_format: Mapping[str, str] | None,
        max_tokens: int,
        *,
        thinking: Mapping[str, str] | None,
        reasoning_effort: str | None,
        temperature: float | None = None,
    ) -> Mapping[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "thinking": dict(thinking or {"type": "disabled"}),
            "temperature": (
                self.temperature if temperature is None else float(temperature)
            ),
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            body["response_format"] = dict(response_format)
        return self._post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            body,
        )

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "METRIC_DASHBOARD_DEEPSEEK_API_KEY is not configured"
            )
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
            raise RuntimeError(
                "METRIC_DASHBOARD_DEEPSEEK_API_KEY is not configured"
            )
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
        opener = request.build_opener(request.ProxyHandler({}))
        return opener.open(req, timeout=self.timeout_seconds)


def _message_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
    ):
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
            _content_error_message(
                payload,
                "deepseek response did not contain a message",
            ),
            _response_metadata(payload),
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekResponseContentError(
            _content_error_message(
                payload,
                "deepseek response did not contain message content",
            ),
            _response_metadata(payload),
        )
    return content


def _content_error_message(payload: Mapping[str, Any], prefix: str) -> str:
    metadata = _response_metadata(payload)
    parts = [prefix]
    if metadata.get("finish_reason") is not None:
        parts.append(f"finish_reason={metadata['finish_reason']}")
    usage = metadata.get("usage", {})
    if isinstance(usage, Mapping) and usage.get("total_tokens") is not None:
        parts.append(f"total_tokens={usage['total_tokens']}")
    if metadata.get("message_keys"):
        parts.append(f"message_keys={','.join(metadata['message_keys'])}")
    return parts[0] if len(parts) == 1 else f"{parts[0]} ({'; '.join(parts[1:])})"


def _response_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    choices = payload.get("choices")
    first = (
        choices[0]
        if isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes))
        and choices
        else {}
    )
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
        "system_fingerprint": payload.get("system_fingerprint"),
        "finish_reason": first.get("finish_reason"),
        "message_keys": sorted(str(key) for key in message),
        "has_content": isinstance(content, str) and bool(content.strip()),
        "content_length": len(content) if isinstance(content, str) else 0,
        "has_reasoning_content": (
            isinstance(reasoning_content, str) and bool(reasoning_content.strip())
        ),
        "reasoning_content_length": (
            len(reasoning_content) if isinstance(reasoning_content, str) else 0
        ),
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


__all__ = [
    "DEEPSEEK_PRO_MODEL",
    "DeepSeekClient",
    "DeepSeekResponseContentError",
]
