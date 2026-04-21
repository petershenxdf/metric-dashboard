from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, Mapping, Sequence
from urllib import error, request

from ..schemas import DeltaOperation, DatasetContext, InstructionDelta, RouterResult, StructuredInstruction, Turn
from .mock import MockLlmProvider


_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "ollama"
_ROUTE_PROMPT_PATH = _PROMPT_DIR / "route_prompt.txt"
_EXTRACT_PROMPT_PATH = _PROMPT_DIR / "extract_prompt.txt"
_ROUTE_PROMPT_TEMPLATE = Template(_ROUTE_PROMPT_PATH.read_text(encoding="utf-8"))
_EXTRACT_PROMPT_TEMPLATE = Template(_EXTRACT_PROMPT_PATH.read_text(encoding="utf-8"))


@dataclass
class OllamaLlmProvider:
    model_name: str = "qwen2.5:14b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 45
    temperature: float = 0.1
    max_output_tokens: int = 800
    allow_mock_fallback: bool = True
    fallback: MockLlmProvider = field(default_factory=MockLlmProvider)
    _last_route: Dict[str, Any] = field(default_factory=dict)
    _last_extract: Dict[str, Any] = field(default_factory=dict)
    _last_health: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"ollama:{self.model_name}"

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "provider_kind": "ollama",
            "label": self.label,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "allow_mock_fallback": self.allow_mock_fallback,
            "prompt_template_files": {
                "route": str(_ROUTE_PROMPT_PATH),
                "extract": str(_EXTRACT_PROMPT_PATH),
            },
            "last_route": dict(self._last_route),
            "last_extract": dict(self._last_extract),
            "last_health": dict(self._last_health),
        }

    def health(self) -> Dict[str, Any]:
        tags_url = f"{self.base_url.rstrip('/')}/api/tags"
        try:
            payload = self._request_json(tags_url)
            models = payload.get("models", ()) if isinstance(payload, Mapping) else ()
            available_models = [
                str(model.get("name") or model.get("model"))
                for model in models
                if isinstance(model, Mapping) and (model.get("name") or model.get("model"))
            ]
            health = {
                "available": True,
                "provider_kind": "ollama",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "model_present": self.model_name in available_models,
                "available_models": available_models,
            }
        except Exception as exc:  # pragma: no cover - exercised in workflow smoke paths
            health = {
                "available": False,
                "provider_kind": "ollama",
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

    def _generate_json(self, prompt: str) -> tuple[Mapping[str, Any], str]:
        body = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        payload = self._post_json(f"{self.base_url.rstrip('/')}/api/generate", body)
        raw_response = payload.get("response", "") if isinstance(payload, Mapping) else ""
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError("ollama response did not contain JSON text")
        parsed = json.loads(raw_response)
        if not isinstance(parsed, Mapping):
            raise ValueError("ollama JSON response must be an object")
        return parsed, raw_response

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request_body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                decoded = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc
        parsed = json.loads(decoded)
        if not isinstance(parsed, Mapping):
            raise ValueError("ollama endpoint did not return a JSON object")
        return parsed

    def _request_json(self, url: str) -> Mapping[str, Any]:
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                decoded = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc
        parsed = json.loads(decoded)
        if not isinstance(parsed, Mapping):
            raise ValueError("ollama endpoint did not return a JSON object")
        return parsed

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
            clarification_question="I could not parse the model response. Please try again.",
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


def _router_result_from_payload(payload: Mapping[str, Any]) -> RouterResult:
    return RouterResult(
        category=str(payload.get("category", "")),
        confidence=float(payload.get("confidence", 0.0)),
        reason=_optional_text(payload.get("reason")),
        clarification_question=_optional_text(payload.get("clarification_question")),
        reply_text=_optional_text(payload.get("reply_text")),
    )


def _instruction_delta_from_payload(
    payload: Mapping[str, Any],
    context: DatasetContext,
) -> InstructionDelta:
    operations_payload = payload.get("operations")
    if operations_payload is None and isinstance(payload.get("delta"), Mapping):
        operations_payload = payload["delta"].get("operations")
    if operations_payload is None:
        operations_payload = ()
    if not isinstance(operations_payload, Sequence) or isinstance(operations_payload, (str, bytes)):
        raise ValueError("operations must be a sequence")
    operations = []
    for item in operations_payload:
        if not isinstance(item, Mapping):
            raise ValueError("operation entries must be objects")
        operations.append(
            DeltaOperation(
                op=str(item.get("op", "")),
                constraint_id=str(item.get("constraint_id", "pending")),
                intent=_normalize_intent_name(item.get("intent")),
                payload=_normalize_operation_payload(
                    _normalize_intent_name(item.get("intent")),
                    item.get("payload") or {},
                    context,
                ),
            )
        )
    return InstructionDelta(operations=tuple(operations))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_INTENT_ALIASES = {
    "split_clusters": "split_cluster",
    "merge_cluster": "merge_clusters",
    "merge_cluster_pair": "merge_clusters",
    "reclassify_outliers": "reclassify_outlier",
}


def _normalize_intent_name(value: Any) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    return _INTENT_ALIASES.get(text, text)


def _trim(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _normalize_operation_payload(
    intent: str | None,
    payload: Any,
    context: DatasetContext,
) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    normalized = dict(payload)
    for key in ("group_a", "group_b", "anchor"):
        if key in normalized:
            normalized[key] = _normalize_group_ref(normalized.get(key), context)
    if "target_groups" in normalized:
        normalized["target_groups"] = _normalize_group_ref_list(normalized.get("target_groups"), context)
    return normalized


def _normalize_group_ref_list(value: Any, context: DatasetContext) -> list[Dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    refs = []
    for item in value:
        normalized = _normalize_group_ref(item, context)
        if normalized is not None:
            refs.append(normalized)
    return refs


def _normalize_group_ref(value: Any, context: DatasetContext) -> Dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        source = _optional_text(value.get("source"))
        ref = _optional_text(value.get("ref"))
        if source and ref:
            return {"source": source, "ref": ref}
        if _optional_text(value.get("cluster_id")):
            return {"source": "cluster", "ref": str(value["cluster_id"])}
        if _optional_text(value.get("point_id")):
            return {"source": "point_id", "ref": str(value["point_id"])}
        if _optional_text(value.get("group_name")):
            return {"source": "selection_group", "ref": str(value["group_name"])}
        return None
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in {"current", "current selection", "selected", "selected points"}:
            return {"source": "selected_points", "ref": "current"}
        if text in context.cluster_ids:
            return {"source": "cluster", "ref": text}
        if text in context.selection_group_names:
            return {"source": "selection_group", "ref": text}
        if text in set((*context.selected_point_ids, *context.unselected_point_ids)):
            return {"source": "point_id", "ref": text}
        if text in context.outlier_point_ids:
            return {"source": "outlier_set", "ref": text}
    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


_ROUTE_FEWSHOT = [
    {
        "message": "how many clusters are there",
        "category": "meta_query",
        "reply_text": "There are 3 clusters: cluster_1, cluster_2, cluster_3.",
        "confidence": 0.92,
    },
    {
        "message": "what classes do we have right now",
        "category": "meta_query",
        "reply_text": "The current classes are the three clusters: cluster_1, cluster_2, cluster_3.",
        "confidence": 0.88,
    },
    {
        "message": "what class are the selected points",
        "category": "meta_query",
        "reply_text": "The selected points alpha_01 and alpha_02 are in cluster_1.",
        "confidence": 0.9,
    },
    {
        "message": "which cluster is p3 in",
        "category": "meta_query",
        "reply_text": "Point p3 is currently in cluster_2.",
        "confidence": 0.9,
    },
    {
        "message": "is outlier_east really an outlier",
        "category": "meta_query",
        "reply_text": "Yes, outlier_east is in the current outlier set.",
        "confidence": 0.88,
    },
    {
        "message": "how many points are selected",
        "category": "meta_query",
        "reply_text": "2 points are currently selected.",
        "confidence": 0.9,
    },
    {
        "message": "which points are selected right now",
        "category": "meta_query",
        "reply_text": "Currently selected points: alpha_01, alpha_02.",
        "confidence": 0.9,
    },
    {
        "message": "tell me about the selection",
        "category": "meta_query",
        "reply_text": "The current selection has 2 points (alpha_01, alpha_02), both in cluster_1.",
        "confidence": 0.85,
    },
    {
        "message": "list the features",
        "category": "meta_query",
        "reply_text": "The dataset features are x and y.",
        "confidence": 0.9,
    },
    {
        "message": "merge clusters 1 and 2",
        "category": "on_topic_actionable",
        "reply_text": None,
        "confidence": 0.95,
    },
    {
        "message": "these points should belong together",
        "category": "on_topic_actionable",
        "reply_text": None,
        "confidence": 0.9,
    },
    {
        "message": "push these apart",
        "category": "on_topic_actionable",
        "reply_text": None,
        "confidence": 0.9,
    },
    {
        "message": "make petal_length more important",
        "category": "on_topic_actionable",
        "reply_text": None,
        "confidence": 0.93,
    },
    {
        "message": "move these together",  # ambiguous when no selection exists
        "category": "on_topic_ambiguous",
        "reply_text": None,
        "clarification_question": "Which points should I use? Nothing is selected right now.",
        "confidence": 0.65,
    },
    {
        "message": "thanks, that's cool",
        "category": "off_topic",
        "reply_text": "No refinement to apply here.",
        "confidence": 0.95,
    },
]


_EXTRACT_FEWSHOT = [
    {
        "message": "merge clusters 1 and 2",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "merge_clusters",
                "payload": {
                    "target_groups": [
                        {"source": "cluster", "ref": "cluster_1"},
                        {"source": "cluster", "ref": "cluster_2"},
                    ]
                },
            }
        ],
    },
    {
        "message": "make petal_length more important",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "feature_weight",
                "payload": {"feature": "petal_length", "direction": "increase"},
            }
        ],
    },
    {
        "message": "these points should belong together",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "group_similar",
                "payload": {
                    "group_a": {"source": "selected_points", "ref": "current"},
                    "group_b": None,
                },
            }
        ],
    },
    {
        "message": "those points should be in one class",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "merge_clusters",
                "payload": {
                    "target_groups": [
                        {"source": "cluster", "ref": "cluster_1"},
                        {"source": "cluster", "ref": "cluster_2"},
                    ]
                },
            }
        ],
    },
    {
        "message": "group the selected points with cluster_2",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "group_similar",
                "payload": {
                    "group_a": {"source": "selected_points", "ref": "current"},
                    "group_b": {"source": "cluster", "ref": "cluster_2"},
                },
            }
        ],
    },
    {
        "message": "push cluster_1 and cluster_3 apart",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "group_dissimilar",
                "payload": {
                    "group_a": {"source": "cluster", "ref": "cluster_1"},
                    "group_b": {"source": "cluster", "ref": "cluster_3"},
                },
            }
        ],
    },
    {
        "message": "separate these points from cluster_3",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "group_dissimilar",
                "payload": {
                    "group_a": {"source": "selected_points", "ref": "current"},
                    "group_b": {"source": "cluster", "ref": "cluster_3"},
                },
            }
        ],
    },
    {
        "message": "ignore cluster 3 for now",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "ignore_cluster",
                "payload": {
                    "target_groups": [{"source": "cluster", "ref": "cluster_3"}]
                },
            }
        ],
    },
    {
        "message": "the selected point should be an outlier",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "reclassify_outlier",
                "payload": {
                    "anchor": {"source": "selected_points", "ref": "current"},
                    "is_outlier": True,
                },
            }
        ],
    },
    {
        "message": "mark the selection as not an outlier",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "reclassify_outlier",
                "payload": {
                    "anchor": {"source": "selected_points", "ref": "current"},
                    "is_outlier": False,
                },
            }
        ],
    },
    {
        "message": "there should be no outlier",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "reclassify_outlier",
                "payload": {
                    "anchor": {"source": "selected_points", "ref": "current"},
                    "is_outlier": False,
                },
            }
        ],
    },
    {
        "message": "none of these are outliers",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "reclassify_outlier",
                "payload": {
                    "anchor": {"source": "selected_points", "ref": "current"},
                    "is_outlier": False,
                },
            }
        ],
    },
    {
        "message": "use the selected point as the anchor for cluster_2",
        "operations": [
            {
                "op": "add",
                "constraint_id": "pending",
                "intent": "anchor_point",
                "payload": {
                    "anchor": {"source": "selected_points", "ref": "current"},
                    "target_groups": [{"source": "cluster", "ref": "cluster_2"}],
                },
            }
        ],
    },
]


def _route_prompt(
    message: str,
    context: DatasetContext,
    history: Sequence[Turn],
    memory_context: Mapping[str, object],
) -> str:
    briefing = _build_briefing(context)
    memory_summary = _compact_memory_summary(memory_context)
    history_text = _history_text(history)
    fewshot_text = "\n\n".join(
        f"User: {example['message']}\n"
        f"JSON: {json.dumps({k: v for k, v in example.items() if k != 'message'}, ensure_ascii=True)}"
        for example in _ROUTE_FEWSHOT
    )
    return f"""You are helping a researcher refine a scatterplot clustering.
Your job is to classify the next user message into exactly one category so the system
knows how to respond.

Categories:
- on_topic_actionable: the user is asking for a refinement (merge, split, weight a feature,
  group points together / apart, ignore a cluster, anchor a point, reclassify an outlier).
- on_topic_ambiguous: the user wants a refinement but the message lacks grounding
  (e.g. "move these together" when nothing is selected).
- meta_query: the user is ASKING about the current state rather than requesting a change.
  This includes:
    * counts ("how many clusters / classes / outliers / selected points")
    * identity / membership ("what class are the selected points",
      "which cluster is p3 in", "what cluster do these points belong to")
    * listings ("what features are there", "list the clusters", "which points are outliers")
    * yes/no state checks ("is outlier_east really an outlier", "is p1 selected")
  Any question about what is currently true in the grounded state is meta_query,
  even if it names selected points, clusters, or outliers.
- off_topic: social chatter, thanks, unrelated tasks, or anything that is neither a
  refinement request nor a question about the grounded state. Do NOT use off_topic for
  questions about clusters / classes / selection / outliers — those are meta_query.

Paraphrase rules:
- "cluster", "class", "category", "group", "label" usually refer to the same clustering.
- "outlier", "anomaly", "noise" mean the outlier set.
- "together" / "belong together" / "same group" → on_topic_actionable (group_similar).
- "apart" / "push apart" / "separate" → on_topic_actionable (group_dissimilar).
- A question ("how many", "what", "which", "where", "is", "are", "do", "list", "show",
  "tell me") about counts, membership, features, selection, or outliers → meta_query.
- If the message is a question AND it mentions a term from the grounded state (cluster,
  class, feature, selected, outlier, point), default to meta_query unless it is clearly
  a request to change something ("merge", "split", "make ... more important", etc.).

For meta_query you MUST draft a short natural language reply_text that directly answers
the question using the grounded state below. Use the real cluster ids, feature names,
and point ids from the grounded state — do not invent names. For off_topic you may set
reply_text to a brief polite refusal. For on_topic_* leave reply_text null.

Grounded state:
{briefing}

Structured memory (summary only):
{memory_summary}

Recent conversation:
{history_text}

Examples:

{fewshot_text}

Now classify this user message. Respond with ONLY a single JSON object with exactly these keys:
{{"category": "...", "confidence": 0.0, "reason": "...", "clarification_question": null, "reply_text": null}}

User: {json.dumps(message, ensure_ascii=True)}
JSON:"""


def _extract_prompt(
    message: str,
    context: DatasetContext,
    history: Sequence[Turn],
    current_instruction: StructuredInstruction,
    memory_context: Mapping[str, object],
) -> str:
    briefing = _build_briefing(context)
    memory_summary = _compact_memory_summary(memory_context)
    history_text = _history_text(history)
    instruction_summary = _compact_instruction_summary(current_instruction)
    fewshot_text = "\n\n".join(
        f"User: {example['message']}\n"
        f"JSON: {json.dumps({'operations': example['operations']}, ensure_ascii=True)}"
        for example in _EXTRACT_FEWSHOT
    )
    return f"""You convert a user message into a structured instruction delta for a
scatterplot clustering refinement system.

Allowed intents (pick exactly one that best matches the user's request):
- feature_weight: raise/lower/ignore a specific feature's influence.
- group_similar: two groups should be pulled closer.
- group_dissimilar: two groups should be pushed apart.
- merge_clusters: two or more existing clusters should become one.
- split_cluster: one existing cluster should be split.
- ignore_cluster: one or more clusters should be dropped from the analysis.
- anchor_point: a specific point represents a cluster.
- reclassify_outlier: a point (or the current selection as a whole) should be marked
  as an outlier or as not an outlier. For collective phrases like "there should be no
  outlier", "none of these are outliers", or "all of these are outliers", emit a single
  reclassify_outlier op with anchor={{"source":"selected_points","ref":"current"}} and
  the correct is_outlier flag — the downstream system expands it across the selection.

Group reference sources (use only these strings for `source`):
- cluster: ref is a cluster id from the grounded state (e.g. "cluster_2").
- outlier_set: ref is an outlier point id from the grounded state.
- selection_group: ref is a saved selection group name.
- selected_points: ref is the literal string "current" when the user points at the
  current selection ("these points", "the selection", "this group").
- point_id: ref is a specific point id visible in the grounded state.

Rules:
- Emit exactly one "add" operation with constraint_id="pending" for the single intent
  the user expressed. If nothing actionable is present, return {{"operations": []}}.
- Never invent cluster ids, point ids, feature names, group names, or outlier ids —
  only use values that appear in the grounded state below or that the user names
  explicitly.
- Pronoun resolution: "these", "those", "them", "those points", "these points",
  "that group", "they" with an active selection → {{"source": "selected_points", "ref": "current"}}.
  Use the recent conversation to resolve what "those" / "them" refers to.
- group_similar vs merge_clusters:
    * group_similar: the user wants specific POINTS or a SELECTION pulled closer together
      ("those points should be together", "put them in the same group", "these belong
      in one class"). Use selected_points or point_id refs.
    * merge_clusters: the user explicitly names two CLUSTER IDs that should become one
      ("merge cluster_1 and cluster_2"). Use cluster refs.
    * When the user says "those points should be in one class" and there IS a current
      selection, use group_similar — NOT merge_clusters.
- If grounding for a slot is missing (e.g. user said "merge clusters" without naming
  which two), leave that slot null or target_groups as [] — do not guess.
- Use the structured memory below only to resolve a user message that answers a
  previously-asked clarification (e.g. the prior draft was an incomplete
  reclassify_outlier and the user now names a point).

Payload shapes (match exactly):
- feature_weight: {{"feature": "<name>", "direction": "increase"|"decrease"|"ignore"}}
- group_similar / group_dissimilar: {{"group_a": GroupRef|null, "group_b": GroupRef|null}}
- merge_clusters / split_cluster / ignore_cluster: {{"target_groups": [GroupRef, ...]}}
- anchor_point: {{"anchor": GroupRef|null, "target_groups": [GroupRef, ...]}}
- reclassify_outlier: {{"anchor": GroupRef|null, "is_outlier": true|false}}

A GroupRef is {{"source": "...", "ref": "..."}}.

Grounded state:
{briefing}

Current compiled instruction:
{instruction_summary}

Structured memory (summary only):
{memory_summary}

Recent conversation:
{history_text}

Examples:

{fewshot_text}

Now extract the delta for this user message. Respond with ONLY a single JSON object:
{{"operations": [{{"op": "add", "constraint_id": "pending", "intent": "...", "payload": {{...}}}}]}}
or {{"operations": []}} if nothing actionable is present.

User: {json.dumps(message, ensure_ascii=True)}
JSON:"""


def _build_briefing(context: DatasetContext) -> str:
    lines = [f"- dataset_id: {context.dataset_id}"]
    if context.feature_names:
        lines.append(f"- features ({len(context.feature_names)}): {', '.join(context.feature_names)}")
    else:
        lines.append("- features: (none)")

    feature_ranges = context.analysis_context.get("feature_ranges") if isinstance(context.analysis_context, Mapping) else None
    if isinstance(feature_ranges, Mapping) and feature_ranges:
        stat_lines = []
        for name in context.feature_names:
            stats = feature_ranges.get(name)
            if isinstance(stats, Mapping):
                stat_lines.append(
                    f"    {name}: min={stats.get('min')} mean={stats.get('mean')} max={stats.get('max')}"
                )
        if stat_lines:
            lines.append("- feature_stats:")
            lines.extend(stat_lines)

    cluster_sizes = context.analysis_context.get("cluster_sizes") if isinstance(context.analysis_context, Mapping) else None
    if context.cluster_ids:
        if isinstance(cluster_sizes, Mapping) and cluster_sizes:
            joined = ", ".join(
                f"{cluster_id}({cluster_sizes.get(cluster_id, 0)})"
                for cluster_id in context.cluster_ids
            )
        else:
            joined = ", ".join(context.cluster_ids)
        lines.append(f"- clusters ({len(context.cluster_ids)}): {joined}")
    else:
        lines.append("- clusters: (none resolved yet)")

    if context.outlier_point_ids:
        preview = ", ".join(context.outlier_point_ids[:8])
        more = "" if len(context.outlier_point_ids) <= 8 else f", ... (+{len(context.outlier_point_ids) - 8} more)"
        lines.append(f"- outliers ({len(context.outlier_point_ids)}): {preview}{more}")
    else:
        lines.append("- outliers: (none)")

    if context.selected_point_ids:
        preview = ", ".join(context.selected_point_ids[:8])
        more = "" if len(context.selected_point_ids) <= 8 else f", ... (+{len(context.selected_point_ids) - 8} more)"
        lines.append(f"- current selection ({len(context.selected_point_ids)} points): {preview}{more}")
        selected_clusters = (
            context.analysis_context.get("selected_point_clusters")
            if isinstance(context.analysis_context, Mapping)
            else None
        )
        if isinstance(selected_clusters, Sequence) and selected_clusters:
            detail_lines = []
            for entry in selected_clusters[:8]:
                if not isinstance(entry, Mapping):
                    continue
                point_id = entry.get("point_id")
                cluster_id = entry.get("cluster_id") or "(unassigned)"
                outlier_flag = " [outlier]" if entry.get("is_outlier") else ""
                features = entry.get("features") or {}
                if isinstance(features, Mapping) and features:
                    feature_text = ", ".join(f"{name}={value}" for name, value in features.items())
                    detail_lines.append(f"    {point_id}: {cluster_id}{outlier_flag} ({feature_text})")
                else:
                    detail_lines.append(f"    {point_id}: {cluster_id}{outlier_flag}")
            if detail_lines:
                lines.append("- selected points by cluster:")
                lines.extend(detail_lines)
    else:
        lines.append("- current selection: (empty)")

    if context.selection_group_names:
        lines.append(f"- saved selection groups: {', '.join(context.selection_group_names)}")
    else:
        lines.append("- saved selection groups: (none)")

    label_summary = context.analysis_context.get("label_summary") if isinstance(context.analysis_context, Mapping) else None
    if isinstance(label_summary, Mapping) and label_summary:
        joined = ", ".join(f"{key}={value}" for key, value in label_summary.items())
        lines.append(f"- manual labels: {joined}")
    elif context.label_annotations:
        lines.append(f"- manual labels: {len(context.label_annotations)} annotation(s)")
    else:
        lines.append("- manual labels: (none)")

    return "\n".join(lines)


def _compact_memory_summary(memory_context: Mapping[str, object]) -> str:
    if not isinstance(memory_context, Mapping) or not memory_context:
        return "(no memory)"

    lines = []
    summary = memory_context.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(f"- summary: {summary.strip()}")

    draft_state = memory_context.get("draft_state")
    if isinstance(draft_state, Mapping) and draft_state.get("candidate_intent"):
        candidate = draft_state.get("candidate_intent")
        pending = draft_state.get("pending_clarification")
        missing = draft_state.get("unresolved_slots") or []
        missing_text = ", ".join(str(slot) for slot in missing) if missing else "none"
        lines.append(
            f"- active_draft: intent={candidate} missing_slots=[{missing_text}]"
            + (f" clarification={pending!r}" if pending else "")
        )

    confirmed = memory_context.get("confirmed_facts") or []
    if isinstance(confirmed, Sequence) and not isinstance(confirmed, (str, bytes)) and confirmed:
        lines.append(f"- confirmed_facts: {len(confirmed)} applied constraint(s)")

    relevant = memory_context.get("relevant_turns") or []
    if isinstance(relevant, Sequence) and not isinstance(relevant, (str, bytes)) and relevant:
        lines.append(f"- relevant_turns: last {len(relevant)} shown in conversation below")

    if not lines:
        return "(no relevant memory)"
    return "\n".join(lines)


def _history_text(history: Sequence[Turn]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{turn.role}: {turn.text}" for turn in history)


def _compact_instruction_summary(current_instruction: StructuredInstruction) -> str:
    constraints = current_instruction.constraints
    if not constraints:
        return f"(version={current_instruction.version}, no constraints yet)"
    lines = [f"(version={current_instruction.version}, {len(constraints)} constraint(s))"]
    for entry in constraints:
        lines.append(f"- {entry.id}: {entry.intent} from \"{entry.raw_message}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File-backed prompt templates and richer grounding context
# ---------------------------------------------------------------------------


def _route_prompt(
    message: str,
    context: DatasetContext,
    history: Sequence[Turn],
    memory_context: Mapping[str, object],
) -> str:
    fewshot_text = "\n\n".join(
        f"User: {example['message']}\n"
        f"JSON: {json.dumps({k: v for k, v in example.items() if k != 'message'}, ensure_ascii=True)}"
        for example in _ROUTE_FEWSHOT
    )
    return _ROUTE_PROMPT_TEMPLATE.substitute(
        briefing=_build_briefing(context),
        context_json=json.dumps(context.to_dict(), ensure_ascii=True, indent=2),
        memory_summary=_compact_memory_summary(memory_context),
        memory_json=json.dumps(memory_context, ensure_ascii=True, indent=2),
        history_text=_history_text(history),
        history_json=json.dumps([turn.to_dict() for turn in history], ensure_ascii=True, indent=2),
        fewshot_text=fewshot_text,
        message_json=json.dumps(message, ensure_ascii=True),
    )


def _extract_prompt(
    message: str,
    context: DatasetContext,
    history: Sequence[Turn],
    current_instruction: StructuredInstruction,
    memory_context: Mapping[str, object],
) -> str:
    fewshot_text = "\n\n".join(
        f"User: {example['message']}\n"
        f"JSON: {json.dumps({'operations': example['operations']}, ensure_ascii=True)}"
        for example in _EXTRACT_FEWSHOT
    )
    return _EXTRACT_PROMPT_TEMPLATE.substitute(
        briefing=_build_briefing(context),
        context_json=json.dumps(context.to_dict(), ensure_ascii=True, indent=2),
        instruction_summary=_compact_instruction_summary(current_instruction),
        instruction_json=json.dumps(current_instruction.to_dict(), ensure_ascii=True, indent=2),
        memory_summary=_compact_memory_summary(memory_context),
        memory_json=json.dumps(memory_context, ensure_ascii=True, indent=2),
        history_text=_history_text(history),
        history_json=json.dumps([turn.to_dict() for turn in history], ensure_ascii=True, indent=2),
        fewshot_text=fewshot_text,
        message_json=json.dumps(message, ensure_ascii=True),
    )


def _build_briefing(context: DatasetContext) -> str:
    lines = [f"- dataset_id: {context.dataset_id}"]
    if context.feature_names:
        lines.append(f"- features ({len(context.feature_names)}): {', '.join(context.feature_names)}")
    else:
        lines.append("- features: (none)")

    feature_ranges = context.analysis_context.get("feature_ranges") if isinstance(context.analysis_context, Mapping) else None
    if isinstance(feature_ranges, Mapping) and feature_ranges:
        stat_lines = []
        for name in context.feature_names:
            stats = feature_ranges.get(name)
            if isinstance(stats, Mapping):
                stat_lines.append(
                    f"    {name}: min={stats.get('min')} mean={stats.get('mean')} max={stats.get('max')}"
                )
        if stat_lines:
            lines.append("- feature_stats:")
            lines.extend(stat_lines)

    cluster_sizes = context.analysis_context.get("cluster_sizes") if isinstance(context.analysis_context, Mapping) else None
    if context.cluster_ids:
        if isinstance(cluster_sizes, Mapping) and cluster_sizes:
            joined = ", ".join(
                f"{cluster_id}({cluster_sizes.get(cluster_id, 0)})"
                for cluster_id in context.cluster_ids
            )
        else:
            joined = ", ".join(context.cluster_ids)
        lines.append(f"- clusters ({len(context.cluster_ids)}): {joined}")
    else:
        lines.append("- clusters: (none resolved yet)")

    point_catalog = context.analysis_context.get("point_catalog") if isinstance(context.analysis_context, Mapping) else None
    if isinstance(point_catalog, Sequence) and not isinstance(point_catalog, (str, bytes)) and point_catalog:
        lines.append(f"- visible point catalog ({len(point_catalog)} points):")
        for entry in point_catalog[:12]:
            if not isinstance(entry, Mapping):
                continue
            point_id = entry.get("point_id")
            cluster_id = entry.get("cluster_id") or "(unassigned)"
            outlier_flag = " [outlier]" if entry.get("is_outlier") else ""
            features = entry.get("features") or {}
            feature_text = ""
            if isinstance(features, Mapping) and features:
                feature_text = " (" + ", ".join(f"{name}={value}" for name, value in features.items()) + ")"
            lines.append(f"    {point_id}: {cluster_id}{outlier_flag}{feature_text}")

    if context.outlier_point_ids:
        preview = ", ".join(context.outlier_point_ids[:8])
        more = "" if len(context.outlier_point_ids) <= 8 else f", ... (+{len(context.outlier_point_ids) - 8} more)"
        lines.append(f"- outliers ({len(context.outlier_point_ids)}): {preview}{more}")
    else:
        lines.append("- outliers: (none)")

    if context.selected_point_ids:
        preview = ", ".join(context.selected_point_ids[:8])
        more = "" if len(context.selected_point_ids) <= 8 else f", ... (+{len(context.selected_point_ids) - 8} more)"
        lines.append(f"- current selection ({len(context.selected_point_ids)} points): {preview}{more}")
        selected_clusters = (
            context.analysis_context.get("selected_point_clusters")
            if isinstance(context.analysis_context, Mapping)
            else None
        )
        if isinstance(selected_clusters, Sequence) and selected_clusters:
            detail_lines = []
            for entry in selected_clusters[:8]:
                if not isinstance(entry, Mapping):
                    continue
                point_id = entry.get("point_id")
                cluster_id = entry.get("cluster_id") or "(unassigned)"
                outlier_flag = " [outlier]" if entry.get("is_outlier") else ""
                features = entry.get("features") or {}
                if isinstance(features, Mapping) and features:
                    feature_text = ", ".join(f"{name}={value}" for name, value in features.items())
                    detail_lines.append(f"    {point_id}: {cluster_id}{outlier_flag} ({feature_text})")
                else:
                    detail_lines.append(f"    {point_id}: {cluster_id}{outlier_flag}")
            if detail_lines:
                lines.append("- selected points by cluster:")
                lines.extend(detail_lines)
    else:
        lines.append("- current selection: (empty)")

    if context.selection_group_names:
        lines.append(f"- saved selection groups: {', '.join(context.selection_group_names)}")
    else:
        lines.append("- saved selection groups: (none)")

    if context.selection_groups:
        point_to_cluster = (
            context.analysis_context.get("point_to_cluster")
            if isinstance(context.analysis_context, Mapping)
            else None
        )
        lines.append("- saved selection group details:")
        for group in context.selection_groups[:8]:
            if not isinstance(group, Mapping):
                continue
            group_name = group.get("group_name") or group.get("group_id") or "(unnamed)"
            point_ids = [str(point_id) for point_id in (group.get("point_ids") or [])[:8]]
            preview = ", ".join(point_ids) if point_ids else "(no points)"
            cluster_summary = []
            if isinstance(point_to_cluster, Mapping):
                for point_id in point_ids:
                    cluster_id = point_to_cluster.get(point_id)
                    if cluster_id and cluster_id not in cluster_summary:
                        cluster_summary.append(cluster_id)
            cluster_text = (
                f" clusters={', '.join(str(cluster_id) for cluster_id in cluster_summary)}"
                if cluster_summary
                else ""
            )
            lines.append(f"    {group_name}: {preview}{cluster_text}")

    label_summary = context.analysis_context.get("label_summary") if isinstance(context.analysis_context, Mapping) else None
    if isinstance(label_summary, Mapping) and label_summary:
        joined = ", ".join(f"{key}={value}" for key, value in label_summary.items())
        lines.append(f"- manual labels: {joined}")
    elif context.label_annotations:
        lines.append(f"- manual labels: {len(context.label_annotations)} annotation(s)")
    else:
        lines.append("- manual labels: (none)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory-aware normalization (kept minimal; does NOT override LLM output)
# ---------------------------------------------------------------------------


_QUESTION_PREFIXES = (
    "what", "which", "where", "who", "whose", "whom", "when", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "should", "will", "shall", "list", "show", "tell", "give",
)

_META_QUERY_TOPIC_TOKENS = (
    "cluster", "clusters", "class", "classes", "category", "categories",
    "group", "groups", "label", "labels",
    "feature", "features", "dimension", "dimensions",
    "outlier", "outliers", "anomaly", "anomalies", "noise",
    "point", "points", "sample", "samples",
    "selected", "selection",
)


def _upgrade_offtopic_to_meta_query(
    result: RouterResult,
    message: str,
) -> RouterResult:
    """Correct the narrow case where the LLM labels a clear meta question as
    ``off_topic``. Fires only when the message looks like a question AND
    mentions a grounded domain term (cluster/class/feature/selection/outlier/
    point) — so genuine social chatter stays off_topic.
    """
    if result.category != "off_topic":
        return result
    text = (message or "").strip().lower()
    if not text:
        return result
    is_question = text.endswith("?") or any(
        text == prefix or text.startswith(f"{prefix} ")
        for prefix in _QUESTION_PREFIXES
    )
    if not is_question:
        return result
    mentions_topic = any(token in text for token in _META_QUERY_TOPIC_TOKENS)
    if not mentions_topic:
        return result
    return RouterResult(
        category="meta_query",
        confidence=max(result.confidence, 0.7),
        reason="question about grounded state misrouted as off_topic",
        clarification_question=None,
        reply_text=result.reply_text,
    )


def _upgrade_router_with_draft(
    result: RouterResult,
    message: str,
    context: DatasetContext,
    memory_context: Mapping[str, object],
) -> RouterResult:
    """Upgrade an ambiguous route to actionable when the message clearly answers an
    open clarification slot. This does not override already-actionable or meta/off
    decisions from the LLM — it only helps resolve the "user just typed a slot value"
    case.
    """
    if result.category == "on_topic_actionable":
        return result
    if result.category in {"meta_query", "off_topic"}:
        return result
    if not _message_resolves_active_draft(message, context, memory_context):
        return result
    return RouterResult(
        category="on_topic_actionable",
        confidence=max(result.confidence, 0.84),
        reason="resolved active draft slot answer",
        clarification_question=None,
        reply_text=result.reply_text,
    )


def _backfill_delta_from_draft(
    message: str,
    delta: InstructionDelta,
    context: DatasetContext,
    memory_context: Mapping[str, object],
) -> InstructionDelta:
    """When the LLM returned an empty delta but the user message is an answer to an
    open clarification slot, synthesize a single-add delta from the active draft.

    Does NOT change the intent the LLM chose when it returned a non-empty delta —
    only fills null slots from the draft when the LLM and draft agree on intent.
    """
    draft_state = _draft_state(memory_context) or {}
    candidate_intent = _optional_text(draft_state.get("candidate_intent"))

    if delta.is_empty:
        if not candidate_intent:
            return delta
        synthesized = _delta_from_active_draft(message, context, draft_state)
        return synthesized if not synthesized.is_empty else delta

    if not candidate_intent:
        return delta

    repaired = []
    for op in delta.operations:
        if op.intent != candidate_intent:
            repaired.append(op)
            continue
        payload = dict(op.payload or {})
        if candidate_intent == "reclassify_outlier":
            payload = _fill_reclassify_slots(payload, message, context, draft_state)
        elif candidate_intent in {"group_similar", "group_dissimilar"}:
            payload = _fill_pair_slots(payload, context, draft_state)
        elif candidate_intent in {"merge_clusters", "split_cluster", "ignore_cluster"}:
            payload = _fill_cluster_slots(payload, context, draft_state)
        elif candidate_intent == "anchor_point":
            payload = _fill_anchor_slots(payload, context, draft_state)
        repaired.append(
            DeltaOperation(
                op=op.op,
                constraint_id=op.constraint_id,
                intent=op.intent,
                payload=payload,
            )
        )
    return InstructionDelta(operations=tuple(repaired))


def _draft_state(memory_context: Mapping[str, object]) -> Mapping[str, object] | None:
    draft_state = memory_context.get("draft_state") if isinstance(memory_context, Mapping) else None
    return draft_state if isinstance(draft_state, Mapping) else None


def _message_resolves_active_draft(
    message: str,
    context: DatasetContext,
    memory_context: Mapping[str, object],
) -> bool:
    draft_state = _draft_state(memory_context)
    if not isinstance(draft_state, Mapping):
        return False
    if not draft_state.get("pending_clarification"):
        return False
    candidate_intent = _optional_text(draft_state.get("candidate_intent"))
    if not candidate_intent:
        return False
    if candidate_intent == "feature_weight":
        return _match_feature_name(message, context) is not None
    if candidate_intent == "reclassify_outlier":
        return bool(_point_like_refs(message, context))
    if candidate_intent in {"group_similar", "group_dissimilar", "anchor_point"}:
        return bool(_message_group_refs(message, context))
    if candidate_intent in {"merge_clusters", "split_cluster", "ignore_cluster"}:
        return bool(_cluster_refs_from_message(message, context))
    return False


def _delta_from_active_draft(
    message: str,
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> InstructionDelta:
    candidate_intent = _optional_text(draft_state.get("candidate_intent"))
    if not candidate_intent:
        return InstructionDelta(operations=())

    draft_payload = _draft_payload(draft_state)
    if candidate_intent == "reclassify_outlier":
        payload = _fill_reclassify_slots(draft_payload, message, context, draft_state)
        return _single_add_delta(candidate_intent, payload)
    if candidate_intent in {"group_similar", "group_dissimilar"}:
        payload = _fill_pair_slots_with_message(draft_payload, message, context, draft_state)
        return _single_add_delta(candidate_intent, payload)
    if candidate_intent in {"merge_clusters", "split_cluster", "ignore_cluster"}:
        payload = _fill_cluster_slots_with_message(draft_payload, message, context, draft_state)
        return _single_add_delta(candidate_intent, payload)
    if candidate_intent == "anchor_point":
        payload = _fill_anchor_slots_with_message(draft_payload, message, context, draft_state)
        return _single_add_delta(candidate_intent, payload)
    if candidate_intent == "feature_weight":
        feature = _match_feature_name(message, context)
        if feature is None:
            return InstructionDelta(operations=())
        payload = dict(draft_payload)
        payload["feature"] = feature
        return _single_add_delta(candidate_intent, payload)
    return InstructionDelta(operations=())


def _single_add_delta(intent: str, payload: Mapping[str, Any]) -> InstructionDelta:
    return InstructionDelta(
        operations=(
            DeltaOperation(
                op="add",
                constraint_id="pending",
                intent=intent,
                payload=dict(payload),
            ),
        )
    )


def _draft_payload(draft_state: Mapping[str, object]) -> Dict[str, Any]:
    proposed_delta = draft_state.get("proposed_delta")
    if not isinstance(proposed_delta, Mapping):
        return {}
    operations = proposed_delta.get("operations")
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)) or not operations:
        return {}
    first = operations[0]
    if not isinstance(first, Mapping):
        return {}
    payload = first.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _fill_reclassify_slots(
    payload: Mapping[str, Any],
    message: str,
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> Dict[str, Any]:
    anchor = _coerce_reclassify_anchor(payload.get("anchor"), context)
    if anchor is None:
        for ref in _point_like_refs(message, context):
            anchor = _coerce_reclassify_anchor(ref, context)
            if anchor is not None:
                break
    if anchor is None:
        draft_ref = _first_draft_ref_by_source(draft_state, context, {"point_id", "selected_points"})
        anchor = _coerce_reclassify_anchor(draft_ref, context)
    return {
        "anchor": anchor,
        "is_outlier": _resolve_is_outlier(payload.get("is_outlier"), message, draft_state),
    }


def _fill_pair_slots(
    payload: Mapping[str, Any],
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> Dict[str, Any]:
    group_a = _normalize_group_ref(payload.get("group_a"), context)
    group_b = _normalize_group_ref(payload.get("group_b"), context)
    draft_ref = _first_draft_ref(draft_state, context)
    if group_a is None and draft_ref is not None:
        group_a = draft_ref
    return {"group_a": group_a, "group_b": group_b}


def _fill_pair_slots_with_message(
    payload: Mapping[str, Any],
    message: str,
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> Dict[str, Any]:
    refs = _message_group_refs(message, context)
    group_a = _normalize_group_ref(payload.get("group_a"), context)
    group_b = _normalize_group_ref(payload.get("group_b"), context)
    draft_ref = _first_draft_ref(draft_state, context)

    if group_a is None and draft_ref is not None:
        group_a = draft_ref
    for ref in refs:
        if group_a is None:
            group_a = ref
            continue
        if group_b is None and ref != group_a:
            group_b = ref
            break
    return {"group_a": group_a, "group_b": group_b}


def _fill_cluster_slots(
    payload: Mapping[str, Any],
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> Dict[str, Any]:
    target_groups = _normalize_group_ref_list(payload.get("target_groups"), context)
    target_groups.extend(_draft_refs_by_source(draft_state, context, {"cluster"}))
    return {"target_groups": _dedupe_refs(target_groups)}


def _fill_cluster_slots_with_message(
    payload: Mapping[str, Any],
    message: str,
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> Dict[str, Any]:
    target_groups = _normalize_group_ref_list(payload.get("target_groups"), context)
    target_groups.extend(_cluster_refs_from_message(message, context))
    target_groups.extend(_draft_refs_by_source(draft_state, context, {"cluster"}))
    return {"target_groups": _dedupe_refs(target_groups)}


def _fill_anchor_slots(
    payload: Mapping[str, Any],
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> Dict[str, Any]:
    anchor = _normalize_group_ref(payload.get("anchor"), context)
    cluster_targets = _normalize_group_ref_list(payload.get("target_groups"), context)
    if anchor is None:
        draft_anchor = _first_draft_ref_by_source(draft_state, context, {"point_id", "selected_points"})
        if draft_anchor is not None:
            anchor = draft_anchor
    cluster_targets.extend(_draft_refs_by_source(draft_state, context, {"cluster"}))
    return {"anchor": anchor, "target_groups": _dedupe_refs(cluster_targets)}


def _fill_anchor_slots_with_message(
    payload: Mapping[str, Any],
    message: str,
    context: DatasetContext,
    draft_state: Mapping[str, object],
) -> Dict[str, Any]:
    anchor = _normalize_group_ref(payload.get("anchor"), context)
    cluster_targets = _normalize_group_ref_list(payload.get("target_groups"), context)

    for ref in _message_group_refs(message, context):
        if anchor is None and ref.get("source") in {"point_id", "selected_points"}:
            anchor = ref
        elif ref.get("source") == "cluster":
            cluster_targets.append(ref)
    if anchor is None:
        draft_anchor = _first_draft_ref_by_source(draft_state, context, {"point_id", "selected_points"})
        if draft_anchor is not None:
            anchor = draft_anchor
    cluster_targets.extend(_draft_refs_by_source(draft_state, context, {"cluster"}))
    return {"anchor": anchor, "target_groups": _dedupe_refs(cluster_targets)}


def _coerce_reclassify_anchor(
    value: Any,
    context: DatasetContext,
) -> Dict[str, str] | None:
    anchor = _normalize_group_ref(value, context)
    if anchor is None:
        return None
    if anchor.get("source") == "selected_points" and anchor.get("ref") == "current":
        if len(context.selected_point_ids) == 1:
            return {"source": "point_id", "ref": context.selected_point_ids[0]}
        return None
    if anchor.get("source") == "point_id":
        return anchor
    return None


def _resolve_is_outlier(
    current_value: Any,
    message: str,
    draft_state: Mapping[str, object],
) -> bool:
    if isinstance(current_value, bool):
        return current_value
    message_value = _message_is_outlier(message)
    if message_value is not None:
        return message_value
    draft_payload = _draft_payload(draft_state)
    if isinstance(draft_payload.get("is_outlier"), bool):
        return bool(draft_payload["is_outlier"])
    return True


def _message_is_outlier(message: str) -> bool | None:
    text = (message or "").strip().lower()
    if not text:
        return None
    if any(token in text for token in ("not an outlier", "not outlier", "non-outlier", "non outlier")):
        return False
    if "outlier" in text:
        return True
    return None


_SELECTION_REFERENCE_TOKENS = (
    "selected point",
    "selected points",
    "current selection",
    "this selection",
    "the selection",
    "these points",
    "those points",
    "these",
    "them",
)


def _point_like_refs(message: str, context: DatasetContext) -> list[Dict[str, str]]:
    refs = _point_refs_from_message(message, context)
    if refs:
        return refs
    if _references_current_selection(message, context):
        return [{"source": "selected_points", "ref": "current"}]
    return []


def _message_group_refs(message: str, context: DatasetContext) -> list[Dict[str, str]]:
    refs = []
    refs.extend(_point_refs_from_message(message, context))
    if _references_current_selection(message, context):
        refs.append({"source": "selected_points", "ref": "current"})
    refs.extend(_selection_group_refs_from_message(message, context))
    refs.extend(_cluster_refs_from_message(message, context))
    return _dedupe_refs(refs)


def _point_refs_from_message(message: str, context: DatasetContext) -> list[Dict[str, str]]:
    text = (message or "").strip().lower()
    if not text:
        return []
    tokens = {
        token
        for token in re.split(r"[\s,;:.!?()]+", text)
        if token
    }
    point_lookup = {
        point_id.lower(): point_id
        for point_id in (*context.selected_point_ids, *context.unselected_point_ids)
    }
    refs = []
    for token in tokens:
        point_id = point_lookup.get(token)
        if point_id:
            refs.append({"source": "point_id", "ref": point_id})
    return _dedupe_refs(refs)


def _selection_group_refs_from_message(message: str, context: DatasetContext) -> list[Dict[str, str]]:
    text = (message or "").strip().lower()
    refs = []
    for group_name in context.selection_group_names:
        if group_name and group_name.lower() in text:
            refs.append({"source": "selection_group", "ref": group_name})
    return _dedupe_refs(refs)


def _cluster_refs_from_message(message: str, context: DatasetContext) -> list[Dict[str, str]]:
    text = (message or "").strip().lower()
    if not text:
        return []
    numeric_suffixes: set[str] = set()
    for match in re.finditer(r"clusters?\s+(\d+(?:\s*(?:,|and|or|&)\s*\d+)*)", text):
        numeric_suffixes.update(re.findall(r"\d+", match.group(1)))
    refs = []
    for cluster_id in context.cluster_ids:
        lower_id = cluster_id.lower()
        suffix = cluster_id.split("_")[-1]
        matches = (
            lower_id in text
            or lower_id.replace("cluster_", "cluster ") in text
            or (suffix and suffix in numeric_suffixes)
        )
        if matches:
            refs.append({"source": "cluster", "ref": cluster_id})
    return _dedupe_refs(refs)


def _references_current_selection(message: str, context: DatasetContext) -> bool:
    text = (message or "").strip().lower()
    return bool(context.selected_point_ids) and any(token in text for token in _SELECTION_REFERENCE_TOKENS)


def _match_feature_name(message: str, context: DatasetContext) -> str | None:
    text = (message or "").strip().lower()
    for feature_name in context.feature_names:
        if feature_name.lower() in text:
            return feature_name
    return None


def _first_draft_ref(draft_state: Mapping[str, object], context: DatasetContext) -> Dict[str, str] | None:
    grounded_refs = draft_state.get("grounded_refs") or ()
    if not isinstance(grounded_refs, Sequence) or isinstance(grounded_refs, (str, bytes)):
        return None
    for item in grounded_refs:
        normalized = _normalize_group_ref(item, context)
        if normalized is not None:
            return normalized
    return None


def _first_draft_ref_by_source(
    draft_state: Mapping[str, object],
    context: DatasetContext,
    allowed_sources: set[str],
) -> Dict[str, str] | None:
    for ref in _draft_refs_by_source(draft_state, context, allowed_sources):
        return ref
    return None


def _draft_refs_by_source(
    draft_state: Mapping[str, object],
    context: DatasetContext,
    allowed_sources: set[str],
) -> list[Dict[str, str]]:
    grounded_refs = draft_state.get("grounded_refs") or ()
    if not isinstance(grounded_refs, Sequence) or isinstance(grounded_refs, (str, bytes)):
        return []
    refs = []
    for item in grounded_refs:
        normalized = _normalize_group_ref(item, context)
        if normalized is not None and normalized.get("source") in allowed_sources:
            refs.append(normalized)
    return _dedupe_refs(refs)


def _dedupe_refs(refs: Sequence[Mapping[str, str]]) -> list[Dict[str, str]]:
    deduped = []
    seen = set()
    for ref in refs:
        source = ref.get("source")
        value = ref.get("ref")
        if not source or not value:
            continue
        key = (source, value)
        if key in seen:
            continue
        deduped.append({"source": str(source), "ref": str(value)})
        seen.add(key)
    return deduped
