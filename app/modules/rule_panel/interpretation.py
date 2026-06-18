from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Tuple

from app.modules.algorithm_adapters.schemas import AnalysisResult
from app.modules.algorithm_adapters.service import cluster_counts
from app.modules.intent_instruction.providers.deepseek import (
    DEEPSEEK_PRO_MODEL,
    DeepSeekLlmProvider,
    DeepSeekResponseContentError,
)
from app.shared.schemas import FeatureMatrix

from .schemas import RULE_INTERPRETATION_CATEGORIES, RuleInterpretation, RuleSet

INTERPRETER_KIND_OPTIONS = ("mock", "deepseek")
DEFAULT_INTERPRETER_KIND = "mock"
_THRESHOLD_TOLERANCE = 1e-6
_TOP_PAIR_LIMIT = 12
_TOP_LABEL_POINT_LIMIT = 12
_RULE_INTERPRETATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts"
    / "rule_interpretation"
    / "deepseek"
    / "label_guidance_prompt.txt"
)
_RULE_INTERPRETATION_PROMPT_VERSION = "rule_label_guidance_v3_plain_language_categories"
_RULE_INTERPRETATION_MIN_OUTPUT_TOKENS = 6000
_RULE_INTERPRETATION_THINKING = {"type": "enabled"}
_RULE_INTERPRETATION_REASONING_EFFORT = "high"
_RULE_INTERPRETATION_RETRY_THINKING = {"type": "disabled"}
_ACTION_TYPES = {
    "inspect_points",
    "merge_clusters",
    "split_cluster",
    "create_cluster",
    "confirm_anomaly",
    "mark_normal",
    "audit_rule",
    "ask_domain_label",
}


@dataclass(frozen=True)
class RuleInterpretationRun:
    interpretation: RuleInterpretation
    request_payload: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interpretation": self.interpretation.to_dict(),
            "request_payload": dict(self.request_payload),
            "diagnostics": dict(self.diagnostics),
        }


class RuleInterpreter(Protocol):
    label: str

    def interpret(
        self,
        rule_set: RuleSet,
        *,
        analysis_result: AnalysisResult | None = None,
        feature_matrix: FeatureMatrix | None = None,
        focus_category: str | None = None,
    ) -> RuleInterpretationRun: ...


class _DeepSeekRuleInterpretationError(RuntimeError):
    def __init__(self, message: str, attempts: Tuple[Mapping[str, Any], ...]):
        super().__init__(message)
        self.attempts = tuple(dict(attempt) for attempt in attempts)


@dataclass
class DeterministicRuleInterpreter:
    label: str = "mock_rule_interpreter"

    def interpret(
        self,
        rule_set: RuleSet,
        *,
        analysis_result: AnalysisResult | None = None,
        feature_matrix: FeatureMatrix | None = None,
        focus_category: str | None = None,
    ) -> RuleInterpretationRun:
        focus_category = _validate_focus_category(focus_category)
        request_payload = build_rule_interpretation_payload(
            rule_set,
            analysis_result=analysis_result,
            feature_matrix=feature_matrix,
            focus_category=focus_category,
        )
        interpretation = (
            _focused_interpretation(
                rule_set,
                focus_category,
                analysis_result=analysis_result,
                feature_matrix=feature_matrix,
            )
            if focus_category is not None
            else _overview_interpretation(
                rule_set,
                analysis_result=analysis_result,
                feature_matrix=feature_matrix,
            )
        )
        interpretation = RuleInterpretation(
            interpretation_id=interpretation.interpretation_id,
            rule_set_id=interpretation.rule_set_id,
            categories=interpretation.categories,
            target_rule_ids=interpretation.target_rule_ids,
            summary=interpretation.summary,
            evidence=interpretation.evidence,
            recommendation=interpretation.recommendation,
            category_explanation=interpretation.category_explanation,
            decision_rationale=interpretation.decision_rationale,
            label_targets=interpretation.label_targets,
            suspicion_reasons=interpretation.suspicion_reasons,
            point_label_guidance=interpretation.point_label_guidance,
            label_outcomes=interpretation.label_outcomes,
            quantitative_findings=interpretation.quantitative_findings,
            suggested_label_actions=interpretation.suggested_label_actions,
            confidence=interpretation.confidence,
            warnings=interpretation.warnings,
            provider_label=self.label,
        )
        return RuleInterpretationRun(
            interpretation=interpretation,
            request_payload=request_payload,
            diagnostics={
                "provider_kind": "mock",
                "provider_label": self.label,
                "used_fallback": False,
                "validation": "grounded_label_guidance",
                "focus_category": focus_category,
            },
        )


@dataclass
class DeepSeekRuleInterpreter:
    client: DeepSeekLlmProvider = field(default_factory=DeepSeekLlmProvider)
    fallback: DeterministicRuleInterpreter = field(default_factory=DeterministicRuleInterpreter)
    _last_run: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.client.label

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "provider_kind": "deepseek",
            "provider_label": self.label,
            "client": self.client.diagnostics(),
            "last_run": dict(self._last_run),
        }

    def interpret(
        self,
        rule_set: RuleSet,
        *,
        analysis_result: AnalysisResult | None = None,
        feature_matrix: FeatureMatrix | None = None,
        focus_category: str | None = None,
    ) -> RuleInterpretationRun:
        focus_category = _validate_focus_category(focus_category)
        request_payload = build_rule_interpretation_payload(
            rule_set,
            analysis_result=analysis_result,
            feature_matrix=feature_matrix,
            focus_category=focus_category,
        )
        prompt = _deepseek_prompt(request_payload)
        try:
            payload, raw_response, attempts, effective_thinking, effective_reasoning = self._generate_rule_json(prompt)
            interpretation = parse_rule_interpretation_payload(
                payload,
                rule_set,
                provider_label=self.label,
            )
            diagnostics = {
                "provider_kind": "deepseek",
                "provider_label": self.label,
                "model_name": self.client.model_name,
                "expected_model_name": DEEPSEEK_PRO_MODEL,
                "using_deepseek_v4_pro": self.client.model_name == DEEPSEEK_PRO_MODEL,
                "thinking": dict(effective_thinking),
                "reasoning_effort": effective_reasoning,
                "deepseek_json_attempts": [dict(attempt) for attempt in attempts],
                "deepseek_retry_used": len(attempts) > 1,
                "used_fallback": False,
                "prompt_char_count": len(prompt),
                "prompt_template_path": str(_RULE_INTERPRETATION_PROMPT_PATH),
                "prompt_template_version": _RULE_INTERPRETATION_PROMPT_VERSION,
                "raw_response": _trim(raw_response),
                "validation": "grounded_label_guidance",
                "focus_category": focus_category,
            }
        except Exception as exc:
            if not self.client.allow_mock_fallback:
                raise
            fallback_run = self.fallback.interpret(
                rule_set,
                analysis_result=analysis_result,
                feature_matrix=feature_matrix,
                focus_category=focus_category,
            )
            interpretation = RuleInterpretation(
                interpretation_id=fallback_run.interpretation.interpretation_id,
                rule_set_id=fallback_run.interpretation.rule_set_id,
                categories=fallback_run.interpretation.categories,
                target_rule_ids=fallback_run.interpretation.target_rule_ids,
                summary=fallback_run.interpretation.summary,
                evidence=fallback_run.interpretation.evidence,
                recommendation=fallback_run.interpretation.recommendation,
                category_explanation=fallback_run.interpretation.category_explanation,
                decision_rationale=fallback_run.interpretation.decision_rationale,
                label_targets=fallback_run.interpretation.label_targets,
                suspicion_reasons=fallback_run.interpretation.suspicion_reasons,
                point_label_guidance=fallback_run.interpretation.point_label_guidance,
                label_outcomes=fallback_run.interpretation.label_outcomes,
                quantitative_findings=fallback_run.interpretation.quantitative_findings,
                suggested_label_actions=fallback_run.interpretation.suggested_label_actions,
                confidence=fallback_run.interpretation.confidence,
                warnings=(*fallback_run.interpretation.warnings, "deepseek_fallback_used"),
                provider_label=f"{self.label}->mock_rule_interpreter",
            )
            diagnostics = {
                **dict(fallback_run.diagnostics),
                "provider_kind": "deepseek",
                "provider_label": self.label,
                "model_name": self.client.model_name,
                "expected_model_name": DEEPSEEK_PRO_MODEL,
                "using_deepseek_v4_pro": self.client.model_name == DEEPSEEK_PRO_MODEL,
                "thinking": dict(_RULE_INTERPRETATION_THINKING),
                "reasoning_effort": _RULE_INTERPRETATION_REASONING_EFFORT,
                "deepseek_json_attempts": [
                    dict(attempt) for attempt in getattr(exc, "attempts", ())
                ],
                "used_fallback": True,
                "error": str(exc),
                "prompt_char_count": len(prompt),
                "prompt_template_path": str(_RULE_INTERPRETATION_PROMPT_PATH),
                "prompt_template_version": _RULE_INTERPRETATION_PROMPT_VERSION,
                "validation": "fallback_preserved_rules",
                "focus_category": focus_category,
            }

        self._last_run = diagnostics
        return RuleInterpretationRun(
            interpretation=interpretation,
            request_payload=request_payload,
            diagnostics=diagnostics,
        )

    def _generate_rule_json(
        self,
        prompt: str,
    ) -> tuple[Mapping[str, Any], str, Tuple[Mapping[str, Any], ...], Mapping[str, str], str | None]:
        max_tokens = max(self.client.max_output_tokens, _RULE_INTERPRETATION_MIN_OUTPUT_TOKENS)
        attempts = []
        attempt_configs = (
            {
                "attempt": "thinking_json",
                "thinking": _RULE_INTERPRETATION_THINKING,
                "reasoning_effort": _RULE_INTERPRETATION_REASONING_EFFORT,
                "max_tokens": max_tokens,
            },
            {
                "attempt": "direct_json_retry",
                "thinking": _RULE_INTERPRETATION_RETRY_THINKING,
                "reasoning_effort": None,
                "max_tokens": max_tokens,
            },
        )
        for index, config in enumerate(attempt_configs):
            try:
                payload, raw_response = self.client._generate_json(
                    prompt,
                    max_tokens=config["max_tokens"],
                    thinking=config["thinking"],
                    reasoning_effort=config["reasoning_effort"],
                )
                attempts.append(
                    {
                        "attempt": config["attempt"],
                        "model_name": self.client.model_name,
                        "thinking": dict(config["thinking"]),
                        "reasoning_effort": config["reasoning_effort"],
                        "max_tokens": config["max_tokens"],
                        "success": True,
                    }
                )
                return (
                    payload,
                    raw_response,
                    tuple(attempts),
                    config["thinking"],
                    config["reasoning_effort"],
                )
            except DeepSeekResponseContentError as exc:
                attempts.append(
                    {
                        "attempt": config["attempt"],
                        "model_name": self.client.model_name,
                        "thinking": dict(config["thinking"]),
                        "reasoning_effort": config["reasoning_effort"],
                        "max_tokens": config["max_tokens"],
                        "success": False,
                        "error": str(exc),
                        "response_metadata": dict(exc.response_metadata),
                    }
                )
                continue
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": config["attempt"],
                        "model_name": self.client.model_name,
                        "thinking": dict(config["thinking"]),
                        "reasoning_effort": config["reasoning_effort"],
                        "max_tokens": config["max_tokens"],
                        "success": False,
                        "error": str(exc),
                    }
                )
                if index < len(attempt_configs) - 1:
                    continue
                raise _DeepSeekRuleInterpretationError(str(exc), tuple(attempts)) from exc
        message = attempts[-1].get("error", "deepseek did not return JSON content") if attempts else "deepseek did not run"
        raise _DeepSeekRuleInterpretationError(str(message), tuple(attempts))


def create_rule_interpreter(provider_kind: str = DEFAULT_INTERPRETER_KIND) -> RuleInterpreter:
    kind = str(provider_kind or DEFAULT_INTERPRETER_KIND).strip().lower()
    if kind == "mock":
        return DeterministicRuleInterpreter()
    if kind == "deepseek":
        return DeepSeekRuleInterpreter()
    raise ValueError(f"provider_kind must be one of: {', '.join(INTERPRETER_KIND_OPTIONS)}")


def build_rule_interpretation_payload(
    rule_set: RuleSet,
    *,
    analysis_result: AnalysisResult | None = None,
    feature_matrix: FeatureMatrix | None = None,
    focus_category: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(rule_set, RuleSet):
        raise ValueError("rule_set must be a RuleSet")
    if analysis_result is not None and not isinstance(analysis_result, AnalysisResult):
        raise ValueError("analysis_result must be an AnalysisResult")
    if feature_matrix is not None and not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")
    focus_category = _validate_focus_category(focus_category)
    guidance_metrics = build_rule_guidance_metrics(
        rule_set,
        analysis_result=analysis_result,
        feature_matrix=feature_matrix,
    )

    return {
        "task": "recommend_next_labels_from_surrogate_rules",
        "dataset_id": rule_set.dataset_id,
        "focus_category": focus_category,
        "rule_set": rule_set.to_dict(),
        "allowed_categories": list(RULE_INTERPRETATION_CATEGORIES),
        "category_descriptions": _category_descriptions(),
        "category_case_status": _category_case_status_from_metrics(rule_set, guidance_metrics),
        "known_rule_ids": [rule.rule_id for rule in rule_set.rules],
        "known_target_ids": sorted({rule.target_id for rule in rule_set.rules}),
        "known_features": _known_features(rule_set, feature_matrix),
        "known_point_ids": sorted(_known_point_ids(rule_set, feature_matrix)),
        "known_thresholds": _known_thresholds(rule_set),
        "rule_guidance_metrics": guidance_metrics,
        "label_candidate_point_profiles": guidance_metrics.get("label_candidate_point_profiles", ()),
        "ssdbcodi": _analysis_summary(analysis_result),
        "instructions": {
            "source_of_truth": "SSDBCODI cluster assignments and outlier flags",
            "decision_tree_role": "explanation_only_surrogate_for_label_guidance",
            "required_llm_provider": "deepseek",
            "required_model": DEEPSEEK_PRO_MODEL,
            "required_thinking": dict(_RULE_INTERPRETATION_THINKING),
            "required_reasoning_effort": _RULE_INTERPRETATION_REASONING_EFFORT,
            "primary_goal": (
                "Guide the user's next labeling or refinement action. Explain what to label, why, and what "
                "cluster/outlier decision the labels could change."
            ),
            "llm_reasoning_role": (
                "Use quantitative metrics as evidence anchors, then synthesize competing hypotheses, expected label outcomes, "
                "decision impact, and uncertainty. The model's value is to turn metrics into user-facing next-step strategy, "
                "not to recalculate metrics."
            ),
            "plain_language_policy": (
                "Write for a careful undergraduate user who is labeling points, not for a data scientist. "
                "Avoid unexplained terms such as Jaccard, margin, support, purity, threshold, and information value in user-facing fields. "
                "When a technical term is necessary, translate it into what the user should look at and why it matters."
            ),
            "must_not_invent": [
                "features",
                "thresholds",
                "rule_ids",
                "cluster_ids",
                "anomaly_ids",
                "point_ids",
            ],
            "output_schema": {
                "categories": [focus_category or "label_priority"],
                "target_rule_ids": ["rule_cluster_1_001"],
                "category_explanation": "what this category examines",
                "summary": "3-6 sentence quantitative explanation",
                "recommendation": "specific next labeling or refinement action",
                "label_targets": [
                    {
                        "priority": "high",
                        "point_ids": ["wine_001"],
                        "rule_ids": ["rule_cluster_1_001"],
                        "label_question": "what the user should decide for these points",
                        "why_label_these_points": "why these labels are informative",
                    }
                ],
                "suspicion_reasons": [
                    {
                        "point_ids": ["wine_001"],
                        "rule_ids": ["rule_cluster_1_001"],
                        "suspicious_signal": "specific reason this is worth checking",
                        "rule_based_reason": "rule evidence that makes these points suspicious or important",
                        "point_based_reason": "point feature values, threshold margins, or outlier score evidence",
                    }
                ],
                "point_label_guidance": [
                    {
                        "point_ids": ["wine_001"],
                        "rule_ids": ["rule_cluster_1_001"],
                        "suggested_label_frame": "label options for the user",
                        "how_to_label": "concrete checklist for assigning the label",
                        "decision_impact": "what the label result would imply",
                        "llm_analysis_note": "DeepSeek point-level interpretation grounded in the supplied data",
                    }
                ],
                "decision_rationale": "why this action is strategically useful beyond the raw metrics",
                "label_outcomes": [
                    {
                        "label_result": "if user labels show one semantic group",
                        "decision_implication": "merge or boundary review becomes more plausible",
                        "rule_ids": ["rule_cluster_1_001"],
                        "point_ids": ["wine_001"],
                    }
                ],
                "quantitative_findings": [
                    {
                        "metric": "pair_jaccard_overlap",
                        "value": 0.0,
                        "rule_ids": ["rule_cluster_1_001", "rule_anomaly_current_outliers_001"],
                        "interpretation": "why this number matters for labeling",
                    }
                ],
                "suggested_label_actions": [
                    {
                        "action_type": "inspect_points",
                        "priority": "high",
                        "rule_ids": ["rule_cluster_1_001"],
                        "point_ids": ["wine_001"],
                        "reason": "why these labels are useful",
                        "hypothesis": "what this action is testing",
                        "why_this_action": "why this action is better than labeling random points",
                        "expected_outcomes": [
                            {
                                "label_result": "labels agree",
                                "decision_implication": "what decision this supports",
                            }
                        ],
                        "risk_note": "what could make this recommendation wrong",
                    }
                ],
                "evidence": [{"rule_ids": ["rule_cluster_1_001"], "feature": "alcohol"}],
                "confidence": 0.0,
                "warnings": [],
            },
            "focus_instruction": (
                f"Focus this response on {focus_category}."
                if focus_category is not None
                else "Choose the best supported categories from the allowed list."
            ),
        },
    }


def parse_rule_interpretation_payload(
    payload: Mapping[str, Any],
    rule_set: RuleSet,
    *,
    provider_label: str,
) -> RuleInterpretation:
    if not isinstance(payload, Mapping):
        raise ValueError("rule interpretation payload must be an object")
    if not isinstance(rule_set, RuleSet):
        raise ValueError("rule_set must be a RuleSet")

    body = payload.get("interpretation", payload)
    if not isinstance(body, Mapping):
        raise ValueError("interpretation must be an object")

    categories = _string_tuple(body.get("categories"), "categories")
    if not categories:
        raise ValueError("categories must be a non-empty list")
    unknown_categories = sorted(set(categories) - set(RULE_INTERPRETATION_CATEGORIES))
    if unknown_categories:
        raise ValueError(f"unknown interpretation categories: {', '.join(unknown_categories)}")

    target_rule_ids = _string_tuple(body.get("target_rule_ids", ()), "target_rule_ids")
    known_rule_ids = {rule.rule_id for rule in rule_set.rules}
    unknown_rules = sorted(set(target_rule_ids) - known_rule_ids)
    if unknown_rules:
        raise ValueError(f"unknown rule id(s): {', '.join(unknown_rules)}")

    evidence = _evidence_tuple(body.get("evidence", ()), rule_set)
    warnings = _string_tuple(body.get("warnings", ()), "warnings")
    summary = body.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string")
    recommendation = body.get("recommendation")
    if not isinstance(recommendation, str) or not recommendation.strip():
        raise ValueError("recommendation must be a non-empty string")
    category_explanation = body.get("category_explanation")
    if not isinstance(category_explanation, str) or not category_explanation.strip():
        raise ValueError("category_explanation must be a non-empty string")
    no_typical_case = _has_no_typical_case_warning(warnings)
    require_label_guidance = not no_typical_case
    label_targets = _reference_tuple(
        body.get("label_targets"),
        rule_set,
        field_name="label_targets",
        require_non_empty=require_label_guidance,
    )
    if label_targets:
        _validate_label_targets(label_targets)
    suspicion_reasons = _reference_tuple(
        body.get("suspicion_reasons"),
        rule_set,
        field_name="suspicion_reasons",
        require_non_empty=require_label_guidance,
    )
    if suspicion_reasons:
        _validate_suspicion_reasons(suspicion_reasons)
    point_label_guidance = _reference_tuple(
        body.get("point_label_guidance"),
        rule_set,
        field_name="point_label_guidance",
        require_non_empty=require_label_guidance,
    )
    if point_label_guidance:
        _validate_point_label_guidance(point_label_guidance)
    decision_rationale = body.get("decision_rationale")
    if not isinstance(decision_rationale, str) or not decision_rationale.strip():
        raise ValueError("decision_rationale must be a non-empty string")
    label_outcomes = _reference_tuple(
        body.get("label_outcomes"),
        rule_set,
        field_name="label_outcomes",
        require_non_empty=require_label_guidance,
    )
    if label_outcomes:
        _validate_label_outcomes(label_outcomes)
    quantitative_findings = _reference_tuple(
        body.get("quantitative_findings"),
        rule_set,
        field_name="quantitative_findings",
        require_non_empty=True,
    )
    suggested_label_actions = _action_tuple(
        body.get("suggested_label_actions"),
        rule_set,
        require_non_empty=require_label_guidance,
    )

    confidence = body.get("confidence", 1.0)
    interpretation = RuleInterpretation(
        interpretation_id=str(body.get("interpretation_id") or f"interp_{rule_set.rule_set_id}"),
        rule_set_id=rule_set.rule_set_id,
        categories=categories,
        target_rule_ids=target_rule_ids,
        summary=summary,
        evidence=evidence,
        recommendation=recommendation,
        category_explanation=category_explanation,
        decision_rationale=decision_rationale,
        label_targets=label_targets,
        suspicion_reasons=suspicion_reasons,
        point_label_guidance=point_label_guidance,
        label_outcomes=label_outcomes,
        quantitative_findings=quantitative_findings,
        suggested_label_actions=suggested_label_actions,
        confidence=confidence,
        warnings=warnings,
        provider_label=provider_label,
    )
    return _normalize_user_facing_terms(interpretation)


def _deepseek_prompt(request_payload: Mapping[str, Any]) -> str:
    focus_category = request_payload.get("focus_category")
    focus_line = (
        f"Focus on exactly this category if possible: {focus_category}.\n"
        if focus_category
        else "Choose one or more supported categories.\n"
    )
    template = _RULE_INTERPRETATION_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{focus_line}", focus_line.rstrip())
        .replace("{payload_json}", json.dumps(request_payload, sort_keys=True, separators=(",", ":")))
    )


def build_rule_guidance_metrics(
    rule_set: RuleSet,
    *,
    analysis_result: AnalysisResult | None = None,
    feature_matrix: FeatureMatrix | None = None,
) -> Dict[str, Any]:
    if not isinstance(rule_set, RuleSet):
        raise ValueError("rule_set must be a RuleSet")
    if analysis_result is not None and not isinstance(analysis_result, AnalysisResult):
        raise ValueError("analysis_result must be an AnalysisResult")
    if feature_matrix is not None and not isinstance(feature_matrix, FeatureMatrix):
        raise ValueError("feature_matrix must be a FeatureMatrix")

    rule_metrics = [_rule_metric(rule) for rule in rule_set.rules]
    pair_metrics = _top_pair_metrics(rule_set.rules)
    label_candidates = _label_candidates(rule_set, pair_metrics)
    point_profiles = _label_candidate_point_profiles(
        rule_set,
        label_candidates,
        analysis_result=analysis_result,
        feature_matrix=feature_matrix,
    )
    feature_usage = rule_set.diagnostics.get("feature_usage", {})
    if not isinstance(feature_usage, Mapping):
        feature_usage = {}

    return {
        "rule_metrics": rule_metrics,
        "pair_metrics": pair_metrics,
        "label_candidate_groups": label_candidates,
        "label_candidate_point_profiles": point_profiles,
        "feature_usage": dict(feature_usage),
        "total_rule_count": len(rule_set.rules),
        "cluster_rule_count": sum(1 for rule in rule_set.rules if rule.target_kind == "cluster"),
        "anomaly_rule_count": sum(1 for rule in rule_set.rules if rule.target_kind == "anomaly"),
        "exception_point_count": len(
            {
                point_id
                for rule in rule_set.rules
                for point_id in rule.exception_point_ids
            }
        ),
    }


def rule_interpretation_category_status(
    rule_set: RuleSet,
    *,
    analysis_result: AnalysisResult | None = None,
    feature_matrix: FeatureMatrix | None = None,
) -> Tuple[Mapping[str, Any], ...]:
    metrics = build_rule_guidance_metrics(
        rule_set,
        analysis_result=analysis_result,
        feature_matrix=feature_matrix,
    )
    return _category_case_status_from_metrics(rule_set, metrics)


def rule_interpretation_category_descriptions() -> Mapping[str, str]:
    return dict(_category_descriptions())


def _overview_interpretation(
    rule_set: RuleSet,
    *,
    analysis_result: AnalysisResult | None = None,
    feature_matrix: FeatureMatrix | None = None,
) -> RuleInterpretation:
    category = "label_priority"
    metrics = build_rule_guidance_metrics(
        rule_set,
        analysis_result=analysis_result,
        feature_matrix=feature_matrix,
    )
    categories = _recommended_categories(metrics)
    return _interpretation_from_metrics(rule_set, category, categories, metrics)


def _focused_interpretation(
    rule_set: RuleSet,
    focus_category: str,
    *,
    analysis_result: AnalysisResult | None = None,
    feature_matrix: FeatureMatrix | None = None,
) -> RuleInterpretation:
    metrics = build_rule_guidance_metrics(
        rule_set,
        analysis_result=analysis_result,
        feature_matrix=feature_matrix,
    )
    return _interpretation_from_metrics(rule_set, focus_category, (focus_category,), metrics)


def _interpretation_from_metrics(
    rule_set: RuleSet,
    primary_category: str,
    categories: Tuple[str, ...],
    metrics: Mapping[str, Any],
) -> RuleInterpretation:
    target_rule_ids = _target_rule_ids_for_category(rule_set, primary_category, metrics)
    category_explanation = _category_descriptions()[primary_category]
    if not _category_has_typical_case(rule_set, primary_category, metrics):
        warnings = (*_warnings_for_category(rule_set, primary_category, metrics), "no_typical_case_for_category")
        return _normalize_user_facing_terms(RuleInterpretation(
            interpretation_id=f"interp_{rule_set.rule_set_id}_{primary_category}",
            rule_set_id=rule_set.rule_set_id,
            categories=categories,
            target_rule_ids=target_rule_ids,
            summary=_no_case_summary(primary_category, metrics),
            evidence=_evidence_for_category(rule_set, primary_category, metrics, target_rule_ids),
            recommendation=_no_case_recommendation(primary_category),
            category_explanation=category_explanation,
            decision_rationale=_no_case_rationale(primary_category),
            label_targets=(),
            suspicion_reasons=(),
            point_label_guidance=(),
            label_outcomes=(),
            quantitative_findings=_findings_for_category(rule_set, primary_category, metrics),
            suggested_label_actions=(),
            confidence=0.35,
            warnings=warnings,
            provider_label="mock_rule_interpreter",
        ))

    summary = _summary_for_category(rule_set, primary_category, metrics)
    recommendation = _recommendation_for_category(primary_category, metrics)
    label_targets = _label_targets_for_category(rule_set, primary_category, metrics, target_rule_ids)
    suspicion_reasons = _suspicion_reasons_for_category(rule_set, primary_category, metrics, target_rule_ids)
    point_label_guidance = _point_label_guidance_for_category(rule_set, primary_category, metrics, target_rule_ids)
    decision_rationale = _decision_rationale_for_category(rule_set, primary_category, metrics)
    label_outcomes = _label_outcomes_for_category(rule_set, primary_category, metrics, target_rule_ids)
    evidence = _evidence_for_category(rule_set, primary_category, metrics, target_rule_ids)
    quantitative_findings = _findings_for_category(rule_set, primary_category, metrics)
    suggested_label_actions = _actions_for_category(rule_set, primary_category, metrics, target_rule_ids)
    warnings = _warnings_for_category(rule_set, primary_category, metrics)

    return _normalize_user_facing_terms(RuleInterpretation(
        interpretation_id=f"interp_{rule_set.rule_set_id}_{primary_category}",
        rule_set_id=rule_set.rule_set_id,
        categories=categories,
        target_rule_ids=target_rule_ids,
        summary=summary,
        evidence=evidence,
        recommendation=recommendation,
        category_explanation=category_explanation,
        decision_rationale=decision_rationale,
        label_targets=label_targets,
        suspicion_reasons=suspicion_reasons,
        point_label_guidance=point_label_guidance,
        label_outcomes=label_outcomes,
        quantitative_findings=quantitative_findings,
        suggested_label_actions=suggested_label_actions,
        confidence=_confidence_for_category(primary_category, metrics),
        warnings=warnings,
        provider_label="mock_rule_interpreter",
    ))


def _rule_metric(rule) -> Dict[str, Any]:
    exception_count = len(rule.exception_point_ids)
    exception_rate = exception_count / rule.support_count if rule.support_count else 0.0
    return {
        "rule_id": rule.rule_id,
        "target_kind": rule.target_kind,
        "target_id": rule.target_id,
        "support_count": rule.support_count,
        "coverage": rule.coverage,
        "purity": rule.purity,
        "exception_count": exception_count,
        "exception_rate": round(exception_rate, 6),
        "condition_count": len(rule.conditions),
        "condition_features": sorted({condition.feature for condition in rule.conditions}),
        "matched_preview_point_ids": list(rule.matched_point_ids[:_TOP_LABEL_POINT_LIMIT]),
        "exception_point_ids": list(rule.exception_point_ids[:_TOP_LABEL_POINT_LIMIT]),
        "rule_confidence_score": round(rule.purity * rule.coverage, 6),
        "quality_warnings": list(rule.diagnostics.get("quality_warnings", ())),
    }


def _top_pair_metrics(rules: Tuple[Any, ...]) -> Tuple[Mapping[str, Any], ...]:
    scored = []
    for left, right in combinations(rules, 2):
        left_points = set(left.matched_point_ids)
        right_points = set(right.matched_point_ids)
        intersection = sorted(left_points & right_points)
        union_count = len(left_points | right_points)
        intersection_count = len(intersection)
        left_support = len(left_points)
        right_support = len(right_points)
        jaccard = intersection_count / union_count if union_count else 0.0
        overlap_share_left = intersection_count / left_support if left_support else 0.0
        overlap_share_right = intersection_count / right_support if right_support else 0.0
        shared_features = sorted(
            {condition.feature for condition in left.conditions}
            & {condition.feature for condition in right.conditions}
        )
        boundary_gaps = _boundary_gaps(left, right, shared_features)
        relation = _pair_relation(
            left,
            right,
            intersection_count=intersection_count,
            shared_features=shared_features,
            boundary_gaps=boundary_gaps,
        )
        priority_score = _pair_priority(
            relation,
            jaccard,
            overlap_share_left,
            overlap_share_right,
            boundary_gaps,
        )
        metric = {
            "rule_ids": [left.rule_id, right.rule_id],
            "target_ids": [left.target_id, right.target_id],
            "target_kinds": [left.target_kind, right.target_kind],
            "relation": relation,
            "intersection_count": intersection_count,
            "union_count": union_count,
            "jaccard_overlap": round(jaccard, 6),
            "overlap_share_a": round(overlap_share_left, 6),
            "overlap_share_b": round(overlap_share_right, 6),
            "point_ids": intersection[:_TOP_LABEL_POINT_LIMIT],
            "shared_features": shared_features,
            "boundary_gaps": boundary_gaps,
            "priority_score": round(priority_score, 6),
        }
        scored.append((priority_score, metric))

    scored.sort(
        key=lambda item: (
            -item[0],
            -item[1]["intersection_count"],
            item[1]["rule_ids"][0],
            item[1]["rule_ids"][1],
        )
    )
    return tuple(metric for _, metric in scored[:_TOP_PAIR_LIMIT])


def _label_candidates(rule_set: RuleSet, pair_metrics: Tuple[Mapping[str, Any], ...]) -> Tuple[Mapping[str, Any], ...]:
    candidates = []
    seen_groups = set()

    for metric in pair_metrics:
        point_ids = tuple(metric.get("point_ids", ()))
        if not point_ids:
            continue
        key = ("pair", tuple(metric["rule_ids"]), point_ids)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        candidates.append(
            {
                "candidate_kind": "overlap_points",
                "priority": "high" if metric["intersection_count"] >= 3 else "medium",
                "rule_ids": list(metric["rule_ids"]),
                "point_ids": list(point_ids[:6]),
                "reason": (
                    f"{metric['intersection_count']} point(s) are shared by the pair; "
                    f"Jaccard={metric['jaccard_overlap']:.2f}."
                ),
            }
        )

    for rule in rule_set.rules:
        if not rule.exception_point_ids:
            continue
        point_ids = tuple(rule.exception_point_ids[:6])
        key = ("exception", rule.rule_id, point_ids)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        candidates.append(
            {
                "candidate_kind": "exception_points",
                "priority": "high",
                "rule_ids": [rule.rule_id],
                "point_ids": list(point_ids),
                "reason": (
                    f"{len(rule.exception_point_ids)} exception point(s) are matched by {rule.rule_id} "
                    f"but disagree with target {rule.target_id}."
                ),
            }
        )

    if not candidates and rule_set.rules:
        strongest_rule = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count))
        candidates.append(
            {
                "candidate_kind": "representative_points",
                "priority": "medium",
                "rule_ids": [strongest_rule.rule_id],
                "point_ids": list(strongest_rule.matched_point_ids[:6]),
                "reason": (
                    f"{strongest_rule.rule_id} has support={strongest_rule.support_count}, "
                    f"purity={strongest_rule.purity:.2f}, coverage={strongest_rule.coverage:.2f}."
                ),
            }
        )

    return tuple(candidates[:_TOP_LABEL_POINT_LIMIT])


def _label_candidate_point_profiles(
    rule_set: RuleSet,
    label_candidates: Tuple[Mapping[str, Any], ...],
    *,
    analysis_result: AnalysisResult | None,
    feature_matrix: FeatureMatrix | None,
) -> Tuple[Mapping[str, Any], ...]:
    ordered_point_ids = []
    for candidate in label_candidates:
        for point_id in candidate.get("point_ids", ()):
            if point_id not in ordered_point_ids:
                ordered_point_ids.append(point_id)
    for rule in rule_set.rules:
        if len(ordered_point_ids) >= _TOP_LABEL_POINT_LIMIT:
            break
        for point_id in (*rule.exception_point_ids, *rule.matched_point_ids):
            if point_id not in ordered_point_ids:
                ordered_point_ids.append(point_id)
            if len(ordered_point_ids) >= _TOP_LABEL_POINT_LIMIT:
                break

    assignments = {}
    outlier_scores = {}
    if analysis_result is not None:
        assignments = {
            assignment.point_id: assignment.cluster_id
            for assignment in analysis_result.cluster_result.assignments
        }
        outlier_scores = {
            score.point_id: {
                "score": round(score.score, 6),
                "is_outlier": score.is_outlier,
            }
            for score in analysis_result.outlier_result.scores
        }

    feature_lookup = {}
    feature_names: Tuple[str, ...] = ()
    if feature_matrix is not None:
        feature_names = tuple(feature_matrix.feature_names)
        feature_lookup = {
            point_id: dict(zip(feature_names, row))
            for point_id, row in zip(feature_matrix.point_ids, feature_matrix.values)
        }

    profiles = []
    for point_id in ordered_point_ids[:_TOP_LABEL_POINT_LIMIT]:
        related_rules = [
            rule
            for rule in rule_set.rules
            if point_id in rule.matched_point_ids or point_id in rule.exception_point_ids
        ]
        condition_features = tuple(
            dict.fromkeys(
                condition.feature
                for rule in related_rules
                for condition in rule.conditions
            )
        )
        values = feature_lookup.get(point_id, {})
        raw_feature_values = {
            feature: round(float(values[feature]), 6)
            for feature in condition_features
            if feature in values
        }
        threshold_margins = []
        for rule in related_rules[:4]:
            for condition in rule.conditions:
                if condition.feature not in values:
                    continue
                value = float(values[condition.feature])
                margin = value - condition.threshold
                satisfied = value <= condition.threshold if condition.operator == "<=" else value > condition.threshold
                threshold_margins.append(
                    {
                        "rule_id": rule.rule_id,
                        "feature": condition.feature,
                        "operator": condition.operator,
                        "threshold": condition.threshold,
                        "point_value": round(value, 6),
                        "signed_margin": round(margin, 6),
                        "absolute_margin": round(abs(margin), 6),
                        "condition_satisfied": satisfied,
                    }
                )
        profiles.append(
            {
                "point_id": point_id,
                "current_cluster_id": assignments.get(point_id),
                "outlier": outlier_scores.get(point_id),
                "related_rule_ids": [rule.rule_id for rule in related_rules[:4]],
                "related_targets": [
                    {
                        "rule_id": rule.rule_id,
                        "target_kind": rule.target_kind,
                        "target_id": rule.target_id,
                    }
                    for rule in related_rules[:4]
                ],
                "raw_feature_values": raw_feature_values,
                "threshold_margins": threshold_margins[:8],
            }
        )
    return tuple(profiles)


def _pair_relation(
    left,
    right,
    *,
    intersection_count: int,
    shared_features: Tuple[str, ...],
    boundary_gaps: Tuple[Mapping[str, Any], ...],
) -> str:
    same_target = left.target_kind == right.target_kind and left.target_id == right.target_id
    cluster_pair = left.target_kind == right.target_kind == "cluster"
    cluster_anomaly_pair = {left.target_kind, right.target_kind} == {"cluster", "anomaly"}

    if intersection_count:
        if same_target:
            return "same_target_overlap"
        if cluster_pair:
            return "cross_cluster_overlap"
        if cluster_anomaly_pair:
            return "cluster_anomaly_overlap"
        return "cross_target_overlap"

    if same_target:
        return "same_target_disjoint_regions"
    if cluster_pair and shared_features and any(item["gap"] == 0 for item in boundary_gaps):
        return "adjacent_cluster_boundary"
    if cluster_pair:
        return "separate_cluster_rules"
    if cluster_anomaly_pair:
        return "cluster_anomaly_separate"
    return "separate_rules"


def _pair_priority(
    relation: str,
    jaccard: float,
    overlap_share_left: float,
    overlap_share_right: float,
    boundary_gaps: Tuple[Mapping[str, Any], ...],
) -> float:
    priority = max(jaccard, overlap_share_left, overlap_share_right)
    if relation in {"cluster_anomaly_overlap", "cross_cluster_overlap", "cross_target_overlap"}:
        priority += 0.5
    if relation in {"same_target_disjoint_regions", "adjacent_cluster_boundary"}:
        priority += 0.25
    if any(item["gap"] == 0 for item in boundary_gaps):
        priority += 0.1
    return priority


def _condition_intervals(rule) -> Dict[str, Tuple[float | None, float | None]]:
    intervals: Dict[str, Tuple[float | None, float | None]] = {}
    for condition in rule.conditions:
        lower, upper = intervals.get(condition.feature, (None, None))
        if condition.operator == ">":
            lower = condition.threshold if lower is None else max(lower, condition.threshold)
        else:
            upper = condition.threshold if upper is None else min(upper, condition.threshold)
        intervals[condition.feature] = (lower, upper)
    return intervals


def _boundary_gaps(left, right, shared_features: Tuple[str, ...]) -> Tuple[Mapping[str, Any], ...]:
    left_intervals = _condition_intervals(left)
    right_intervals = _condition_intervals(right)
    gaps = []
    for feature in shared_features:
        left_lower, left_upper = left_intervals.get(feature, (None, None))
        right_lower, right_upper = right_intervals.get(feature, (None, None))
        gap = _interval_gap(left_lower, left_upper, right_lower, right_upper)
        gaps.append(
            {
                "feature": feature,
                "rule_a_interval": _interval_payload(left_lower, left_upper),
                "rule_b_interval": _interval_payload(right_lower, right_upper),
                "gap": gap,
            }
        )
    return tuple(gaps)


def _interval_gap(
    left_lower: float | None,
    left_upper: float | None,
    right_lower: float | None,
    right_upper: float | None,
) -> float | None:
    if left_upper is not None and right_lower is not None and left_upper < right_lower:
        return round(right_lower - left_upper, 6)
    if right_upper is not None and left_lower is not None and right_upper < left_lower:
        return round(left_lower - right_upper, 6)
    return 0.0


def _interval_payload(lower: float | None, upper: float | None) -> Mapping[str, Any]:
    return {
        "lower_exclusive": lower,
        "upper_inclusive": upper,
    }


def _recommended_categories(metrics: Mapping[str, Any]) -> Tuple[str, ...]:
    categories = ["label_priority", "feature_label_strategy", "rule_confidence_audit"]
    pair_relations = {metric["relation"] for metric in metrics.get("pair_metrics", ())}
    if pair_relations & {"cross_cluster_overlap", "cluster_anomaly_overlap", "cross_target_overlap", "same_target_overlap"}:
        categories.append("overlap_merge_signal")
    if pair_relations & {"same_target_disjoint_regions", "adjacent_cluster_boundary", "separate_cluster_rules"}:
        categories.append("boundary_review")
        categories.append("split_or_new_cluster_signal")
    if metrics.get("anomaly_rule_count", 0):
        categories.append("anomaly_label_review")
    if metrics.get("exception_point_count", 0):
        categories.append("exception_relabel_review")
    return tuple(dict.fromkeys(categories))


def _category_case_status_from_metrics(rule_set: RuleSet, metrics: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    descriptions = _category_descriptions()
    return tuple(
        {
            "category": category,
            "label": category.replace("_", " ").title(),
            "description": descriptions[category],
            "has_typical_case": _category_has_typical_case(rule_set, category, metrics),
            "point_count": len(_candidate_point_ids_for_category(rule_set, metrics, _target_rule_ids_for_category(rule_set, category, metrics)))
            if _category_has_typical_case(rule_set, category, metrics)
            else 0,
            "reason": _category_case_reason(rule_set, category, metrics),
        }
        for category in RULE_INTERPRETATION_CATEGORIES
    )


def _category_has_typical_case(rule_set: RuleSet, category: str, metrics: Mapping[str, Any]) -> bool:
    if not rule_set.rules:
        return False
    pair = _best_pair(metrics, category)
    if category == "overlap_merge_signal":
        return pair is not None and pair.get("intersection_count", 0) > 0
    if category == "boundary_review":
        return pair is not None and pair.get("relation") in {
            "adjacent_cluster_boundary",
            "cross_cluster_overlap",
            "cluster_anomaly_overlap",
        }
    if category == "split_or_new_cluster_signal":
        return pair is not None and pair.get("relation") in {
            "same_target_disjoint_regions",
            "adjacent_cluster_boundary",
            "separate_cluster_rules",
        }
    if category == "anomaly_label_review":
        return metrics.get("anomaly_rule_count", 0) > 0
    if category == "exception_relabel_review":
        return metrics.get("exception_point_count", 0) > 0
    if category == "feature_label_strategy":
        feature_usage = metrics.get("feature_usage", {})
        return isinstance(feature_usage, Mapping) and bool(feature_usage)
    return category in {"label_priority", "rule_confidence_audit"}


def _category_case_reason(rule_set: RuleSet, category: str, metrics: Mapping[str, Any]) -> str:
    if _category_has_typical_case(rule_set, category, metrics):
        point_count = len(_candidate_point_ids_for_category(rule_set, metrics, _target_rule_ids_for_category(rule_set, category, metrics)))
        return f"{point_count} candidate point(s) are available for this category."
    return _no_case_recommendation(category)


def _no_case_summary(category: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"This category is available as a lens, but the current rule set does not contain a strong example for it. "
        f"The current rules include {metrics.get('cluster_rule_count', 0)} cluster rule(s), "
        f"{metrics.get('anomaly_rule_count', 0)} anomaly rule(s), and "
        f"{metrics.get('exception_point_count', 0)} exception point(s). "
        "Do not force a label recommendation here; use another category that has candidate points."
    )


def _no_case_recommendation(category: str) -> str:
    return {
        "label_priority": "No generated rule is available, so there is no useful next point for this category yet.",
        "boundary_review": "No clear boundary example is available right now; use this category after two rules form a visible boundary.",
        "overlap_merge_signal": "No two rules currently share matched points, so there is no typical merge example to label.",
        "split_or_new_cluster_signal": "No separated rule region is strong enough for a split/new-cluster example right now.",
        "anomaly_label_review": "No anomaly rule is available, so there is no anomaly-specific point to review in this category.",
        "exception_relabel_review": "No exception point is available, so there is no relabel case to inspect here.",
        "feature_label_strategy": "No raw feature threshold dominates the current rule cards, so this category has no clear checklist example.",
        "rule_confidence_audit": "No rule is available to audit yet.",
    }[category]


def _no_case_rationale(category: str) -> str:
    return (
        f"{category.replace('_', ' ').title()} should only ask the user to label points when the rule cards show a concrete case. "
        "The current evidence is too weak for this category, so the safest user-facing behavior is to say that no typical case is available."
    )


def _target_rule_ids_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
) -> Tuple[str, ...]:
    if not rule_set.rules:
        return ()
    pair = _best_pair(metrics, category)
    if pair is not None:
        return tuple(pair["rule_ids"])
    if category == "anomaly_label_review":
        anomaly_rules = [rule.rule_id for rule in rule_set.rules if rule.target_kind == "anomaly"]
        return tuple(anomaly_rules[:3])
    if category == "exception_relabel_review":
        exception_rules = [rule.rule_id for rule in rule_set.rules if rule.exception_point_ids]
        if exception_rules:
            return tuple(exception_rules[:3])
    strongest_rule = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count))
    return (strongest_rule.rule_id,)


def _best_pair(metrics: Mapping[str, Any], category: str) -> Mapping[str, Any] | None:
    pair_metrics = tuple(metrics.get("pair_metrics", ()))
    if not pair_metrics:
        return None
    if category == "overlap_merge_signal":
        for metric in pair_metrics:
            if metric["intersection_count"] > 0:
                return metric
        return None
    preferred = {
        "label_priority": {
            "cluster_anomaly_overlap",
            "cross_cluster_overlap",
            "cross_target_overlap",
            "same_target_overlap",
            "adjacent_cluster_boundary",
        },
        "boundary_review": {"adjacent_cluster_boundary", "cross_cluster_overlap", "cluster_anomaly_overlap"},
        "overlap_merge_signal": {
            "cross_cluster_overlap",
            "cluster_anomaly_overlap",
            "cross_target_overlap",
            "same_target_overlap",
        },
        "split_or_new_cluster_signal": {"same_target_disjoint_regions", "adjacent_cluster_boundary", "separate_cluster_rules"},
    }.get(category)
    if preferred is None:
        return None
    for metric in pair_metrics:
        if metric["relation"] in preferred:
            return metric
    return pair_metrics[0]


def _summary_for_category(rule_set: RuleSet, category: str, metrics: Mapping[str, Any]) -> str:
    if not rule_set.rules:
        return "There are 0 generated rules, so there is no quantitative rule evidence for labeling yet."

    strongest_rule = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count))
    pair = _best_pair(metrics, category)
    pair_sentence = _pair_sentence(pair)
    candidate_sentence = _candidate_sentence(metrics)
    feature_sentence = _feature_sentence(metrics)
    exception_count = metrics.get("exception_point_count", 0)
    anomaly_count = metrics.get("anomaly_rule_count", 0)

    summaries = {
        "label_priority": (
            f"Start labeling from the highest-ambiguity rule evidence, not from random points. {pair_sentence} "
            f"The strongest single rule is {strongest_rule.rule_id} with support {strongest_rule.support_count}, "
            f"purity {strongest_rule.purity:.2f}, and coverage {strongest_rule.coverage:.2f}. {candidate_sentence}"
        ),
        "boundary_review": (
            f"Use the rule boundaries to decide whether the current analysis separated nearby regions correctly. {pair_sentence} "
            f"Boundary labeling should sample both sides of the shared feature cutoffs and compare at least "
            f"{min(6, strongest_rule.support_count)} representative point(s) from the linked rules."
        ),
        "overlap_merge_signal": (
            f"Overlap is a label-driven merge signal only when rules share matched points. "
            f"{pair_sentence} If there is no sample-level overlap, do not merge from the rule cards alone; label adjacent or representative "
            f"points first and treat merge/new-cluster decisions as hypotheses."
        ),
        "split_or_new_cluster_signal": (
            f"Split/new-cluster review is strongest when one target needs multiple separated rules or when adjacent rules explain "
            f"different targets with weak separation. {pair_sentence} The current rule set has {metrics.get('cluster_rule_count', 0)} "
            f"cluster rule(s), {anomaly_count} anomaly rule(s), and {exception_count} exception point(s), so labels should focus on "
            "whether separated regions actually share one concept."
        ),
        "anomaly_label_review": (
            f"Anomaly labels should be reviewed where outlier rules overlap cluster rules or have very small support. "
            f"The rule set has {anomaly_count} anomaly rule(s) and {exception_count} exception point(s). {pair_sentence} "
            "Label these points as true anomaly versus normal cluster member before changing the outlier interpretation."
        ),
        "exception_relabel_review": (
            f"Exception points are the most direct relabel candidates because they are matched by a rule but disagree with its target. "
            f"The current rule set exposes {exception_count} unique exception point(s). {candidate_sentence} "
            "If several exception points receive the same user label, they become evidence for a boundary fix or a new cluster."
        ),
        "feature_label_strategy": (
            f"Use raw feature cutoffs as the labeling checklist rather than projected x/y positions. {feature_sentence} "
            f"The strongest rule still needs label validation because its confidence score is "
            f"{strongest_rule.purity * strongest_rule.coverage:.2f} from purity {strongest_rule.purity:.2f} and coverage {strongest_rule.coverage:.2f}."
        ),
        "rule_confidence_audit": (
            f"Treat the surrogate rules as ranking signals, not ground truth. {strongest_rule.rule_id} has support "
            f"{strongest_rule.support_count}, purity {strongest_rule.purity:.2f}, coverage {strongest_rule.coverage:.2f}, "
            f"and {len(strongest_rule.exception_point_ids)} exception point(s). Rules with low coverage or many exceptions should trigger labels before any merge/split decision."
        ),
    }
    return summaries[category]


def _pair_sentence(pair: Mapping[str, Any] | None) -> str:
    if pair is None:
        return "There is no rule pair available, so the first label pass should use representative points from the strongest rule."
    return (
        f"The top pair {pair['rule_ids'][0]} vs {pair['rule_ids'][1]} has relation {pair['relation']}, "
        f"{pair['intersection_count']} shared matched point(s), Jaccard {pair['jaccard_overlap']:.2f}, "
        f"and overlap shares {pair['overlap_share_a']:.2f}/{pair['overlap_share_b']:.2f}."
    )


def _candidate_sentence(metrics: Mapping[str, Any]) -> str:
    candidates = tuple(metrics.get("label_candidate_groups", ()))
    if not candidates:
        return "No candidate point group was produced, so ask for labels on representative matched points."
    top = candidates[0]
    return (
        f"First candidate group contains {len(top.get('point_ids', ())) } point id(s): "
        f"{', '.join(top.get('point_ids', ())[:6])}."
    )


def _feature_sentence(metrics: Mapping[str, Any]) -> str:
    usage = metrics.get("feature_usage", {})
    if not isinstance(usage, Mapping) or not usage:
        return "No dominant raw feature cutoff was found in the rule diagnostics."
    top_items = list(usage.items())[:3]
    formatted = ", ".join(f"{feature} used {count} time(s)" for feature, count in top_items)
    return f"The most repeated raw cutoffs are {formatted}."


def _recommendation_for_category(category: str, metrics: Mapping[str, Any]) -> str:
    candidates = tuple(metrics.get("label_candidate_groups", ()))
    point_ids = []
    rule_ids = []
    if candidates:
        point_ids = list(candidates[0].get("point_ids", ()))[:6]
        rule_ids = list(candidates[0].get("rule_ids", ()))
    pair = _best_pair(metrics, category)
    if pair is not None and not rule_ids:
        rule_ids = list(pair["rule_ids"])
    point_text = ", ".join(point_ids) if point_ids else "the representative matched points"
    rule_text = ", ".join(rule_ids) if rule_ids else "the highest-confidence rule"

    overlap_has_pair = category != "overlap_merge_signal" or pair is not None
    recommendations = {
        "label_priority": f"Label {point_text} first, then compare those labels against {rule_text} before changing clusters.",
        "boundary_review": f"Label points from both sides of {rule_text}; use agreement to decide whether the boundary is meaningful.",
        "overlap_merge_signal": (
            f"Label the overlap candidates for {rule_text}; merge only if user labels show the same semantic group."
            if overlap_has_pair
            else f"No rule pair has sample-level overlap, so label {point_text} and use boundary_review before considering any merge."
        ),
        "split_or_new_cluster_signal": f"Label separated regions covered by {rule_text}; create a new cluster only if labels reveal a consistent concept not captured by existing targets.",
        "anomaly_label_review": f"Label {point_text} as true anomaly versus normal cluster member before trusting the anomaly rule.",
        "exception_relabel_review": f"Give explicit labels to {point_text}; repeated exception labels should drive relabel or boundary refinement.",
        "feature_label_strategy": (
            f"Label {point_text} by checking whether their raw feature values satisfy {rule_text}'s cutoff story; "
            "the doubt is whether that cutoff story matches the user's human label, not whether the raw feature itself is wrong."
        ),
        "rule_confidence_audit": f"Audit {rule_text} by labeling a small matched sample and every exception point before using it for merge/split decisions.",
    }
    return recommendations[category]


def _label_targets_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    point_ids = _candidate_point_ids_for_category(rule_set, metrics, target_rule_ids)
    rule_ids = _candidate_rule_ids_for_category(metrics, target_rule_ids)
    priority = "high" if category in {"label_priority", "overlap_merge_signal", "exception_relabel_review"} else "medium"
    label_question = {
        "label_priority": "Which current analysis target do these points belong to after a human checks them?",
        "boundary_review": "Do points on this rule boundary receive the same human label or different labels?",
        "overlap_merge_signal": "Do the candidate points support one shared human group across the cited rules?",
        "split_or_new_cluster_signal": "Do these points reveal a distinct human group that should be split or created?",
        "anomaly_label_review": "Are these points true anomalies or normal members of a visible cluster?",
        "exception_relabel_review": "Are these exception-like points mislabeled, boundary cases, or members of a hidden group?",
        "feature_label_strategy": "Do the user's labels agree with the raw feature cutoff story used by the cited rule?",
        "rule_confidence_audit": "Does this high-confidence surrogate rule also match human labels?",
    }[category]
    why = _why_action_for_category(category, metrics)
    return (
        {
            "priority": priority,
            "point_ids": point_ids,
            "rule_ids": rule_ids,
            "label_question": label_question,
            "why_label_these_points": why,
        },
    )


def _suspicion_reasons_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    point_ids = _candidate_point_ids_for_category(rule_set, metrics, target_rule_ids)
    rule_ids = _candidate_rule_ids_for_category(metrics, target_rule_ids)
    pair = _best_pair(metrics, category)
    profiles = _profiles_by_point(metrics)
    pair_reason = (
        f"The linked rule relation is {pair['relation']} with intersection {pair['intersection_count']} "
        f"and Jaccard {pair['jaccard_overlap']:.2f}."
        if pair is not None
        else "No stronger overlap pair is available, so these points are representative validation samples."
    )
    suspicious_signal = {
        "label_priority": "Start here because these labels can quickly confirm or challenge the current rule story.",
        "boundary_review": "Check whether points near this rule boundary still feel like the same wine group to a human.",
        "overlap_merge_signal": "Check whether the same points being covered by two rules actually deserve one shared label.",
        "split_or_new_cluster_signal": "Check whether a separated region is a real new group or just a numerical split.",
        "anomaly_label_review": "Check whether the model's unusual points are truly unusual to the user.",
        "exception_relabel_review": "Check these points because the rule and the current assignment disagree.",
        "feature_label_strategy": "Check whether the raw-feature checklist describes a meaningful wine group, not just a numeric cutoff.",
        "rule_confidence_audit": "Check whether a confident rule also makes sense to a human labeler.",
    }[category]
    rule_reason = f"{pair_reason} {_rule_target_reason(rule_set, rule_ids)}"
    return tuple(
        {
            "point_ids": [point_id],
            "rule_ids": rule_ids,
            "suspicious_signal": f"{point_id}: {suspicious_signal}",
            "rule_based_reason": rule_reason,
            "point_based_reason": _point_profile_reason(profiles.get(point_id)),
        }
        for point_id in point_ids[:6]
    )


def _point_label_guidance_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    point_ids = _candidate_point_ids_for_category(rule_set, metrics, target_rule_ids)
    rule_ids = _candidate_rule_ids_for_category(metrics, target_rule_ids)
    profiles = _profiles_by_point(metrics)
    target_ids = _target_ids_for_rules(rule_set, rule_ids)
    target_text = ", ".join(target_ids) if target_ids else "the cited rule target"
    frame = {
        "label_priority": f"Choose whether the point belongs to {target_text}, another cluster, an anomaly, or uncertain.",
        "boundary_review": f"Choose the human label first, then see whether points on the two sides of {target_text}'s rule boundary get different labels.",
        "overlap_merge_signal": f"Choose whether these points should share one label across {target_text} or stay separated.",
        "split_or_new_cluster_signal": "Choose an existing cluster, a possible new cluster, or uncertain for this separated group.",
        "anomaly_label_review": "Choose true anomaly, normal cluster member, or uncertain.",
        "exception_relabel_review": "Choose corrected cluster/anomaly label, boundary case, or uncertain.",
        "feature_label_strategy": f"Choose whether the point's human label agrees with the raw-feature story for {target_text}.",
        "rule_confidence_audit": f"Choose whether the matched points really belong to {target_text} when judged by human meaning.",
    }[category]
    how_to_label = {
        "label_priority": "Look at the listed wine ids first. Give each one the label that best matches its actual raw features and domain meaning.",
        "boundary_review": "Label the points without trusting the rule first. Then check whether the rule boundary separates different human labels.",
        "overlap_merge_signal": "If these shared points get the same human label, merging or shared-boundary review becomes reasonable. If not, keep the groups separate.",
        "split_or_new_cluster_signal": "If several points get the same label but that label does not fit any current cluster, mark them as possible new-cluster candidates.",
        "anomaly_label_review": "Decide whether the point is genuinely unusual, or whether it is a normal member that the model happened to score highly.",
        "exception_relabel_review": "If multiple exception points receive the same corrected label, treat that as stronger evidence than one isolated mismatch.",
        "feature_label_strategy": "Use the raw feature rule as a checklist, then decide whether that checklist matches a meaningful wine label.",
        "rule_confidence_audit": "Use a small human-labeled sample to decide whether the rule is genuinely useful, not merely numerically tidy.",
    }[category]
    return tuple(
        {
            "point_ids": [point_id],
            "rule_ids": rule_ids,
            "suggested_label_frame": frame,
            "how_to_label": f"{how_to_label} {_profile_threshold_sentence(profiles.get(point_id))}",
            "decision_impact": _decision_impact_for_category(category),
            "llm_analysis_note": _fallback_llm_note(point_id, profiles.get(point_id)),
        }
        for point_id in point_ids[:6]
    )


def _candidate_point_ids_for_category(
    rule_set: RuleSet,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> list[str]:
    candidates = tuple(metrics.get("label_candidate_groups", ()))
    if candidates:
        point_ids = list(candidates[0].get("point_ids", ()))[:6]
        if point_ids:
            return point_ids
    rules_by_id = {rule.rule_id: rule for rule in rule_set.rules}
    seed_rule = rules_by_id.get(target_rule_ids[0]) if target_rule_ids else (rule_set.rules[0] if rule_set.rules else None)
    if seed_rule is None:
        return []
    return list((*seed_rule.exception_point_ids, *seed_rule.matched_point_ids))[:6]


def _candidate_rule_ids_for_category(metrics: Mapping[str, Any], target_rule_ids: Tuple[str, ...]) -> list[str]:
    candidates = tuple(metrics.get("label_candidate_groups", ()))
    if candidates:
        rule_ids = list(candidates[0].get("rule_ids", ()))[:3]
        if rule_ids:
            return rule_ids
    return list(target_rule_ids[:3])


def _profiles_by_point(metrics: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(profile.get("point_id")): profile
        for profile in metrics.get("label_candidate_point_profiles", ())
        if isinstance(profile, Mapping) and profile.get("point_id")
    }


def _point_profile_reason(profile: Mapping[str, Any] | None) -> str:
    if not profile:
        return "This point was selected because it appears in the rule cards, but no detailed feature profile is available."
    cluster_id = profile.get("current_cluster_id") or "unassigned"
    outlier = profile.get("outlier")
    outlier_text = ""
    if isinstance(outlier, Mapping):
        status = "currently marked as an outlier" if outlier.get("is_outlier") else "not currently marked as an outlier"
        outlier_text = f" The current analysis gives it outlier score {outlier.get('score')} and it is {status}."
    margins = profile.get("threshold_margins")
    if isinstance(margins, (list, tuple)) and margins:
        closest = min(
            (margin for margin in margins if isinstance(margin, Mapping) and margin.get("absolute_margin") is not None),
            key=lambda margin: margin.get("absolute_margin", 0),
            default=None,
        )
        if closest is not None:
            direction = "passes" if closest.get("condition_satisfied") else "does not pass"
            return (
                f"{profile.get('point_id')} is currently placed in {cluster_id}.{outlier_text} "
                f"The closest rule check is {closest.get('feature')} {closest.get('operator')} {closest.get('threshold')}; "
                f"this wine has {closest.get('feature')}={closest.get('point_value')}, so it {direction} that check. "
                "That makes it useful for asking whether this raw-feature cutoff matches the user's idea of the group."
            )
    raw_values = profile.get("raw_feature_values")
    if isinstance(raw_values, Mapping) and raw_values:
        first_feature, first_value = next(iter(raw_values.items()))
        return (
            f"{profile.get('point_id')} is currently placed in {cluster_id}.{outlier_text} "
            f"One visible raw feature is {first_feature}={first_value}; use this as a concrete clue when assigning the human label."
        )
    return f"{profile.get('point_id')} is currently placed in {cluster_id}.{outlier_text}"


def _profile_threshold_sentence(profile: Mapping[str, Any] | None) -> str:
    if not profile:
        return "No detailed raw-feature checklist is available for this point."
    margins = profile.get("threshold_margins")
    if isinstance(margins, (list, tuple)) and margins:
        closest = min(
            (margin for margin in margins if isinstance(margin, Mapping) and margin.get("absolute_margin") is not None),
            key=lambda margin: margin.get("absolute_margin", 0),
            default=None,
        )
        if closest is not None:
            direction = "satisfies" if closest.get("condition_satisfied") else "does not satisfy"
            return (
                f"For {profile.get('point_id')}, compare {closest.get('feature')}={closest.get('point_value')} "
                f"with the rule check {closest.get('feature')} {closest.get('operator')} {closest.get('threshold')}. "
                f"The point {direction} the check, so the human label tells us whether this cutoff is meaningful."
            )
    return f"For {profile.get('point_id')}, compare its raw feature values with the cited rule conditions, then label by human meaning."


def _fallback_llm_note(point_id: str, profile: Mapping[str, Any] | None) -> str:
    if not profile:
        return (
            f"{point_id}: this is fallback guidance because DeepSeek V4 Pro did not return a live answer. "
            "Treat it as a checklist, not as the final model-written interpretation."
        )
    cluster_id = profile.get("current_cluster_id") or "unassigned"
    outlier = profile.get("outlier")
    outlier_text = ""
    if isinstance(outlier, Mapping):
        status = "marked outlier" if outlier.get("is_outlier") else "not marked outlier"
        outlier_text = f"; outlier score {outlier.get('score')}; {status}"
    return (
        f"{point_id}: fallback guidance. The current analysis placement is {cluster_id}{outlier_text}. "
        "A live DeepSeek V4 Pro answer should turn this evidence into a more specific label hypothesis."
    )


def _rule_target_reason(rule_set: RuleSet, rule_ids: list[str]) -> str:
    rules_by_id = {rule.rule_id: rule for rule in rule_set.rules}
    parts = []
    for rule_id in rule_ids[:2]:
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        condition_text = ", ".join(
            f"{condition.feature} {condition.operator} {condition.threshold:g}"
            for condition in rule.conditions[:3]
        ) or "all matched points"
        parts.append(
            f"{rule.rule_id} targets {rule.target_id} with support {rule.support_count}, "
            f"purity {rule.purity:.2f}, coverage {rule.coverage:.2f}, conditions [{condition_text}]"
        )
    return " ".join(parts)


def _target_ids_for_rules(rule_set: RuleSet, rule_ids: list[str]) -> Tuple[str, ...]:
    rules_by_id = {rule.rule_id: rule for rule in rule_set.rules}
    return tuple(
        dict.fromkeys(
            rule.target_id
            for rule_id in rule_ids
            for rule in (rules_by_id.get(rule_id),)
            if rule is not None
        )
    )


def _decision_impact_for_category(category: str) -> str:
    return {
        "label_priority": "Consistent labels establish the next trusted anchor; mixed labels move the workflow to boundary or exception review.",
        "boundary_review": "Same labels across the boundary weaken the split; different labels support keeping the boundary.",
        "overlap_merge_signal": "Same labels make merge review plausible; mixed labels suggest keeping targets separate or creating a new cluster.",
        "split_or_new_cluster_signal": "A repeated new label supports split/new-cluster review; existing labels support current clusters.",
        "anomaly_label_review": "True-anomaly labels support the outlier rule; normal-member labels support mark-normal feedback.",
        "exception_relabel_review": "Repeated exception labels support relabel or boundary correction; mixed labels require more local labels.",
        "feature_label_strategy": "Agreement validates the threshold checklist; disagreement means the raw threshold story is not enough.",
        "rule_confidence_audit": "Agreement lets the rule guide later checks; disagreement proves rule fidelity to SSDBCODI is not semantic validity.",
    }[category]


def _decision_rationale_for_category(rule_set: RuleSet, category: str, metrics: Mapping[str, Any]) -> str:
    pair = _best_pair(metrics, category)
    candidate_count = len(metrics.get("label_candidate_groups", ()))
    strongest = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count)) if rule_set.rules else None
    pair_clause = (
        f"The most relevant rule pair has relation {pair['relation']}, intersection {pair['intersection_count']}, "
        f"and Jaccard {pair['jaccard_overlap']:.2f}."
        if pair is not None
        else "No sample-level overlap pair is available, so the recommendation must rely on representative or boundary labels."
    )
    strongest_clause = (
        f"The strongest single-rule anchor is {strongest.rule_id} with support {strongest.support_count}, "
        f"purity {strongest.purity:.2f}, and coverage {strongest.coverage:.2f}."
        if strongest is not None
        else "There is no single-rule anchor because no rules were generated."
    )
    category_clause = {
        "label_priority": "The strategy is to maximize label information gain before asking the user to change model state.",
        "boundary_review": "The strategy is to test whether a threshold boundary corresponds to a real semantic distinction.",
        "overlap_merge_signal": "The strategy is to treat merge as a hypothesis that requires direct shared-point labels.",
        "split_or_new_cluster_signal": "The strategy is to test whether separated rule regions share one concept or deserve separate labels.",
        "anomaly_label_review": "The strategy is to distinguish true anomalies from normal cluster members near anomaly rules.",
        "exception_relabel_review": "The strategy is to resolve direct disagreements between rule matches and current SSDBCODI targets.",
        "feature_label_strategy": "The strategy is to move user attention from projected position to raw feature thresholds used by rules.",
        "rule_confidence_audit": "The strategy is to audit rule reliability before any expensive merge, split, or anomaly decision.",
    }[category]
    return f"{category_clause} {pair_clause} {strongest_clause} The current payload exposes {candidate_count} candidate label group(s), so the suggested action is grounded but still needs human labels for semantic validity."


def _label_outcomes_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    pair = _best_pair(metrics, category)
    candidates = tuple(metrics.get("label_candidate_groups", ()))
    point_ids = list(candidates[0].get("point_ids", ()))[:6] if candidates else []
    rule_ids = list(pair["rule_ids"]) if pair is not None else list(target_rule_ids[:3])
    if not point_ids and rule_set.rules:
        seed_rule = next((rule for rule in rule_set.rules if rule.rule_id in rule_ids), rule_set.rules[0])
        point_ids = list(seed_rule.matched_point_ids[:6])

    outcome_templates = {
        "label_priority": (
            ("candidate labels are consistent with the cited rule target", "use the rule as a stable reference before reviewing harder boundaries"),
            ("candidate labels are mixed or disagree with the target", "prioritize boundary or exception review before any state change"),
        ),
        "boundary_review": (
            ("labels on both sides of the boundary agree", "the boundary may be too sharp and merge/shared-boundary review becomes plausible"),
            ("labels differ by side of the boundary", "the current threshold boundary is likely meaningful"),
        ),
        "overlap_merge_signal": (
            ("overlap labels show one semantic group", "merge or shared-boundary review becomes plausible"),
            ("overlap labels are mixed", "keep targets separate or inspect whether a new cluster explains the mixed region"),
        ),
        "split_or_new_cluster_signal": (
            ("separated regions receive different labels", "split or new-cluster review becomes plausible"),
            ("separated regions receive the same label", "keep the current cluster but audit the surrogate boundary"),
        ),
        "anomaly_label_review": (
            ("points are labeled true anomaly", "retain or strengthen the anomaly explanation for these rule regions"),
            ("points are labeled normal cluster members", "review outlier flags or mark-normal feedback for these points"),
        ),
        "exception_relabel_review": (
            ("exception points share a consistent user label", "use them as relabel or boundary-fix evidence"),
            ("exception points have mixed labels", "treat them as local ambiguity and gather more labels nearby"),
        ),
        "feature_label_strategy": (
            ("labels align with the raw feature thresholds", "use these thresholds as a domain-facing label checklist"),
            ("labels do not align with the raw feature thresholds", "downgrade feature-threshold explanation and rely on more labels"),
        ),
        "rule_confidence_audit": (
            ("sample labels agree with high-confidence rules", "rules can guide follow-up review but still should not mutate state alone"),
            ("sample labels disagree with high-confidence rules", "the surrogate is faithful to SSDBCODI but not to user semantics, so refinement is needed"),
        ),
    }[category]

    outcomes = []
    for label_result, implication in outcome_templates:
        outcomes.append(
            {
                "label_result": label_result,
                "decision_implication": implication,
                "rule_ids": rule_ids,
                "point_ids": point_ids,
            }
        )
    return tuple(outcomes)


def _evidence_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    evidence = []
    pair = _best_pair(metrics, category)
    if pair is not None:
        evidence.append(
            {
                "rule_ids": list(pair["rule_ids"]),
                "target_ids": list(pair["target_ids"]),
                "relation": pair["relation"],
                "intersection_count": pair["intersection_count"],
                "jaccard_overlap": pair["jaccard_overlap"],
                "point_ids": list(pair.get("point_ids", ())[:6]),
            }
        )

    rules_by_id = {rule.rule_id: rule for rule in rule_set.rules}
    for rule_id in target_rule_ids[:3]:
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        entry = {
            "rule_id": rule.rule_id,
            "target_kind": rule.target_kind,
            "target_id": rule.target_id,
            "support_count": rule.support_count,
            "coverage": rule.coverage,
            "purity": rule.purity,
            "exception_count": len(rule.exception_point_ids),
        }
        if rule.conditions:
            condition = rule.conditions[0]
            entry.update(
                {
                    "feature": condition.feature,
                    "operator": condition.operator,
                    "threshold": condition.threshold,
                }
            )
        evidence.append(entry)
    return tuple(evidence)


def _findings_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    findings = []
    pair = _best_pair(metrics, category)
    if pair is not None:
        findings.extend(
            [
                {
                    "metric": "pair_intersection_count",
                    "value": pair["intersection_count"],
                    "rule_ids": list(pair["rule_ids"]),
                    "interpretation": "number of matched points shared by both rules",
                },
                {
                    "metric": "pair_jaccard_overlap",
                    "value": pair["jaccard_overlap"],
                    "rule_ids": list(pair["rule_ids"]),
                    "interpretation": "shared points divided by the union of both rule matches",
                },
            ]
        )

    if rule_set.rules:
        strongest_rule = max(rule_set.rules, key=lambda rule: (rule.purity, rule.coverage, rule.support_count))
        findings.extend(
            [
                {
                    "metric": "strongest_rule_support",
                    "value": strongest_rule.support_count,
                    "rule_ids": [strongest_rule.rule_id],
                    "interpretation": "number of points covered by the strongest rule",
                },
                {
                    "metric": "strongest_rule_confidence_score",
                    "value": round(strongest_rule.purity * strongest_rule.coverage, 6),
                    "rule_ids": [strongest_rule.rule_id],
                    "interpretation": "purity multiplied by coverage for prioritizing label checks",
                },
            ]
        )

    findings.append(
        {
            "metric": "exception_point_count",
            "value": metrics.get("exception_point_count", 0),
            "interpretation": "unique rule exception points that should be considered for relabel review",
        }
    )
    return tuple(findings)


def _actions_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
    target_rule_ids: Tuple[str, ...],
) -> Tuple[Mapping[str, Any], ...]:
    candidates = tuple(metrics.get("label_candidate_groups", ()))
    top_candidate = candidates[0] if candidates else {}
    point_ids = list(top_candidate.get("point_ids", ()))
    rule_ids = list(top_candidate.get("rule_ids", target_rule_ids))
    if not point_ids and rule_set.rules:
        rules_by_id = {rule.rule_id: rule for rule in rule_set.rules}
        seed_rule = rules_by_id.get(target_rule_ids[0]) if target_rule_ids else rule_set.rules[0]
        point_ids = list(seed_rule.matched_point_ids[:6])
        rule_ids = [seed_rule.rule_id]

    action_type = {
        "anomaly_label_review": "confirm_anomaly",
        "exception_relabel_review": "ask_domain_label",
        "rule_confidence_audit": "audit_rule",
        "split_or_new_cluster_signal": "create_cluster",
        "overlap_merge_signal": "merge_clusters",
    }.get(category, "inspect_points")
    if category == "overlap_merge_signal" and _best_pair(metrics, category) is None:
        action_type = "inspect_points"

    return (
        {
            "action_type": action_type,
            "priority": "high" if category in {"label_priority", "overlap_merge_signal", "exception_relabel_review"} else "medium",
            "rule_ids": rule_ids[:3],
            "point_ids": point_ids[:6],
            "reason": _recommendation_for_category(category, metrics),
            "hypothesis": _action_hypothesis_for_category(category, metrics),
            "why_this_action": _why_action_for_category(category, metrics),
            "expected_outcomes": list(_expected_outcomes_for_category(category)),
            "risk_note": _risk_note_for_category(category, metrics),
        },
    )


def _action_hypothesis_for_category(category: str, metrics: Mapping[str, Any]) -> str:
    pair = _best_pair(metrics, category)
    pair_text = (
        f"the selected rule pair relation is {pair['relation']} with Jaccard {pair['jaccard_overlap']:.2f}"
        if pair is not None
        else "no sample-overlap pair is available"
    )
    return {
        "label_priority": f"The selected points are expected to reduce the most uncertainty because {pair_text}.",
        "boundary_review": f"The raw-threshold boundary may or may not correspond to a real semantic boundary because {pair_text}.",
        "overlap_merge_signal": f"Merge is only plausible if shared or representative points receive the same human label; currently {pair_text}.",
        "split_or_new_cluster_signal": f"Separated rule regions may represent distinct concepts rather than one target; currently {pair_text}.",
        "anomaly_label_review": "The anomaly rule may contain true anomalies, boundary cases, or normal cluster members that need human confirmation.",
        "exception_relabel_review": "Exception points may reveal mislabeled points, a broken boundary, or a small hidden group.",
        "feature_label_strategy": "The repeated raw thresholds may be a useful human labeling checklist, but only if user labels align with them.",
        "rule_confidence_audit": "The rule may be faithful to SSDBCODI output without being semantically correct for the user.",
    }[category]


def _why_action_for_category(category: str, metrics: Mapping[str, Any]) -> str:
    candidate_groups = tuple(metrics.get("label_candidate_groups", ()))
    candidate_clause = (
        f"The action uses {len(candidate_groups[0].get('point_ids', ())) } candidate point(s) from the highest-ranked group."
        if candidate_groups
        else "The action falls back to representative matched points because no higher-ambiguity candidate group exists."
    )
    metric_clause = f"The rule set contains {metrics.get('cluster_rule_count', 0)} cluster rule(s), {metrics.get('anomaly_rule_count', 0)} anomaly rule(s), and {metrics.get('exception_point_count', 0)} exception point(s)."
    return f"{candidate_clause} {metric_clause} These labels are more informative than random labels because they test the rule relation that would most affect the next refinement decision."


def _expected_outcomes_for_category(category: str) -> Tuple[Mapping[str, str], ...]:
    return {
        "label_priority": (
            {"label_result": "labels agree with the current rule target", "decision_implication": "use the rule as an anchor for later review"},
            {"label_result": "labels disagree or are mixed", "decision_implication": "switch to boundary, exception, or split review before changing state"},
        ),
        "boundary_review": (
            {"label_result": "both sides get the same label", "decision_implication": "boundary may be artificial and merge/shared-boundary review becomes plausible"},
            {"label_result": "each side gets a different label", "decision_implication": "boundary is likely meaningful and should be preserved"},
        ),
        "overlap_merge_signal": (
            {"label_result": "overlap labels agree", "decision_implication": "merge or shared-boundary review becomes plausible"},
            {"label_result": "overlap labels are mixed", "decision_implication": "keep separate targets or investigate a new cluster"},
        ),
        "split_or_new_cluster_signal": (
            {"label_result": "regions get different labels", "decision_implication": "split or new-cluster review becomes plausible"},
            {"label_result": "regions get the same label", "decision_implication": "current target may be coherent despite multiple rules"},
        ),
        "anomaly_label_review": (
            {"label_result": "points are true anomalies", "decision_implication": "retain or strengthen the anomaly rule explanation"},
            {"label_result": "points are normal members", "decision_implication": "review outlier flags or mark them normal"},
        ),
        "exception_relabel_review": (
            {"label_result": "exceptions share one label", "decision_implication": "use them as relabel or boundary-fix evidence"},
            {"label_result": "exceptions are mixed", "decision_implication": "collect more nearby labels before changing structure"},
        ),
        "feature_label_strategy": (
            {"label_result": "labels align with raw thresholds", "decision_implication": "use these thresholds in the labeling checklist"},
            {"label_result": "labels do not align with raw thresholds", "decision_implication": "treat the surrogate feature story as weak"},
        ),
        "rule_confidence_audit": (
            {"label_result": "labels agree with high-confidence rules", "decision_implication": "rules are usable as guidance for follow-up checks"},
            {"label_result": "labels disagree with high-confidence rules", "decision_implication": "rule fidelity to SSDBCODI is not enough; user semantics require refinement"},
        ),
    }[category]


def _risk_note_for_category(category: str, metrics: Mapping[str, Any]) -> str:
    if category == "overlap_merge_signal" and _best_pair(metrics, category) is None:
        return "There is no sample-level overlap, so this action must not be treated as merge evidence yet."
    if category == "exception_relabel_review" and not metrics.get("exception_point_count", 0):
        return "No exception points are present, so the action relies on representative points rather than direct contradictions."
    if category == "anomaly_label_review" and not metrics.get("anomaly_rule_count", 0):
        return "No anomaly rule is present, so anomaly review should wait for outlier evidence."
    return "The recommendation is grounded in surrogate rules, but semantic correctness still depends on user labels."


def _warnings_for_category(
    rule_set: RuleSet,
    category: str,
    metrics: Mapping[str, Any],
) -> Tuple[str, ...]:
    warnings = []
    if not rule_set.rules:
        warnings.append("empty_rule_set")
    if category == "anomaly_label_review" and not metrics.get("anomaly_rule_count", 0):
        warnings.append("no_anomaly_rule_generated")
    if category == "exception_relabel_review" and not metrics.get("exception_point_count", 0):
        warnings.append("no_exception_points_in_rules")
    if category == "overlap_merge_signal":
        pair = _best_pair(metrics, category)
        if pair is None or pair["intersection_count"] == 0:
            warnings.append("no_sample_overlap_in_top_pair")
    if any(rule.purity < 0.8 for rule in rule_set.rules):
        warnings.append("some_rules_have_low_purity")
    return tuple(warnings)


def _confidence_for_category(category: str, metrics: Mapping[str, Any]) -> float:
    pair = _best_pair(metrics, category)
    candidate_bonus = 0.08 if metrics.get("label_candidate_groups") else 0.0
    if pair is None:
        return round(0.58 + candidate_bonus, 2)
    overlap_signal = max(pair["jaccard_overlap"], pair["overlap_share_a"], pair["overlap_share_b"])
    return round(min(0.9, 0.62 + overlap_signal * 0.22 + candidate_bonus), 2)


def _category_descriptions() -> Mapping[str, str]:
    return {
        "label_priority": "Rank the next points or rule regions the user should label first.",
        "boundary_review": "Use rule boundaries to decide whether neighboring regions need labels.",
        "overlap_merge_signal": "Explain whether overlapping rules suggest merge/shared-boundary review.",
        "split_or_new_cluster_signal": "Explain whether separated or weakly covered rule regions suggest split or new-cluster review.",
        "anomaly_label_review": "Identify outlier-rule points that need true-anomaly versus normal-member labels.",
        "exception_relabel_review": "Prioritize rule exception points for relabel or boundary correction.",
        "feature_label_strategy": "Turn raw feature thresholds into a checklist for manual labeling.",
        "rule_confidence_audit": "Audit support, coverage, purity, exceptions, and warnings before refinement.",
    }


def _validate_focus_category(focus_category: str | None) -> str | None:
    if focus_category in (None, ""):
        return None
    if not isinstance(focus_category, str):
        raise ValueError("focus_category must be a string")
    cleaned = focus_category.strip()
    if cleaned not in RULE_INTERPRETATION_CATEGORIES:
        raise ValueError(f"focus_category must be one of: {', '.join(RULE_INTERPRETATION_CATEGORIES)}")
    return cleaned


def _analysis_summary(analysis_result: AnalysisResult | None) -> Mapping[str, Any]:
    if analysis_result is None:
        return {}
    score_values = [score.score for score in analysis_result.outlier_result.scores]
    return {
        "analysis_run_id": analysis_result.analysis_run_id,
        "cluster_counts": cluster_counts(analysis_result.cluster_result),
        "outlier_point_ids": list(analysis_result.outlier_result.outlier_point_ids),
        "outlier_score_min": min(score_values) if score_values else None,
        "outlier_score_max": max(score_values) if score_values else None,
        "provider": analysis_result.diagnostics.get("provider"),
    }


def _known_features(rule_set: RuleSet, feature_matrix: FeatureMatrix | None) -> Tuple[str, ...]:
    if feature_matrix is not None:
        return tuple(feature_matrix.feature_names)
    raw_features = rule_set.diagnostics.get("raw_feature_names")
    if isinstance(raw_features, (list, tuple)):
        return tuple(str(feature) for feature in raw_features)
    return tuple(sorted({condition.feature for rule in rule_set.rules for condition in rule.conditions}))


def _known_point_ids(rule_set: RuleSet, feature_matrix: FeatureMatrix | None) -> set[str]:
    if feature_matrix is not None:
        return set(feature_matrix.point_ids)
    point_ids = set()
    for rule in rule_set.rules:
        point_ids.update(rule.matched_point_ids)
        point_ids.update(rule.exception_point_ids)
    return point_ids


def _known_thresholds(rule_set: RuleSet) -> Tuple[Mapping[str, Any], ...]:
    thresholds = []
    seen = set()
    for rule in rule_set.rules:
        for condition in rule.conditions:
            key = (condition.feature, condition.operator, condition.threshold)
            if key in seen:
                continue
            seen.add(key)
            thresholds.append(
                {
                    "feature": condition.feature,
                    "operator": condition.operator,
                    "threshold": condition.threshold,
                }
            )
    return tuple(thresholds)


def _string_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a list of strings")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    cleaned = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        cleaned.append(item.strip())
    return tuple(cleaned)


_USER_TEXT_REPLACEMENTS = (
    ("unusualness score", "outlier score"),
    ("Unusualness score", "Outlier score"),
    ("anomaly score", "outlier score"),
    ("Anomaly score", "Outlier score"),
    ("semantic label", "human label"),
    ("semantic labels", "human labels"),
    ("semantic group", "human group"),
    ("semantic groups", "human groups"),
    ("semantic class", "human class"),
    ("SSDBCODI", "current analysis"),
    ("Jaccard", "overlap score"),
    ("threshold", "cutoff"),
    ("Threshold", "Cutoff"),
)


def _normalize_user_facing_terms(interpretation: RuleInterpretation) -> RuleInterpretation:
    return RuleInterpretation(
        interpretation_id=interpretation.interpretation_id,
        rule_set_id=interpretation.rule_set_id,
        categories=interpretation.categories,
        target_rule_ids=interpretation.target_rule_ids,
        summary=_normalize_user_text(interpretation.summary),
        evidence=interpretation.evidence,
        recommendation=_normalize_user_text(interpretation.recommendation),
        category_explanation=_normalize_user_text(interpretation.category_explanation),
        decision_rationale=_normalize_user_text(interpretation.decision_rationale),
        label_targets=tuple(_normalize_mapping_text(item) for item in interpretation.label_targets),
        suspicion_reasons=tuple(_normalize_mapping_text(item) for item in interpretation.suspicion_reasons),
        point_label_guidance=tuple(_normalize_mapping_text(item) for item in interpretation.point_label_guidance),
        label_outcomes=tuple(_normalize_mapping_text(item) for item in interpretation.label_outcomes),
        quantitative_findings=interpretation.quantitative_findings,
        suggested_label_actions=tuple(_normalize_mapping_text(item) for item in interpretation.suggested_label_actions),
        confidence=interpretation.confidence,
        warnings=interpretation.warnings,
        provider_label=interpretation.provider_label,
    )


def _normalize_mapping_text(item: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = {}
    for key, value in item.items():
        if isinstance(value, str):
            normalized[key] = _normalize_user_text(value)
        elif isinstance(value, Mapping):
            normalized[key] = _normalize_mapping_text(value)
        elif isinstance(value, (list, tuple)):
            normalized[key] = [
                _normalize_mapping_text(child)
                if isinstance(child, Mapping)
                else _normalize_user_text(child)
                if isinstance(child, str)
                else child
                for child in value
            ]
        else:
            normalized[key] = value
    return normalized


def _normalize_user_text(value: str | None) -> str | None:
    if value is None:
        return value
    normalized = value
    for source, target in _USER_TEXT_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return normalized


def _has_no_typical_case_warning(warnings: Tuple[str, ...]) -> bool:
    return any(
        warning in {"no_typical_case_for_category", "no_related_points_for_category"}
        for warning in warnings
    )


def _evidence_tuple(value: Any, rule_set: RuleSet) -> Tuple[Mapping[str, Any], ...]:
    return _reference_tuple(value, rule_set, field_name="evidence", require_non_empty=False)


def _reference_tuple(
    value: Any,
    rule_set: RuleSet,
    *,
    field_name: str,
    require_non_empty: bool,
) -> Tuple[Mapping[str, Any], ...]:
    if value is None:
        if require_non_empty:
            raise ValueError(f"{field_name} must be a non-empty list")
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    if require_non_empty and not value:
        raise ValueError(f"{field_name} must be a non-empty list")

    known_rule_ids = {rule.rule_id for rule in rule_set.rules}
    known_target_ids = {rule.target_id for rule in rule_set.rules}
    known_features = set(_known_features(rule_set, None))
    known_point_ids = _known_point_ids(rule_set, None)
    known_thresholds = _known_thresholds(rule_set)

    entries = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name} entries must be objects")
        entry = dict(item)
        _validate_evidence_rule_id(entry, known_rule_ids)
        _validate_evidence_feature(entry, known_features)
        _validate_evidence_target(entry, known_target_ids)
        _validate_evidence_points(entry, known_point_ids)
        _validate_evidence_threshold(entry, known_thresholds)
        entries.append(entry)
    return tuple(entries)


def _action_tuple(
    value: Any,
    rule_set: RuleSet,
    *,
    require_non_empty: bool,
) -> Tuple[Mapping[str, Any], ...]:
    actions = _reference_tuple(
        value,
        rule_set,
        field_name="suggested_label_actions",
        require_non_empty=require_non_empty,
    )
    for action in actions:
        action_type = action.get("action_type")
        if action_type not in _ACTION_TYPES:
            raise ValueError(f"suggested_label_actions contains unknown action_type: {action_type}")
        priority = action.get("priority")
        if priority is not None and priority not in {"high", "medium", "low"}:
            raise ValueError("suggested_label_actions priority must be high, medium, or low")
        reason = action.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("suggested_label_actions reason must be a non-empty string")
        hypothesis = action.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("suggested_label_actions hypothesis must be a non-empty string")
        why_this_action = action.get("why_this_action")
        if not isinstance(why_this_action, str) or not why_this_action.strip():
            raise ValueError("suggested_label_actions why_this_action must be a non-empty string")
        risk_note = action.get("risk_note")
        if not isinstance(risk_note, str) or not risk_note.strip():
            raise ValueError("suggested_label_actions risk_note must be a non-empty string")
        expected_outcomes = action.get("expected_outcomes")
        if not isinstance(expected_outcomes, (list, tuple)) or not expected_outcomes:
            raise ValueError("suggested_label_actions expected_outcomes must be a non-empty list")
        _validate_label_outcomes(tuple(dict(item) if isinstance(item, Mapping) else item for item in expected_outcomes))
    return actions


def _validate_label_outcomes(outcomes: Tuple[Any, ...]) -> None:
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            raise ValueError("label_outcomes entries must be objects")
        label_result = outcome.get("label_result")
        if not isinstance(label_result, str) or not label_result.strip():
            raise ValueError("label_outcomes label_result must be a non-empty string")
        implication = outcome.get("decision_implication")
        if not isinstance(implication, str) or not implication.strip():
            raise ValueError("label_outcomes decision_implication must be a non-empty string")


def _validate_label_targets(targets: Tuple[Mapping[str, Any], ...]) -> None:
    for target in targets:
        priority = target.get("priority")
        if priority is not None and priority not in {"high", "medium", "low"}:
            raise ValueError("label_targets priority must be high, medium, or low")
        point_ids = target.get("point_ids")
        if not isinstance(point_ids, (list, tuple)) or not point_ids:
            raise ValueError("label_targets point_ids must be a non-empty list")
        label_question = target.get("label_question")
        if not isinstance(label_question, str) or not label_question.strip():
            raise ValueError("label_targets label_question must be a non-empty string")
        why = target.get("why_label_these_points")
        if not isinstance(why, str) or not why.strip():
            raise ValueError("label_targets why_label_these_points must be a non-empty string")


def _validate_suspicion_reasons(reasons: Tuple[Mapping[str, Any], ...]) -> None:
    for reason in reasons:
        point_ids = reason.get("point_ids")
        if not isinstance(point_ids, (list, tuple)) or not point_ids:
            raise ValueError("suspicion_reasons point_ids must be a non-empty list")
        for field_name in ("suspicious_signal", "rule_based_reason", "point_based_reason"):
            value = reason.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"suspicion_reasons {field_name} must be a non-empty string")


def _validate_point_label_guidance(guidance: Tuple[Mapping[str, Any], ...]) -> None:
    for item in guidance:
        point_ids = item.get("point_ids")
        if not isinstance(point_ids, (list, tuple)) or not point_ids:
            raise ValueError("point_label_guidance point_ids must be a non-empty list")
        for field_name in ("suggested_label_frame", "how_to_label", "decision_impact", "llm_analysis_note"):
            value = item.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"point_label_guidance {field_name} must be a non-empty string")


def _validate_evidence_rule_id(entry: Mapping[str, Any], known_rule_ids: set[str]) -> None:
    rule_ids = []
    if entry.get("rule_id") is not None:
        rule_ids.append(entry.get("rule_id"))
    if entry.get("rule_ids") is not None:
        value = entry.get("rule_ids")
        if not isinstance(value, (list, tuple)):
            raise ValueError("rule_ids must be a list")
        rule_ids.extend(value)
    unknown = sorted(str(rule_id) for rule_id in rule_ids if rule_id not in known_rule_ids)
    if unknown:
        raise ValueError(f"entry references unknown rule id(s): {', '.join(unknown)}")


def _validate_evidence_feature(entry: Mapping[str, Any], known_features: set[str]) -> None:
    feature = entry.get("feature")
    if feature is not None and feature not in known_features:
        raise ValueError(f"entry references unknown feature: {feature}")


def _validate_evidence_target(entry: Mapping[str, Any], known_target_ids: set[str]) -> None:
    target_ids = []
    if entry.get("target_id") is not None:
        target_ids.append(entry.get("target_id"))
    if entry.get("target_ids") is not None:
        value = entry.get("target_ids")
        if not isinstance(value, (list, tuple)):
            raise ValueError("target_ids must be a list")
        target_ids.extend(value)
    unknown = sorted(str(target_id) for target_id in target_ids if target_id not in known_target_ids)
    if unknown:
        raise ValueError(f"entry references unknown target id(s): {', '.join(unknown)}")


def _validate_evidence_points(entry: Mapping[str, Any], known_point_ids: set[str]) -> None:
    point_ids = []
    if entry.get("point_id") is not None:
        point_ids.append(entry.get("point_id"))
    if entry.get("point_ids") is not None:
        value = entry.get("point_ids")
        if not isinstance(value, (list, tuple)):
            raise ValueError("point_ids must be a list")
        point_ids.extend(value)
    unknown = sorted(str(point_id) for point_id in point_ids if point_id not in known_point_ids)
    if unknown:
        raise ValueError(f"entry references unknown point id(s): {', '.join(unknown)}")


def _validate_evidence_threshold(entry: Mapping[str, Any], known_thresholds: Tuple[Mapping[str, Any], ...]) -> None:
    if "threshold" not in entry:
        return
    threshold = entry.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be numeric")

    feature = entry.get("feature")
    operator = entry.get("operator")
    for known in known_thresholds:
        if feature is not None and known["feature"] != feature:
            continue
        if operator is not None and known["operator"] != operator:
            continue
        if abs(float(known["threshold"]) - float(threshold)) <= _THRESHOLD_TOLERANCE:
            return
    raise ValueError(f"entry references unknown threshold: {threshold}")


def _trim(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...[trimmed]"
