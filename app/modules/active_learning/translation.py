from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from app.shared.deepseek import DEEPSEEK_PRO_MODEL, DeepSeekClient

from .schemas import PointGuidance, TranslationPacket


PROMPT_VERSION = "active_learning_round_translation_v5"
PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts"
    / "active_learning"
    / "deepseek"
    / "round_guidance_prompt.txt"
)
MAX_OUTPUT_TOKENS = 6000
_USER_FACING_BANNED_PHRASES = (
    "affected-region score",
    "affected region score",
    "candidate rank",
    "category evidence",
    "cluster",
    "cluster lineage",
    "computer agreement alone",
    "deterministic",
    "deserves to be trusted",
    "deserves trust",
    "high information value",
    "human verification",
    "inspect the values",
    "jaccard",
    "outlier score",
    "purity",
    "ranking score",
    "seed",
    "semantic class",
    "semantic label",
    "ssdbcodi",
    "unusualness score",
)
_RAW_ASSIGNMENT_PATTERN = re.compile(
    r"\b[a-zA-Z][\w.-]*\s*=\s*-?\d+(?:\.\d+)?\b"
)


def build_translation_packet(
    *,
    plan: Mapping[str, Any],
    display_rule_set: Mapping[str, Any],
    round_delta: Mapping[str, Any],
    previous_label_events: Sequence[Mapping[str, Any]],
    entity_name: str,
) -> TranslationPacket:
    target_ids = set(plan.get("target_rule_ids", ()))
    target_rules = []
    relevant_features = []
    for rule in display_rule_set.get("rules", ()):
        if rule.get("rule_id") not in target_ids:
            continue
        conditions = [
            condition.get("display_text")
            or (
                f"{condition.get('source_feature') or condition.get('feature')} "
                f"{condition.get('display_operator') or condition.get('operator')} "
                f"{condition.get('display_value', condition.get('threshold'))}"
            )
            for condition in rule.get("conditions", ())
        ]
        for condition in rule.get("conditions", ()):
            feature_name = condition.get("source_feature") or condition.get("feature")
            if feature_name and feature_name not in relevant_features:
                relevant_features.append(feature_name)
        target_rules.append(
            {
                "rule_id": rule.get("rule_id"),
                "target_kind": rule.get("target_kind"),
                "target_id": rule.get("target_id"),
                "conditions": conditions,
                "plain_scope": _scope_label(rule.get("support_count")),
                "plain_reliability": _rule_reliability(
                    rule.get("coverage"),
                    rule.get("purity"),
                ),
                "exception_count": len(rule.get("exception_point_ids", ())),
                "technical_evidence": {
                    "support_count": rule.get("support_count"),
                    "coverage": rule.get("coverage"),
                    "purity": rule.get("purity"),
                },
            }
        )
    previous_diff = dict(plan.get("previous_plan_diff", {}))
    compact_plan = {
        "plan_id": plan.get("plan_id"),
        "plan_version": plan.get("plan_version"),
        "session_id": plan.get("session_id"),
        "round_id": plan.get("round_id"),
        "focus_category": plan.get("focus_category"),
        "recommended_point_ids": list(plan.get("recommended_point_ids", ())),
        "target_rule_ids": list(plan.get("target_rule_ids", ())),
        "previous_plan_diff": {
            "added_point_ids": list(previous_diff.get("added_point_ids", ())),
            "removed_point_ids": list(previous_diff.get("removed_point_ids", ())),
            "retained_point_ids": list(previous_diff.get("retained_point_ids", ())),
        },
        "candidate_rankings": [
            {
                "point_id": item.get("point_id"),
                "recheck_context": _plain_recheck_context(
                    item.get("recheck_reason", "")
                ),
            }
            for item in plan.get("candidate_rankings", ())
            if item.get("point_id") in set(plan.get("recommended_point_ids", ()))
        ],
        "immutable_fields": [
            "plan_id",
            "focus_category",
            "recommended_point_ids",
            "target_rule_ids",
        ],
        "evidence_policy_version": plan.get("evidence_policy_version"),
    }
    profiles = tuple(
        _translation_profile(item, relevant_features=relevant_features)
        for item in plan.get("point_profiles", ())
    )
    evidence_cards = tuple(
        _translation_evidence_card(item)
        for item in plan.get("category_evidence_cards", ())
    )
    return TranslationPacket(
        task="translate_deterministic_active_learning_plan",
        plan=compact_plan,
        target_rules=tuple(target_rules),
        point_profiles=profiles,
        category_evidence_cards=evidence_cards,
        label_options=tuple(plan.get("label_options", ())),
        round_delta={
            "baseline": bool(round_delta.get("baseline")),
            "plain_summary": _plain_round_delta(round_delta),
            "changed_cluster_point_ids": list(
                round_delta.get("changed_cluster_point_ids", ())
            ),
            "outlier_added_point_ids": list(
                round_delta.get("outlier_added_point_ids", ())
            ),
            "outlier_removed_point_ids": list(
                round_delta.get("outlier_removed_point_ids", ())
            ),
        },
        previous_label_events=tuple(dict(item) for item in previous_label_events),
        entity_name=entity_name,
    )


def _translation_evidence_card(
    item: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "point_id": item.get("point_id"),
        "category": item.get("category"),
        "evidence_category": item.get("evidence_category"),
        "category_explanation": item.get("category_explanation"),
        "evidence_policy_version": item.get("evidence_policy_version"),
        "evidence_bullets": [
            _translation_evidence_bullet(bullet)
            for bullet in item.get("evidence_bullets", ())
        ],
        "comparison_targets": [
            {
                "point_id": target.get("point_id"),
                "relation": target.get("relation"),
                "source": target.get("source"),
                "human_label": target.get("human_label"),
                "confirmed_unusual": target.get("confirmed_unusual"),
                "features_to_compare": list(
                    target.get("features_to_compare", ())
                ),
            }
            for target in item.get("comparison_targets", ())
        ],
        "round_context": dict(item.get("round_context", {})),
    }


def _translation_evidence_bullet(
    bullet: Mapping[str, Any],
) -> Dict[str, Any]:
    plain_fact = str(bullet.get("plain_fact", "")).strip()
    point_connection = str(
        bullet.get("point_connection", "")
    ).strip()
    payload = {
        "dimension_id": bullet.get("dimension_id"),
        "question": bullet.get("question"),
        "status": bullet.get("status"),
        "headline": bullet.get("headline"),
        "plain_fact": plain_fact,
        "labeling_value": bullet.get("labeling_value"),
        "evidence_fact_ids": list(
            bullet.get("evidence_fact_ids", ())
        ),
    }
    if point_connection and point_connection != plain_fact:
        payload["point_connection"] = point_connection
    return payload


def _translation_profile(
    item: Mapping[str, Any],
    *,
    relevant_features: Sequence[str],
) -> Dict[str, Any]:
    evidence = dict(item.get("plain_language_evidence", {}))
    raw_features = dict(item.get("raw_features", {}))
    selected_features = []
    for name in relevant_features:
        if name in raw_features and name not in selected_features:
            selected_features.append(name)
    for name in raw_features:
        if name not in selected_features:
            selected_features.append(name)
        if len(selected_features) >= 12:
            break
    return {
        "point_id": item.get("point_id"),
        "current_group_id": item.get("current_cluster_id"),
        "is_currently_flagged_unusual": bool(item.get("is_outlier")),
        "supporting_values": {
            name: raw_features[name] for name in selected_features[:12]
        },
        "supporting_metadata": dict(
            list(dict(item.get("metadata", {})).items())[:6]
        ),
        "plain_language_evidence": evidence,
        "related_rule_ids": list(item.get("related_rule_ids", ())),
        "does_not_fit_a_current_rule": bool(item.get("is_rule_exception")),
        "group_changed_after_previous_labels": bool(item.get("cluster_changed")),
        "unusual_status_changed_after_previous_labels": bool(
            item.get("outlier_changed")
        ),
        "covered_categories": list(item.get("covered_categories", ())),
    }


def _scope_label(value: Any) -> str:
    count = int(value or 0)
    if count >= 30:
        return "many records"
    if count >= 8:
        return "several records"
    return "a few records"


def _rule_reliability(coverage: Any, purity: Any) -> str:
    coverage_value = float(coverage or 0.0)
    purity_value = float(purity or 0.0)
    if coverage_value >= 0.7 and purity_value >= 0.9:
        return "strong in the current analysis"
    if coverage_value >= 0.4 and purity_value >= 0.75:
        return "moderate in the current analysis"
    return "limited in the current analysis"


def _plain_recheck_context(value: Any) -> str:
    return {
        "cluster_changed_after_label": (
            "The record was labeled before, but the latest analysis moved it "
            "to a different group."
        ),
        "outlier_status_changed_after_label": (
            "The record was labeled before, but the latest analysis changed "
            "whether it treats the record as unusual."
        ),
        "current_rule_conflicts_with_existing_label": (
            "The current rule does not agree with the existing human label."
        ),
    }.get(str(value or ""), "")


def _plain_round_delta(round_delta: Mapping[str, Any]) -> str:
    if round_delta.get("baseline"):
        return "This is the first round, before any human labels have changed the analysis."
    group_changes = len(round_delta.get("changed_cluster_point_ids", ()))
    unusual_changes = len(round_delta.get("outlier_added_point_ids", ())) + len(
        round_delta.get("outlier_removed_point_ids", ())
    )
    if group_changes and unusual_changes:
        return (
            "The previous labels changed some group assignments and also changed "
            "which records the system treats as unusual."
        )
    if group_changes:
        return "The previous labels changed some group assignments."
    if unusual_changes:
        return (
            "The previous labels changed which records the system treats as unusual."
        )
    return (
        "The previous labels did not change the main results, so this round checks "
        "the questions that remain unresolved."
    )


def translate_plan(
    packet: TranslationPacket,
    *,
    provider_kind: str,
    deterministic_fallback: Mapping[str, Any],
) -> Mapping[str, Any]:
    provider_kind = str(provider_kind or "mock").strip().lower()
    if provider_kind == "mock":
        return {
            "guidance": dict(deterministic_fallback),
            "diagnostics": {
                "provider_kind": "mock",
                "provider_label": "deterministic_active_learning_translator",
                "used_fallback": False,
                "prompt_template_version": PROMPT_VERSION,
                "translation_packet_char_count": len(
                    json.dumps(packet.to_dict(), sort_keys=True)
                ),
            },
        }
    if provider_kind != "deepseek":
        raise ValueError("provider_kind must be mock or deepseek")

    client = DeepSeekClient(
        model_name=DEEPSEEK_PRO_MODEL,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    prompt = _prompt(packet)
    attempts = []
    last_error = None
    previous_output = ""
    for attempt in ("direct_json", "schema_repair"):
        attempt_prompt = prompt
        if attempt == "schema_repair":
            attempt_prompt = (
                f"{prompt}\n\nThe previous response failed validation: {last_error}. "
                f"Repair this JSON without changing immutable IDs:\n{previous_output}"
            )
        try:
            payload, raw_response, metadata = client.generate_json_with_metadata(
                attempt_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                thinking={"type": "disabled"},
                reasoning_effort=None,
                temperature=0.0,
            )
            response_model = metadata.get("model")
            if response_model != DEEPSEEK_PRO_MODEL:
                raise ValueError(
                    f"DeepSeek returned model {response_model}; "
                    f"expected {DEEPSEEK_PRO_MODEL}"
                )
            previous_output = raw_response
            partial_fallback_fields = []
            try:
                guidance = validate_guidance(payload, packet)
            except ValueError as validation_error:
                guidance, partial_fallback_fields = _merge_partial_guidance(
                    payload,
                    deterministic_fallback,
                    packet,
                )
                if not partial_fallback_fields:
                    raise validation_error
            attempts.append(
                {
                    "attempt": attempt,
                    "success": True,
                    "partial_fallback_fields": list(
                        partial_fallback_fields
                    ),
                    "response_metadata": dict(metadata),
                }
            )
            return {
                "guidance": guidance,
                "diagnostics": {
                    "provider_kind": "deepseek",
                    "provider_label": f"deepseek:{DEEPSEEK_PRO_MODEL}",
                    "requested_model_name": DEEPSEEK_PRO_MODEL,
                    "model_name": response_model,
                    "using_deepseek_v4_pro": True,
                    "used_fallback": False,
                    "partial_fallback": bool(partial_fallback_fields),
                    "partial_fallback_fields": list(
                        partial_fallback_fields
                    ),
                    "temperature": 0.0,
                    "thinking": {"type": "disabled"},
                    "prompt_template_path": str(PROMPT_PATH),
                    "prompt_template_version": PROMPT_VERSION,
                    "prompt_char_count": len(attempt_prompt),
                    "translation_packet_char_count": len(
                        json.dumps(packet.to_dict(), sort_keys=True)
                    ),
                    "attempts": attempts,
                    "token_usage": dict(metadata.get("usage", {})),
                    "finish_reason": metadata.get("finish_reason"),
                    "message_keys": list(metadata.get("message_keys", ())),
                    "system_fingerprint": metadata.get(
                        "system_fingerprint"
                    ),
                },
            }
        except Exception as exc:
            last_error = str(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "success": False,
                    "error": last_error,
                }
            )
    return {
        "guidance": {
            **dict(deterministic_fallback),
            "provider_kind": "deterministic_fallback",
            "warnings": [
                *list(deterministic_fallback.get("warnings", ())),
                "deepseek_fallback_used",
            ],
        },
        "diagnostics": {
            "provider_kind": "deepseek",
            "provider_label": f"deepseek:{DEEPSEEK_PRO_MODEL}->deterministic",
            "requested_model_name": DEEPSEEK_PRO_MODEL,
            "model_name": None,
            "using_deepseek_v4_pro": False,
            "used_fallback": True,
            "error": last_error,
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
            "prompt_template_path": str(PROMPT_PATH),
            "prompt_template_version": PROMPT_VERSION,
            "prompt_char_count": len(prompt),
            "translation_packet_char_count": len(
                json.dumps(packet.to_dict(), sort_keys=True)
            ),
            "attempts": attempts,
        },
    }


def validate_guidance(
    payload: Mapping[str, Any],
    packet: TranslationPacket,
) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("DeepSeek guidance must be a JSON object")
    body = payload.get("guidance", payload)
    if not isinstance(body, Mapping):
        raise ValueError("guidance must be an object")
    plan = packet.plan
    if body.get("plan_id") != plan.get("plan_id"):
        raise ValueError("DeepSeek changed plan_id")
    if body.get("category") != plan.get("focus_category"):
        raise ValueError("DeepSeek changed the focus category")
    expected_ids = tuple(plan.get("recommended_point_ids", ()))
    returned_ids = tuple(body.get("recommended_point_ids", ()))
    if returned_ids != expected_ids:
        raise ValueError("DeepSeek changed or reordered recommended point IDs")
    expected_rules = tuple(plan.get("target_rule_ids", ()))
    returned_rules = tuple(body.get("target_rule_ids", ()))
    if returned_rules != expected_rules:
        raise ValueError("DeepSeek changed or reordered target rule IDs")
    expected_label_options = tuple(packet.label_options)
    returned_label_options = tuple(body.get("label_options", ()))
    if returned_label_options != expected_label_options:
        raise ValueError("DeepSeek changed or reordered label options")
    guidance_items = body.get("point_guidance")
    if not isinstance(guidance_items, list):
        raise ValueError("point_guidance must be a list")
    evidence_cards = {
        item.get("point_id"): dict(item)
        for item in packet.category_evidence_cards
    }
    guidance_by_id = {}
    for raw_item in guidance_items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("point_guidance entries must be objects")
        point_id = raw_item.get("point_id")
        if point_id in guidance_by_id:
            raise ValueError("point_guidance must contain each point exactly once")
        for field_name in (
            "why_selected",
            "what_changed_since_last_round",
            "how_to_label",
        ):
            if not isinstance(raw_item.get(field_name), str) or not raw_item[field_name].strip():
                raise ValueError(f"point_guidance {field_name} must be non-empty")
            _validate_plain_language(
                raw_item[field_name],
                field_name=f"point_guidance.{field_name}",
            )
        outcomes = raw_item.get("possible_outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise ValueError("point_guidance possible_outcomes must be non-empty")
        if len(outcomes) > 2:
            raise ValueError("point_guidance possible_outcomes must contain at most two items")
        for outcome in outcomes:
            _validate_plain_language(
                str(outcome),
                field_name="point_guidance.possible_outcomes",
            )
        evidence_bullets = raw_item.get("evidence_bullets")
        if not isinstance(evidence_bullets, list):
            raise ValueError(
                "point_guidance evidence_bullets must be a list"
            )
        expected_card = evidence_cards.get(point_id)
        if expected_card is None:
            raise ValueError(
                "point_guidance references a point without an evidence card"
            )
        expected_bullets = tuple(
            expected_card.get("evidence_bullets", ())
        )
        if len(evidence_bullets) != len(expected_bullets):
            raise ValueError(
                "point_guidance evidence_bullets must cover every fixed dimension"
            )
        validated_bullets = []
        for returned_bullet, expected_bullet in zip(
            evidence_bullets,
            expected_bullets,
        ):
            if not isinstance(returned_bullet, Mapping):
                raise ValueError(
                    "point_guidance evidence bullet must be an object"
                )
            if returned_bullet.get("dimension_id") != expected_bullet.get(
                "dimension_id"
            ):
                raise ValueError(
                    "DeepSeek changed or reordered evidence dimensions"
                )
            if returned_bullet.get("status") != expected_bullet.get("status"):
                raise ValueError("DeepSeek changed an evidence status")
            if tuple(returned_bullet.get("evidence_fact_ids", ())) != tuple(
                expected_bullet.get("evidence_fact_ids", ())
            ):
                raise ValueError(
                    "DeepSeek changed evidence fact references"
                )
            for field_name in (
                "headline",
                "explanation",
                "why_this_point",
            ):
                value = returned_bullet.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"point_guidance evidence bullet {field_name} "
                        "must be non-empty"
                    )
                _validate_plain_language(
                    value,
                    field_name=(
                        "point_guidance.evidence_bullets."
                        f"{field_name}"
                    ),
                )
            _validate_labeling_value(
                returned_bullet["why_this_point"],
                field_name=(
                    "point_guidance.evidence_bullets."
                    "why_this_point"
                ),
            )
            validated_bullets.append(
                {
                    "dimension_id": expected_bullet["dimension_id"],
                    "question": expected_bullet["question"],
                    "status": expected_bullet["status"],
                    "headline": returned_bullet["headline"].strip(),
                    "explanation": returned_bullet[
                        "explanation"
                    ].strip(),
                    "why_this_point": returned_bullet[
                        "why_this_point"
                    ].strip(),
                    "evidence_fact_ids": list(
                        expected_bullet.get("evidence_fact_ids", ())
                    ),
                    "technical_details": dict(
                        expected_bullet.get("technical_details", {})
                    ),
                }
            )
        expected_target_ids = tuple(
            item.get("point_id")
            for item in expected_card.get("comparison_targets", ())
        )
        returned_target_ids = tuple(
            raw_item.get("comparison_target_ids", ())
        )
        if returned_target_ids != expected_target_ids:
            raise ValueError(
                "DeepSeek changed or reordered comparison target IDs"
            )
        guidance_by_id[point_id] = PointGuidance(
            point_id=point_id,
            why_selected=raw_item["why_selected"].strip(),
            what_changed_since_last_round=raw_item[
                "what_changed_since_last_round"
            ].strip(),
            how_to_label=raw_item["how_to_label"].strip(),
            possible_outcomes=tuple(str(value).strip() for value in outcomes),
            evidence_bullets=tuple(validated_bullets),
            comparison_targets=tuple(
                dict(item)
                for item in expected_card.get("comparison_targets", ())
            ),
        )
    if tuple(guidance_by_id) != expected_ids:
        raise ValueError("point_guidance must cover recommended points in order")
    summary = body.get("summary")
    explanation = body.get("category_explanation")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be non-empty")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("category_explanation must be non-empty")
    expected_explanation = next(
        (
            str(item.get("category_explanation", "")).strip()
            for item in packet.category_evidence_cards
            if item.get("category_explanation")
        ),
        "",
    )
    if expected_explanation and explanation.strip() != expected_explanation:
        raise ValueError("DeepSeek changed the fixed category explanation")
    _validate_plain_language(summary, field_name="summary")
    _validate_plain_language(explanation, field_name="category_explanation")
    return {
        "provider_kind": "deepseek",
        "plan_id": plan["plan_id"],
        "category": plan["focus_category"],
        "recommended_point_ids": list(expected_ids),
        "target_rule_ids": list(expected_rules),
        "label_options": list(expected_label_options),
        "category_explanation": (
            expected_explanation or explanation.strip()
        ),
        "summary": summary.strip(),
        "point_guidance": [
            guidance_by_id[point_id].to_dict() for point_id in expected_ids
        ],
        "warnings": [
            str(value) for value in body.get("warnings", ())
        ],
    }


def _merge_partial_guidance(
    payload: Mapping[str, Any],
    deterministic_fallback: Mapping[str, Any],
    packet: TranslationPacket,
) -> tuple[Dict[str, Any], list[str]]:
    if not isinstance(payload, Mapping):
        raise ValueError("DeepSeek guidance must be a JSON object")
    body = payload.get("guidance", payload)
    if not isinstance(body, Mapping):
        raise ValueError("guidance must be an object")
    plan = packet.plan
    immutable_pairs = (
        ("plan_id", plan.get("plan_id")),
        ("category", plan.get("focus_category")),
        (
            "recommended_point_ids",
            list(plan.get("recommended_point_ids", ())),
        ),
        ("target_rule_ids", list(plan.get("target_rule_ids", ()))),
        ("label_options", list(packet.label_options)),
    )
    for field_name, expected in immutable_pairs:
        if body.get(field_name) != expected:
            raise ValueError(
                f"DeepSeek changed immutable field {field_name}"
            )

    raw_items = body.get("point_guidance")
    if not isinstance(raw_items, list):
        raise ValueError("point_guidance must be a list")
    raw_by_id = {
        item.get("point_id"): item
        for item in raw_items
        if isinstance(item, Mapping)
    }
    expected_ids = tuple(plan.get("recommended_point_ids", ()))
    if tuple(
        item.get("point_id")
        for item in raw_items
        if isinstance(item, Mapping)
    ) != expected_ids:
        raise ValueError(
            "DeepSeek changed or reordered recommended point guidance"
        )
    fallback_by_id = {
        item.get("point_id"): item
        for item in deterministic_fallback.get("point_guidance", ())
        if isinstance(item, Mapping)
    }
    cards_by_id = {
        item.get("point_id"): item
        for item in packet.category_evidence_cards
    }
    fallback_fields = []

    def text_or_fallback(
        raw_value: Any,
        fallback_value: Any,
        field_path: str,
    ) -> str:
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                _validate_plain_language(
                    raw_value,
                    field_name=field_path,
                )
                return raw_value.strip()
            except ValueError:
                pass
        fallback_fields.append(field_path)
        return str(fallback_value or "").strip()

    merged_items = []
    for point_id in expected_ids:
        raw_item = raw_by_id[point_id]
        fallback_item = fallback_by_id.get(point_id, {})
        card = cards_by_id[point_id]
        expected_bullets = tuple(card.get("evidence_bullets", ()))
        raw_bullets = raw_item.get("evidence_bullets")
        raw_bullets_by_dimension = {
            item.get("dimension_id"): item
            for item in raw_bullets
            if isinstance(item, Mapping)
        } if isinstance(raw_bullets, list) else {}
        fallback_bullets_by_dimension = {
            item.get("dimension_id"): item
            for item in fallback_item.get("evidence_bullets", ())
            if isinstance(item, Mapping)
        }
        merged_bullets = []
        for expected_bullet in expected_bullets:
            dimension_id = expected_bullet.get("dimension_id")
            raw_bullet = raw_bullets_by_dimension.get(dimension_id, {})
            fallback_bullet = fallback_bullets_by_dimension.get(
                dimension_id,
                {},
            )
            contract_matches = (
                raw_bullet.get("status") == expected_bullet.get("status")
                and tuple(raw_bullet.get("evidence_fact_ids", ()))
                == tuple(expected_bullet.get("evidence_fact_ids", ()))
            )
            headline = ""
            explanation = ""
            why_this_point = ""
            if contract_matches:
                headline = text_or_fallback(
                    raw_bullet.get("headline"),
                    fallback_bullet.get("headline"),
                    (
                        f"point_guidance.{point_id}."
                        f"{dimension_id}.headline"
                    ),
                )
                explanation = text_or_fallback(
                    raw_bullet.get("explanation"),
                    fallback_bullet.get("explanation"),
                    (
                        f"point_guidance.{point_id}."
                        f"{dimension_id}.explanation"
                    ),
                )
                why_this_point = text_or_fallback(
                    raw_bullet.get("why_this_point"),
                    fallback_bullet.get("why_this_point"),
                    (
                        f"point_guidance.{point_id}."
                        f"{dimension_id}.why_this_point"
                    ),
                )
                try:
                    _validate_labeling_value(
                        why_this_point,
                        field_name=(
                            f"point_guidance.{point_id}."
                            f"{dimension_id}.why_this_point"
                        ),
                    )
                except ValueError:
                    fallback_fields.append(
                        f"point_guidance.{point_id}."
                        f"{dimension_id}.why_this_point"
                    )
                    why_this_point = str(
                        fallback_bullet.get("why_this_point", "")
                    ).strip()
            else:
                fallback_fields.append(
                    f"point_guidance.{point_id}.{dimension_id}"
                )
                headline = str(
                    fallback_bullet.get("headline")
                    or expected_bullet.get("headline", "")
                ).strip()
                explanation = str(
                    fallback_bullet.get("explanation")
                    or " ".join(
                        (
                            str(expected_bullet.get("plain_fact", "")),
                            str(expected_bullet.get("why_it_matters", "")),
                        )
                    )
                ).strip()
                why_this_point = str(
                    fallback_bullet.get("why_this_point")
                    or expected_bullet.get("labeling_value", "")
                ).strip()
            merged_bullets.append(
                {
                    "dimension_id": dimension_id,
                    "status": expected_bullet.get("status"),
                    "headline": headline,
                    "explanation": explanation,
                    "why_this_point": why_this_point,
                    "evidence_fact_ids": list(
                        expected_bullet.get("evidence_fact_ids", ())
                    ),
                }
            )

        outcomes = raw_item.get("possible_outcomes")
        valid_outcomes = (
            isinstance(outcomes, list)
            and 0 < len(outcomes) <= 2
            and all(
                isinstance(value, str) and value.strip()
                for value in outcomes
            )
        )
        if valid_outcomes:
            try:
                for value in outcomes:
                    _validate_plain_language(
                        value,
                        field_name=(
                            f"point_guidance.{point_id}."
                            "possible_outcomes"
                        ),
                    )
            except ValueError:
                valid_outcomes = False
        if not valid_outcomes:
            fallback_fields.append(
                f"point_guidance.{point_id}.possible_outcomes"
            )
            outcomes = list(
                fallback_item.get("possible_outcomes", ())
            )
        merged_items.append(
            {
                "point_id": point_id,
                "why_selected": text_or_fallback(
                    raw_item.get("why_selected"),
                    fallback_item.get("why_selected"),
                    f"point_guidance.{point_id}.why_selected",
                ),
                "what_changed_since_last_round": text_or_fallback(
                    raw_item.get("what_changed_since_last_round"),
                    fallback_item.get("what_changed_since_last_round"),
                    (
                        f"point_guidance.{point_id}."
                        "what_changed_since_last_round"
                    ),
                ),
                "how_to_label": text_or_fallback(
                    raw_item.get("how_to_label"),
                    fallback_item.get("how_to_label"),
                    f"point_guidance.{point_id}.how_to_label",
                ),
                "possible_outcomes": outcomes,
                "evidence_bullets": merged_bullets,
                "comparison_target_ids": [
                    item.get("point_id")
                    for item in card.get("comparison_targets", ())
                ],
            }
        )
    expected_category_explanation = next(
        (
            str(item.get("category_explanation", "")).strip()
            for item in packet.category_evidence_cards
            if item.get("category_explanation")
        ),
        str(
            deterministic_fallback.get("category_explanation", "")
        ).strip(),
    )
    if body.get("category_explanation") != expected_category_explanation:
        fallback_fields.append("category_explanation")
    merged = {
        "plan_id": plan.get("plan_id"),
        "category": plan.get("focus_category"),
        "recommended_point_ids": list(expected_ids),
        "target_rule_ids": list(plan.get("target_rule_ids", ())),
        "label_options": list(packet.label_options),
        "category_explanation": expected_category_explanation,
        "summary": text_or_fallback(
            body.get("summary"),
            deterministic_fallback.get("summary"),
            "summary",
        ),
        "point_guidance": merged_items,
        "warnings": [
            str(value) for value in body.get("warnings", ())
        ],
    }
    return validate_guidance(merged, packet), fallback_fields


def _validate_plain_language(value: str, *, field_name: str) -> None:
    text = str(value).strip()
    lowered = text.lower()
    banned = next(
        (phrase for phrase in _USER_FACING_BANNED_PHRASES if phrase in lowered),
        None,
    )
    if banned:
        raise ValueError(
            f"{field_name} contains user-facing technical phrase: {banned}"
        )
    if _RAW_ASSIGNMENT_PATTERN.search(text):
        raise ValueError(
            f"{field_name} lists raw numeric assignments instead of explaining them"
        )
    if len(text) > 560:
        raise ValueError(f"{field_name} is too long for the guidance card")


def _validate_labeling_value(value: str, *, field_name: str) -> None:
    lowered = str(value).lower()
    if not any(
        cue in lowered
        for cue in (
            "label",
            "human",
            "answer",
            "decision",
            "judgment",
            "type",
        )
    ):
        raise ValueError(
            f"{field_name} does not explain what a human label would clarify"
        )


def _prompt(packet: TranslationPacket) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace(
        "{translation_packet_json}",
        json.dumps(packet.to_dict(), sort_keys=True, separators=(",", ":")),
    )
