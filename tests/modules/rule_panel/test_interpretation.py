import json
import unittest
from unittest.mock import patch

from app.modules.intent_instruction.providers.deepseek import DeepSeekLlmProvider
from app.modules.rule_panel.fixtures import rule_panel_fixture_state
from app.modules.rule_panel.interpretation import (
    DeepSeekRuleInterpreter,
    DeterministicRuleInterpreter,
    build_rule_guidance_metrics,
    build_rule_interpretation_payload,
    parse_rule_interpretation_payload,
)
from app.modules.rule_panel.schemas import RULE_INTERPRETATION_CATEGORIES


class RuleInterpretationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = rule_panel_fixture_state()
        cls.rule_set = cls.state["rule_set"]
        cls.analysis = cls.state["analysis"]
        cls.feature_matrix = cls.state["feature_matrix"]
        cls.rule = next(rule for rule in cls.rule_set.rules if rule.conditions)
        cls.condition = cls.rule.conditions[0]

    def valid_payload(self, **overrides):
        payload = {
            "categories": ["label_priority", "feature_label_strategy"],
            "target_rule_ids": [self.rule.rule_id],
            "summary": (
                "This recommendation uses 1 rule and 1 labeled candidate group. "
                "The rule is grounded in wine raw feature thresholds."
            ),
            "recommendation": "Label the cited wine point before changing the cluster boundary.",
            "category_explanation": "Label priority identifies the next points that reduce labeling uncertainty.",
            "label_targets": [
                {
                    "priority": "high",
                    "rule_ids": [self.rule.rule_id],
                    "point_ids": [self.rule.matched_point_ids[0]],
                    "label_question": "Does this point semantically belong to the cited rule target?",
                    "why_label_these_points": "it is tied to a known rule and can validate the next label step",
                }
            ],
            "suspicion_reasons": [
                {
                    "rule_ids": [self.rule.rule_id],
                    "point_ids": [self.rule.matched_point_ids[0]],
                    "suspicious_signal": "the rule has not been validated by human labels",
                    "rule_based_reason": "the cited rule is a surrogate for SSDBCODI and needs semantic validation",
                    "point_based_reason": "the point is a matched member of the cited rule",
                }
            ],
            "point_label_guidance": [
                {
                    "rule_ids": [self.rule.rule_id],
                    "point_ids": [self.rule.matched_point_ids[0]],
                    "suggested_label_frame": "choose cited cluster, neighboring cluster, anomaly, or uncertain",
                    "how_to_label": "compare the point's raw features with the cited rule threshold",
                    "decision_impact": "agreement supports the rule; disagreement triggers boundary review",
                    "llm_analysis_note": "DeepSeek should ground this label advice in point-level features",
                }
            ],
            "decision_rationale": (
                "This action tests whether the rule's quantitative agreement with SSDBCODI also matches "
                "the user's semantic label."
            ),
            "label_outcomes": [
                {
                    "label_result": "labels agree with the rule target",
                    "decision_implication": "use the rule as a stable reference for later boundary review",
                    "rule_ids": [self.rule.rule_id],
                    "point_ids": [self.rule.matched_point_ids[0]],
                },
                {
                    "label_result": "labels disagree with the rule target",
                    "decision_implication": "audit the boundary before changing cluster state",
                    "rule_ids": [self.rule.rule_id],
                    "point_ids": [self.rule.matched_point_ids[0]],
                },
            ],
            "quantitative_findings": [
                {
                    "metric": "strongest_rule_support",
                    "value": self.rule.support_count,
                    "rule_ids": [self.rule.rule_id],
                    "interpretation": "support count used to prioritize labeling",
                }
            ],
            "suggested_label_actions": [
                {
                    "action_type": "inspect_points",
                    "priority": "high",
                    "rule_ids": [self.rule.rule_id],
                    "point_ids": [self.rule.matched_point_ids[0]],
                    "reason": "this point anchors the next manual label check",
                    "hypothesis": "the cited point is representative of the rule target",
                    "why_this_action": "this point is grounded in a known rule instead of a random projected location",
                    "expected_outcomes": [
                        {
                            "label_result": "labels agree",
                            "decision_implication": "keep using the rule as guidance",
                        },
                        {
                            "label_result": "labels disagree",
                            "decision_implication": "downgrade this rule's refinement value",
                        },
                    ],
                    "risk_note": "the surrogate can agree with SSDBCODI while still missing user semantics",
                }
            ],
            "evidence": [
                {
                    "rule_id": self.rule.rule_id,
                    "feature": self.condition.feature,
                    "operator": self.condition.operator,
                    "threshold": self.condition.threshold,
                    "point_id": self.rule.matched_point_ids[0],
                }
            ],
            "confidence": 0.82,
            "warnings": [],
        }
        payload.update(overrides)
        return payload

    def test_valid_model_output_is_parsed_into_rule_interpretation(self):
        interpretation = parse_rule_interpretation_payload(
            self.valid_payload(),
            self.rule_set,
            provider_label="deepseek:unit",
        )

        self.assertEqual(interpretation.provider_label, "deepseek:unit")
        self.assertIn("label_priority", interpretation.categories)
        self.assertEqual(interpretation.target_rule_ids, (self.rule.rule_id,))
        self.assertEqual(interpretation.evidence[0]["feature"], self.condition.feature)
        self.assertIn("Label", interpretation.recommendation)
        self.assertIn("Label priority", interpretation.category_explanation)
        self.assertEqual(interpretation.label_targets[0]["point_ids"], [self.rule.matched_point_ids[0]])
        self.assertIn("surrogate", interpretation.suspicion_reasons[0]["rule_based_reason"])
        self.assertIn("raw features", interpretation.point_label_guidance[0]["how_to_label"])
        self.assertIn("human label", interpretation.decision_rationale)
        self.assertEqual(interpretation.label_outcomes[0]["label_result"], "labels agree with the rule target")
        self.assertEqual(interpretation.quantitative_findings[0]["metric"], "strongest_rule_support")
        self.assertEqual(interpretation.suggested_label_actions[0]["action_type"], "inspect_points")
        self.assertIn("hypothesis", interpretation.suggested_label_actions[0])

    def test_user_facing_terms_are_normalized(self):
        interpretation = parse_rule_interpretation_payload(
            self.valid_payload(
                recommendation="The unusualness score and anomaly score should be checked against SSDBCODI before using a semantic label.",
                label_targets=[
                    {
                        "priority": "high",
                        "rule_ids": [self.rule.rule_id],
                        "point_ids": [self.rule.matched_point_ids[0]],
                        "label_question": "Does this semantic label match the threshold story?",
                        "why_label_these_points": "SSDBCODI marked this point near a threshold.",
                    }
                ],
            ),
            self.rule_set,
            provider_label="deepseek:unit",
        )

        combined = " ".join(
            [
                interpretation.recommendation,
                interpretation.label_targets[0]["label_question"],
                interpretation.label_targets[0]["why_label_these_points"],
            ]
        )
        self.assertIn("outlier score", combined)
        self.assertIn("current analysis", combined)
        self.assertIn("human label", combined)
        self.assertIn("cutoff", combined)
        self.assertNotIn("unusualness score", combined)
        self.assertNotIn("anomaly score", combined)
        self.assertNotIn("semantic label", combined)

    def test_unknown_categories_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown interpretation categories"):
            parse_rule_interpretation_payload(
                self.valid_payload(categories=["made_up_category"]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "categories must be a non-empty list"):
            parse_rule_interpretation_payload(
                self.valid_payload(categories=[]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

    def test_unknown_rule_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown rule id"):
            parse_rule_interpretation_payload(
                self.valid_payload(target_rule_ids=["rule_missing"]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

    def test_unknown_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown feature"):
            parse_rule_interpretation_payload(
                self.valid_payload(evidence=[{"rule_id": self.rule.rule_id, "feature": "x"}]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

    def test_unknown_point_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown point id"):
            parse_rule_interpretation_payload(
                self.valid_payload(evidence=[{"rule_id": self.rule.rule_id, "point_id": "wine_999"}]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

    def test_unknown_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown threshold"):
            parse_rule_interpretation_payload(
                self.valid_payload(
                    evidence=[
                        {
                            "rule_id": self.rule.rule_id,
                            "feature": self.condition.feature,
                            "threshold": self.condition.threshold + 12345,
                        }
                    ]
                ),
                self.rule_set,
                provider_label="deepseek:unit",
            )

    def test_missing_action_guidance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "recommendation"):
            parse_rule_interpretation_payload(
                self.valid_payload(recommendation=""),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "quantitative_findings"):
            parse_rule_interpretation_payload(
                self.valid_payload(quantitative_findings=[]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "decision_rationale"):
            parse_rule_interpretation_payload(
                self.valid_payload(decision_rationale=""),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "category_explanation"):
            parse_rule_interpretation_payload(
                self.valid_payload(category_explanation=""),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "label_targets"):
            parse_rule_interpretation_payload(
                self.valid_payload(label_targets=[]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "suspicion_reasons"):
            parse_rule_interpretation_payload(
                self.valid_payload(suspicion_reasons=[]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "point_label_guidance"):
            parse_rule_interpretation_payload(
                self.valid_payload(point_label_guidance=[]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "label_outcomes"):
            parse_rule_interpretation_payload(
                self.valid_payload(label_outcomes=[]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "why_this_action"):
            parse_rule_interpretation_payload(
                self.valid_payload(
                    suggested_label_actions=[
                        {
                            "action_type": "inspect_points",
                            "priority": "high",
                            "rule_ids": [self.rule.rule_id],
                            "point_ids": [self.rule.matched_point_ids[0]],
                            "reason": "missing why field",
                            "hypothesis": "valid hypothesis",
                            "expected_outcomes": [
                                {
                                    "label_result": "labels agree",
                                    "decision_implication": "valid implication",
                                }
                            ],
                            "risk_note": "valid risk note",
                        }
                    ]
                ),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "suggested_label_actions"):
            parse_rule_interpretation_payload(
                self.valid_payload(suggested_label_actions=[]),
                self.rule_set,
                provider_label="deepseek:unit",
            )

    def test_no_typical_case_category_can_return_empty_guidance(self):
        interpretation = parse_rule_interpretation_payload(
            self.valid_payload(
                categories=["overlap_merge_signal"],
                target_rule_ids=[],
                recommendation="No two rules currently share matched points, so do not force a merge example.",
                label_targets=[],
                suspicion_reasons=[],
                point_label_guidance=[],
                label_outcomes=[],
                suggested_label_actions=[],
                warnings=["no_typical_case_for_category"],
            ),
            self.rule_set,
            provider_label="deepseek:unit",
        )

        self.assertEqual(interpretation.categories, ("overlap_merge_signal",))
        self.assertEqual(interpretation.label_targets, ())
        self.assertIn("no_typical_case_for_category", interpretation.warnings)

    def test_unknown_action_references_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown rule id"):
            parse_rule_interpretation_payload(
                self.valid_payload(
                    suggested_label_actions=[
                        {
                            "action_type": "inspect_points",
                            "priority": "high",
                            "rule_ids": ["rule_missing"],
                            "point_ids": [self.rule.matched_point_ids[0]],
                            "reason": "bad rule reference",
                        }
                    ]
                ),
                self.rule_set,
                provider_label="deepseek:unit",
            )

        with self.assertRaisesRegex(ValueError, "unknown point id"):
            parse_rule_interpretation_payload(
                self.valid_payload(
                    suggested_label_actions=[
                        {
                            "action_type": "inspect_points",
                            "priority": "high",
                            "rule_ids": [self.rule.rule_id],
                            "point_ids": ["wine_999"],
                            "reason": "bad point reference",
                        }
                    ]
                ),
                self.rule_set,
                provider_label="deepseek:unit",
            )

    def test_rule_guidance_metrics_include_pair_and_label_candidates(self):
        metrics = build_rule_guidance_metrics(self.rule_set)

        self.assertIn("pair_metrics", metrics)
        self.assertIn("label_candidate_groups", metrics)
        self.assertGreater(len(metrics["rule_metrics"]), 0)
        self.assertIn("jaccard_overlap", metrics["pair_metrics"][0])
        self.assertIn("rule_confidence_score", metrics["rule_metrics"][0])

    def test_mocked_provider_output_can_use_every_category(self):
        interpretation = parse_rule_interpretation_payload(
            self.valid_payload(categories=list(RULE_INTERPRETATION_CATEGORIES)),
            self.rule_set,
            provider_label="mock",
        )

        self.assertEqual(set(interpretation.categories), set(RULE_INTERPRETATION_CATEGORIES))

    def test_request_payload_contains_rule_state_without_secret_fields(self):
        payload = build_rule_interpretation_payload(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.feature_matrix,
        )

        encoded = json.dumps(payload)
        self.assertIn("wine_mat", encoded)
        self.assertIn("known_rule_ids", payload)
        self.assertIn("ssdbcodi", payload)
        self.assertNotIn("api_key", encoded.lower())
        self.assertNotIn("sk-", encoded)

    def test_request_payload_records_focus_category(self):
        payload = build_rule_interpretation_payload(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.feature_matrix,
            focus_category="anomaly_label_review",
        )

        self.assertEqual(payload["focus_category"], "anomaly_label_review")
        self.assertIn("rule_guidance_metrics", payload)
        self.assertIn("category_descriptions", payload)
        self.assertIn("label_candidate_point_profiles", payload)
        self.assertGreater(len(payload["label_candidate_point_profiles"]), 0)
        self.assertIn("raw_feature_values", payload["label_candidate_point_profiles"][0])
        self.assertEqual(payload["instructions"]["required_model"], "deepseek-v4-pro")
        self.assertIn("Focus this response on anomaly_label_review", payload["instructions"]["focus_instruction"])

    def test_deterministic_interpreter_returns_focused_category(self):
        run = DeterministicRuleInterpreter().interpret(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.feature_matrix,
            focus_category="boundary_review",
        )

        self.assertEqual(run.interpretation.categories, ("boundary_review",))
        self.assertIn("Label", run.interpretation.recommendation)
        self.assertGreater(len(run.interpretation.quantitative_findings), 0)
        self.assertGreater(len(run.interpretation.suggested_label_actions), 0)
        self.assertEqual(run.request_payload["focus_category"], "boundary_review")
        self.assertEqual(run.diagnostics["focus_category"], "boundary_review")

    def test_invalid_focus_category_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "focus_category"):
            DeterministicRuleInterpreter().interpret(
                self.rule_set,
                analysis_result=self.analysis,
                feature_matrix=self.feature_matrix,
                focus_category="not_a_category",
            )

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_deepseek_provider_uses_chat_completions_json_mode(self, mock_post_json):
        mock_post_json.return_value = {
            "choices": [{"message": {"content": json.dumps(self.valid_payload())}}]
        }
        interpreter = DeepSeekRuleInterpreter(
            client=DeepSeekLlmProvider(api_key="test-key", allow_mock_fallback=False)
        )

        run = interpreter.interpret(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.feature_matrix,
            focus_category="label_priority",
        )

        self.assertFalse(run.diagnostics["used_fallback"])
        self.assertEqual(run.interpretation.provider_label, "deepseek:deepseek-v4-pro")
        url, request_payload = mock_post_json.call_args[0]
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request_payload["response_format"], {"type": "json_object"})
        self.assertEqual(request_payload["model"], "deepseek-v4-pro")
        self.assertEqual(request_payload["thinking"], {"type": "enabled"})
        self.assertEqual(request_payload["reasoning_effort"], "high")
        self.assertGreaterEqual(request_payload["max_tokens"], 6000)
        self.assertIn("recommend_next_labels_from_surrogate_rules", request_payload["messages"][0]["content"])
        self.assertIn("labeling/refinement analyst", request_payload["messages"][0]["content"])
        self.assertIn("rule_guidance_metrics", request_payload["messages"][0]["content"])
        self.assertIn("label_candidate_point_profiles", request_payload["messages"][0]["content"])
        self.assertIn("Focus on exactly this category if possible: label_priority", request_payload["messages"][0]["content"])
        self.assertIn("Use DeepSeek V4 Pro as an analyst, not a metric echo", request_payload["messages"][0]["content"])
        self.assertIn("no_typical_case_for_category", request_payload["messages"][0]["content"])
        self.assertIn("careful undergraduate user", request_payload["messages"][0]["content"])
        self.assertIn("prompt_template_path", run.diagnostics)
        self.assertTrue(run.diagnostics["using_deepseek_v4_pro"])
        self.assertFalse(run.diagnostics["deepseek_retry_used"])

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_deepseek_provider_retries_v4_pro_when_thinking_response_has_no_content(self, mock_post_json):
        mock_post_json.side_effect = [
            {
                "id": "first",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "reasoning used the token budget",
                        },
                    }
                ],
                "usage": {"total_tokens": 6000, "completion_tokens": 6000},
            },
            {
                "choices": [{"message": {"content": json.dumps(self.valid_payload())}}],
            },
        ]
        interpreter = DeepSeekRuleInterpreter(
            client=DeepSeekLlmProvider(api_key="test-key", allow_mock_fallback=False)
        )

        run = interpreter.interpret(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.feature_matrix,
            focus_category="label_priority",
        )

        self.assertFalse(run.diagnostics["used_fallback"])
        self.assertTrue(run.diagnostics["using_deepseek_v4_pro"])
        self.assertTrue(run.diagnostics["deepseek_retry_used"])
        self.assertEqual(mock_post_json.call_count, 2)
        first_payload = mock_post_json.call_args_list[0][0][1]
        retry_payload = mock_post_json.call_args_list[1][0][1]
        self.assertEqual(first_payload["model"], "deepseek-v4-pro")
        self.assertEqual(retry_payload["model"], "deepseek-v4-pro")
        self.assertEqual(first_payload["thinking"], {"type": "enabled"})
        self.assertEqual(retry_payload["thinking"], {"type": "disabled"})
        self.assertEqual(run.diagnostics["deepseek_json_attempts"][0]["response_metadata"]["finish_reason"], "length")

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_deepseek_provider_retries_v4_pro_when_thinking_json_is_malformed(self, mock_post_json):
        mock_post_json.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"categories":["label_priority"],"summary":"unterminated'
                        }
                    }
                ]
            },
            {
                "choices": [{"message": {"content": json.dumps(self.valid_payload())}}],
            },
        ]
        interpreter = DeepSeekRuleInterpreter(
            client=DeepSeekLlmProvider(api_key="test-key", allow_mock_fallback=False)
        )

        run = interpreter.interpret(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.feature_matrix,
            focus_category="label_priority",
        )

        self.assertFalse(run.diagnostics["used_fallback"])
        self.assertTrue(run.diagnostics["deepseek_retry_used"])
        self.assertEqual(mock_post_json.call_count, 2)
        retry_payload = mock_post_json.call_args_list[1][0][1]
        self.assertEqual(retry_payload["model"], "deepseek-v4-pro")
        self.assertEqual(retry_payload["thinking"], {"type": "disabled"})

    def test_provider_errors_return_diagnostics_without_changing_rule_set(self):
        interpreter = DeepSeekRuleInterpreter(
            client=DeepSeekLlmProvider(api_key="", allow_mock_fallback=True)
        )

        run = interpreter.interpret(
            self.rule_set,
            analysis_result=self.analysis,
            feature_matrix=self.feature_matrix,
        )

        self.assertTrue(run.diagnostics["used_fallback"])
        self.assertEqual(run.request_payload["rule_set"], self.rule_set.to_dict())
        self.assertEqual(self.rule_set.to_dict(), self.state["rule_set"].to_dict())


if __name__ == "__main__":
    unittest.main()
