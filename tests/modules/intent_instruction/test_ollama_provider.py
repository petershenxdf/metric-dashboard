from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.modules.intent_instruction.providers.ollama import OllamaLlmProvider
from app.modules.intent_instruction.schemas import DatasetContext, StructuredInstruction


class OllamaProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = OllamaLlmProvider(allow_mock_fallback=False)
        self.context = DatasetContext(
            dataset_id="ollama_test",
            feature_names=("x", "y"),
            cluster_ids=("cluster_1", "cluster_2", "cluster_3"),
            outlier_point_ids=("outlier_1",),
            selection_group_names=("group A",),
            selection_groups=(
                {
                    "group_id": "group_001",
                    "group_name": "group A",
                    "point_ids": ("p1", "p2"),
                },
            ),
            analysis_context={
                "point_to_cluster": {
                    "p1": "cluster_1",
                    "p2": "cluster_1",
                    "p3": "cluster_2",
                    "p4": "cluster_3",
                },
                "selected_point_clusters": (
                    {"point_id": "p1", "cluster_id": "cluster_1", "is_outlier": False, "features": {"x": 1.0, "y": 2.0}},
                    {"point_id": "p2", "cluster_id": "cluster_1", "is_outlier": False, "features": {"x": 1.5, "y": 2.5}},
                ),
                "point_catalog": (
                    {"point_id": "p1", "cluster_id": "cluster_1", "is_outlier": False, "features": {"x": 1.0, "y": 2.0}},
                    {"point_id": "p2", "cluster_id": "cluster_1", "is_outlier": False, "features": {"x": 1.5, "y": 2.5}},
                    {"point_id": "p3", "cluster_id": "cluster_2", "is_outlier": False, "features": {"x": 3.0, "y": 4.0}},
                ),
            },
            selected_point_ids=("p1", "p2"),
            unselected_point_ids=("p3", "p4"),
        )

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_route_parses_model_json(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "category": "meta_query",
                "confidence": 0.83,
                "reason": "asks about current clusters",
                "clarification_question": None,
            },
            '{"category":"meta_query"}',
        )
        result = self.provider.route("how many clusters are there now", self.context, ())
        self.assertEqual(result.category, "meta_query")
        self.assertAlmostEqual(result.confidence, 0.83, places=2)

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_extract_parses_delta_json(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
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
                ]
            },
            '{"operations":[{"op":"add"}]}',
        )
        delta = self.provider.extract(
            "merge clusters 1 and 2",
            self.context,
            (),
            StructuredInstruction(version=0, constraints=()),
        )
        self.assertEqual(delta.operations[0].intent, "merge_clusters")
        self.assertEqual(
            delta.operations[0].payload["target_groups"],
            [
                {"source": "cluster", "ref": "cluster_1"},
                {"source": "cluster", "ref": "cluster_2"},
            ],
        )

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_extract_normalizes_string_group_refs(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "operations": [
                    {
                        "op": "add",
                        "constraint_id": "pending",
                        "intent": "merge_clusters",
                        "payload": {"target_groups": ["cluster_1", "cluster_2"]},
                    }
                ]
            },
            '{"operations":[{"payload":{"target_groups":["cluster_1","cluster_2"]}}]}',
        )
        delta = self.provider.extract(
            "merge clusters 1 and 2",
            self.context,
            (),
            StructuredInstruction(version=0, constraints=()),
        )
        self.assertEqual(
            delta.operations[0].payload["target_groups"],
            [
                {"source": "cluster", "ref": "cluster_1"},
                {"source": "cluster", "ref": "cluster_2"},
            ],
        )

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_route_uses_active_draft_to_accept_slot_answer(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "category": "on_topic_ambiguous",
                "confidence": 0.41,
                "reason": "short reply",
                "clarification_question": "Which point?",
            },
            '{"category":"on_topic_ambiguous"}',
        )
        result = self.provider.route(
            "p1",
            self.context,
            (),
            memory_context={
                "draft_state": {
                    "candidate_intent": "reclassify_outlier",
                    "pending_clarification": "Which point should I reclassify?",
                    "grounded_refs": [],
                }
            },
        )
        self.assertEqual(result.category, "on_topic_actionable")
        self.assertGreaterEqual(result.confidence, 0.84)

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_route_preserves_llm_category_even_when_keywords_overlap(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "category": "meta_query",
                "confidence": 0.6,
                "reason": "question about outliers",
                "clarification_question": None,
                "reply_text": "There is 1 outlier: outlier_1.",
            },
            '{"category":"meta_query"}',
        )
        result = self.provider.route(
            "which point is an outlier right now?",
            self.context,
            (),
        )
        self.assertEqual(result.category, "meta_query")
        self.assertEqual(result.reply_text, "There is 1 outlier: outlier_1.")

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_route_prompt_includes_group_details_and_memory_json(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "category": "meta_query",
                "confidence": 0.7,
                "reason": "test",
                "clarification_question": None,
                "reply_text": "2 points are selected.",
            },
            '{"category":"meta_query"}',
        )
        self.provider.route(
            "what are the selected points now",
            self.context,
            (),
            memory_context={
                "summary": "User is asking about the current selection.",
                "recent_turns": [{"role": "user", "text": "what are the selected points now"}],
            },
        )
        prompt = mock_generate_json.call_args[0][0]
        route_prompt_path = self.provider.diagnostics()["prompt_template_files"]["route"]
        self.assertIn("prompts", route_prompt_path)
        self.assertIn("route_prompt.txt", route_prompt_path)
        self.assertIn('"selection_groups"', prompt)
        self.assertIn('"group_name": "group A"', prompt)
        self.assertIn('"recent_turns"', prompt)
        self.assertIn('"point_catalog"', prompt)

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_extract_synthesizes_reclassify_from_active_draft(self, mock_generate_json):
        mock_generate_json.return_value = (
            {"operations": []},
            '{"operations":[]}',
        )
        delta = self.provider.extract(
            "p1",
            self.context,
            (),
            StructuredInstruction(version=0, constraints=()),
            memory_context={
                "draft_state": {
                    "candidate_intent": "reclassify_outlier",
                    "pending_clarification": "Which point should I reclassify?",
                    "grounded_refs": [],
                    "proposed_delta": {
                        "operations": [
                            {
                                "op": "add",
                                "constraint_id": "pending",
                                "intent": "reclassify_outlier",
                                "payload": {"anchor": None, "is_outlier": True},
                            }
                        ]
                    },
                }
            },
        )
        self.assertEqual(delta.operations[0].intent, "reclassify_outlier")
        self.assertEqual(
            delta.operations[0].payload,
            {"anchor": {"source": "point_id", "ref": "p1"}, "is_outlier": True},
        )

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_extract_preserves_llm_intent_without_override(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "operations": [
                    {
                        "op": "add",
                        "constraint_id": "pending",
                        "intent": "ignore_cluster",
                        "payload": {
                            "target_groups": [{"source": "cluster", "ref": "cluster_1"}],
                        },
                    }
                ]
            },
            '{"operations":[{"intent":"ignore_cluster"}]}',
        )
        delta = self.provider.extract(
            "ignore cluster_1, it looks like an outlier dump",
            self.context,
            (),
            StructuredInstruction(version=0, constraints=()),
        )
        self.assertEqual(delta.operations[0].intent, "ignore_cluster")
        self.assertEqual(
            delta.operations[0].payload["target_groups"],
            [{"source": "cluster", "ref": "cluster_1"}],
        )

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_route_upgrades_offtopic_to_meta_query_for_grounded_question(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "category": "off_topic",
                "confidence": 0.5,
                "reason": "ignored",
                "clarification_question": None,
                "reply_text": None,
            },
            '{"category":"off_topic"}',
        )
        result = self.provider.route(
            "what class are the selected points",
            self.context,
            (),
        )
        self.assertEqual(result.category, "meta_query")

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_route_keeps_offtopic_for_social_message(self, mock_generate_json):
        mock_generate_json.return_value = (
            {
                "category": "off_topic",
                "confidence": 0.9,
                "reason": "social",
                "clarification_question": None,
                "reply_text": "No refinement to apply here.",
            },
            '{"category":"off_topic"}',
        )
        result = self.provider.route("thanks, that is cool", self.context, ())
        self.assertEqual(result.category, "off_topic")

    @patch.object(OllamaLlmProvider, "_generate_json")
    def test_route_falls_back_when_enabled(self, mock_generate_json):
        provider = OllamaLlmProvider(allow_mock_fallback=True)
        mock_generate_json.side_effect = ValueError("bad json")
        result = provider.route("merge clusters 1 and 2", self.context, ())
        self.assertEqual(result.category, "on_topic_actionable")
        self.assertTrue(provider.diagnostics()["last_route"]["used_fallback"])

    @patch.object(OllamaLlmProvider, "_generate_text")
    def test_freeform_reply_uses_root_reply_prompt(self, mock_generate_text):
        mock_generate_text.return_value = "I understand this as a request to merge cluster_1 and cluster_2."

        reply = self.provider.freeform_reply(
            "merge clusters 1 and 2",
            self.context,
            (),
            response={
                "reply": "Okay, I recorded a merge for cluster_1, cluster_2.",
                "router_category": "on_topic_actionable",
            },
            provider_trace={
                "router_result": {"category": "on_topic_actionable"},
                "proposed_delta": {
                    "operations": [
                        {
                            "intent": "merge_clusters",
                            "payload": {
                                "target_groups": [
                                    {"source": "cluster", "ref": "cluster_1"},
                                    {"source": "cluster", "ref": "cluster_2"},
                                ]
                            },
                        }
                    ]
                },
            },
            memory_context={"summary": "User wants to merge two clusters."},
        )

        self.assertIn("merge cluster_1 and cluster_2", reply.lower())
        diagnostics = self.provider.diagnostics()
        self.assertIn("reply_prompt.txt", diagnostics["prompt_template_files"]["reply"])
        self.assertIn("Current processed workflow response JSON", diagnostics["last_reply"]["prompt_text"])
        self.assertEqual(diagnostics["last_reply"]["result"]["reply"], reply)

    def test_provider_reads_model_defaults_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "METRIC_DASHBOARD_LLM_MODEL": "llama3.1:8b",
                "METRIC_DASHBOARD_OLLAMA_KEEP_ALIVE": "45m",
            },
            clear=False,
        ):
            provider = OllamaLlmProvider()

        self.assertEqual(provider.model_name, "llama3.1:8b")
        self.assertEqual(provider.keep_alive, "45m")

    @patch.object(OllamaLlmProvider, "_post_json")
    def test_generate_json_sends_keep_alive_to_ollama(self, mock_post_json):
        mock_post_json.return_value = {"response": '{"category":"meta_query","confidence":0.9}'}

        self.provider._generate_json("hello")

        request_payload = mock_post_json.call_args[0][1]
        self.assertEqual(request_payload["keep_alive"], self.provider.keep_alive)

    @patch.object(OllamaLlmProvider, "_request_json")
    def test_health_uses_cached_result_until_explicit_refresh(self, mock_request_json):
        mock_request_json.return_value = {"models": [{"name": self.provider.model_name}]}

        first = self.provider.health()
        second = self.provider.health()
        refreshed = self.provider.health(force_refresh=True)

        self.assertTrue(first["available"])
        self.assertEqual(mock_request_json.call_count, 2)
        self.assertEqual(second, first)
        self.assertEqual(refreshed["model_name"], self.provider.model_name)


if __name__ == "__main__":
    unittest.main()
