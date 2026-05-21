from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.modules.intent_instruction.providers.deepseek import (
    DEEPSEEK_FLASH_MODEL,
    DEEPSEEK_PRO_MODEL,
    DeepSeekLlmProvider,
)
from app.modules.intent_instruction.schemas import DatasetContext, StructuredInstruction


class DeepSeekProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = DeepSeekLlmProvider(
            api_key="test-key",
            allow_mock_fallback=False,
        )
        self.context = DatasetContext(
            dataset_id="deepseek_test",
            feature_names=("x", "y"),
            cluster_ids=("cluster_1", "cluster_2"),
            outlier_point_ids=("p3",),
            selected_point_ids=("p1",),
            unselected_point_ids=("p2", "p3"),
            analysis_context={
                "point_to_cluster": {"p1": "cluster_1", "p2": "cluster_2"},
                "point_catalog": (
                    {"point_id": "p1", "cluster_id": "cluster_1", "is_outlier": False},
                    {"point_id": "p2", "cluster_id": "cluster_2", "is_outlier": False},
                ),
                "ssdbcodi": {
                    "parameters": {
                        "min_pts": 3,
                        "alpha": 0.4,
                        "beta": 0.3,
                        "contamination": 0.13,
                        "rscore_weight": 0.5,
                    },
                    "cluster_counts": {"cluster_1": 1, "cluster_2": 1},
                    "outlier_point_ids": ["p3"],
                    "seeds": [
                        {
                            "point_id": "p1",
                            "cluster_id": "cluster_1",
                            "source": "kmeans_bootstrap",
                        }
                    ],
                    "point_scores": [
                        {
                            "point_id": "p1",
                            "cluster_id": "cluster_1",
                            "r_score": 1.0,
                            "l_score": 0.1,
                            "sim_score": 0.0,
                            "t_score": 0.3,
                            "c_dist": 0.0,
                            "e_max": 0.0,
                        }
                    ],
                },
            },
        )

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_route_uses_chat_completions_json_mode(self, mock_post_json):
        mock_post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"category":"meta_query","confidence":0.9,'
                            '"reason":"asks about clusters"}'
                        )
                    }
                }
            ]
        }

        result = self.provider.route("how many clusters?", self.context, ())

        self.assertEqual(result.category, "meta_query")
        url, request_payload = mock_post_json.call_args[0]
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request_payload["model"], DEEPSEEK_PRO_MODEL)
        self.assertEqual(request_payload["response_format"], {"type": "json_object"})
        self.assertEqual(request_payload["thinking"], {"type": "disabled"})
        self.assertEqual(request_payload["messages"][0]["role"], "user")

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_route_prompt_teaches_cluster_count_recommendations(self, mock_post_json):
        mock_post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"category":"meta_query","confidence":0.9,'
                            '"reason":"asks for a recommendation"}'
                        )
                    }
                }
            ]
        }

        self.provider.route("how many clusters shoukd be", self.context, ())

        request_payload = mock_post_json.call_args[0][1]
        prompt = request_payload["messages"][0]["content"]
        self.assertIn("how many clusters should there be", prompt)
        self.assertIn("how many clusters shoukd be", prompt)
        self.assertIn("do not answer with only the current count", prompt)
        self.assertIn("ssdbcodi cluster_counts", prompt)

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_extract_parses_delta_json(self, mock_post_json):
        mock_post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"operations":[{"op":"add","constraint_id":"pending",'
                            '"intent":"merge_clusters","payload":{"target_groups":'
                            '["cluster_1","cluster_2"]}}]}'
                        )
                    }
                }
            ]
        }

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

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_freeform_reply_uses_text_mode(self, mock_post_json):
        mock_post_json.return_value = {
            "choices": [{"message": {"content": "I would merge cluster_1 and cluster_2."}}]
        }

        reply = self.provider.freeform_reply(
            "merge clusters 1 and 2",
            self.context,
            (),
            response={"reply": "Okay, I recorded a merge."},
            provider_trace={},
        )

        self.assertIn("merge", reply)
        request_payload = mock_post_json.call_args[0][1]
        self.assertNotIn("response_format", request_payload)
        self.assertEqual(request_payload["thinking"], {"type": "disabled"})

    @patch.object(DeepSeekLlmProvider, "_post_json")
    def test_reply_prompt_can_correct_current_count_for_should_question(self, mock_post_json):
        mock_post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I would compare the current 2-cluster result with nearby K values."
                    }
                }
            ]
        }

        reply = self.provider.freeform_reply(
            "how many clusters shoukd be",
            self.context,
            (),
            response={
                "reply": "There are currently 2 clusters: cluster_1, cluster_2.",
                "router_category": "meta_query",
            },
            provider_trace={},
        )

        self.assertIn("nearby K", reply)
        request_payload = mock_post_json.call_args[0][1]
        prompt = request_payload["messages"][0]["content"]
        self.assertIn("processed workflow response only reports the current count", prompt)
        self.assertIn("words like \"should\"", prompt)
        self.assertIn("ssdbcodi cluster_counts", prompt)

    def test_health_reports_missing_api_key(self):
        provider = DeepSeekLlmProvider(api_key="")

        health = provider.health(force_refresh=True)

        self.assertFalse(health["available"])
        self.assertEqual(health["provider_kind"], "deepseek")
        self.assertIn("API_KEY", health["error"])

    def test_provider_reads_deepseek_defaults_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "METRIC_DASHBOARD_LLM_MODEL": DEEPSEEK_FLASH_MODEL,
                "METRIC_DASHBOARD_DEEPSEEK_BASE_URL": "https://example.test",
                "METRIC_DASHBOARD_DEEPSEEK_API_KEY": "env-key",
            },
            clear=False,
        ):
            provider = DeepSeekLlmProvider()

        self.assertEqual(provider.model_name, DEEPSEEK_FLASH_MODEL)
        self.assertEqual(provider.base_url, "https://example.test")
        self.assertEqual(provider.api_key, "env-key")


if __name__ == "__main__":
    unittest.main()
