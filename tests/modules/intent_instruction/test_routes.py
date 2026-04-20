from __future__ import annotations

import unittest

from app import create_app
from app.modules.intent_instruction.state import reset_debug_provider


class IntentInstructionRouteTests(unittest.TestCase):
    def setUp(self):
        reset_debug_provider()
        self.app = create_app()
        self.client = self.app.test_client()

    def tearDown(self):
        reset_debug_provider()

    def test_debug_page_returns_200(self):
        response = self.client.get("/modules/intent-instruction/")
        self.assertEqual(response.status_code, 200)

    def test_health_returns_200_with_provider_diagnostics(self):
        response = self.client.get("/modules/intent-instruction/health")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("llm_provider", body["diagnostics"])

    def test_route_api_returns_router_category(self):
        response = self.client.post(
            "/modules/intent-instruction/api/route",
            json={"message": "merge clusters 1 and 2"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"]["router_result"]["category"], "on_topic_actionable")

    def test_route_api_rejects_empty_message(self):
        response = self.client.post(
            "/modules/intent-instruction/api/route",
            json={"message": "   "},
        )
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertFalse(body["ok"])

    def test_compile_api_returns_delta_and_advances_version(self):
        response = self.client.post(
            "/modules/intent-instruction/api/compile",
            json={"message": "merge clusters 1 and 2"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"]["response"]["router_category"], "on_topic_actionable")
        self.assertEqual(body["data"]["response"]["intent_type"], "merge_clusters")
        self.assertGreaterEqual(body["data"]["response"]["current_instruction_version"], 1)

    def test_compile_api_off_topic_does_not_advance_state(self):
        response = self.client.post(
            "/modules/intent-instruction/api/compile",
            json={"message": "what's the weather"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["response"]["router_category"], "off_topic")
        self.assertIsNone(body["data"]["response"]["delta"])

    def test_state_api_returns_current_instruction(self):
        response = self.client.get("/modules/intent-instruction/api/state")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("current_instruction", body["data"])
        self.assertIn("snapshot", body["data"])

    def test_reset_api_clears_state(self):
        self.client.post(
            "/modules/intent-instruction/api/compile",
            json={"message": "merge clusters 1 and 2"},
        )
        response = self.client.post("/modules/intent-instruction/api/reset")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["current_instruction"]["version"], 0)

    def test_examples_api_lists_phase_one_intents(self):
        response = self.client.get("/modules/intent-instruction/api/examples")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        intents = {item["intent"] for item in body["data"]["examples"]}
        self.assertIn("feature_weight", intents)
        self.assertIn("split_cluster", intents)


if __name__ == "__main__":
    unittest.main()
