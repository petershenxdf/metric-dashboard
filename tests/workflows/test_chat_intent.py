from __future__ import annotations

import unittest

from app import create_app


class ChatIntentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.post("/workflows/chat-intent/api/reset")

    def test_index_returns_200(self):
        response = self.client.get("/workflows/chat-intent/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 8 Intent Compilation", response.data)

    def test_message_actionable_round_trip(self):
        response = self.client.post(
            "/workflows/chat-intent/api/messages",
            json={"message": "merge clusters 1 and 2"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"]["response"]["router_category"], "on_topic_actionable")
        self.assertEqual(body["data"]["response"]["intent_type"], "merge_clusters")
        self.assertGreaterEqual(body["data"]["current_instruction"]["version"], 1)
        self.assertEqual(
            body["data"]["response"]["delta"]["operations"][0]["payload"]["target_groups"],
            [
                {"source": "cluster", "ref": "cluster_1"},
                {"source": "cluster", "ref": "cluster_2"},
            ],
        )

    def test_meta_query_reports_grounded_clusters_not_constraint_count(self):
        response = self.client.post(
            "/workflows/chat-intent/api/messages",
            json={"message": "how many clusters are there now"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["response"]["router_category"], "meta_query")
        self.assertIn("clusters", body["data"]["response"]["reply"])
        self.assertNotIn("constraints", body["data"]["response"]["reply"].lower())

    def test_incomplete_selection_reference_does_not_commit_constraint(self):
        self.client.post("/modules/selection/api/clear")
        self.client.post("/workflows/chat-intent/api/reset")
        response = self.client.post(
            "/workflows/chat-intent/api/messages",
            json={"message": "move these together"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["response"]["router_category"], "on_topic_ambiguous")
        self.assertTrue(body["data"]["response"]["requires_followup"])
        self.assertIsNone(body["data"]["response"]["delta"])
        self.assertEqual(body["data"]["current_instruction"]["version"], 0)

    def test_off_topic_message_is_recorded_but_state_unchanged(self):
        response = self.client.post(
            "/workflows/chat-intent/api/messages",
            json={"message": "what's the weather today?"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["response"]["router_category"], "off_topic")
        self.assertEqual(body["data"]["current_instruction"]["version"], 0)

    def test_reset_clears_state_and_history(self):
        self.client.post(
            "/workflows/chat-intent/api/messages",
            json={"message": "merge clusters 1 and 2"},
        )
        response = self.client.post("/workflows/chat-intent/api/reset")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["state"]["turn_count"], 0)
        self.assertEqual(body["data"]["current_instruction"]["version"], 0)


if __name__ == "__main__":
    unittest.main()
