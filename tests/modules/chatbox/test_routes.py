from __future__ import annotations

import unittest

from app import create_app
from app.modules.chatbox.fixtures import current_selection_context
from app.modules.chatbox.state import reset_debug_store_for_context
from app.modules.labeling.state import reset_debug_store_for_context as reset_labeling
from app.modules.selection.state import reset_debug_store


class ChatboxRouteTests(unittest.TestCase):
    def setUp(self):
        reset_debug_store()
        context = current_selection_context()
        reset_labeling(context)
        reset_debug_store_for_context(context)
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/chatbox/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Chatbox", response.data)
        self.assertIn(b"Selection Context", response.data)
        self.assertIn(b"Instruction Snapshot", response.data)
        self.assertIn(b"Step 7 is strategy-agnostic", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/chatbox/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "chatbox")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_context_api_returns_selection_label_and_chips(self):
        response = self.client.get("/modules/chatbox/api/context")

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["dataset_id"], "selection_iris_debug")
        self.assertIn("selection_context", data)
        self.assertIn("label_context", data)
        self.assertIn("selection_groups", data)
        self.assertIn("suggestion_chips", data)
        self.assertIn("instruction_snapshot", data)
        intent_types = {chip["intent_type"] for chip in data["suggestion_chips"]}
        self.assertIn("split_cluster", intent_types)

    def test_history_api_starts_empty_after_reset(self):
        response = self.client.get("/modules/chatbox/api/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["turn_count"], 0)

    def test_messages_api_rejects_empty_message(self):
        response = self.client.post("/modules/chatbox/api/messages", json={"message": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "invalid_chat_message")

    def test_messages_api_handles_actionable_message(self):
        response = self.client.post(
            "/modules/chatbox/api/messages",
            json={"message": "Make petal_length more important"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["response"]["router_category"], "on_topic_actionable")
        self.assertEqual(data["response"]["intent_type"], "feature_weight")
        self.assertEqual(data["state"]["turn_count"], 2)

    def test_messages_api_handles_off_topic_message(self):
        response = self.client.post(
            "/modules/chatbox/api/messages",
            json={"message": "What's the weather today?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["response"]["router_category"], "off_topic")
        self.assertIsNone(response.json["data"]["response"]["delta"])

    def test_reset_api_clears_chat_state(self):
        self.client.post(
            "/modules/chatbox/api/messages",
            json={"message": "merge clusters 1 and 2"},
        )

        response = self.client.post("/modules/chatbox/api/reset", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["turn_count"], 0)
        self.assertEqual(response.json["data"]["instruction_snapshot"]["version"], 0)

    def test_clear_api_clears_turns_only(self):
        self.client.post(
            "/modules/chatbox/api/messages",
            json={"message": "ignore cluster 5"},
        )

        response = self.client.post("/modules/chatbox/api/clear", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["turn_count"], 0)

    def test_chat_selection_workflow_loads(self):
        response = self.client.get("/workflows/chat-selection/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 7 Chat Intake", response.data)
        self.assertIn(b"Suggestion Chips", response.data)
        self.assertIn(b"Instruction Snapshot", response.data)


if __name__ == "__main__":
    unittest.main()
