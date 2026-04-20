from __future__ import annotations

import unittest

from app import create_app


class IntentRuntimeValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.post("/workflows/intent-runtime-validation/api/reset")

    def test_index_returns_200(self):
        response = self.client.get("/workflows/intent-runtime-validation/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 8.5 Runtime Validation", response.data)

    def test_state_api_exposes_grounded_runtime_payload(self):
        response = self.client.get("/workflows/intent-runtime-validation/api/state")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("analysis_context", body["data"]["state"])
        self.assertIn("memory_state", body["data"])
        self.assertIn("evaluation_results", body["data"])

    def test_message_round_trip_uses_same_grounded_cluster_resolution(self):
        response = self.client.post(
            "/workflows/intent-runtime-validation/api/messages",
            json={"message": "merge clusters 1 and 2"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["response"]["intent_type"], "merge_clusters")
        self.assertEqual(
            body["data"]["response"]["delta"]["operations"][0]["payload"]["target_groups"],
            [
                {"source": "cluster", "ref": "cluster_1"},
                {"source": "cluster", "ref": "cluster_2"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
