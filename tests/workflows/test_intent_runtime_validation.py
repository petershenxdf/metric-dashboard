from __future__ import annotations

import unittest

from app import create_app
from app.modules.labeling.state import reset_debug_store_for_context as reset_labeling_store_for_context
from app.modules.selection.service import get_selection_context
from app.modules.selection.state import reset_debug_store_for_dataset
from app.shared.fixtures import analysis_selection_dataset, analysis_selection_initial_selected_point_ids


class IntentRuntimeValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        dataset = analysis_selection_dataset()
        selection_store = reset_debug_store_for_dataset(
            dataset,
            analysis_selection_initial_selected_point_ids(dataset.dataset_id),
        )
        reset_labeling_store_for_context(get_selection_context(selection_store))
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.post(
            "/workflows/intent-runtime-validation/api/reset",
            json={"provider_kind": "mock"},
        )

    def test_index_returns_200(self):
        response = self.client.get("/workflows/intent-runtime-validation/?provider_kind=mock")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 8.5 Runtime Validation", response.data)
        self.assertIn(b"Chatbox", response.data)

    def test_state_api_exposes_grounded_runtime_payload(self):
        response = self.client.get("/workflows/intent-runtime-validation/api/state?provider_kind=mock")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("analysis_context", body["data"]["state"])
        self.assertIn("memory_state", body["data"])
        self.assertIn("evaluation_results", body["data"])
        self.assertIn("storage", body["data"])
        self.assertEqual(body["data"]["runtime_diagnostics"]["provider_kind"], "mock")

    def test_message_round_trip_uses_same_grounded_cluster_resolution(self):
        response = self.client.post(
            "/workflows/intent-runtime-validation/api/messages",
            json={"message": "merge clusters 1 and 2", "provider_kind": "mock"},
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
        self.assertIn("storage", body["data"]["runtime"])
        self.assertEqual(
            [turn["role"] for turn in body["data"]["runtime"]["chat_state"]["turns"]],
            ["user", "assistant"],
        )

    def test_index_renders_user_and_assistant_turns_after_message(self):
        self.client.post(
            "/workflows/intent-runtime-validation/api/messages",
            json={"message": "merge clusters 1 and 2", "provider_kind": "mock"},
        )
        response = self.client.get("/workflows/intent-runtime-validation/?provider_kind=mock")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"merge clusters 1 and 2", response.data)
        self.assertIn(b"Okay, I recorded a merge for cluster_1, cluster_2.", response.data)
        self.assertIn(b"runtime-message-row user", response.data)
        self.assertIn(b"runtime-message-row assistant", response.data)

    def test_saved_selection_groups_are_forwarded_with_chat_payload(self):
        save_response = self.client.post(
            "/workflows/intent-runtime-validation/api/groups",
            json={"provider_kind": "mock", "group_name": "bridge"},
        )
        self.assertEqual(save_response.status_code, 200)

        response = self.client.post(
            "/workflows/intent-runtime-validation/api/messages",
            json={"message": "treat bridge as similar to cluster 2", "provider_kind": "mock"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        selection_groups = body["data"]["forwarded_payload"]["selection_groups"]
        self.assertTrue(selection_groups)
        self.assertEqual(selection_groups[0]["group_name"], "bridge")

    def test_label_updates_refresh_runtime_state(self):
        response = self.client.post(
            "/workflows/intent-runtime-validation/api/label",
            json={"provider_kind": "mock", "action": "assign_cluster", "label_value": "cluster_2"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreaterEqual(body["data"]["runtime"]["state"]["label_context"]["annotation_count"], 1)
        self.assertGreaterEqual(body["data"]["runtime"]["state"]["analysis_context"]["selected_count"], 1)

    def test_selection_updates_refresh_runtime_state(self):
        clear_response = self.client.post(
            "/workflows/intent-runtime-validation/api/clear-selection",
            json={"provider_kind": "mock"},
        )
        self.assertEqual(clear_response.status_code, 200)
        response = self.client.post(
            "/workflows/intent-runtime-validation/api/select",
            json={
                "provider_kind": "mock",
                "point_ids": ["gamma_01", "gamma_02"],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(
            body["data"]["runtime"]["state"]["selection_context"]["selected_point_ids"],
            ["gamma_01", "gamma_02"],
        )
        self.assertEqual(
            body["data"]["runtime"]["state"]["analysis_context"]["selected_point_ids"],
            ["gamma_01", "gamma_02"],
        )

    def test_runtime_config_api_switches_provider(self):
        response = self.client.post(
            "/workflows/intent-runtime-validation/api/runtime-config",
            json={"provider_kind": "mock", "model_name": "debug-model"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["data"]["runtime_diagnostics"]["provider_kind"], "mock")
        self.assertEqual(body["data"]["runtime_diagnostics"]["runtime_config"]["model_name"], "debug-model")


if __name__ == "__main__":
    unittest.main()
