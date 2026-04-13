import unittest

from app import create_app
from app.modules.selection.state import reset_debug_store


class SelectionRouteTests(unittest.TestCase):
    def setUp(self):
        reset_debug_store()
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/selection/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Selection", response.data)
        self.assertIn(b"Supported Selection Types", response.data)
        self.assertIn(b"Saved Selection Groups", response.data)
        self.assertIn(b"setosa_001", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/selection/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "selection")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_state_api_returns_selection_state(self):
        response = self.client.get("/modules/selection/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["dataset_id"], "selection_iris_debug")
        self.assertIn("selected_point_ids", data)
        self.assertIn("unselected_point_ids", data)

    def test_select_api_adds_points(self):
        response = self.client.post(
            "/modules/selection/api/select",
            json={"point_ids": ["setosa_002"], "source": "api"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertIn("setosa_002", selected)

    def test_deselect_api_removes_points(self):
        response = self.client.post(
            "/modules/selection/api/deselect",
            json={"point_ids": ["setosa_001"], "source": "api"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertNotIn("setosa_001", selected)

    def test_replace_api_replaces_selection(self):
        response = self.client.post(
            "/modules/selection/api/replace",
            json={"point_ids": ["virginica_001"], "source": "manual_list"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertEqual(selected, ["virginica_001"])

    def test_toggle_api_switches_selection(self):
        response = self.client.post(
            "/modules/selection/api/toggle",
            json={"point_ids": ["setosa_001", "virginica_002"], "source": "point_click"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertNotIn("setosa_001", selected)
        self.assertIn("virginica_002", selected)

    def test_clear_api_clears_selection(self):
        response = self.client.post("/modules/selection/api/clear", json={"source": "api"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["state"]["selected_point_ids"], [])

    def test_unknown_point_returns_error_envelope(self):
        response = self.client.post(
            "/modules/selection/api/select",
            json={"point_ids": ["not-real"], "source": "api"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_selection_action")

    def test_groups_api_starts_empty(self):
        response = self.client.get("/modules/selection/api/groups")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["groups"], [])
        self.assertEqual(response.json["data"]["group_count"], 0)

    def test_save_group_api_saves_current_selection(self):
        response = self.client.post(
            "/modules/selection/api/groups",
            json={"group_name": "Starter pair"},
        )

        self.assertEqual(response.status_code, 200)
        group = response.json["data"]["group"]
        self.assertEqual(group["group_name"], "Starter pair")
        self.assertEqual(group["point_ids"], ["setosa_001", "versicolor_001"])

    def test_select_group_api_replaces_selection_with_saved_group(self):
        save_response = self.client.post(
            "/modules/selection/api/groups",
            json={
                "group_name": "Virginica focus",
                "point_ids": ["virginica_001", "virginica_002"],
            },
        )
        group_id = save_response.json["data"]["group"]["group_id"]

        response = self.client.post(f"/modules/selection/api/groups/{group_id}/select")

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["selection"]["state"]["selected_point_ids"]
        self.assertEqual(selected, ["virginica_001", "virginica_002"])

    def test_delete_group_api_removes_saved_group(self):
        save_response = self.client.post(
            "/modules/selection/api/groups",
            json={"group_name": "Delete me", "point_ids": ["setosa_002"]},
        )
        group_id = save_response.json["data"]["group"]["group_id"]

        response = self.client.delete(f"/modules/selection/api/groups/{group_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["deleted_group"]["group_id"], group_id)
        self.assertEqual(response.json["data"]["groups"], [])

    def test_duplicate_group_name_returns_error_envelope(self):
        self.client.post(
            "/modules/selection/api/groups",
            json={"group_name": "Focus", "point_ids": ["setosa_002"]},
        )
        response = self.client.post(
            "/modules/selection/api/groups",
            json={"group_name": "focus", "point_ids": ["setosa_003"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_selection_group")

    def test_selection_context_workflow_loads(self):
        response = self.client.get("/workflows/selection-context/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Selection Context", response.data)
        self.assertIn(b"Selection Context Payload", response.data)


if __name__ == "__main__":
    unittest.main()
