import unittest

from app import create_app


class AnalysisSelectionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.post("/workflows/analysis-selection/api/reset", json={})

    def test_workflow_page_loads(self):
        response = self.client.get("/workflows/analysis-selection/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 1-4 Visual Test", response.data)
        self.assertIn(b"wide_gap_analysis_debug", response.data)
        self.assertIn(b"alpha_01", response.data)
        self.assertIn(b"drag a rectangle", response.data)
        self.assertIn(b"Saved Selection Groups", response.data)

    def test_combined_state_api_returns_step_1_to_4_payloads(self):
        response = self.client.get("/workflows/analysis-selection/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["dataset"]["dataset_id"], "wide_gap_analysis_debug")
        self.assertIn("projection", data)
        self.assertIn("outliers", data)
        self.assertIn("clusters", data)
        self.assertIn("selection", data)
        self.assertIn("selection_context", data)

    def test_select_adds_to_workflow_selection(self):
        response = self.client.post(
            "/workflows/analysis-selection/api/select",
            json={"point_ids": ["alpha_02"], "source": "point_click", "mode": "additive"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertIn("alpha_02", selected)

    def test_deselect_api_supports_remove_mode(self):
        response = self.client.post(
            "/workflows/analysis-selection/api/deselect",
            json={"point_ids": ["alpha_01"], "source": "rectangle", "mode": "subtractive"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertNotIn("alpha_01", selected)

    def test_group_round_trip_works_on_workflow_dataset(self):
        save_response = self.client.post(
            "/workflows/analysis-selection/api/groups",
            json={
                "group_name": "Workflow focus",
                "point_ids": ["beta_01", "outlier_west"],
            },
        )
        group_id = save_response.json["data"]["group"]["group_id"]

        select_response = self.client.post(f"/workflows/analysis-selection/api/groups/{group_id}/select")
        selected = select_response.json["data"]["selection"]["state"]["selected_point_ids"]
        self.assertEqual(selected, ["beta_01", "outlier_west"])

        delete_response = self.client.delete(f"/workflows/analysis-selection/api/groups/{group_id}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json["data"]["groups"], [])

    def test_dataset_dropdown_can_use_original_outlier_fixture(self):
        response = self.client.get("/workflows/analysis-selection/?dataset_id=default_analysis_outlier_debug")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"default_analysis_outlier_debug", response.data)
        self.assertIn(b"cluster_a_001", response.data)

    def test_invalid_dataset_id_falls_back_to_default_workflow_dataset(self):
        response = self.client.get("/workflows/analysis-selection/api/state?dataset_id=not-real")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["dataset"]["dataset_id"], "wide_gap_analysis_debug")


if __name__ == "__main__":
    unittest.main()
