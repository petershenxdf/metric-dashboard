import unittest

from app import create_app


class AnalysisLabelingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.post("/workflows/analysis-labeling/api/reset-selection", json={})
        self.client.post("/workflows/analysis-labeling/api/reset-labels", json={})

    def test_workflow_page_loads(self):
        response = self.client.get("/workflows/analysis-labeling/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 1-5 Visual Test", response.data)
        self.assertIn(b"wide_gap_analysis_debug", response.data)
        self.assertIn(b"alpha_01", response.data)
        self.assertIn(b"Structured Feedback", response.data)
        self.assertIn(b"Label Selected Points", response.data)

    def test_combined_state_api_returns_step_1_to_5_payloads(self):
        response = self.client.get("/workflows/analysis-labeling/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["dataset"]["dataset_id"], "wide_gap_analysis_debug")
        self.assertIn("projection", data)
        self.assertIn("outliers", data)
        self.assertIn("clusters", data)
        self.assertIn("selection", data)
        self.assertIn("selection_context", data)
        self.assertIn("labeling", data)
        self.assertIn("raw_clusters", data)
        self.assertIn("raw_outliers", data)

    def test_page_only_offers_cluster_and_outlier_labels(self):
        response = self.client.get("/workflows/analysis-labeling/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'value="cluster_1"', response.data)
        self.assertIn(b'value="cluster_2"', response.data)
        self.assertIn(b'value="cluster_3"', response.data)
        self.assertIn(b'value="outlier"', response.data)
        self.assertNotIn(b"interesting_group", response.data)
        self.assertNotIn(b"Assign New Class", response.data)

    def test_select_adds_to_integrated_selection(self):
        response = self.client.post(
            "/workflows/analysis-labeling/api/select",
            json={"point_ids": ["alpha_02"], "source": "point_click", "mode": "additive"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertIn("alpha_02", selected)

    def test_label_api_creates_annotation_for_selected_points(self):
        response = self.client.post(
            "/workflows/analysis-labeling/api/label",
            json={"action": "assign_cluster", "label_value": "cluster_2", "n_clusters": 3},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["annotation"]["label_type"], "cluster")
        self.assertEqual(data["annotation"]["label_value"], "cluster_2")
        self.assertEqual(data["state"]["labeling"]["annotation_count"], 1)
        self.assertEqual(data["state"]["labeling"]["structured_feedback"][0]["instruction_type"], "assign_cluster")

    def test_cluster_label_changes_effective_cluster_state(self):
        response = self.client.post(
            "/workflows/analysis-labeling/api/label",
            json={"action": "assign_cluster", "label_value": "cluster_2", "n_clusters": 3},
        )

        self.assertEqual(response.status_code, 200)
        assignments = {
            assignment["point_id"]: assignment["cluster_id"]
            for assignment in response.json["data"]["state"]["clusters"]["assignments"]
        }
        raw_assignments = {
            assignment["point_id"]: assignment["cluster_id"]
            for assignment in response.json["data"]["state"]["raw_clusters"]["assignments"]
        }
        self.assertEqual(assignments["alpha_01"], "cluster_2")
        self.assertEqual(raw_assignments["alpha_01"], "cluster_1")
        self.assertNotIn("alpha_01", response.json["data"]["state"]["outliers"]["outlier_point_ids"])

    def test_outlier_label_changes_effective_outlier_state(self):
        self.client.post(
            "/workflows/analysis-labeling/api/select",
            json={"point_ids": ["beta_01"], "source": "point_click", "mode": "additive"},
        )
        response = self.client.post(
            "/workflows/analysis-labeling/api/label",
            json={"action": "mark_outlier", "n_clusters": 3, "point_ids": ["beta_01"]},
        )

        self.assertEqual(response.status_code, 200)
        state = response.json["data"]["state"]
        assignments = {
            assignment["point_id"]: assignment["cluster_id"]
            for assignment in state["clusters"]["assignments"]
        }
        self.assertIn("beta_01", state["outliers"]["outlier_point_ids"])
        self.assertNotIn("beta_01", state["raw_outliers"]["outlier_point_ids"])
        self.assertNotIn("beta_01", assignments)

    def test_rejects_labels_outside_current_cluster_options(self):
        response = self.client.post(
            "/workflows/analysis-labeling/api/label",
            json={"action": "assign_cluster", "label_value": "cluster_4", "n_clusters": 3},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "invalid_labeling_action")

    def test_rejects_new_class_labels_in_analysis_labeling_workflow(self):
        response = self.client.post(
            "/workflows/analysis-labeling/api/label",
            json={"action": "assign_new_class", "label_value": "interesting_group", "n_clusters": 3},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "invalid_labeling_action")

    def test_clear_labels_keeps_selection_but_removes_annotations(self):
        self.client.post(
            "/workflows/analysis-labeling/api/label",
            json={"action": "mark_outlier", "n_clusters": 3},
        )

        response = self.client.post("/workflows/analysis-labeling/api/clear-labels", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["state"]["labeling"]["annotation_count"], 0)


if __name__ == "__main__":
    unittest.main()
