import unittest

from app import create_app
from app.modules.labeling.state import reset_debug_store_for_context
from app.modules.selection.service import get_selection_context
from app.modules.selection.state import get_debug_store, reset_debug_store


class LabelingRouteTests(unittest.TestCase):
    def setUp(self):
        reset_debug_store()
        context = get_selection_context(get_debug_store())
        reset_debug_store_for_context(context)
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/labeling/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Labeling", response.data)
        self.assertIn(b"Current Selection Context", response.data)
        self.assertIn(b"Structured Feedback", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/labeling/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "labeling")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_state_api_returns_annotation_state(self):
        response = self.client.get("/modules/labeling/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["dataset_id"], "selection_iris_debug")
        self.assertEqual(response.json["data"]["annotation_count"], 0)

    def test_apply_api_creates_cluster_annotation(self):
        response = self.client.post(
            "/modules/labeling/api/apply",
            json={"action": "assign_cluster", "label_value": "cluster_2"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["annotation"]["label_type"], "cluster")
        self.assertEqual(data["annotation"]["label_value"], "cluster_2")
        self.assertEqual(data["state"]["structured_feedback"][0]["instruction_type"], "assign_cluster")

    def test_apply_api_creates_outlier_annotation(self):
        response = self.client.post(
            "/modules/labeling/api/apply",
            json={"action": "mark_outlier"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["annotation"]["label_type"], "outlier")
        self.assertEqual(response.json["data"]["state"]["structured_feedback"][0]["instruction_type"], "is_outlier")

    def test_unknown_point_returns_error_envelope(self):
        response = self.client.post(
            "/modules/labeling/api/apply",
            json={"action": "assign_cluster", "label_value": "cluster_2", "point_ids": ["not-real"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_labeling_action")

    def test_reset_api_clears_annotations(self):
        self.client.post(
            "/modules/labeling/api/apply",
            json={"action": "assign_cluster", "label_value": "cluster_2"},
        )

        response = self.client.post("/modules/labeling/api/reset", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["annotation_count"], 0)

    def test_selection_labeling_workflow_loads(self):
        response = self.client.get("/workflows/selection-labeling/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Selection and Labeling", response.data)
        self.assertIn(b"Structured Feedback", response.data)


if __name__ == "__main__":
    unittest.main()
