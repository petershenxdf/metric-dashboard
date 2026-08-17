import unittest

from app import create_app
from app.modules.labeling.state import reset_debug_store_for_context
from app.modules.selection.service import get_selection_context
from app.modules.selection.state import reset_debug_store_for_dataset
from app.shared.fixtures import (
    DEFAULT_WORKFLOW_DATASET_ID,
    analysis_selection_dataset,
    analysis_selection_initial_selected_point_ids,
)


class ScatterplotRouteTests(unittest.TestCase):
    def setUp(self):
        dataset = analysis_selection_dataset(DEFAULT_WORKFLOW_DATASET_ID)
        store = reset_debug_store_for_dataset(
            dataset,
            analysis_selection_initial_selected_point_ids(dataset.dataset_id),
        )
        context = get_selection_context(store)
        reset_debug_store_for_context(context)
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/scatterplot/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Scatterplot", response.data)
        self.assertIn(b"Render Payload", response.data)
        self.assertIn(b"wide_gap_analysis_debug", response.data)
        self.assertIn(b"data-selection-rect", response.data)
        self.assertIn(b"Saved Selection Groups", response.data)
        self.assertIn(b"SSDBCODI bootstrap clusters", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/scatterplot/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "scatterplot")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_render_payload_api_returns_points(self):
        response = self.client.get("/modules/scatterplot/api/render-payload")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["dataset_id"], "wide_gap_analysis_debug")
        self.assertGreater(len(data["points"]), 0)
        self.assertIn("selected", data["points"][0])
        self.assertIn("is_outlier", data["points"][0])

    def test_toggle_api_updates_selection_through_selection_boundary(self):
        response = self.client.post(
            "/modules/scatterplot/api/toggle",
            json={"point_ids": ["alpha_02"], "source": "point_click"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["selection"]["state"]["selected_point_ids"]
        self.assertIn("alpha_02", selected)

    def test_rectangle_select_api_keeps_rectangle_source(self):
        response = self.client.post(
            "/modules/scatterplot/api/select",
            json={"point_ids": ["alpha_02", "beta_01"], "source": "rectangle", "mode": "additive"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]["selection"]
        self.assertEqual(data["action"]["source"], "rectangle")
        self.assertIn("beta_01", data["state"]["selected_point_ids"])

    def test_scatterplot_group_api_round_trip(self):
        save_response = self.client.post(
            "/modules/scatterplot/api/groups",
            json={"group_name": "Scatter focus", "point_ids": ["alpha_02", "beta_01"]},
        )
        self.assertEqual(save_response.status_code, 200)
        group_id = save_response.json["data"]["group"]["group_id"]

        select_response = self.client.post(f"/modules/scatterplot/api/groups/{group_id}/select")
        self.assertEqual(select_response.status_code, 200)
        self.assertEqual(
            select_response.json["data"]["selection"]["state"]["selected_point_ids"],
            ["alpha_02", "beta_01"],
        )

        delete_response = self.client.delete(f"/modules/scatterplot/api/groups/{group_id}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json["data"]["groups"], [])

if __name__ == "__main__":
    unittest.main()
