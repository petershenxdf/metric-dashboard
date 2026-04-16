import unittest

from app import create_app
from app.modules.ssdbcodi.fixtures import (
    MOONS_FIXTURE_DATASET_ID,
    ssdbcodi_dataset_options,
)
from app.modules.ssdbcodi.store import reset_debug_store


class SsdbcodiRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        for option in ssdbcodi_dataset_options():
            reset_debug_store(option["dataset_id"])
            self.client.post(
                "/modules/ssdbcodi/api/reset",
                json={"dataset_id": option["dataset_id"]},
            )

    def test_module_is_registered(self):
        response = self.client.get("/modules/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SSDBCODI", response.data)

    def test_debug_page_loads_with_bootstrap_run(self):
        response = self.client.get("/modules/ssdbcodi/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SSDBCODI", response.data)
        self.assertIn(b"Dataset", response.data)
        self.assertIn(b"Two moons", response.data)
        self.assertIn(b"Bootstrap k", response.data)
        self.assertIn(b"updateSelectionUi", response.data)
        self.assertIn(b"label saved and previewed", response.data)

    def test_debug_page_can_preview_moons_dataset(self):
        response = self.client.get(
            f"/modules/ssdbcodi/?dataset_id={MOONS_FIXTURE_DATASET_ID}&n_clusters=2"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(MOONS_FIXTURE_DATASET_ID.encode("utf-8"), response.data)
        self.assertIn(b"moon_upper_01", response.data)

    def test_debug_page_does_not_store_history_until_explicit_run(self):
        self.client.get("/modules/ssdbcodi/")
        response = self.client.get("/modules/ssdbcodi/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json["data"]["latest_run_id"])
        self.assertEqual(response.json["data"]["history"], [])

    def test_health_api_reports_working(self):
        response = self.client.get("/modules/ssdbcodi/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "ssdbcodi")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_run_api_returns_scores(self):
        response = self.client.post(
            "/modules/ssdbcodi/api/run",
            json={"n_clusters": 3, "min_pts": 3, "alpha": 0.4, "beta": 0.3},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertIn("run_id", data)
        self.assertIn("point_scores", data)
        self.assertGreater(len(data["point_scores"]), 0)
        self.assertEqual(data["cluster_counts"], {
            "cluster_1": 6,
            "cluster_2": 6,
            "cluster_3": 6,
        })
        for score in data["point_scores"]:
            self.assertIn("r_score", score)
            self.assertIn("l_score", score)
            self.assertIn("sim_score", score)
            self.assertIn("t_score", score)

    def test_run_api_supports_dataset_selection(self):
        response = self.client.post(
            "/modules/ssdbcodi/api/run",
            json={"dataset_id": MOONS_FIXTURE_DATASET_ID, "n_clusters": 2},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        point_ids = {score["point_id"] for score in response.json["data"]["point_scores"]}
        self.assertIn("moon_upper_01", point_ids)

    def test_state_api_after_run_includes_summary(self):
        self.client.post("/modules/ssdbcodi/api/run", json={})
        response = self.client.get("/modules/ssdbcodi/api/state")

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["module"], "ssdbcodi")
        self.assertIsNotNone(data["latest_run_id"])
        self.assertIn("cluster_counts", data)

    def test_scores_api_requires_prior_run(self):
        response = self.client.get("/modules/ssdbcodi/api/scores")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "no_result")

    def test_label_api_then_rerun_uses_user_seeds(self):
        self.client.post(
            "/modules/ssdbcodi/api/select",
            json={"point_ids": ["ring_a_01"], "source": "point_click"},
        )
        label_response = self.client.post(
            "/modules/ssdbcodi/api/label",
            json={
                "action": "assign_cluster",
                "label_value": "cluster_1",
            },
        )
        self.assertTrue(label_response.json["ok"])
        self.assertTrue(label_response.json["data"]["pending_run"])
        self.assertNotIn("result", label_response.json["data"])
        state_after_label = self.client.get("/modules/ssdbcodi/api/state")
        self.assertIsNone(state_after_label.json["data"]["latest_run_id"])

        self.client.post("/modules/ssdbcodi/api/clear-selection", json={})
        self.client.post(
            "/modules/ssdbcodi/api/select",
            json={"point_ids": ["ring_b_01"], "source": "point_click"},
        )
        self.client.post(
            "/modules/ssdbcodi/api/label",
            json={
                "action": "assign_cluster",
                "label_value": "cluster_2",
            },
        )

        run_response = self.client.post("/modules/ssdbcodi/api/run", json={})
        self.assertTrue(run_response.json["ok"])
        data = run_response.json["data"]
        self.assertTrue(data["bootstrap_used"])
        cluster_ids = {score["cluster_id"] for score in data["point_scores"] if not score["is_outlier"]}
        self.assertIn("cluster_1", cluster_ids)
        self.assertIn("cluster_2", cluster_ids)

    def test_relabeling_one_point_preserves_far_bootstrap_clusters(self):
        self.client.post(
            "/modules/ssdbcodi/api/select",
            json={"point_ids": ["ring_c_04"], "source": "point_click"},
        )
        label_response = self.client.post(
            "/modules/ssdbcodi/api/label",
            json={
                "action": "assign_cluster",
                "label_value": "cluster_3",
                "n_clusters": 4,
            },
        )
        self.assertTrue(label_response.json["ok"])

        run_response = self.client.post(
            "/modules/ssdbcodi/api/run",
            json={"n_clusters": 4},
        )

        self.assertTrue(run_response.json["ok"])
        scores = run_response.json["data"]["point_scores"]
        cluster_ids = {score["cluster_id"] for score in scores if not score["is_outlier"]}
        self.assertIn("cluster_1", cluster_ids)
        self.assertIn("cluster_2", cluster_ids)
        ring_c_04 = next(score for score in scores if score["point_id"] == "ring_c_04")
        self.assertEqual(ring_c_04["cluster_id"], "cluster_3")

    def test_label_api_requires_current_selection(self):
        response = self.client.post(
            "/modules/ssdbcodi/api/label",
            json={"action": "assign_cluster", "label_value": "cluster_1"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])

    def test_label_api_rejects_unknown_point_selection(self):
        response = self.client.post(
            "/modules/ssdbcodi/api/select",
            json={"point_ids": ["missing_point"], "source": "point_click"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])

    def test_selection_group_roundtrip_uses_selection_module_contract(self):
        self.client.post(
            "/modules/ssdbcodi/api/select",
            json={"point_ids": ["ring_a_01", "ring_a_02"], "source": "rectangle"},
        )
        save = self.client.post(
            "/modules/ssdbcodi/api/groups",
            json={"group_name": "ring a pair"},
        )
        self.assertTrue(save.json["ok"])
        group_id = save.json["data"]["group"]["group_id"]

        self.client.post("/modules/ssdbcodi/api/clear-selection", json={})
        select = self.client.post(f"/modules/ssdbcodi/api/groups/{group_id}/select")

        self.assertTrue(select.json["ok"])
        self.assertEqual(
            select.json["data"]["selection"]["state"]["selected_point_ids"],
            ["ring_a_01", "ring_a_02"],
        )

    def test_invalid_parameters_return_error_envelope(self):
        response = self.client.post(
            "/modules/ssdbcodi/api/run",
            json={"contamination": 0.99},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_parameters")

    def test_reset_clears_store_and_labels(self):
        self.client.post("/modules/ssdbcodi/api/run", json={})
        self.client.post(
            "/modules/ssdbcodi/api/select",
            json={"point_ids": ["ring_a_01"], "source": "point_click"},
        )
        self.client.post(
            "/modules/ssdbcodi/api/label",
            json={
                "action": "assign_cluster",
                "label_value": "cluster_1",
            },
        )
        reset = self.client.post("/modules/ssdbcodi/api/reset")
        self.assertTrue(reset.json["ok"])

        scores = self.client.get("/modules/ssdbcodi/api/scores")
        self.assertEqual(scores.status_code, 400)


if __name__ == "__main__":
    unittest.main()
