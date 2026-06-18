import unittest

from app import create_app


class WineDashboardWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.post("/workflows/wine-dashboard/api/reset-selection", json={})
        self.client.post("/workflows/wine-dashboard/api/reset-labels", json={})

    def dashboard_url(self, path="/"):
        return f"/workflows/wine-dashboard{path}?provider_kind=mock"

    def test_dashboard_page_loads_with_wine_data_and_rule_panel(self):
        response = self.client.get(self.dashboard_url("/"))

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 8.8 Integrated Rule Dashboard", response.data)
        self.assertIn(b"wine_mat", response.data)
        self.assertIn(b"alcohol", response.data)
        self.assertIn(b"Wine Projection", response.data)
        self.assertIn(b"Rule Cards", response.data)
        self.assertIn(b"Raw feature conditions", response.data)
        self.assertIn(b"Rule Interpretation", response.data)
        self.assertIn(b"DeepSeek V4 Pro used", response.data)
        self.assertIn(b"No typical case now", response.data)
        self.assertIn(b"Recommended points are highlighted on the scatterplot", response.data)
        self.assertIn(b"data-guide-point-id", response.data)
        self.assertIn(b"guidance-ring", response.data)
        self.assertIn(b"data-selection-count", response.data)
        self.assertIn(b"updateSelectionUi", response.data)
        self.assertIn(b"Why These Points Matter", response.data)
        self.assertIn(b"Label These Points", response.data)
        self.assertIn(b"How To Label", response.data)
        self.assertIn(b"Open State API", response.data)
        self.assertIn(b"Show Inline Debug JSON", response.data)
        self.assertNotIn(b'"interpretation_preview"', response.data)
        self.assertNotIn(b"unusualness score", response.data)

    def test_state_api_exposes_integrated_step_1_to_8_6_payload(self):
        response = self.client.get(self.dashboard_url("/api/state"))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["workflow"], "wine-dashboard")
        self.assertEqual(data["dataset"]["dataset_id"], "wine_mat")
        self.assertEqual(len(data["dataset"]["points"]), 129)
        self.assertIn("alcohol", data["feature_matrix"]["feature_names"])
        self.assertIn("projection", data)
        self.assertIn("scatterplot", data)
        self.assertIn("selection", data)
        self.assertIn("labeling", data)
        self.assertIn("clusters", data)
        self.assertIn("outliers", data)
        self.assertIn("rule_set", data)
        self.assertIn("interpretation_preview", data)
        self.assertIn("interpretation_request", data)
        self.assertIn("interpretation_diagnostics", data)
        self.assertIn("guidance_point_ids", data)
        self.assertIn("category_status", data)
        self.assertEqual(data["provider_kind"], "mock")
        self.assertTrue(any(item["has_typical_case"] for item in data["category_status"]))
        self.assertTrue(any(not item["has_typical_case"] for item in data["category_status"]))
        self.assertGreater(len(data["guidance_point_ids"]), 0)
        self.assertGreater(len(data["interpretation_preview"]["label_targets"]), 0)
        self.assertGreater(len(data["interpretation_preview"]["point_label_guidance"]), 0)
        self.assertFalse(data["chatbox"]["included"])

    def test_category_without_typical_case_shows_empty_guidance(self):
        response = self.client.get(
            "/workflows/wine-dashboard/?provider_kind=mock&focus_category=overlap_merge_signal"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No Typical Case For This Category", response.data)
        self.assertIn(b"No two rules currently share matched points", response.data)

    def test_debug_json_is_opt_in(self):
        response = self.client.get(
            "/workflows/wine-dashboard/?provider_kind=mock&focus_category=label_priority&debug=1"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"interpretation_preview"', response.data)

    def test_deepseek_is_skipped_when_category_has_no_typical_case(self):
        response = self.client.get(
            "/workflows/wine-dashboard/api/state?provider_kind=deepseek&focus_category=overlap_merge_signal"
        )

        self.assertEqual(response.status_code, 200)
        diagnostics = response.json["data"]["interpretation_diagnostics"]
        self.assertTrue(diagnostics["deepseek_skipped"])
        self.assertEqual(diagnostics["skip_reason"], "no_typical_case_for_category")
        self.assertFalse(diagnostics["used_fallback"])

    def test_rule_conditions_use_wine_raw_features_not_projection_axes(self):
        response = self.client.get(self.dashboard_url("/api/state"))

        self.assertEqual(response.status_code, 200)
        rules = response.json["data"]["rule_set"]["rules"]
        condition_features = {
            condition["feature"]
            for rule in rules
            for condition in rule["conditions"]
        }
        self.assertGreater(len(condition_features), 0)
        self.assertNotIn("x", condition_features)
        self.assertNotIn("y", condition_features)
        self.assertIn("alcohol", response.json["data"]["rule_set"]["diagnostics"]["raw_feature_names"])

    def test_select_adds_wine_points(self):
        response = self.client.post(
            "/workflows/wine-dashboard/api/select",
            json={"point_ids": ["wine_010"], "source": "point_click", "mode": "additive", "provider_kind": "mock"},
        )

        self.assertEqual(response.status_code, 200)
        selected = response.json["data"]["state"]["selected_point_ids"]
        self.assertIn("wine_010", selected)
        self.assertIn("context", response.json["data"])
        self.assertNotIn("interpretation_preview", response.json["data"])
        self.assertNotIn("rule_set", response.json["data"])

    def test_label_api_updates_effective_state_and_rule_set(self):
        response = self.client.post(
            "/workflows/wine-dashboard/api/label",
            json={"action": "assign_cluster", "label_value": "cluster_2", "n_clusters": 3, "provider_kind": "mock"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["annotation"]["label_type"], "cluster")
        self.assertEqual(data["state"]["labeling"]["annotation_count"], 1)
        assignments = {
            assignment["point_id"]: assignment["cluster_id"]
            for assignment in data["state"]["clusters"]["assignments"]
        }
        self.assertEqual(assignments["wine_001"], "cluster_2")
        self.assertEqual(data["state"]["rule_set"]["dataset_id"], "wine_mat")

    def test_invalid_tree_settings_return_error_envelope(self):
        response = self.client.get("/workflows/wine-dashboard/api/state?max_depth=0&provider_kind=mock")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_parameters")


if __name__ == "__main__":
    unittest.main()
