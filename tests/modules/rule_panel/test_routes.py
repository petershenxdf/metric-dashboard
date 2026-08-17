import unittest

from app import create_app


class RulePanelRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/rule-panel/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rule Panel", response.data)
        self.assertIn(b"SSDBCODI", response.data)
        self.assertIn(b"explanation-only", response.data)
        self.assertIn(b"wine_mat", response.data)
        self.assertIn(b"alcohol", response.data)
        self.assertIn(b"Raw feature conditions", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/rule-panel/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "rule-panel")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_rules_api_returns_rule_set(self):
        response = self.client.get("/modules/rule-panel/api/rules?max_depth=3&min_samples_leaf=1")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["model"]["source_of_truth"], "ssdbcodi")
        self.assertEqual(data["model"]["role"], "explanation_only")
        self.assertEqual(data["dataset_id"], "wine_mat")
        self.assertIn("alcohol", data["diagnostics"]["raw_feature_names"])
        self.assertEqual(data["diagnostics"]["source_point_count"], 129)
        self.assertGreater(len(data["rules"]), 0)
        condition_features = {
            condition["feature"]
            for rule in data["rules"]
            for condition in rule["conditions"]
        }
        self.assertGreater(len(condition_features), 0)
        self.assertNotIn("x", condition_features)
        self.assertNotIn("y", condition_features)

    def test_invalid_tree_depth_returns_error_envelope(self):
        response = self.client.get("/modules/rule-panel/api/rules?max_depth=0")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_parameters")


if __name__ == "__main__":
    unittest.main()
