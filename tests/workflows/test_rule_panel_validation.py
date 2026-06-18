import unittest

from app import create_app


class RulePanelValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_workflow_page_loads(self):
        response = self.client.get("/workflows/rule-panel-validation/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 8.6 Rule Panel Validation", response.data)
        self.assertIn(b"SSDBCODI", response.data)
        self.assertIn(b"Rule Cards", response.data)
        self.assertIn(b"wine_mat", response.data)
        self.assertIn(b"alcohol", response.data)
        self.assertIn(b"Raw feature conditions", response.data)

    def test_state_api_exposes_analysis_and_rule_set(self):
        response = self.client.get("/workflows/rule-panel-validation/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["workflow"], "rule-panel-validation")
        self.assertIn("analysis", data)
        self.assertIn("rule_set", data)
        self.assertIn("interpretation_preview", data)
        self.assertEqual(data["rule_set"]["dataset_id"], "wine_mat")
        self.assertIn("alcohol", data["feature_matrix"]["feature_names"])
        self.assertEqual(data["rule_set"]["diagnostics"]["source_point_count"], 129)
        condition_features = {
            condition["feature"]
            for rule in data["rule_set"]["rules"]
            for condition in rule["conditions"]
        }
        self.assertGreater(len(condition_features), 0)
        self.assertNotIn("x", condition_features)
        self.assertNotIn("y", condition_features)
        self.assertEqual(response.json["diagnostics"]["source_of_truth"], "ssdbcodi")


if __name__ == "__main__":
    unittest.main()
