import unittest

from app import create_app


class RuleInterpretationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_workflow_page_loads(self):
        response = self.client.get("/workflows/rule-interpretation/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 8.7 Rule Interpretation", response.data)
        self.assertIn(b"wine_mat", response.data)
        self.assertIn(b"Categorized Label Guidance", response.data)
        self.assertIn(b"Rule Cards", response.data)
        self.assertIn(b"Raw feature conditions", response.data)
        self.assertIn(b"Interpretation Categories", response.data)
        self.assertIn(b"label_priority", response.data)
        self.assertIn(b"overlap_merge_signal", response.data)
        self.assertIn(b"Rank the next points", response.data)
        self.assertIn(b"1. Label These Points", response.data)
        self.assertIn(b"2. Why These Points Need Checking", response.data)
        self.assertIn(b"3. How To Label Them", response.data)
        self.assertIn(b"Suggested Label Actions", response.data)
        self.assertIn(b"Quantitative Findings", response.data)
        self.assertIn(b"Decision Rationale", response.data)
        self.assertIn(b"Label Outcome Branches", response.data)

    def test_state_api_exposes_rule_interpretation_contract(self):
        response = self.client.get("/workflows/rule-interpretation/api/state?provider_kind=mock")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["workflow"], "rule-interpretation")
        self.assertEqual(data["dataset"]["dataset_id"], "wine_mat")
        self.assertIn("rule_set", data)
        self.assertIn("interpretation", data)
        self.assertIn("interpretation_request", data)
        self.assertIn("interpretation_diagnostics", data)
        self.assertEqual(data["interpretation"]["provider_label"], "mock_rule_interpreter")
        self.assertIn("label_priority", data["interpretation"]["categories"])
        self.assertIn("known_features", data["interpretation_request"])
        self.assertIn("alcohol", data["interpretation_request"]["known_features"])
        self.assertIn("rule_guidance_metrics", data["interpretation_request"])
        self.assertGreater(len(data["interpretation"]["quantitative_findings"]), 0)
        self.assertGreater(len(data["interpretation"]["suggested_label_actions"]), 0)
        self.assertIn("category_explanation", data["interpretation"])
        self.assertGreater(len(data["interpretation"]["label_targets"]), 0)
        self.assertGreater(len(data["interpretation"]["suspicion_reasons"]), 0)
        self.assertGreater(len(data["interpretation"]["point_label_guidance"]), 0)
        self.assertIn("label_candidate_point_profiles", data["interpretation_request"])
        self.assertIn("decision_rationale", data["interpretation"])
        self.assertGreater(len(data["interpretation"]["label_outcomes"]), 0)
        action = data["interpretation"]["suggested_label_actions"][0]
        self.assertIn("hypothesis", action)
        self.assertIn("why_this_action", action)
        self.assertIn("expected_outcomes", action)
        self.assertIn("risk_note", action)

    def test_focus_category_button_path_returns_category_specific_explanation(self):
        response = self.client.get(
            "/workflows/rule-interpretation/api/state?provider_kind=mock&focus_category=feature_label_strategy"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["focus_category"], "feature_label_strategy")
        self.assertEqual(data["interpretation"]["categories"], ["feature_label_strategy"])
        self.assertIn("raw feature cutoffs", data["interpretation"]["summary"])
        self.assertEqual(data["interpretation_request"]["focus_category"], "feature_label_strategy")

    def test_invalid_focus_category_returns_error_envelope(self):
        response = self.client.get("/workflows/rule-interpretation/api/state?focus_category=not_real")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_parameters")

    def test_invalid_provider_returns_error_envelope(self):
        response = self.client.get("/workflows/rule-interpretation/api/state?provider_kind=ollama")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_parameters")


if __name__ == "__main__":
    unittest.main()
