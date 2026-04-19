import unittest

from app import create_app


class ProviderFeedbackWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_workflow_page_loads(self):
        response = self.client.get("/workflows/provider-feedback/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Step 6.5 Provider Feedback Lab", response.data)
        self.assertIn(b"algorithm_adapters", response.data)
        self.assertIn(b"SSDBCODI", response.data)

    def test_state_api_exposes_adapter_and_standalone_provider_payloads(self):
        response = self.client.get("/workflows/provider-feedback/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["module_boundary"], "algorithm_adapters")
        self.assertEqual(data["active_provider"], "ssdbcodi")
        self.assertIn("adapter_analysis", data)
        self.assertIn("ssdbcodi_result", data)
        self.assertIn("point_scores", data["ssdbcodi_result"])


if __name__ == "__main__":
    unittest.main()
