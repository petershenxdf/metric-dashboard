import unittest

from app import create_app


class AlgorithmAdapterRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/algorithm-adapters/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Algorithm Adapters", response.data)
        self.assertIn(b"SSDBCODI", response.data)
        self.assertIn(b"ssdbcodi", response.data)
        self.assertIn(b"default_analysis_outlier_debug", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/algorithm-adapters/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "algorithm-adapters")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_outliers_api_returns_scores(self):
        response = self.client.get("/modules/algorithm-adapters/api/outliers")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["algorithm"], "ssdbcodi_numpy")
        self.assertEqual(len(data["scores"]), 18)
        self.assertIn("outlier_point_ids", data)
        self.assertGreaterEqual(len(data["outlier_point_ids"]), 1)

    def test_clusters_api_returns_assignments_for_non_outliers(self):
        response = self.client.get("/modules/algorithm-adapters/api/clusters?n_clusters=2")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["algorithm"], "ssdbcodi_numpy")
        self.assertEqual(data["n_clusters"], 2)
        self.assertGreater(len(data["assignments"]), 0)
        self.assertIn("excluded_outlier_point_ids", data)

    def test_analysis_api_returns_execution_order(self):
        response = self.client.get("/modules/algorithm-adapters/api/analysis?n_clusters=3")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        diagnostics = response.json["data"]["diagnostics"]
        self.assertEqual(diagnostics["provider"], "ssdbcodi")
        self.assertEqual(diagnostics["legacy_provider"], "sequential_lof_then_kmeans")
        self.assertEqual(diagnostics["execution_order"], ["kmeans_bootstrap", "ssdbcodi_integrated"])

    def test_invalid_cluster_count_returns_error_envelope(self):
        response = self.client.get("/modules/algorithm-adapters/api/clusters?n_clusters=0")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["error"]["code"], "invalid_parameters")

    def test_default_analysis_workflow_loads(self):
        response = self.client.get("/workflows/default-analysis/?n_clusters=2")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Default Analysis", response.data)
        self.assertIn(b"SSDBCODI", response.data)
        self.assertIn(b"default_analysis_outlier_debug", response.data)


if __name__ == "__main__":
    unittest.main()
