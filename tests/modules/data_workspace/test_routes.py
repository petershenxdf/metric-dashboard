import unittest

from app import create_app


class DataWorkspaceRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/data-workspace/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data Workspace", response.data)
        self.assertIn(b"iris_debug_sample", response.data)
        self.assertIn(b"setosa_001", response.data)
        self.assertIn(b"Feature Matrix Preview", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/data-workspace/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "data-workspace")
        self.assertEqual(response.json["data"]["status"], "working")
        self.assertEqual(response.json["diagnostics"]["dependency_mode"], "fixture")

    def test_dataset_api_returns_dataset_contract(self):
        response = self.client.get("/modules/data-workspace/api/dataset")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["dataset_id"], "iris_debug_sample")
        self.assertEqual(len(data["points"]), 15)
        self.assertEqual(
            data["feature_names"],
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )

    def test_matrix_api_returns_feature_matrix_contract(self):
        response = self.client.get("/modules/data-workspace/api/matrix")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(len(data["point_ids"]), 15)
        self.assertEqual(len(data["values"]), 15)
        self.assertEqual(len(data["values"][0]), 4)

    def test_state_api_summarizes_fixture(self):
        response = self.client.get("/modules/data-workspace/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["point_count"], 15)
        self.assertEqual(response.json["data"]["feature_count"], 4)
        self.assertEqual(response.json["data"]["matrix_shape"], [15, 4])


if __name__ == "__main__":
    unittest.main()
