import unittest

from app import create_app


class ProjectionRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_debug_page_loads(self):
        response = self.client.get("/modules/projection/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Projection", response.data)
        self.assertIn(b"MDS Debug Plot", response.data)
        self.assertIn(b"Coordinate Table", response.data)
        self.assertIn(b"setosa_001", response.data)

    def test_health_api_reports_working_module(self):
        response = self.client.get("/modules/projection/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "projection")
        self.assertEqual(response.json["data"]["status"], "working")

    def test_projection_api_returns_projection_contract(self):
        response = self.client.get("/modules/projection/api/projection")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        data = response.json["data"]
        self.assertEqual(data["method"], "mds")
        self.assertEqual(len(data["coordinates"]), 15)
        self.assertEqual(data["coordinates"][0]["point_id"], "setosa_001")
        self.assertIn("x", data["coordinates"][0])
        self.assertIn("y", data["coordinates"][0])

    def test_state_api_summarizes_projection(self):
        response = self.client.get("/modules/projection/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(response.json["data"]["module"], "projection")
        self.assertEqual(response.json["data"]["method"], "mds")
        self.assertEqual(response.json["data"]["coordinate_count"], 15)

    def test_data_projection_workflow_loads(self):
        response = self.client.get("/workflows/data-projection/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data and Projection", response.data)
        self.assertIn(b"Projection Plot", response.data)
        self.assertIn(b"15 rows by 4 columns", response.data)


if __name__ == "__main__":
    unittest.main()
